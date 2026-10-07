"""Train -> select on validation -> evaluate on test once -> predict with the selected model.

* Every model reads the same split (``split.csv``) and the registry stores its fingerprint.
* The best model is chosen on VALIDATION macro-F1 among the models that were really trained.
* ``predict`` loads that best model, which carries its own normalizer, so there is no train/serve skew.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data.split import part
from .evaluate import macro_f1, mcnemar, scores, slice_scores
from .models import build_model, load_model
from .models.linear import TfidfModel
from .text.normalize import Normalizer, has_emoji_or_emoticon, has_negation, has_slang

REGISTRY_FILE = "registry.json"
LABEL_IDS = {"negative": 0, "positive": 1}
ID_LABELS = {v: k for k, v in LABEL_IDS.items()}


def _xy(df: pd.DataFrame):
    return df["text"].tolist(), np.array([LABEL_IDS[label] for label in df["label"]])


def train_models(split_df: pd.DataFrame, manifest: dict, names: list[str], model_dir: str | Path, seed: int = 42,
                 model_options: dict | None = None, log=print) -> dict:
    model_dir = Path(model_dir)
    tr_x, tr_y = _xy(part(split_df, "train"))
    va_x, va_y = _xy(part(split_df, "val"))
    entries = {}
    for name in names:
        start = time.time()
        model = build_model(name, seed=seed, **(model_options or {}).get(name, {}))
        model.fit(tr_x, tr_y, va_x, va_y)
        val_f1 = macro_f1(va_y, model.predict_proba(va_x).argmax(1))
        model.save(model_dir / name)
        entries[name] = {"val_macro_f1": round(val_f1, 4), "seconds": round(time.time() - start, 2)}
        log(f"{name}: val macro-F1 {val_f1:.4f} ({entries[name]['seconds']} s)")
    best = max(entries, key=lambda n: entries[n]["val_macro_f1"])
    registry = {"split_fingerprint": manifest["fingerprint"], "seed": seed, "models": entries, "best": best,
                "selected_on": "val_macro_f1"}
    old = model_dir / REGISTRY_FILE
    if old.exists():
        prev = json.loads(old.read_text(encoding="utf-8"))
        if prev.get("split_fingerprint") == manifest["fingerprint"]:
            registry["models"] = {**prev["models"], **entries}
            registry["best"] = max(registry["models"], key=lambda n: registry["models"][n]["val_macro_f1"])
    model_dir.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    log(f"best on validation: {registry['best']}")
    return registry


def _read_registry(model_dir: Path) -> dict:
    path = Path(model_dir) / REGISTRY_FILE
    if not path.exists():
        raise FileNotFoundError(f"no {REGISTRY_FILE} in {model_dir}: run 'tweetmood train' first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_slices(test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    texts = test_df["text"].tolist()
    masks = {f"source={s}": (test_df["source"] == s).to_numpy() for s in sorted(test_df["source"].unique())}
    masks["has_negation"] = np.array([has_negation(t) for t in texts])
    masks["has_slang"] = np.array([has_slang(t) for t in texts])
    masks["has_emoji_or_emoticon"] = np.array([has_emoji_or_emoticon(t) for t in texts])
    return masks


def evaluate_models(split_df: pd.DataFrame, manifest: dict, model_dir: str | Path, n_boot: int = 1000) -> dict:
    model_dir = Path(model_dir)
    registry = _read_registry(model_dir)
    if registry["split_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("the models were trained on another split (fingerprint differs)")
    test = part(split_df, "test")
    te_x, te_y = _xy(test)
    masks = test_slices(test)
    preds, report = {}, {"split_fingerprint": manifest["fingerprint"], "best": registry["best"], "models": {}}
    for name in registry["models"]:
        proba = load_model(model_dir / name).predict_proba(te_x)
        preds[name] = proba.argmax(1)
        report["models"][name] = {"val_macro_f1": registry["models"][name]["val_macro_f1"],
                                  "test": scores(te_y, proba, n_boot=n_boot),
                                  "slices": slice_scores(te_y, preds[name], masks)}
    best = registry["best"]
    report["mcnemar_vs_best"] = {n: mcnemar(te_y, preds[best], preds[n]) for n in preds if n != best}
    (model_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (model_dir / "evaluation.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    lines = [f"# Test evaluation (best on validation: `{report['best']}`)", "",
             "| model | val macro-F1 | test macro-F1 | 95 % CI | accuracy | ROC-AUC | McNemar p vs best |",
             "|---|---|---|---|---|---|---|"]
    order = sorted(report["models"], key=lambda n: -report["models"][n]["test"]["macro_f1"])
    for n in order:
        m = report["models"][n]
        t = m["test"]
        p = report["mcnemar_vs_best"].get(n, {}).get("p_value")
        lines.append(f"| {n} | {m['val_macro_f1']:.4f} | {t['macro_f1']:.4f} | {t['macro_f1_ci95'][0]:.3f} – "
                     f"{t['macro_f1_ci95'][1]:.3f} | {t['accuracy']:.4f} | {t.get('roc_auc', float('nan')):.4f} | "
                     f"{'-' if p is None else f'{p:.4f}'} |")
    slices = sorted({s for m in report["models"].values() for s in m["slices"]})
    lines += ["", "## Test macro-F1 per slice", "", "| model | " + " | ".join(slices) + " |",
              "|---|" + "---|" * len(slices)]
    for n in order:
        sl = report["models"][n]["slices"]
        cells = [f"{sl[s]['macro_f1']:.3f} (n={sl[s]['n']})" if s in sl else "-" for s in slices]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def predict(texts: list[str], model_dir: str | Path, model_name: str | None = None) -> list[dict]:
    model_dir = Path(model_dir)
    name = model_name or _read_registry(model_dir)["best"]
    proba = load_model(model_dir / name).predict_proba(texts)
    return [{"text": t, "label": ID_LABELS[int(p.argmax())], "p_positive": round(float(p[1]), 4), "model": name}
            for t, p in zip(texts, proba)]


# --- ablation of the normalizer -------------------------------------------------------------------

_STOP = {"i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its", "they", "them", "a", "an", "the",
         "and", "but", "if", "or", "as", "of", "at", "by", "for", "with", "to", "from", "in", "on", "is", "are",
         "was", "were", "be", "been", "do", "does", "did", "this", "that", "these", "those", "so", "than", "too",
         "very", "can", "will", "just", "not", "no", "nor", "never", "don't", "dont", "isn't", "aint"}


class StopwordNormalizer(Normalizer):
    """Ablation only: the prototype's order (lower-case first) plus stop-word removal, which deletes negations."""

    def normalize(self, text: str) -> str:
        lowered = str(text).lower()
        lowered = re.sub(r"https?://\S+|@\w+", " ", lowered)
        return " ".join(w for w in re.findall(r"[a-z0-9']+", lowered) if w not in _STOP)


ABLATIONS = {
    "full": Normalizer(negation_scope=True),
    "no-negation-scope": Normalizer(negation_scope=False),
    "no-slang": Normalizer(negation_scope=True, slang=False),
    "no-emoji-emoticons": Normalizer(negation_scope=True, emoji=False, emoticons=False),
    "stopword-removal": StopwordNormalizer(),
}


def ablate(split_df: pd.DataFrame, model: str = "tfidf-logreg", seed: int = 42) -> pd.DataFrame:
    """Train one TF-IDF model per normalizer variant. Report val and test macro-F1 and the test slices."""
    tr_x, tr_y = _xy(part(split_df, "train"))
    va_x, va_y = _xy(part(split_df, "val"))
    test = part(split_df, "test")
    te_x, te_y = _xy(test)
    masks = test_slices(test)
    rows = []
    for variant, norm in ABLATIONS.items():
        m = TfidfModel(model, seed=seed, normalizer=norm).fit(tr_x, tr_y, va_x, va_y)
        pred = m.predict_proba(te_x).argmax(1)
        sl = slice_scores(te_y, pred, masks)
        row = {"variant": variant, "val_macro_f1": round(m.selected["val_macro_f1"], 4),
               "test_macro_f1": round(macro_f1(te_y, pred), 4)}
        for key in ("has_negation", "has_slang", "has_emoji_or_emoticon"):
            row[key] = round(sl[key]["macro_f1"], 4) if key in sl else None
        for key in [k for k in sl if k.startswith("source=")]:
            row[key] = round(sl[key]["macro_f1"], 4)
        rows.append(row)
    return pd.DataFrame(rows)

