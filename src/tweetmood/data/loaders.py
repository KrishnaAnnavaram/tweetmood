"""Loaders for the public tweet corpora. Each returns ``text, label, source`` with binary labels.

Only the text and the label are read. User names, ids and dates are never read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

LABELS = ("negative", "positive")


class SchemaError(ValueError):
    """Raised when a frame does not match ``text, label, source``."""


@dataclass
class LoadReport:
    source: str
    rows_read: int = 0
    rows_kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, n: int) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("text", "label", "source"):
        if col not in df.columns:
            raise SchemaError(f"missing column {col!r}")
    out = df.copy()
    out["text"] = out["text"].astype(str)
    if (out["text"].str.strip() == "").any():
        raise SchemaError("empty text")
    if not out["label"].isin(LABELS).all():
        raise SchemaError(f"labels must be {LABELS}")
    if out["source"].isna().any():
        raise SchemaError("missing source")
    return out.reset_index(drop=True)


def _finish(source: str, text: pd.Series, label: pd.Series, mapping: dict) -> tuple[pd.DataFrame, LoadReport]:
    report = LoadReport(source, rows_read=len(text))
    lab = label.astype(str).str.strip().str.lower().map(mapping)
    txt = text.astype("string").str.strip()
    empty = txt.isna() | (txt == "")
    report.add("empty_text", int(empty.sum()))
    unmapped = lab.isna() & ~empty
    report.add("label_not_binary", int(unmapped.sum()))
    keep = ~empty & ~unmapped
    df = pd.DataFrame({"text": txt[keep].astype(str), "label": lab[keep], "source": source})
    report.rows_kept = len(df)
    return df.reset_index(drop=True), report


def load_sentiment140(path: str | Path, sample: int | None = None, seed: int = 42):
    """Sentiment140: six columns, no header, latin-1. ``0`` = negative, ``4`` = positive.

    ``sample`` draws a class-balanced sample WITHOUT replacement after exact duplicates are removed.
    """
    raw = pd.read_csv(path, header=None, usecols=[0, 5], names=["target", "text"], dtype=str, encoding="latin-1")
    df, report = _finish("sentiment140", raw["text"], raw["target"], {"0": "negative", "4": "positive"})
    before = len(df)
    df = df.drop_duplicates("text").reset_index(drop=True)
    report.add("exact_duplicate", before - len(df))
    if sample is not None and sample < len(df):
        per_class = sample // 2
        parts = [g.sample(n=min(per_class, len(g)), replace=False, random_state=seed) for _, g in df.groupby("label")]
        df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    report.rows_kept = len(df)
    return df, report


def load_tweeteval(folder: str | Path):
    """TweetEval sentiment: ``{train,val,test}_text.txt`` and ``_labels.txt`` (0 neg, 1 neutral, 2 pos).

    Neutral tweets are dropped. All three files are pooled. The shared splitter makes the split.
    """
    folder = Path(folder)
    texts, labels = [], []
    for part in ("train", "val", "test"):
        tpath, lpath = folder / f"{part}_text.txt", folder / f"{part}_labels.txt"
        if not tpath.exists():
            continue
        texts += tpath.read_text(encoding="utf-8").splitlines()
        labels += lpath.read_text(encoding="utf-8").splitlines()
    if len(texts) != len(labels):
        raise SchemaError("TweetEval text and label files differ in length")
    return _finish("tweeteval", pd.Series(texts), pd.Series(labels), {"0": "negative", "2": "positive"})


def load_social_sentiments(path: str | Path):
    """The small social-media sentiment set: columns ``Text`` and ``Sentiment`` (about 190 emotion names).

    Only rows labelled exactly Positive or Negative are kept. The report shows how few they are.
    """
    raw = pd.read_csv(path, dtype=str)
    return _finish("social_recent", raw["Text"], raw["Sentiment"], {"negative": "negative", "positive": "positive"})


def load_unified(path: str | Path):
    raw = pd.read_csv(path, dtype=str)
    if "source" not in raw.columns:
        raw["source"] = Path(path).stem
    frames, total = [], LoadReport(Path(path).stem, rows_read=len(raw))
    for src, part in raw.groupby("source"):
        df, rep = _finish(str(src), part["text"], part["label"], {"negative": "negative", "positive": "positive"})
        frames.append(df)
        for k, v in rep.dropped.items():
            total.add(k, v)
    out = pd.concat(frames, ignore_index=True)
    total.rows_kept = len(out)
    return out, total


LOADERS = {
    "sentiment140": load_sentiment140,
    "tweeteval": load_tweeteval,
    "social": load_social_sentiments,
    "unified": load_unified,
}


def load_many(specs: list[str], sample: int | None = None, seed: int = 42):
    frames, reports = [], []
    for spec in specs:
        kind, _, path = spec.partition("=")
        if kind not in LOADERS or not path:
            raise ValueError(f"source must be KIND=PATH with KIND in {sorted(LOADERS)}, got {spec!r}")
        if kind == "sentiment140":
            df, rep = load_sentiment140(path, sample=sample, seed=seed)
        else:
            df, rep = LOADERS[kind](path)
        frames.append(df)
        reports.append(rep)
    return pd.concat(frames, ignore_index=True), reports
