"""The one shared split. Every model (lexicon, TF-IDF, hybrid, fine-tuned) reads the same ids.

The prototype split the neural models with seed 42 and the TF-IDF models with seed 52, so the two
families were scored on different test tweets. Here ``prepare`` writes the split once, with a
fingerprint, and every command reads it from disk.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..text.normalize import Normalizer
from .loaders import validate

SPLIT_FILE = "split.csv"
MANIFEST_FILE = "manifest.json"
_KEY_DROP = re.compile(r"@user|\burl\b|[^\w\s]")
_KEY_NORMALIZER = Normalizer()


def dedup_key(text: str) -> str:
    """The normalized text without mentions, URLs and punctuation. Emoticon and emoji tokens stay."""
    return " ".join(_KEY_DROP.sub(" ", _KEY_NORMALIZER(text)).split())


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates by key. If copies disagree on the label, drop all of them."""
    work = df.copy()
    work["key"] = work["text"].map(dedup_key)
    work = work[work["key"] != ""]
    n_labels = work.groupby("key")["label"].transform("nunique")
    conflicts = int((n_labels > 1).sum())
    work = work[n_labels == 1]
    before = len(work)
    work = work.drop_duplicates("key").reset_index(drop=True)
    return work, {"rows_in": len(df), "conflicting_rows": conflicts, "duplicates": before - len(work),
                  "rows_out": len(work)}


def make_split(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.1, test_frac: float = 0.2,
               holdout_sources: tuple[str, ...] = ()) -> pd.DataFrame:
    """Stratified (label x source) split. ``holdout_sources`` go to test only (out-of-domain check)."""
    df = validate(df)
    df, _ = deduplicate(df)
    df["id"] = [hashlib.sha1(k.encode("utf-8")).hexdigest()[:12] for k in df["key"]]
    held = df["source"].isin(holdout_sources)
    main = df[~held].copy()
    strata = main["label"] + "|" + main["source"]
    if strata.value_counts().min() < 3:
        strata = main["label"]
    rest, test = train_test_split(main, test_size=test_frac, stratify=strata, random_state=seed)
    rest_strata = strata.loc[rest.index]
    train, val = train_test_split(rest, test_size=val_frac / (1 - test_frac), stratify=rest_strata,
                                  random_state=seed + 1)
    parts = [train.assign(split="train"), val.assign(split="val"), test.assign(split="test"),
             df[held].assign(split="test")]
    out = pd.concat(parts, ignore_index=True)
    return out[["id", "text", "label", "source", "split"]]


def fingerprint(df: pd.DataFrame) -> str:
    rows = sorted(zip(df["id"], df["split"]))
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()[:16]


def save_split(df: pd.DataFrame, folder: str | Path, extra: dict | None = None) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / SPLIT_FILE, index=False)
    manifest = {
        "fingerprint": fingerprint(df),
        "rows": df["split"].value_counts().to_dict(),
        "labels": {s: g["label"].value_counts().to_dict() for s, g in df.groupby("split")},
        "sources": {s: g["source"].value_counts().to_dict() for s, g in df.groupby("split")},
        **(extra or {}),
    }
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(folder: str | Path) -> tuple[pd.DataFrame, dict]:
    folder = Path(folder)
    df = pd.read_csv(folder / SPLIT_FILE, dtype=str)
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    if fingerprint(df) != manifest["fingerprint"]:
        raise ValueError("split.csv does not match its manifest fingerprint")
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids in split.csv")
    return df, manifest


def part(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df[df["split"] == name].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"the {name} split is empty")
    return out
