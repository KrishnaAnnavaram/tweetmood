"""The model interface and the factory.

Every model has ``fit(train_texts, train_y, val_texts, val_y)``, ``predict_proba(texts) -> (n, 2)``,
``save(folder)`` and ``load(folder)``. Every model normalizes raw text itself with the same
``Normalizer``, so the training path and the prediction path cannot differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

CLASSIC = ("lexicon", "tfidf-logreg", "tfidf-svm", "tfidf-nb")
NEURAL = ("hybrid-bilstm", "hybrid-transformer", "finetune")
MODEL_NAMES = CLASSIC + NEURAL
META_FILE = "model.json"


class SentimentModel(Protocol):
    name: str

    def fit(self, train_texts, train_y, val_texts, val_y) -> "SentimentModel": ...

    def predict_proba(self, texts) -> np.ndarray: ...

    def save(self, folder: Path) -> None: ...


def build_model(name: str, seed: int = 42, **options) -> SentimentModel:
    if name == "lexicon":
        from .lexicon import LexiconModel

        return LexiconModel(**options)
    if name.startswith("tfidf-"):
        from .linear import TfidfModel

        return TfidfModel(name, seed=seed, **options)
    if name.startswith("hybrid-"):
        from .hybrid import HybridModel  # torch

        return HybridModel(head=name.split("-", 1)[1], seed=seed, **options)
    if name == "finetune":
        from .finetune import FineTuneModel  # torch + transformers

        return FineTuneModel(seed=seed, **options)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def write_meta(folder: Path, name: str, extra: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / META_FILE).write_text(json.dumps({"name": name, **extra}, indent=2), encoding="utf-8")


def load_model(folder: str | Path) -> SentimentModel:
    folder = Path(folder)
    meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    name = meta["name"]
    if name == "lexicon":
        from .lexicon import LexiconModel

        return LexiconModel.load(folder)
    if name.startswith("tfidf-"):
        from .linear import TfidfModel

        return TfidfModel.load(folder)
    if name.startswith("hybrid-"):
        from .hybrid import HybridModel

        return HybridModel.load(folder)
    if name == "finetune":
        from .finetune import FineTuneModel

        return FineTuneModel.load(folder)
    raise ValueError(f"unknown model in {folder}: {name!r}")
