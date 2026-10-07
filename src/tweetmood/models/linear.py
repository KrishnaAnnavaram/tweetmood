"""Sparse TF-IDF models (scikit-learn). The normalizer is the first step of the pipeline.

* The matrix stays sparse: there is no ``.toarray()``. 30 000 tweets x 10 000 terms as a dense
  float64 matrix needs 2.4 GB, the sparse matrix needs a few MB.
* The vectorizer is fit inside the pipeline on the training split only.
* The regularization strength is selected on the validation split.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..evaluate import macro_f1
from ..text.normalize import Normalizer
from .base import write_meta

STRENGTHS = {"tfidf-logreg": (0.3, 1.0, 3.0, 10.0), "tfidf-svm": (0.03, 0.1, 0.3, 1.0), "tfidf-nb": (0.1, 0.3, 1.0)}


class NormalizeText(BaseEstimator, TransformerMixin):
    """scikit-learn step that applies the shared ``Normalizer``."""

    def __init__(self, normalizer: Normalizer | None = None):
        self.normalizer = normalizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        norm = self.normalizer or Normalizer()
        return [norm(t) for t in X]


def make_pipeline(name: str, strength: float, normalizer: Normalizer, seed: int, max_features: int = 50000):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, sublinear_tf=True,
                          token_pattern=r"(?u)[^\s]+", lowercase=False)
    if name == "tfidf-logreg":
        clf = LogisticRegression(C=strength, max_iter=3000, random_state=seed)
    elif name == "tfidf-svm":
        clf = CalibratedClassifierCV(LinearSVC(C=strength, random_state=seed), cv=3)
    elif name == "tfidf-nb":
        clf = ComplementNB(alpha=strength)
    else:
        raise ValueError(f"unknown TF-IDF model {name!r}")
    return Pipeline([("normalize", NormalizeText(normalizer)), ("tfidf", vec), ("clf", clf)])


class TfidfModel:
    def __init__(self, name: str, seed: int = 42, normalizer: Normalizer | None = None, max_features: int = 50000):
        if name not in STRENGTHS:
            raise ValueError(f"unknown TF-IDF model {name!r}")
        self.name = name
        self.seed = seed
        self.normalizer = normalizer or Normalizer(negation_scope=True)
        self.max_features = max_features
        self.pipeline: Pipeline | None = None
        self.selected: dict = {}

    def fit(self, train_texts, train_y, val_texts, val_y) -> "TfidfModel":
        best = None
        for strength in STRENGTHS[self.name]:
            pipe = make_pipeline(self.name, strength, self.normalizer, self.seed, self.max_features)
            pipe.fit(list(train_texts), np.asarray(train_y))
            score = macro_f1(np.asarray(val_y), pipe.predict(list(val_texts)))
            if best is None or score > best[1]:
                best = (pipe, score, strength)
        self.pipeline, val_score, strength = best
        self.selected = {"strength": strength, "val_macro_f1": val_score}
        return self

    def predict_proba(self, texts) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("fit the model first")
        return self.pipeline.predict_proba(list(texts))

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        write_meta(folder, self.name, {"seed": self.seed, "normalizer": asdict(self.normalizer),
                                       "normalizer_fingerprint": self.normalizer.fingerprint(), **self.selected})
        joblib.dump(self.pipeline, folder / "pipeline.joblib")

    @classmethod
    def load(cls, folder: Path) -> "TfidfModel":
        import json

        meta = json.loads((Path(folder) / "model.json").read_text(encoding="utf-8"))
        model = cls(meta["name"], seed=meta["seed"], normalizer=Normalizer(**meta["normalizer"]))
        model.pipeline = joblib.load(Path(folder) / "pipeline.joblib")
        model.selected = {k: meta[k] for k in ("strength", "val_macro_f1") if k in meta}
        return model
