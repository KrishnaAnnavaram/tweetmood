"""A transparent rule baseline: word polarities, negation flips and a threshold chosen on train.

It needs no training data for the polarities, so it shows how far plain rules go on new slang
(the slang map turns "mid" into "mediocre", which the polarity list knows).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..text.normalize import Normalizer
from .base import write_meta

POLARITY: dict[str, float] = {
    # positive
    "good": 1, "great": 1.5, "excellent": 2, "amazing": 2, "awesome": 2, "love": 2, "lovely": 1.5, "perfect": 2,
    "wonderful": 2, "best": 1.5, "greatest": 2, "admirable": 1.5, "exciting": 1.5, "memorable": 1.5, "happy": 1.5,
    "fun": 1, "nice": 1, "beautiful": 1.5, "brilliant": 2, "win": 1.5, "laughing": 1, "special": 1, "charm": 1,
    "improvement": 1, "relatable": 1, "acceptable": 0.5, "thanks": 1, "glad": 1.5, "enjoy": 1.5, "enjoyed": 1.5,
    "emo_smile": 1.5, "emo_laugh": 1.5, "emo_heart": 2, "emo_happy": 1.5, "emo_wink": 1, "emo_tongue": 0.5,
    "emoji_fire": 1.5, "emoji_smiling_face_with_heart_shaped_eyes": 2, "emoji_person_raising_both_hands_in_celebration": 1.5,
    "emoji_sparkles": 1, "emoji_face_with_tears_of_joy": 1, "emoji_red_heart": 2, "emoji_heavy_black_heart": 2,
    "emoji_thumbs_up_sign": 1.5,
    # negative
    "bad": -1, "awful": -2, "terrible": -2, "horrible": -2, "worst": -2, "hate": -2, "mediocre": -1.5,
    "embarrassing": -1.5, "failure": -1.5, "failed": -1.5, "disappointing": -1.5, "disappointed": -1.5,
    "suspicious": -1, "bitter": -1, "disgust": -1.5, "awkward": -1, "delusional": -1, "loss": -1.5, "sad": -1.5,
    "boring": -1.5, "annoyed": -1.5, "annoying": -1.5, "broken": -1.5, "rejected": -1, "lie": -0.5, "poor": -1.5,
    "emo_sad": -1.5, "emo_cry": -1.5, "emo_skeptical": -1, "emo_annoyed": -1, "emo_angry": -2,
    "emo_broken_heart": -2, "emoji_face_with_rolling_eyes": -1.5, "emoji_unamused_face": -1.5,
    "emoji_loudly_crying_face": -1, "emoji_face_with_uneven_eyes_and_wavy_mouth": -1, "emoji_thumbs_down_sign": -1.5,
}


class LexiconModel:
    name = "lexicon"

    def __init__(self, normalizer: Normalizer | None = None, threshold: float = 0.0):
        self.normalizer = normalizer or Normalizer(negation_scope=True)
        if not self.normalizer.negation_scope:
            self.normalizer = Normalizer(**{**asdict(self.normalizer), "negation_scope": True})
        self.threshold = threshold

    def score(self, text: str) -> float:
        total = 0.0
        for tok in self.normalizer(text).split():
            if tok.startswith("NEG_"):
                total -= 0.8 * POLARITY.get(tok[4:], 0.0)  # a negated word flips and weakens
            else:
                total += POLARITY.get(tok, 0.0)
        return total

    def fit(self, train_texts, train_y, val_texts=None, val_y=None) -> "LexiconModel":
        scores = np.array([self.score(t) for t in train_texts])
        y = np.asarray(train_y)
        candidates = np.unique(np.concatenate([[0.0], scores]))
        accs = [((scores > c).astype(int) == y).mean() for c in candidates]
        self.threshold = float(candidates[int(np.argmax(accs))])
        return self

    def predict_proba(self, texts) -> np.ndarray:
        s = np.array([self.score(t) for t in texts]) - self.threshold
        p = 1.0 / (1.0 + np.exp(-1.5 * s))
        return np.column_stack([1 - p, p])

    def save(self, folder: Path) -> None:
        write_meta(Path(folder), self.name, {"threshold": self.threshold, "normalizer": asdict(self.normalizer)})

    @classmethod
    def load(cls, folder: Path) -> "LexiconModel":
        meta = json.loads((Path(folder) / "model.json").read_text(encoding="utf-8"))
        return cls(Normalizer(**meta["normalizer"]), threshold=meta["threshold"])
