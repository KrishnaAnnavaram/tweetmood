"""Synthetic tweets for the offline demo and the tests.

Two sources:

* ``synthetic_classic`` - plain-English tweets with emoticons, like the 2009 Sentiment140 era.
* ``synthetic_genz``    - the same opinions in Gen-Z slang and emoji ("mid", "bussin", "W", "💀").

The slang words are rare or absent in the classic source. The slang map turns them into words that
the classic source uses ("mid" -> "mediocre"). The demo holds out ``synthetic_genz`` as a test-only
source to measure how much the slang map helps on new slang.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TOPICS = ["this album", "the new episode", "my order", "the update", "this cafe", "the match", "that movie",
           "the concert", "my new phone", "the season", "this game", "the trailer", "the menu", "the new song"]
_CLASSIC = {
    "positive": ["excellent", "great", "the greatest", "admirable", "exciting", "perfect", "memorable",
                 "wonderful", "good", "lovely"],
    "negative": ["mediocre", "awful", "embarrassing", "a failure", "disappointing", "suspicious", "bitter",
                 "terrible", "bad", "awkward"],
}
_GENZ = {
    "positive": ["bussin", "goated", "slay", "lit", "elite", "iconic", "a big W", "hits different", "valid"],
    "negative": ["mid", "cringe", "sus", "trash", "a flop", "an L", "delulu", "big yikes"],
}
_NEG_FLIP = {"positive": "negative", "negative": "positive"}
_EMOTICON = {"positive": [":)", ":D", "<3", "XD", ""], "negative": [":(", ":'(", ":/", "-_-", ""]}
_EMOJI = {"positive": ["\U0001F525", "\U0001F60D", "\U0001F64C", "✨", ""],
          "negative": ["\U0001F644", "\U0001F612", "\U0001F62D", "\U0001F974", ""]}
_FILLER = ["", "", "ngl", "fr", "tbh", "lowkey", "honestly", "omg"]


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _opinion(rng, label: str, vocab: dict, negator: str) -> str:
    topic = _pick(rng, _TOPICS)
    if rng.random() < 0.25:
        # negated opinion: the word has the opposite polarity of the label
        return f"{topic} is {negator} {_pick(rng, vocab[_NEG_FLIP[label]])}"
    return f"{topic} is {_pick(rng, vocab[label])}"


def make_tweets(n: int = 4000, genz_share: float = 0.25, noise: float = 0.04, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        label = "positive" if rng.random() < 0.5 else "negative"
        if rng.random() < genz_share:
            body = _opinion(rng, label, _GENZ, _pick(rng, ["not", "dont think its", "aint"]))
            text = " ".join(w for w in (_pick(rng, _FILLER), body, _pick(rng, _EMOJI[label])) if w)
            source = "synthetic_genz"
        else:
            body = _opinion(rng, label, _CLASSIC, _pick(rng, ["not", "never", "hardly"]))
            text = " ".join(w for w in (body, _pick(rng, _EMOTICON[label])) if w)
            source = "synthetic_classic"
        if rng.random() < 0.2:
            text = "@" + _pick(rng, ["mia", "dev_k", "radio1"]) + " " + text
        if rng.random() < 0.1:
            text = text.replace(" is ", " is soooo ", 1)
        if rng.random() < noise:
            label = _NEG_FLIP[label]
        rows.append((text, label, source))
    return pd.DataFrame(rows, columns=["text", "label", "source"])
