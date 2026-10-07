"""The ONE text normalizer. Training, evaluation and ``tweetmood predict`` all call it.

Order of steps (the order matters):

1. decode HTML entities, replace URLs with ``url`` and user mentions with ``@user``
2. map emoticons while the case is still known (``:D`` is a laugh, ``:d`` is not an emoticon)
3. map the upper-case single letters ``W`` and ``L``
4. name each emoji (``😂`` -> ``emoji_face_with_tears_of_joy``). Underscores stay, so the name is one token
5. split hashtags (``#NoCap`` -> ``no cap``)
6. lower-case
7. expand negation contractions (``dont`` -> ``do not``). Negations are NEVER removed
8. shorten letter runs (``soooo`` -> ``soo``)
9. expand slang with one compiled pattern (longest phrase first)

There is no stop-word removal: a stop list deletes "not", "no" and "never".
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import asdict, dataclass

from .lexicon import CASED_SLANG, EMOTICONS, NEGATION_FORMS, NEGATORS, SLANG

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_USER = re.compile(r"(?<![\w@])@\w{1,30}")
_HASHTAG = re.compile(r"#(\w+)")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)")
_RUN = re.compile(r"(\w)\1{2,}")
_PUNCT_RUN = re.compile(r"([!?.])\1{2,}")
_SPACE = re.compile(r"\s+")
_SKIP_CODEPOINTS = {0xFE0F, 0xFE0E, 0x200D, 0x20E3}


def _phrase_pattern(phrases, flags=0) -> re.Pattern:
    ordered = sorted(phrases, key=len, reverse=True)
    body = "|".join(re.escape(p) for p in ordered)
    return re.compile(rf"(?<![\w'])(?:{body})(?![\w'])", flags)


_EMOTICON_RX = re.compile(
    r"(?:(?<=\s)|^)(" + "|".join(re.escape(e) for e in sorted(EMOTICONS, key=len, reverse=True)) + r")(?=\s|$|[.,!?])"
)
_CASED_RX = re.compile(r"(?<![\w])(" + "|".join(CASED_SLANG) + r")(?![\w])")
_NEG_RX = _phrase_pattern(NEGATION_FORMS)
_SLANG_RX = _phrase_pattern(SLANG)
_NEG_STOP = re.compile(r"^[.,!?;:]+$")


def emoji_name(ch: str) -> str | None:
    """``emoji_<unicode name>`` for a pictographic symbol, ``""`` for a joiner or modifier, else None."""
    cp = ord(ch)
    if cp in _SKIP_CODEPOINTS or 0x1F3FB <= cp <= 0x1F3FF:
        return ""
    if unicodedata.category(ch) != "So" or cp < 0x2100:
        return None
    name = unicodedata.name(ch, "")
    if not name:
        return None
    return "emoji_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass(frozen=True)
class Normalizer:
    emoticons: bool = True
    emoji: bool = True
    hashtags: bool = True
    slang: bool = True
    elongation: bool = True
    negation_scope: bool = False  # mark up to 3 words after a negator with NEG_ (bag-of-words models only)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        t = html.unescape(str(text))
        t = _URL.sub(" url ", t)
        t = _USER.sub(" @user ", t)
        if self.emoticons:
            t = _EMOTICON_RX.sub(lambda m: f" {EMOTICONS[m.group(1)]} ", t)
        if self.slang:
            t = _CASED_RX.sub(lambda m: f" {CASED_SLANG[m.group(1)]} ", t)
        if self.emoji:
            out = []
            for ch in t:
                name = emoji_name(ch)
                out.append(ch if name is None else f" {name} " if name else " ")
            t = "".join(out)
        if self.hashtags:
            t = _HASHTAG.sub(lambda m: " " + _CAMEL.sub(" ", m.group(1)) + " ", t)
        t = t.lower().replace("’", "'")
        t = _NEG_RX.sub(lambda m: NEGATION_FORMS[m.group(0)], t)
        if self.elongation:
            t = _RUN.sub(r"\1\1", t)
            t = _PUNCT_RUN.sub(r"\1\1", t)
        if self.slang:
            t = _SLANG_RX.sub(lambda m: SLANG[m.group(0)], t)
        t = re.sub(r"([.,!?;:])", r" \1 ", t)
        t = _SPACE.sub(" ", t).strip()
        if self.negation_scope:
            t = mark_negation(t)
        return t


def mark_negation(text: str, window: int = 3) -> str:
    """Prefix up to ``window`` words after a negator with ``NEG_``. Punctuation ends the scope."""
    out, left = [], 0
    for tok in text.split():
        if _NEG_STOP.match(tok):
            left = 0
            out.append(tok)
        elif tok in NEGATORS:
            left = window
            out.append(tok)
        elif left > 0:
            out.append("NEG_" + tok)
            left -= 1
        else:
            out.append(tok)
    return " ".join(out)


def has_negation(text: str) -> bool:
    return any(tok in NEGATORS for tok in Normalizer(slang=False).normalize(text).split())


def has_slang(text: str) -> bool:
    lowered = str(text).lower()
    return bool(_SLANG_RX.search(lowered)) or bool(_CASED_RX.search(str(text)))


def has_emoji_or_emoticon(text: str) -> bool:
    return bool(_EMOTICON_RX.search(str(text))) or any(emoji_name(ch) for ch in str(text))
