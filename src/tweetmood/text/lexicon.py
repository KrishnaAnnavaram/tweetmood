"""Hand-written maps: emoticons, social-media slang and negation words.

The maps are data, so a reviewer can read and extend them. Keys of ``SLANG`` are lower-case phrases.
Values are plain-English expansions. A value keeps the sentiment of the slang term in plain words
("mid" -> "mediocre"), so a model that learned "mediocre" from older text can read the new term.
"""

from __future__ import annotations

# Emoticons are matched BEFORE lower-casing (":D" and ":d" differ, "XD" is a laugh).
EMOTICONS: dict[str, str] = {
    ":)": "emo_smile", ":-)": "emo_smile", "(:": "emo_smile", "=)": "emo_smile", ":]": "emo_smile",
    ":D": "emo_laugh", ":-D": "emo_laugh", "XD": "emo_laugh", "xD": "emo_laugh", "=D": "emo_laugh",
    ";)": "emo_wink", ";-)": "emo_wink",
    ":P": "emo_tongue", ":-P": "emo_tongue", ":p": "emo_tongue",
    ":(": "emo_sad", ":-(": "emo_sad", "):": "emo_sad", "=(": "emo_sad", ":[": "emo_sad",
    ":'(": "emo_cry", "T_T": "emo_cry", ";_;": "emo_cry",
    ":/": "emo_skeptical", ":-/": "emo_skeptical", ":\\": "emo_skeptical",
    ":|": "emo_flat", "-_-": "emo_annoyed",
    ">:(": "emo_angry", ">:-(": "emo_angry",
    ":O": "emo_surprise", ":o": "emo_surprise", ":-O": "emo_surprise",
    "<3": "emo_heart", "</3": "emo_broken_heart", "^_^": "emo_happy", "^^": "emo_happy",
}

# Case-sensitive single letters: "W" is a win and "L" is a loss only in upper case and alone.
CASED_SLANG: dict[str, str] = {"W": "win", "L": "loss"}

SLANG: dict[str, str] = {
    # emphasis and honesty markers
    "no cap": "honestly", "cap": "lie", "fr": "for real", "frfr": "for real", "deadass": "seriously",
    "ngl": "honestly", "tbh": "honestly", "imo": "in my opinion", "imho": "in my opinion",
    "lowkey": "somewhat", "highkey": "very", "periodt": "period", "on god": "truly",
    # positive slang
    "slay": "excellent", "slayed": "excellent", "slaps": "is excellent", "bussin": "excellent",
    "goated": "the greatest", "goat": "the greatest", "based": "admirable", "lit": "exciting",
    "pog": "excellent", "poggers": "excellent", "hits different": "feels special", "ate that": "did excellently",
    "glow up": "improvement", "big mood": "relatable", "vibes": "good feeling", "rizz": "charm",
    "iconic": "memorable", "valid": "acceptable", "elite": "excellent", "chef's kiss": "perfect",
    # negative slang
    "mid": "mediocre", "sus": "suspicious", "cringe": "embarrassing", "salty": "bitter", "ick": "disgust",
    "flop": "failure", "flopped": "failed", "delulu": "delusional", "smh": "disappointed",
    "fml": "my life is bad", "ffs": "annoyed", "trash": "awful", "ratio": "rejected",
    "touch grass": "go outside", "yikes": "awkward", "big yikes": "very awkward",
    # laughter (positive in context)
    "lol": "laughing", "lmao": "laughing", "lmfao": "laughing", "rofl": "laughing", "im dead": "laughing",
    "i'm dead": "laughing", "dead": "laughing",
    # abbreviations
    "idk": "i do not know", "idc": "i do not care", "ily": "i love you", "omg": "oh my god", "wtf": "what the hell",
    "iykyk": "if you know you know", "rn": "right now", "af": "very", "bc": "because", "u": "you", "ur": "your",
    "r": "are", "pls": "please", "plz": "please", "thx": "thanks", "ty": "thank you", "luv": "love",
    "gr8": "great", "b4": "before", "2day": "today", "2nite": "tonight", "w/": "with", "w/o": "without",
}

# Contractions and apostrophe-less spellings that hide a negation.
NEGATION_FORMS: dict[str, str] = {
    "don't": "do not", "dont": "do not", "doesn't": "does not", "doesnt": "does not", "didn't": "did not",
    "didnt": "did not", "can't": "can not", "cant": "can not", "cannot": "can not", "won't": "will not",
    "wont": "will not", "isn't": "is not", "isnt": "is not", "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not", "weren't": "were not", "werent": "were not", "ain't": "is not",
    "aint": "is not", "shouldn't": "should not", "shouldnt": "should not", "wouldn't": "would not",
    "wouldnt": "would not", "couldn't": "could not", "couldnt": "could not", "haven't": "have not",
    "havent": "have not", "hasn't": "has not", "hasnt": "has not", "nah": "no",
}

NEGATORS = frozenset({"not", "no", "never", "nothing", "nobody", "none", "nor", "neither", "without", "hardly"})
