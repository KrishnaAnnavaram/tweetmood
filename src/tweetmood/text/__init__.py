"""Text normalization: emoticons, emoji names, hashtags, negations and slang."""

from .normalize import Normalizer, mark_negation

__all__ = ["Normalizer", "mark_negation"]
