"""Text utilities shared by data preparation, training, evaluation and inference."""

from __future__ import annotations

import re
import unicodedata

# Kazakh-specific Cyrillic letters: their presence is the strongest signal that
# a comment is written in Kazakh rather than Russian.
KK_SPECIFIC = set("әғқңөұүһі")

_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_USERNAME = re.compile(r"(?<!\w)@\w{3,}")
_PHONE = re.compile(r"(?<!\d)(?:\+?7|8)?[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-()]?\d{2}[\s\-()]?\d{2}(?!\d)")
_LONG_DIGITS = re.compile(r"\b\d{7,}\b")
_MULTISPACE = re.compile(r"\s+")


def model_normalize(text: str) -> str:
    """Normalisation applied to model input (train AND inference, identical)."""
    text = unicodedata.normalize("NFKC", text)
    return _MULTISPACE.sub(" ", text).strip()


def detect_lang(text: str) -> str:
    """Return 'kk', 'ru', 'en', 'mixed' or 'other'.

    Rule: any Kazakh-specific letter in a Cyrillic-dominated text -> 'kk'
    (Kazakh comments routinely mix in Russian loanwords).  Cyrillic without
    Kazakh-specific letters -> 'ru'.  Latin without Cyrillic -> 'en'.
    """
    cyr = len(_CYR.findall(text))
    lat = len(_LAT.findall(text))
    total = cyr + lat
    if total == 0:
        return "other"
    has_kk = any(ch in KK_SPECIFIC for ch in text.lower())
    if has_kk and cyr / total >= 0.3:
        return "kk"
    if cyr / total >= 0.6:
        return "ru"
    if lat / total >= 0.6:
        return "en"
    return "mixed"


def strip_pii(text: str, names: list[str] | None = None) -> tuple[str, int]:
    """Remove URLs, e-mails, phone numbers and personal names.

    Returns the cleaned text and the number of substitutions performed.
    Names are matched case-insensitively as whole tokens; only names of length
    >= 4 are used to limit accidental replacement of short common words.
    """
    removed = 0
    for pattern in (_URL, _EMAIL, _USERNAME, _PHONE, _LONG_DIGITS):
        text, n = pattern.subn(" ", text)
        removed += n
    if names:
        for name in names:
            if len(name) < 4:
                continue
            pat = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
            text, n = pat.subn("[ИМЯ]", text)
            removed += n
    text = _MULTISPACE.sub(" ", text).strip()
    return text, removed


def near_dup_key(text: str) -> str:
    """Order- and punctuation-insensitive signature used for near-duplicate groups."""
    norm = model_normalize(text).casefold()
    tokens = re.findall(r"\w+", norm)
    return " ".join(sorted(set(tokens)))


def has_pii(text: str) -> bool:
    return bool(_URL.search(text) or _EMAIL.search(text) or _PHONE.search(text))
