"""Lexicon utilities: normalisation, profanity matching (plain + masked forms).

The lexicon is the annotation protocol for this dataset: it decides which rows
are labelled 1 (Kazakh profanity present) and which RU/EN rows are true hard
negatives.  It is never used as a feature of the classifier itself.

Matching rules (documented in README, "Labeling rules"):

1. Normalisation: NFKC, casefold, typographic quotes/dashes -> ASCII,
   zero-width characters removed, ``ё`` -> ``е``, whitespace collapsed,
   repeated letters collapsed (``сууук`` -> ``сук``).
2. A second (Cyrillic) view maps Latin homoglyphs to Cyrillic
   (``c``->``с``, ``a``->``а`` ...) so obfuscated ``cік`` matches ``сік``.
3. Plain match: a lexicon *root* matches when it is a prefix of a token
   (Kazakh is agglutinative: ``сік`` matches ``сікті``).  Roots flagged
   ``infix`` (Russian) may match inside a token.  Roots flagged ``exact``
   (short roots, English words) must equal the whole token.
   Each entry has an ``exceptions`` list of innocent words that share the
   same letters (``сукно``, ``хлеб``, ``көтерді``).
4. Masked match: for tokens containing ``*``/``•``/``_``..., the visible
   letters must be a subsequence of the root, start with the root's first
   letter and be at least 2 letters long; the token and root must use the
   same script (Cyrillic ``с**к`` never matches English ``cock``).
5. A word counts once even if several variants match.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

MASK_CHARS = set("*•·_.#@$%&+=~^`'\"“”‘’")
CYR_RANGE = ("\u0400", "\u04ff")

LAT_TO_CYR = str.maketrans({
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х",
    "y": "у", "k": "к", "m": "м", "t": "т", "h": "н", "b": "в",
    "u": "и", "0": "о", "3": "е", "6": "б", "9": "д", "4": "ч",
})
_PUNCT_MAP = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."})
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_REPEATS = re.compile(r"(.)\1+")
_SPACE = re.compile(r"\s+")
_CYR_CHAR = re.compile(r"[\u0400-\u04ff]")
_LAT_CHAR = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.translate(_PUNCT_MAP).casefold().replace("ё", "е")
    return _SPACE.sub(" ", text).strip()


def collapse_repeats(text: str) -> str:
    return _REPEATS.sub(r"\1", text)


def cyrillic_view(text: str) -> str:
    return text.translate(LAT_TO_CYR)


def _strip_outer_punct(token: str) -> str:
    return token.strip(".,;:!?()[]{}<>\"'«»—–-…")


def is_masked(token: str) -> bool:
    return any(ch in MASK_CHARS for ch in token)


def _is_subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(ch in it for ch in small)


@dataclass
class Match:
    entry_id: str
    lang: str
    severity: str
    surface: str
    mode: str  # "plain" | "masked"


@dataclass
class Lexicon:
    entries: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._prep = []
        for e in self.entries:
            root = collapse_repeats(normalize(e["root"]))
            self._prep.append({
                "e": e,
                "root": root,
                "root_cyr": cyrillic_view(root),
                "surfaces": [root] + [collapse_repeats(normalize(s)) for s in e.get("surfaces", [])],
                "exceptions": [collapse_repeats(normalize(x)) for x in e.get("exceptions", [])],
                "infix": bool(e.get("infix")),
                "exact": bool(e.get("exact")),
                "cyr": bool(_CYR_CHAR.search(root)),
            })

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(entries=data["entries"])

    # ------------------------------------------------------------------
    def match(self, text: str) -> list[Match]:
        norm = normalize(text)
        flat = collapse_repeats(norm)
        flat_cyr = cyrillic_view(flat)
        tokens = [_strip_outer_punct(t) for t in flat.split(" ")]
        masked_tokens = [t for t in tokens if is_masked(t)]
        has_cyr = bool(_CYR_CHAR.search(flat))
        has_lat = bool(_LAT_CHAR.search(flat))

        found: dict[str, Match] = {}
        for p in self._prep:
            if p["cyr"] and not has_cyr:
                continue
            if not p["cyr"] and not has_lat:
                continue
            e = p["e"]
            hit = self._plain(flat, flat_cyr if p["cyr"] else None, p)
            if hit:
                found[e["id"]] = Match(e["id"], e["lang"], e["severity"], hit, "plain")
                continue
            if masked_tokens:
                hit = self._masked(masked_tokens, p)
                if hit:
                    found[e["id"]] = Match(e["id"], e["lang"], e["severity"], hit, "masked")
        return list(found.values())

    # ------------------------------------------------------------------
    @staticmethod
    def _plain(flat: str, flat_cyr: str | None, p: dict) -> str | None:
        views = (flat,) if flat_cyr is None else (flat, flat_cyr)
        for view in views:
            for s in p["surfaces"]:
                if len(s) < 2:
                    continue
                if p["exact"]:
                    for m in re.finditer(rf"(?<!\w){re.escape(s)}(?!\w)", view):
                        token = m.group(0)
                        if any(token.startswith(x) for x in p["exceptions"]):
                            continue
                        return s
                    continue
                if p["infix"]:
                    idx = view.find(s)
                    while idx >= 0:
                        start = view.rfind(" ", 0, idx) + 1
                        end = view.find(" ", idx + len(s))
                        token = view[start: end if end != -1 else len(view)]
                        if not any(token.startswith(x) for x in p["exceptions"]):
                            return s
                        idx = view.find(s, idx + 1)
                    continue
                for m in re.finditer(re.escape(s), view):
                    start = m.start()
                    if start > 0 and view[start - 1].isalnum():
                        continue
                    end = m.end()
                    while end < len(view) and view[end].isalnum():
                        end += 1
                    token = view[start:end]
                    if any(token.startswith(x) for x in p["exceptions"]):
                        continue
                    return s
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _masked(tokens: list[str], p: dict) -> str | None:
        for tok in tokens:
            tok_cyr = bool(_CYR_CHAR.search(tok))
            if tok_cyr != p["cyr"]:
                continue
            visible = "".join(ch for ch in (cyrillic_view(tok) if tok_cyr else tok) if ch.isalpha())
            if len(visible) < 2:
                continue
            root_view = p["root_cyr"] if tok_cyr else p["root"]
            if visible[0] == root_view[0] and _is_subsequence(visible, root_view):
                return tok
        return None

    # ------------------------------------------------------------------
    def label_text(self, text: str) -> tuple[int, list[Match]]:
        """Kazakh-task label: 1 if any *Kazakh* profanity matches."""
        matches = self.match(text)
        return (1 if any(m.lang == "kk" for m in matches) else 0), matches


def profanity_languages(matches: list[Match]) -> str:
    return ",".join(sorted({m.lang for m in matches})) or "none"
