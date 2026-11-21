"""Synthetic review generation (explicitly flagged ``synthetic=True`` in the data).

Purpose
-------
Real publicly available Kazakh *product reviews containing Kazakh profanity*
are extremely scarce, so the minority class is balanced with generated
review-style sentences.  The generator covers four phenomena:

* kk positive      : Kazakh review frame + Kazakh profanity (base / masked /
                     stretched / homoglyph spelling variants)
* kk negative      : Kazakh review frame, no profanity (incl. negative sentiment)
* hard negative ru : Kazakh frame + *Russian* profanity   -> label 0
* hard negative en : English frame (or Kazakh frame) + *English* profanity -> 0
* mixed positive   : Kazakh frame with Russian loanwords + Kazakh profanity -> 1

Every generated row is validated against the lexicon (positive rows must match
a Kazakh entry, negative rows must not), so the synthetic labels follow exactly
the same annotation rules as the real data.
"""

from __future__ import annotations

import random
from itertools import cycle

import pandas as pd

from lexicon_utils import Lexicon

PRODUCTS = [
    "телефон", "ноутбук", "көйлек", "аяқ киім", "донер", "лаваш", "кофе",
    "суши", "пицца", "жиһаз", "кітап", "дәрі", "такси", "жеткізу", "дүкен",
    "мейрамхана", "кафе", "қонақүй", "монша", "мектеп", "дәріхана", "базар",
    "шаш қию", "массаж", "спортзал", "тамақ", "көлік",
]

EN_PRODUCTS = ["phone", "laptop", "dress", "sneakers", "pizza", "coffee",
               "headphones", "backpack", "chair", "monitor"]

# Common English profanity used for generation.  The full LDNOOBW list stays in
# the lexicon for detection, but its long tail (fetish terms, slurs) produces
# nonsense review sentences and is deliberately not used for generation.
COMMON_EN = ["fuck", "fucking", "fucker", "shit", "shitty", "bullshit",
             "bitch", "asshole", "bastard", "dick", "prick", "dumbass",
             "crap", "piss", "douchebag", "jackass", "twat", "wanker",
             "whore", "cunt", "cock", "pussy"]

NEG_FRAMES = [
    "{p} жақсы, бағасы да қолжетімді.",
    "{p} туралы пікірім жақсы.",
    "{p} сапасы орташа.",
    "{p} күтілгендей болды.",
    "{p} алдым, әзірге көңілім толды.",
    "Жеткізу тез болды, {p} жақсы.",
    "{p} бағасы қымбат, бірақ сапасы жақсы.",
    "Қызмет көрсету жақсы, {p} ұнады.",
    "{p} бұрынғыдан жақсы екен.",
    "{p} үшін рахмет, тағы аламын.",
    "{p} нашар екен, өкінішке орай.",
    "{p} сапасы нашар, ақшамды қайтарыңыздар.",
    "Жеткізу өте ұзақ болды, {p} әлі келмеді.",
    "{p} күтілгендей емес, бағасына тұрарлық емес.",
    "Қызмет көрсету нашар, {p} тағы алмаймын.",
    "{p} дәмсіз, қайта келмеймін.",
    "Тапсырыс қате келді, {p} ауыстырыңыздар.",
    "{p} екі апта күттім, ешкім хабарласпады.",
    "{p} бағасы өте қымбат.",
    "{p} суреттегідей емес.",
]

POS_FRAMES = [
    "{p} {x}, сапасы нашар.",
    "{x}, {p} күтілгендей емес.",
    "Жеткізу {x} ұзақ болды, {p} әлі келмеді.",
    "{p} {x} екен, ақшамды қайтарыңдар.",
    "{x}, бұл {p} үшін қымбат.",
    "Сапасы {x}, қайта келмеймін.",
    "{p} алдым, {x}, бірақ бағасы жақсы.",
    "{x}! {p} мүлде нашар екен.",
    "{p} {x}, дүкенге кеңес бермеймін.",
    "{x}, {p} туралы айтқандарының бәрі өтірік.",
    "Мен {p} алдым, {x}, енді қайтарғым келеді.",
    "{p} {x}, қызмет көрсетуі төмен.",
    "{x}, жеткізуші {p} қате әкелді.",
    "{p} {x} болып шықты, өкінішке орай.",
    "{x}, дүкен {p} туралы ешнәрсе білмейді.",
]

MILD_INSULTS = ["ақымақ", "сасық", "арсыз", "жынды", "есер", "қоқыс", "құбыжық"]

INSULT_NEG_FRAMES = [
    "{p} {x} екен, бірақ бағасы жақсы.",
    "{x} сатушы, {p} нашар.",
    "{p} нашар, {x} қызмет көрсету.",
    "Менің ойымша, {p} {x} болып шықты.",
    "{x} жеткізуші, {p} қате әкелді.",
]

MIXED_POS_FRAMES = [
    "{p} {x}, но качество плохое.",
    "{x}, доставка {p} опять задержалась.",
    "{p} пришел, {x}, размер не тот.",
    "{x}, заказ {p} так и не пришел.",
    "{p} {x}, продавец не отвечает.",
    "{x}, заказ {p} пришел с опозданием на неделю.",
]

RU_HARD_NEG_FRAMES = [
    "{p} {x}, сапасы нашар.",
    "Жеткізу {x} ұзақ болды, {p} әлі жоқ.",
    "{x}, {p} күтілгендей емес.",
    "Сапасы {x}, қайта келмеймін.",
    "{p} {x} екен, ақшамды қайтарыңдар.",
]

EN_FRAMES = [
    "The {p} is {x}, never buying again.",
    "{p} quality is {x}.",
    "Delivery was {x} slow, the {p} never arrived.",
    "Bought this {p}, {x}, returning it.",
    "{x}, this {p} is not worth the price.",
]

MIXED_EN_FRAMES = [
    "Жеткізу {x} болды, {p} келмеді.",
    "{p} {x}, сапасы нашар.",
    "{x}, бұл {p} үшін тым қымбат.",
]

VOWELS = "аәеёиоөұүуыіэюя"
CYR_TO_LAT = {"с": "c", "а": "a", "е": "e", "о": "o", "р": "p", "х": "x",
              "у": "y", "к": "k", "м": "m", "т": "t", "н": "h", "в": "b"}


def _masked(word: str) -> str:
    if len(word) >= 4:
        return word[0] + "*" * (len(word) - 2) + word[-1]
    if len(word) == 3:
        return word[0] + "*" + word[-1]
    return word[0] + "*"


def _stretched(word: str) -> str:
    for i, ch in enumerate(word):
        if ch in VOWELS:
            return word[: i + 1] + ch + word[i + 1:]
    return word + word[-1]


def _homoglyph(word: str) -> str:
    for i, ch in enumerate(word):
        if ch in CYR_TO_LAT:
            return word[:i] + CYR_TO_LAT[ch] + word[i + 1:]
    return word


def _variant(word: str, rng: random.Random) -> str:
    kind = rng.choices(["base", "masked", "stretched", "homoglyph"], weights=[5, 2, 2, 1])[0]
    if kind == "masked":
        return _masked(word)
    if kind == "stretched":
        return _stretched(word)
    if kind == "homoglyph":
        return _homoglyph(word)
    return word


def _surfaces(entries: list[dict], severity_ok=("severe", "moderate")) -> list[str]:
    out = []
    for e in entries:
        if e.get("severity") not in severity_ok:
            continue
        out.extend(e.get("insert_forms", [e["root"]]))
    return sorted(set(out))


def generate(lexicon: Lexicon, n_pos: int, n_neg: int, n_hard_ru: int, n_hard_en: int,
             seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    kk_words = _surfaces([e for e in lexicon.entries if e["lang"] == "kk"])
    ru_words = _surfaces([e for e in lexicon.entries if e["lang"] == "ru"])
    en_all = set(_surfaces([e for e in lexicon.entries if e["lang"] == "en"]))
    en_words = [w for w in COMMON_EN if w in en_all]
    if not (kk_words and ru_words and en_words):
        raise SystemExit("lexicon is missing kk/ru/en entries - run 02_build_lexicon.py first")

    rows: list[dict] = []

    def add(text: str, label: int, lang: str, template_id: str) -> None:
        rows.append({"text": text, "label": label, "lang": lang,
                     "source": "synthetic", "synthetic": True, "template_id": template_id})

    for _ in range(n_pos):
        frame = rng.choice(POS_FRAMES)
        add(frame.format(p=rng.choice(PRODUCTS), x=_variant(rng.choice(kk_words), rng)),
            1, "kk", "kk_pos_profane")
    for _ in range(max(1, n_pos // 5)):
        frame = rng.choice(MIXED_POS_FRAMES)
        add(frame.format(p=rng.choice(PRODUCTS), x=_variant(rng.choice(kk_words), rng)),
            1, "kk", "kk_pos_profane_mixed_ru_words")
    for _ in range(n_neg):
        frame = rng.choice(NEG_FRAMES)
        add(frame.format(p=rng.choice(PRODUCTS)), 0, "kk", "kk_negative_clean")
    for _ in range(max(1, n_neg // 6)):
        frame = rng.choice(INSULT_NEG_FRAMES)
        add(frame.format(p=rng.choice(PRODUCTS), x=rng.choice(MILD_INSULTS)),
            0, "kk", "kk_negative_insult_without_profanity")
    for _ in range(n_hard_ru):
        frame = rng.choice(RU_HARD_NEG_FRAMES)
        add(frame.format(p=rng.choice(PRODUCTS), x=_variant(rng.choice(ru_words), rng)),
            0, "kk", "hard_neg_ru_profanity_in_kk_frame")
    for _ in range(n_hard_en // 2):
        frame = rng.choice(EN_FRAMES)
        add(frame.format(p=rng.choice(EN_PRODUCTS), x=_variant(rng.choice(en_words), rng)),
            0, "en", "hard_neg_en_profanity")
    for _ in range(n_hard_en - n_hard_en // 2):
        frame = rng.choice(MIXED_EN_FRAMES)
        add(frame.format(p=rng.choice(PRODUCTS), x=_variant(rng.choice(en_words), rng)),
            0, "kk", "hard_neg_en_profanity_in_kk_frame")

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # validate every synthetic row against the annotation rules
    keep, drop = [], 0
    for _, r in df.iterrows():
        label, matches = lexicon.label_text(r["text"])
        if label != r["label"]:
            drop += 1
            continue
        keep.append({**r.to_dict(), "lexicon_hits": ",".join(sorted(m.entry_id for m in matches))})
    df = pd.DataFrame(keep)
    print(f"synthetic rows kept={len(df):,} dropped_by_validation={drop}")
    return df
