"""Step 02 - build the annotation lexicon.

The lexicon is the **annotation protocol**, not a model feature.  It defines
what counts as profanity for this dataset.

Origin rule
-----------
* ``kk`` entries: profanity of **Kazakh origin / Kazakh orthography**
  (қотақ, сік/сіг, ам, көт, бок, жәлеп, қаншық, тасақ, ...).  A Kazakh
  comment containing one of these is labelled 1.
* ``ru`` entries: Russian мат (хуй, пизд, блядь, еб, сук, ...).  Used only to
  (a) verify that the Russian toxic corpus really contains profanity and
  (b) generate hard negatives - such rows stay labelled 0.
* ``en`` entries: English profanity (LDNOOBW list).  Used only for hard
  negatives and synthetic English rows - label 0.
* mild insults that are **not** profanity (ақымақ "fool", сасық "stinky",
  арсыз "shameless", ...) are written to ``non_profane_insults`` and used as
  *negative* examples - an insult is not automatically profanity.

Sources
-------
* Rustem-Kaimolla/kazakh-swear-words (HF, MIT) - Kazakh words/phrases; every
  entry below was manually reviewed.  Items of Russian origin in that dataset
  (нахуй, похуй, долбоёб) were moved to the RU list, i.e. their labels were
  revised for this task.
* LDNOOBW list of profanity (public, English).
* Manual curation for roots/variants/exceptions (documented per entry).

Usage:
    python src/02_build_lexicon.py
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lexicon.json"

LDNOOBW_EN = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
)

SRC_SWEAR = "Rustem-Kaimolla/kazakh-swear-words (MIT, manually verified)"
SRC_MANUAL = "manual curation"

# --------------------------------------------------------------------------
# Kazakh-origin profanity.  severity: severe = obscene (sexual/anatomical),
# moderate = profanity-grade slur.  "insert_forms" are natural word forms used
# by the synthetic generator.
# --------------------------------------------------------------------------
KK_ENTRIES = [
    {
        "id": "kk_kotaq", "lang": "kk", "root": "қотақ", "severity": "severe",
        "category": "vulgar",
        "surfaces": ["қотағы", "қотағым", "қотағың", "қотағын", "қотақбас",
                     "қотақпас", "қотақты", "қотақтар"],
        "insert_forms": ["қотақ", "қотағы", "қотақбас", "қотағың"],
        "exceptions": [], "source": SRC_SWEAR,
    },
    {
        "id": "kk_sik", "lang": "kk", "root": "сік", "severity": "severe",
        "category": "vulgar",
        "surfaces": ["сігу", "сікті", "сікпейді", "сікпе", "сик", "сык"],
        "insert_forms": ["сік", "сігу", "сікпейді"],
        "exceptions": ["сікір", "сіксіз", "сіккіш"], "source": SRC_SWEAR,
    },
    {
        "id": "kk_sig", "lang": "kk", "root": "сіг", "severity": "severe",
        "category": "offensive",
        "surfaces": ["сігем", "сігіл", "сігема", "сігейін", "сігейн", "сігеді",
                     "сігесің", "сігіп", "сігер", "сыгем", "сыгейн"],
        "insert_forms": ["сігейін", "сігем", "сігіл"],
        "exceptions": ["сігір"], "source": SRC_SWEAR,
    },
    {
        "id": "kk_am", "lang": "kk", "root": "ам", "severity": "severe",
        "category": "vulgar", "exact": True,
        "surfaces": ["амы", "амың", "амын", "амды", "амға", "амда", "амнан",
                     "амның", "амдар", "амым", "амына", "амынды"],
        "insert_forms": ["ам", "амың", "амын"],
        "exceptions": [], "source": SRC_SWEAR,
    },
    {
        "id": "kk_ambas", "lang": "kk", "root": "амбас", "severity": "severe",
        "category": "vulgar", "surfaces": ["амбас", "амбасты", "амбассың"],
        "insert_forms": ["амбас"], "exceptions": [], "source": SRC_SWEAR,
    },
    {
        "id": "kk_amshyq", "lang": "kk", "root": "амшық", "severity": "severe",
        "category": "vulgar", "surfaces": ["амшық", "амшықтар"],
        "insert_forms": ["амшық"], "exceptions": [], "source": SRC_SWEAR,
    },
    {
        "id": "kk_amshelek", "lang": "kk", "root": "амщелек", "severity": "severe",
        "category": "vulgar", "surfaces": ["амщелек"],
        "insert_forms": ["амщелек"], "exceptions": [], "source": SRC_SWEAR,
    },
    {
        "id": "kk_kot", "lang": "kk", "root": "көт", "severity": "severe",
        "category": "vulgar",
        "surfaces": ["көті", "көтің", "көтіңді", "көтім", "көтін", "көтке",
                     "көттер", "котин", "котын", "коты"],
        "insert_forms": ["көт", "көтің", "көтіңді", "көтім"],
        "exceptions": ["көтер", "көтерме", "көтеріл", "көтерлік"],
        "source": SRC_SWEAR,
    },
    {
        "id": "kk_boq", "lang": "kk", "root": "бок", "severity": "moderate",
        "category": "vulgar",
        "surfaces": ["бок", "богы", "бокқа", "боктан", "бокты"],
        "insert_forms": ["бок"], "exceptions": ["бокс", "боксер", "боксты"],
        "source": SRC_SWEAR,
    },
    {
        "id": "kk_zhalep", "lang": "kk", "root": "жәлеп", "severity": "severe",
        "category": "offensive", "surfaces": ["жәлептер", "жәлепсің"],
        "insert_forms": ["жәлеп"], "exceptions": [], "source": SRC_SWEAR,
    },
    {
        "id": "kk_qanshyq", "lang": "kk", "root": "қаншық", "severity": "severe",
        "category": "insult", "surfaces": ["қаншықтар", "каншык", "каншыктар"],
        "insert_forms": ["қаншық", "каншык"], "exceptions": [],
        "source": SRC_SWEAR,
    },
    {
        "id": "kk_tasaq", "lang": "kk", "root": "тасақ", "severity": "severe",
        "category": "vulgar", "surfaces": ["тасақ", "тасағы"],
        "insert_forms": ["тасақ"], "exceptions": [], "source": SRC_SWEAR,
    },
    {
        "id": "kk_saldaqy", "lang": "kk", "root": "салдақы", "severity": "moderate",
        "category": "insult", "surfaces": ["салдақы", "салдакы", "салдақылар"],
        "insert_forms": ["салдақы"], "exceptions": [], "source": SRC_SWEAR,
    },
    {
        "id": "kk_shshs", "lang": "kk", "root": "шшс", "severity": "severe",
        "category": "offensive", "exact": True, "surfaces": ["шшс"],
        "insert_forms": ["шшс"], "exceptions": [],
        "source": SRC_SWEAR + " (abbreviation of a profane phrase)",
    },
]

# --------------------------------------------------------------------------
# Russian мат - used ONLY to verify hard negatives / generate them.
# --------------------------------------------------------------------------
RU_ENTRIES = [
    {"id": "ru_khui", "lang": "ru", "root": "хуй", "severity": "severe",
     "category": "vulgar", "infix": True,
     "surfaces": ["хуя", "хуе", "хуё", "хуи", "хуйн", "хуйня"],
     "insert_forms": ["хуй", "хуйня", "нахуй"], "exceptions": [],
     "source": SRC_MANUAL},
    {"id": "ru_pizd", "lang": "ru", "root": "пизд", "severity": "severe",
     "category": "vulgar", "infix": True,
     "surfaces": ["пизда", "пиздец", "пизде"],
     "insert_forms": ["пиздец", "пизда"], "exceptions": [],
     "source": SRC_MANUAL},
    {"id": "ru_blyad", "lang": "ru", "root": "бляд", "severity": "severe",
     "category": "vulgar", "infix": True,
     "surfaces": ["блядь", "бляди", "блять", "бля"],
     "insert_forms": ["блядь", "бля"], "exceptions": [],
     "source": SRC_MANUAL},
    {"id": "ru_eb", "lang": "ru", "root": "еб", "severity": "severe",
     "category": "vulgar", "infix": True,
     "surfaces": ["ебать", "ебал", "еблан", "заебал", "уебал", "долбоеб", "ёб", "ёбнутый", "ёбаный"],
     "insert_forms": ["ебать", "заебал", "долбоеб", "ёб"],
     "exceptions": ["себе", "тебе", "хлеб", "неб", "греб", "погреб",
                    "требу", "колеб", "учеб", "жреб", "лебед", "стеб", "потреб",
                    "употреб", "себ", "теб", "хреб", "волшеб", "реб", "веб",
                    "серебр"],
     "source": SRC_MANUAL},
    {"id": "ru_suk", "lang": "ru", "root": "сук", "severity": "severe",
     "category": "insult",
     "surfaces": ["сука", "суки", "сукин", "сукой", "суке"],
     "insert_forms": ["сука", "суки"],
     "exceptions": ["сукно", "сукцесс", "суккул", "сукре"],
     "source": SRC_MANUAL},
    {"id": "ru_mud", "lang": "ru", "root": "муд", "severity": "severe",
     "category": "insult",
     "surfaces": ["мудак", "мудила", "мудозвон"],
     "insert_forms": ["мудак", "мудила"],
     "exceptions": ["мудр", "мудрец", "мудрос"],
     "source": SRC_MANUAL},
    {"id": "ru_zalup", "lang": "ru", "root": "залуп", "severity": "severe",
     "category": "vulgar", "surfaces": ["залупа", "залупу"],
     "insert_forms": ["залупа"], "exceptions": [], "source": SRC_MANUAL},
    {"id": "ru_gandon", "lang": "ru", "root": "гандон", "severity": "severe",
     "category": "insult", "surfaces": ["гандон", "гондон"],
     "insert_forms": ["гандон"], "exceptions": [], "source": SRC_MANUAL},
    {"id": "ru_pidor", "lang": "ru", "root": "пидор", "severity": "severe",
     "category": "insult", "surfaces": ["пидор", "пидар", "пидорас"],
     "insert_forms": ["пидор"], "exceptions": [], "source": SRC_MANUAL},
    {"id": "ru_shlyukh", "lang": "ru", "root": "шлюх", "severity": "severe",
     "category": "insult", "surfaces": ["шлюха", "шлюхи"],
     "insert_forms": ["шлюха"], "exceptions": [], "source": SRC_MANUAL},
]

# Kazakh insults that are NOT profanity: used as negative examples only.
NON_PROFANE_INSULTS = ["ақымақ", "сасық", "арсыз", "жынды", "есер",
                       "қоқыс", "құбыжық", "қарақшы", "пұшық"]


def fetch_english_profanity() -> list[dict]:
    raw = urllib.request.urlopen(LDNOOBW_EN, timeout=60).read().decode("utf-8")
    words = sorted({w.strip().lower() for w in raw.splitlines()
                    if w.strip().isalpha() and 3 <= len(w.strip()) <= 20})
    return [{
        "id": f"en_{w}", "lang": "en", "root": w, "severity": "severe",
        "category": "profanity", "surfaces": [], "insert_forms": [w],
        "exceptions": [], "exact": True,  # English is not agglutinative
        "source": "LDNOOBW (public English profanity list)",
    } for w in words]


def merge_dataset_forms(entries: list[dict]) -> None:
    """Attach manually verified word forms from the Kazakh swear-words dataset."""
    import urllib.request

    url = "https://huggingface.co/datasets/Rustem-Kaimolla/kazakh-swear-words/resolve/main/data.jsonl"
    try:
        raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"!! could not fetch swear-words dataset ({exc}); using curated forms only")
        return
    items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    added = 0
    for item in items:
        if item.get("label") != 1 or item.get("severity") == "mild":
            continue
        text = item["text"].strip().lower()
        if not text or " " in text or len(text) > 15:
            continue
        for e in entries:
            if e["lang"] != "kk":
                continue
            root = e["root"]
            if text == root or text.startswith(root[:3]) and root[:3] in text:
                if text not in e["surfaces"] and text != root:
                    e["surfaces"].append(text)
                    added += 1
                break
    print(f"swear-words dataset: added {added} verified surface forms to kk entries")


def main() -> int:
    entries = [*KK_ENTRIES, *RU_ENTRIES]
    merge_dataset_forms(entries)
    en = fetch_english_profanity()
    print(f"english entries from LDNOOBW: {len(en)}")
    entries.extend(en)

    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate entry ids"
    for e in entries:
        assert e["root"], e
        assert e["lang"] in {"kk", "ru", "en"}, e

    payload = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "description": "Annotation lexicon for the Kazakh profanity task. "
                       "Contains profanity - required to reproduce the labels.",
        "entries": entries,
        "non_profane_insults": NON_PROFANE_INSULTS,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    by_lang = {l: sum(1 for e in entries if e["lang"] == l) for l in ("kk", "ru", "en")}
    print(f"lexicon written to {OUT}")
    print(f"entries: {len(entries)}  (kk={by_lang['kk']}, ru={by_lang['ru']}, en={by_lang['en']})")
    print("non-profane insults kept as negatives:", ", ".join(NON_PROFANE_INSULTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
