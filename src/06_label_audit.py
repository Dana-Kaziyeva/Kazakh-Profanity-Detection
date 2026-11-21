"""Step 06 - produce a human-reviewable label audit (does NOT change labels).

For every row the audit lists *why* its label is suspicious:

    MODEL_SAYS_PROFANE      label 0 but the trained model predicts >= 0.9
    MODEL_SAYS_CLEAN        label 1 but the model predicts <= 0.1
    RULE_ZERO_DESPITE_KK    label 0 although a Kazakh profanity match exists,
                            only because the language heuristic tagged it 'ru'
    KK_MATCH_IN_RU_TEXT     label 1 from a Kazakh root inside a Russian/mixed text
    AMBIGUOUS_SHORT_ROOT    the match is an inherently ambiguous short root
                            (ам / бок / шшс) outside clean Kazakh text
    WEAK_MASKED_MATCH       label 1 supported only by a mask with <= 2 visible letters
    PROPER_NOUN             the matched surface sits inside a capitalised name
    HARD_NEGATIVE           label 0 with Russian/English profanity (correct by
                            rule - listed for completeness, not an error)

Outputs
    data/processed/label_review.csv      one row per suspicious example, sorted by priority
    data/processed/label_review_summary.json
    data/label_fixes.csv                 empty template: id,label,note  (fill it, then
                                         re-run 03_prepare_data.py --refresh)

Usage:
    python src/06_label_audit.py [--all]   # --all: score every row (slower)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon_utils import Lexicon  # noqa: E402
from model import ProfanityClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AMBIGUOUS = {"kk_am", "kk_boq", "kk_shshs"}
PRIORITY = {
    "MODEL_SAYS_PROFANE": 5,
    "MODEL_SAYS_CLEAN": 5,
    "RULE_ZERO_DESPITE_KK": 4,
    "KK_MATCH_IN_RU_TEXT": 3,
    "AMBIGUOUS_SHORT_ROOT": 2,
    "WEAK_MASKED_MATCH": 2,
    "PROPER_NOUN": 1,
    "HARD_NEGATIVE": 0,
}


def _visible_letters(token: str) -> int:
    return sum(ch.isalpha() for ch in token)


def _proper_noun_hit(text: str, surface: str) -> bool:
    """True if the matched surface appears inside a capitalised, non-initial word."""
    if len(surface) < 3:
        return False
    for m in re.finditer(re.escape(surface), text, flags=re.IGNORECASE):
        start = m.start()
        tok_start = text.rfind(" ", 0, start) + 1
        token = text[tok_start:m.end()]
        rest = text[m.end():]
        word = rest.split(" ")[0] if rest else ""
        word = token + word
        word = re.split(r"[^\w]", word)[0]
        if word[:1].isupper() and tok_start > 0:
            return True
    return False


def audit_row(row, matches, prob) -> tuple[list[str], str, list[str]]:
    kk = [m for m in matches if m.lang == "kk"]
    other = [m for m in matches if m.lang != "kk"]
    label = int(row["label"])
    reasons, notes = [], []

    if prob is not None and not np.isnan(prob):
        if label == 0 and prob >= 0.9:
            reasons.append("MODEL_SAYS_PROFANE")
        if label == 1 and prob <= 0.1:
            reasons.append("MODEL_SAYS_CLEAN")

    if label == 0 and kk and row["lang"] == "ru":
        reasons.append("RULE_ZERO_DESPITE_KK")
        notes.append(" Kazakh profanity present, but the language heuristic tagged the text 'ru'")

    if label == 1 and kk and row["lang"] in ("ru", "mixed"):
        reasons.append("KK_MATCH_IN_RU_TEXT")

    if label == 1 and kk:
        ids = {m.entry_id for m in kk}
        if ids & AMBIGUOUS and row["lang"] != "kk":
            reasons.append("AMBIGUOUS_SHORT_ROOT")
        if all(m.mode == "masked" for m in kk):
            vis = max(_visible_letters(m.surface) for m in kk)
            if vis <= 2:
                reasons.append("WEAK_MASKED_MATCH")
        for m in kk:
            if _proper_noun_hit(str(row["text"]), m.surface):
                reasons.append("PROPER_NOUN")
                break

    if label == 0 and other and not kk:
        reasons.append("HARD_NEGATIVE")

    suggest = ""
    if "MODEL_SAYS_PROFANE" in reasons and "HARD_NEGATIVE" not in reasons:
        suggest = 1
    elif "RULE_ZERO_DESPITE_KK" in reasons:
        suggest = 1
    elif "MODEL_SAYS_CLEAN" in reasons or "KK_MATCH_IN_RU_TEXT" in reasons:
        suggest = 0
    return reasons, suggest, notes


AMBIGUOUS_ENTRIES = {"kk_am", "kk_boq", "kk_shshs"}


def _has_ambiguous_hit(row) -> bool:
    hits = str(row.get("lexicon_hits") or "").split(",")
    return any(h in AMBIGUOUS_ENTRIES for h in hits)


def review_type(row) -> str:
    """Group a flagged row by the kind of decision it needs."""
    r = set(str(row["reasons"]).split("|"))
    if "MODEL_SAYS_PROFANE" in r and "HARD_NEGATIVE" in r:
        return "model_boundary"          # label is right, the model is wrong
    if "MODEL_SAYS_CLEAN" in r or "MODEL_SAYS_PROFANE" in r:
        return "label_fix"
    if r & {"RULE_ZERO_DESPITE_KK", "KK_MATCH_IN_RU_TEXT"} and not _has_ambiguous_hit(row):
        return "systematic_lang_rule"    # one language-detector fix, not per-row edits
    if r == {"HARD_NEGATIVE"}:
        return "informational"           # correct by rule, listed for completeness
    return "ambiguous"


def suggested_label(row) -> str:
    t = row["review_type"]
    r = set(str(row["reasons"]).split("|"))
    if t == "label_fix":
        return 0 if "MODEL_SAYS_CLEAN" in r else 1
    if t == "systematic_lang_rule" and "RULE_ZERO_DESPITE_KK" in r:
        return 1
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="score every row with the model (slower)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    lexicon = Lexicon.load(ROOT / "data" / "lexicon.json")
    proc = ROOT / "data" / "processed"

    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(proc / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"rows: {len(df):,}")

    print("matching lexicon ...")
    matches = [lexicon.match(t) for t in df["text"]]

    clf = ProfanityClassifier(ROOT / "artifacts" / "model")
    df["model_prob"] = np.nan
    need = args.all | df["label"].eq(1).to_numpy() | np.array([bool(m) for m in matches])
    idx = np.where(need)[0]
    print(f"scoring {len(idx):,} rows with the model (threshold {clf.threshold}) ...")
    probs = clf.predict_proba(df.loc[idx, "text"].tolist(), batch_size=32)
    df.loc[idx, "model_prob"] = probs

    rows, summary = [], {}
    for i, row in df.iterrows():
        reasons, suggest, notes = audit_row(row, matches[i], row["model_prob"])
        for r in reasons:
            summary[r] = summary.get(r, 0) + 1
        if not reasons:
            continue
        kk = [m for m in matches[i] if m.lang == "kk"]
        rows.append({
            "id": row["id"], "priority": max(PRIORITY[r] for r in reasons),
            "reasons": "|".join(reasons), "label": int(row["label"]),
            "suggested_label": suggest, "model_prob": (None if np.isnan(row["model_prob"])
                                                       else round(float(row["model_prob"]), 4)),
            "lang": row["lang"], "source": row["source"], "synthetic": bool(row["synthetic"]),
            "split": row["split"], "text": row["text"],
            "lexicon_hits": row["lexicon_hits"], "match_modes": row["match_modes"],
            "note": ("" if not notes else ";" + "".join(notes)),
        })

    out = pd.DataFrame(rows)
    out["review_type"] = out.apply(review_type, axis=1)
    out["suggested_label"] = out.apply(suggested_label, axis=1)
    out = out.sort_values(["priority", "model_prob"], ascending=[False, False])
    # utf-8-sig: the BOM lets Excel open the Kazakh/Russian text correctly
    out.to_csv(proc / "label_review.csv", index=False, encoding="utf-8-sig")
    actionable = out[out["review_type"] != "informational"]
    actionable.to_csv(proc / "label_review_actionable.csv", index=False, encoding="utf-8-sig")

    payload = {
        "rows_total": int(len(df)),
        "rows_flagged": int(len(out)),
        "rows_actionable": int(len(actionable)),
        "by_reason": {k: int(v) for k, v in summary.items()},
        "by_review_type": {str(k): int(v) for k, v in actionable["review_type"].value_counts().items()},
        "by_suggested_label": {str(k): int(v) for k, v in actionable["suggested_label"].value_counts().items()},
        "model_threshold": clf.threshold,
    }
    (proc / "label_review_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fixes = ROOT / "data" / "label_fixes.csv"
    if not fixes.exists():
        fixes.write_text("id,label,note\n", encoding="utf-8-sig")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nreview file     : {proc / 'label_review.csv'}")
    print(f"actionable file : {proc / 'label_review_actionable.csv'}  ({len(actionable)} rows)")
    print(f"fix template    : {fixes}  (id,label,note)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
