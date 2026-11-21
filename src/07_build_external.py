"""Step 07 - build the external evaluation sets (never used for training).

E1  NEW DATASETS (genuinely different corpora, never downloaded before)
    * mangi-llm/Kazakh_Multi-Task_corpus -> sentiment_analysis.jsonl (180 curated
      opinion sentences: restaurants, movies, schools ...)
    * Arailym-tleubayeva/small_kazakh_corpus (25,922 long scientific Kazakh texts;
      a random sample of 500) - a deliberately *very* different domain, to measure
      false positives on formal prose.
    Both sources are CC-BY / Apache-2.0 and are NOT part of any training split.

Note on recall: no open, non-gated Kazakh dataset with real profanity exists (the
100K KZ movie-review corpus is gated; see README).  This external set therefore
measures the false-positive rate on never-seen domains, not recall.  Recall on
real profanity is measured on the 26 real Telegram positives of the held-out test
split (see 05_evaluate.py).

Labels are produced with exactly the same protocol as the training data
(lexicon + language rule) and PII is stripped.

Usage:
    python src/07_build_external.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon_utils import Lexicon, profanity_languages  # noqa: E402
from text_utils import detect_lang, has_pii, model_normalize, strip_pii  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "raw"
OUT = ROOT / "data" / "external"

# Manual corrections after reviewing the external labels: both rows are scientific
# Kazakh prose that the lexicon matched accidentally (numeric/math notation and a
# short-root false positive).  Recorded here so the external set is reproducible.
MANUAL_OVERRIDES = {"x00400": 0, "x00429": 0}


def load_names() -> list[str]:
    d = ROOT / "data" / "raw" / "kazakh_names"
    df = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))], ignore_index=True)
    col = "name" if "name" in df.columns else df.columns[0]
    return sorted({str(x).strip() for x in df[col].dropna() if 4 <= len(str(x).strip()) <= 30})


def build_e1(names: list[str], lexicon: Lexicon, seed: int) -> pd.DataFrame:
    rows = []

    sent = [json.loads(l) for l in (RAW / "mangi_sentiment.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in sent:
        rows.append({"text": str(r["text"]), "source": "mangi_sentiment", "external_profile": "E1_new_dataset",
                     "orig_label": str(r.get("sentiment", ""))})

    sci = pd.read_csv(RAW / "small_kazakh_corpus.csv")
    sample = sci.sample(n=500, random_state=seed)
    for _, r in sample.iterrows():
        rows.append({"text": str(r["content"]), "source": "small_kazakh_corpus",
                     "external_profile": "E1_new_dataset", "orig_label": ""})

    df = pd.DataFrame(rows)
    df["text"] = df["text"].map(lambda t: strip_pii(model_normalize(t), names)[0])
    df = df[df["text"].str.len() >= 3].reset_index(drop=True)
    df["pii_after"] = df["text"].map(has_pii)
    return df


def label(df: pd.DataFrame, lexicon: Lexicon) -> pd.DataFrame:
    labels, langs, hits, prof = [], [], [], []
    for t in df["text"]:
        lang = detect_lang(t)
        label, matches = lexicon.label_text(t)
        if lang == "ru":
            label = 0
        labels.append(label)
        langs.append(lang)
        hits.append(",".join(sorted(m.entry_id for m in matches)))
        prof.append(profanity_languages(matches))
    return df.assign(label=labels, lang=langs, lexicon_hits=hits, profanity_langs=prof)


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed = cfg["seed"]
    lexicon = Lexicon.load(ROOT / "data" / "lexicon.json")
    names = load_names()
    OUT.mkdir(parents=True, exist_ok=True)

    e1 = build_e1(names, lexicon, seed)
    df = pd.concat([e1], ignore_index=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df = label(df, lexicon)
    df["id"] = [f"x{i:05d}" for i in range(len(df))]
    if MANUAL_OVERRIDES:
        before = int(df["label"].sum())
        df["label"] = [MANUAL_OVERRIDES.get(i, l) for i, l in zip(df["id"], df["label"])]
        print(f"manual overrides applied: {len(MANUAL_OVERRIDES)} rows, positives {before} -> {int(df['label'].sum())}")
    df.to_csv(OUT / "external_labeled.csv", index=False, encoding="utf-8-sig")

    # review file: all positives + a random sample of negatives per profile
    pos = df[df.label == 1]
    neg_parts = [g.sample(n=min(25, len(g)), random_state=seed)
                 for _, g in df[df.label == 0].groupby("external_profile")]
    neg = pd.concat(neg_parts) if neg_parts else df.iloc[0:0]
    review = pd.concat([pos, neg]).sort_values(["label", "external_profile"], ascending=[False, True])
    review.to_csv(OUT / "external_review.csv", index=False, encoding="utf-8-sig")

    print(df.groupby(["external_profile", "label"]).size().to_string())
    print("\nPII remaining:", int(df["pii_after"].sum()) if "pii_after" in df else "n/a")
    print("positives by source:")
    print(df[df.label == 1].groupby("source").size().to_string())
    print(f"\nwrote {OUT / 'external_labeled.csv'} ({len(df)} rows) and external_review.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
