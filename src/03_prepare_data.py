"""Step 03 - build the labelled dataset, deduplicate, group and split.

Pipeline
--------
1. load raw sources (step 01 output)
2. strip PII: URLs, e-mails, phones, personal names (Kazakh name list)
3. label with the lexicon (step 02 output) + the language-of-comment rule:
       label = 1  <=>  text contains Kazakh profanity AND is not pure Russian
   Russian/English profanity in a Kazakh/Russian/English comment -> label 0
   (hard negatives; see README "Labeling rules")
4. add synthetic rows (synthetic.py), validated against the same lexicon
5. exact de-duplication + near-duplicate grouping (token-set signature)
6. stratified group-aware split 70/15/15 (seed from config.yaml)
7. write train/val/test CSVs, split report and audit files

Usage:
    python src/03_prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lexicon_utils import Lexicon, profanity_languages
from synthetic import generate
from text_utils import detect_lang, has_pii, model_normalize, near_dup_key, strip_pii

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def read_parquets(slug: str) -> pd.DataFrame:
    d = ROOT / "data" / "raw" / slug
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise SystemExit(f"missing raw files for {slug} - run src/01_download_data.py first")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def pick_text_column(df: pd.DataFrame) -> str:
    best, best_len = None, -1.0
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(200)
            if len(sample) and sample.str.len().mean() > best_len:
                best, best_len = col, sample.str.len().mean()
    return best or "text"


def load_names() -> list[str]:
    df = read_parquets("kazakh_names")
    col = pick_text_column(df)
    names = {str(x).strip() for x in df[col].dropna() if 4 <= len(str(x).strip()) <= 30}
    names = sorted(names)
    print(f"PII name list: {len(names):,} entries (column '{col}')")
    return names


def build_real_frame(cfg: dict, names: list[str], lexicon: Lexicon, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"sources": {}}
    frames = []

    def add(df: pd.DataFrame, text_col: str, source: str, lang_hint: str | None = None) -> None:
        t = df[text_col].dropna().astype(str).map(model_normalize)
        t = t.str.replace(r"^\[\d+\]\s*", "", regex=True)  # dataset artefacts like "[1] "
        t = t[t.str.len() >= 3]
        frames.append(pd.DataFrame({"text": t, "source": source, "lang_hint": lang_hint or ""}))
        stats["sources"][source] = {"rows_loaded": int(len(df)), "rows_kept_text": int(len(t))}

    # --- real Kazakh reviews -------------------------------------------------
    for slug, source, cap in [
        ("kazakh_reviews_2gis", "2gis_reviews", None),
        ("kazakh_entertainment", "entertainment_reviews", cfg["data"]["max_kazakh_reviews"]),
    ]:
        df = read_parquets(slug)
        col = pick_text_column(df)
        df = df.sample(n=min(cap or len(df), len(df)), random_state=cfg["seed"]) if cap and len(df) > cap else df
        add(df, col, source, "kk")

    # --- real Russian product reviews (clean text) ---------------------------
    df = read_parquets("russian_reviews")
    col = pick_text_column(df)
    cap = cfg["data"]["max_russian_reviews"]
    if len(df) > cap:
        df = df.sample(n=cap, random_state=cfg["seed"])
    add(df, col, "ru_reviews", "ru")

    # --- real Russian toxic corpus: keep rows verified to contain RU profanity
    df = read_parquets("russian_toxic")
    col = pick_text_column(df)
    n_scan = min(cfg["data"]["ru_scan_rows"], len(df))
    scan = df.sample(n=n_scan, random_state=cfg["seed"])
    ru_lex = Lexicon([e for e in lexicon.entries if e["lang"] == "ru"])
    hits = []
    for _, row in scan.iterrows():
        text = model_normalize(str(row[col]))
        if len(text) < 3:
            continue
        if ru_lex.match(text):
            hits.append(text)
        if len(hits) >= cfg["data"]["max_hard_negatives"]:
            break
    print(f"russian_toxic: scanned {n_scan:,} rows, kept {len(hits):,} rows with Russian profanity")
    frames.append(pd.DataFrame({"text": hits, "source": "ru_toxic", "lang_hint": "ru"}))
    stats["sources"]["ru_toxic"] = {"rows_loaded": int(len(df)), "rows_scanned": n_scan, "rows_kept": len(hits)}

    # --- real Kazakh profanity mined from the Telegram chat corpus -----------
    cache = ROOT / "data" / "raw" / "_derived"
    cache.mkdir(parents=True, exist_ok=True)
    pos_cache, neg_cache = cache / "telegram_pos.parquet", cache / "telegram_clean.parquet"
    if pos_cache.exists() and neg_cache.exists() and not refresh:
        positives = pd.read_parquet(pos_cache)["text"].tolist()
        negatives = pd.read_parquet(neg_cache)["text"].tolist()
        print(f"telegram: loaded from cache ({len(positives)} positives / {len(negatives)} clean negatives)")
    else:
        tg = read_parquets("kazakh_telegram")
        col = pick_text_column(tg)
        messages = tg[col].dropna().astype(str).map(model_normalize)
        messages = messages[(messages.str.len() >= 12) & (messages.str.len() <= 600)].reset_index(drop=True)
        kk_letters = messages.str.contains(r"[әғқңөұүһі]", regex=True)
        triggers = sorted({e["root"][:2] for e in lexicon.entries
                           if e["lang"] in ("kk", "ru") and len(e["root"]) >= 2})
        pat = re.compile("|".join(re.escape(x) for x in triggers))
        cand = messages[kk_letters & messages.str.contains(pat, regex=True, na=False)]
        print(f"telegram: {len(messages):,} messages, {int(kk_letters.sum()):,} Kazakh script, "
              f"{len(cand):,} candidates - scanning ...", flush=True)
        positives, t0 = [], time.time()
        for i, text in enumerate(cand, 1):
            label, _ = lexicon.label_text(text)
            if label == 1:
                positives.append(text)
            if i % 1000 == 0:
                print(f"  scanned {i:,}/{len(cand):,}  positives={len(positives):,} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        positives = list(dict.fromkeys(positives))
        random.Random(cfg["seed"]).shuffle(positives)
        positives = positives[: cfg["data"]["max_telegram_positives"]]
        clean = messages[kk_letters & (~messages.index.isin(cand.index)) & (messages.str.len() >= 30)]
        negatives = clean.sample(n=min(cfg["data"]["max_telegram_negatives"], len(clean)),
                                 random_state=cfg["seed"]).tolist()
        pd.DataFrame({"text": positives}).to_parquet(pos_cache, index=False)
        pd.DataFrame({"text": negatives}).to_parquet(neg_cache, index=False)
    print(f"telegram: {len(positives):,} real positives, {len(negatives):,} clean negatives")
    frames.append(pd.DataFrame({"text": positives, "source": "telegram_pos", "lang_hint": "kk"}))
    frames.append(pd.DataFrame({"text": negatives, "source": "telegram_clean", "lang_hint": "kk"}))
    stats["sources"]["telegram_pos"] = {"rows_kept": len(positives)}
    stats["sources"]["telegram_clean"] = {"rows_kept": len(negatives)}

    # --- author-written Kazakh profanity phrases from the swear-words dataset
    #     (only those whose profanity is Kazakh-origin are kept: the dataset
    #      labels Russian мат phrases as 'profanity' too - their labels are
    #      revised here, see README)
    sw = read_parquets("kazakh_swear_words")
    phrase_pos = []
    for _, row in sw[sw["label"] == 1].iterrows():
        text = model_normalize(str(row["text"]))
        if " " in text and lexicon.label_text(text)[0] == 1:
            phrase_pos.append(text)
    phrase_pos = sorted(set(phrase_pos))
    frames.append(pd.DataFrame({"text": phrase_pos, "source": "swear_words_phrases", "lang_hint": "kk"}))
    stats["sources"]["swear_words_phrases"] = {"rows_kept": len(phrase_pos)}
    print(f"swear-words phrases kept as positives (Kazakh-origin only): {len(phrase_pos)}")

    real = pd.concat(frames, ignore_index=True)

    # --- PII stripping --------------------------------------------------------
    removed_total, cleaned = 0, []
    for text in real["text"]:
        clean, n = strip_pii(text, names)
        removed_total += n
        cleaned.append(clean)
    real = real.assign(text=cleaned)
    real = real[real["text"].str.len() >= 3].reset_index(drop=True)
    real["has_pii_after"] = real["text"].map(has_pii)
    stats["pii"] = {"substitutions": int(removed_total),
                    "rows_still_with_pii": int(real["has_pii_after"].sum())}

    # exact duplicates
    before = len(real)
    real["norm"] = real["text"].map(model_normalize)
    real = real.drop_duplicates(subset="norm").drop(columns="norm")
    stats["exact_duplicates_removed"] = before - len(real)
    return real.reset_index(drop=True), stats


def label_real(real: pd.DataFrame, lexicon: Lexicon) -> pd.DataFrame:
    labels, langs, hits, prof_langs, modes = [], [], [], [], []
    for text in real["text"]:
        lang = detect_lang(text)
        label, matches = lexicon.label_text(text)
        # Language-of-comment rule: pure Russian text is never Kazakh profanity,
        # even when a Kazakh-lexicon word form coincides (e.g. loanwords).
        if lang == "ru":
            label = 0
        labels.append(label)
        langs.append(lang)
        hits.append(",".join(sorted(m.entry_id for m in matches)))
        prof_langs.append(profanity_languages(matches))
        modes.append(",".join(sorted({m.mode for m in matches})))
    out = real.copy()
    out["label"] = labels
    out["lang"] = langs
    out["lexicon_hits"] = hits
    out["profanity_langs"] = prof_langs
    out["match_modes"] = modes
    out["synthetic"] = False
    out["template_id"] = ""
    return out


def split_grouped(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> tuple[pd.Series, pd.Series]:
    """Group-aware stratified split: whole near-duplicate groups land in one split.

    Returns (group_key, split) aligned to ``df``.
    """
    df = df.copy()
    df["group"] = df["text"].map(near_dup_key)
    g = (df.groupby("group")
           .agg(label=("label", "max"), n=("label", "size"))
           .reset_index()
           .sample(frac=1.0, random_state=seed)
           .reset_index(drop=True))
    totals = g.groupby("label")["n"].sum().to_dict()
    counts = {s: {0: 0, 1: 0} for s in fractions}
    assign: dict[str, str] = {}
    for _, row in g.iterrows():
        lab = int(row["label"])
        target_total = totals.get(lab, 0)
        best_split, best_score = None, -1e9
        for s, frac in fractions.items():
            deficit = frac * target_total - counts[s][lab]
            score = deficit / frac
            if score > best_score:
                best_split, best_score = s, score
        assign[row["group"]] = best_split
        counts[best_split][lab] += int(row["n"])
    return df["group"], df["group"].map(assign)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute derived caches (Telegram mining)")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    lexicon = Lexicon.load(ROOT / "data" / "lexicon.json")

    print("loading raw data ...")
    names = load_names()
    real, stats = build_real_frame(cfg, names, lexicon, refresh=args.refresh)
    print(f"real rows after cleaning: {len(real):,}")

    real = label_real(real, lexicon)
    print(f"real rows labelled: {len(real):,}  positives={int(real['label'].sum()):,}")
    print(real.groupby(["source", "label"]).size().to_string())

    sc = cfg["data"]["synthetic"]
    syn = generate(lexicon, sc["positives"], sc["negatives"],
                   sc["hard_negatives_ru"], sc["hard_negatives_en"], seed)
    syn["profanity_langs"] = [
        profanity_languages(lexicon.match(t)) for t in syn["text"]
    ]
    syn["match_modes"] = [
        ",".join(sorted({m.mode for m in lexicon.match(t)})) for t in syn["text"]
    ]
    syn["lang"] = syn["text"].map(detect_lang)

    df = pd.concat([real, syn], ignore_index=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["id"] = [f"r{i:06d}" for i in range(len(df))]
    print(f"combined rows: {len(df):,}  (synthetic share {df['synthetic'].mean():.1%})")

    # --- manual label corrections (data/label_fixes.csv: id,label,note) -------
    fixes_path = ROOT / "data" / "label_fixes.csv"
    if fixes_path.exists():
        fixes = pd.read_csv(fixes_path, encoding="utf-8-sig")
        fixes = fixes.dropna(subset=["id", "label"]) if len(fixes) else fixes
        if len(fixes):
            mapping = dict(zip(fixes["id"].astype(str), fixes["label"].astype(int)))
            unknown = set(mapping) - set(df["id"])
            if unknown:
                print(f"  !! {len(unknown)} label-fix ids are not in the dataset (ignored)")
            new_label = df["id"].map(mapping)
            changed = new_label.notna()
            before = df.loc[changed, "label"]
            df.loc[changed, "label"] = new_label[changed].astype(int)
            n01 = int(((before == 0) & (df.loc[changed, "label"] == 1)).sum())
            n10 = int(((before == 1) & (df.loc[changed, "label"] == 0)).sum())
            print(f"label fixes applied: {int(changed.sum())} rows (0->1: {n01}, 1->0: {n10})")

    df["group"], df["split"] = split_grouped(df, cfg["data"]["splits"], seed)

    cols = ["id", "text", "label", "lang", "source", "synthetic", "template_id",
            "lexicon_hits", "match_modes", "profanity_langs", "group", "split"]
    df = df[cols].sort_values("id").reset_index(drop=True)
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].drop(columns=["group", "split"])
        part.to_csv(proc / f"{split}.csv", index=False, encoding="utf-8")
        print(f"  {split:5s}: {len(part):5,} rows  positives={int(part['label'].sum()):4,} "
              f"({part['label'].mean():.1%})  synthetic={part['synthetic'].mean():.1%}")

    # audit material for manual verification
    real_only = df[~df["synthetic"].astype(bool)]
    real_only[real_only["label"] == 1].to_csv(proc / "audit_positives.csv", index=False, encoding="utf-8-sig")
    neg_sample = real_only[real_only["label"] == 0].sample(
        n=min(150, int((real_only["label"] == 0).sum())), random_state=seed)
    neg_sample.to_csv(proc / "audit_negatives_sample.csv", index=False, encoding="utf-8-sig")

    report = {
        "seed": seed,
        "total_rows": int(len(df)),
        "class_distribution": df["label"].value_counts().sort_index().to_dict(),
        "synthetic_share": float(df["synthetic"].mean()),
        "splits": {
            s: {"rows": int((df["split"] == s).sum()),
                "positives": int(df.loc[df["split"] == s, "label"].sum()),
                "pos_rate": float(df.loc[df["split"] == s, "label"].mean())}
            for s in ["train", "val", "test"]},
        "language_distribution": df["lang"].value_counts().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict(),
        "unique_near_dup_groups": int(df["group"].nunique()),
        "largest_group": int(df.groupby("group").size().max()),
        "raw_stats": stats,
    }
    (proc / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    if stats["pii"]["rows_still_with_pii"]:
        print("!! WARNING: PII patterns remain in some rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
