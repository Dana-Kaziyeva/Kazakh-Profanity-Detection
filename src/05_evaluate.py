"""Step 05 - evaluate the trained model ONCE on the held-out test set.

Reported:
* precision / recall / F1 for the profanity class (+ macro F1, accuracy)
* confusion matrix and the number of test examples
* the Option-B boundary probe: correct-0 rate on Russian/English profanity
  hard negatives, recall on mixed Kazakh+Russian comments that DO contain
  Kazakh profanity, masked-form recall
* tokenizer probe: [UNK] rates per language (KazakhBERTmulti was pretrained on
  Kazakh only - this quantifies how it sees Russian/English text)
* one end-to-end trace (raw text -> tokens -> logits -> probability -> label)

Usage:
    python src/05_evaluate.py [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProfanityClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def metrics_block(y: np.ndarray, p: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
    return {
        "n": int(len(y)),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "f1_macro": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    art = ROOT / "artifacts"
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    clf = ProfanityClassifier(ROOT / "artifacts" / "model")
    thr = args.threshold if args.threshold is not None else clf.threshold
    print(f"model={clf.base}  max_len={clf.max_len}  threshold={thr}  test rows={len(test)}")

    probs = clf.predict_proba(test["text"].tolist())
    pred = (probs >= thr).astype(int)
    y = test["label"].to_numpy()

    overall = metrics_block(y, pred)
    print("\n=== overall test metrics")
    print(json.dumps(overall, indent=2))

    # ---------------- boundary probe (Option B) ----------------
    def subset_stats(name: str, mask: pd.Series, positive: bool) -> dict:
        mask = mask & (test["label"] == (1 if positive else 0))
        sub = test[mask]
        if len(sub) == 0:
            return {"subset": name, "n": 0}
        pr = (probs[mask.to_numpy()] >= thr).astype(int)
        yy = sub["label"].to_numpy()
        if positive:
            rate = float((pr == 1).mean())  # recall on the profanity rows
            metric = "recall"
        else:
            rate = float((pr == 0).mean())  # correct-0 rate on the clean/hard-negative rows
            metric = "correct_zero_rate"
        return {"subset": name, "n": int(len(sub)), metric: round(rate, 4)}

    syn = test["synthetic"].astype(bool)
    tmpl = test["template_id"].fillna("")
    real_kk = (~syn) & test["source"].isin(["2gis_reviews", "entertainment_reviews"])
    probe = [
        subset_stats("real Kazakh reviews labelled 0", real_kk & (test["label"] == 0), False),
        subset_stats("real Kazakh reviews labelled 1", real_kk & (test["label"] == 1), True),
        subset_stats("RU hard negatives (real toxic corpus)", test["source"] == "ru_toxic", False),
        subset_stats("RU hard negatives (synthetic, KK frame)", tmpl.str.startswith("hard_neg_ru"), False),
        subset_stats("EN hard negatives (synthetic)", tmpl.str.startswith("hard_neg_en"), False),
        subset_stats("mixed KK+RU comments WITH KK profanity", tmpl == "kk_pos_profane_mixed_ru_words", True),
        subset_stats("masked-form positives (all)", test["match_modes"].fillna("").str.contains("masked"), True),
        subset_stats("synthetic positives (all)", syn & (test["label"] == 1), True),
        subset_stats("real Telegram positives", test["source"] == "telegram_pos", True),
        subset_stats("insult-without-profanity negatives", tmpl == "kk_negative_insult_without_profanity", False),
    ]
    probe = [p for p in probe if p and p.get("n", 0) > 0]
    print("\n=== boundary probe")
    for p in probe:
        print(" ", p)

    # per-language breakdown of the overall metrics
    by_lang = {}
    for lang in sorted(test["lang"].unique()):
        m = test["lang"] == lang
        if m.sum() >= 10:
            by_lang[lang] = metrics_block(y[m.to_numpy()], pred[m.to_numpy()])
    print("\n=== metrics by language")
    print(json.dumps(by_lang, indent=2))

    # ---------------- tokenizer probe ----------------
    tok_probe = {}
    for lang in ["kk", "ru", "en"]:
        texts = test.loc[test["lang"] == lang, "text"].head(50).tolist()
        if lang == "en" and not texts:
            texts = test.loc[test["source"] == "synthetic", "text"].head(50).tolist()
        unk, tot = 0, 0
        for t in texts:
            ids = clf.tok(t, truncation=True, max_length=clf.max_len)["input_ids"]
            toks = clf.tok.convert_ids_to_tokens(ids)
            unk += sum(1 for x in toks if x == clf.tok.unk_token)
            tot += len(toks)
        tok_probe[lang] = {"texts": len(texts), "unk_tokens": unk, "total_tokens": tot,
                           "unk_rate": round(unk / tot, 4) if tot else None}
    print("\n=== tokenizer probe ([UNK] share per language)")
    print(json.dumps(tok_probe, indent=2))

    # ---------------- save predictions and errors ----------------
    out = test[["id", "text", "label", "lang", "source", "synthetic", "template_id",
                "lexicon_hits", "match_modes", "profanity_langs"]].copy()
    out["prob_profanity"] = probs.round(6)
    out["prediction"] = pred
    out["correct"] = (pred == y).astype(int)
    out.to_csv(art / "sample_predictions.csv", index=False, encoding="utf-8-sig")
    errors = out[out["correct"] == 0].copy()
    errors["margin"] = (errors["prob_profanity"] - thr).abs().round(6)
    errors.sort_values("margin").to_csv(art / "errors.csv", index=False, encoding="utf-8-sig")

    # ---------------- one end-to-end trace ----------------
    mixed = test[tmpl == "kk_pos_profane_mixed_ru_words"]
    trace_row = mixed.iloc[0] if len(mixed) else test.iloc[0]
    trace = clf.explain(trace_row["text"])
    trace["true_label"] = int(trace_row["label"])
    trace["predicted_label"] = int(probs[test.index.get_loc(trace_row.name)] >= thr)
    (art / "trace_example.txt").write_text(
        "END-TO-END TRACE (one test comment)\n"
        f"text: {trace['text']}\n"
        f"normalised: {trace['normalised']}\n"
        f"tokens ({trace['n_tokens']}, truncated={trace['truncated']}): {' | '.join(trace['tokens'])}\n"
        f"[UNK] tokens: {trace['unk_tokens']}\n"
        f"logits: {trace['logits']}\n"
        f"probabilities: {trace['probabilities']}\n"
        f"p(profanity)={trace['p_profanity']:.4f}  threshold={thr}  "
        f"predicted={trace['predicted_label']}  true={trace['true_label']}\n",
        encoding="utf-8")

    cm_text = (
        f"Confusion matrix (rows = true, cols = predicted), threshold={thr}\n"
        f"                 pred 0    pred 1\n"
        f"  true 0 (clean) {overall['confusion']['tn']:8d}  {overall['confusion']['fp']:8d}\n"
        f"  true 1 (prof.) {overall['confusion']['fn']:8d}  {overall['confusion']['tp']:8d}\n"
        f"\nTest examples: {overall['n']}\n"
        f"Precision={overall['precision']:.4f} Recall={overall['recall']:.4f} "
        f"F1={overall['f1']:.4f} F1_macro={overall['f1_macro']:.4f} Accuracy={overall['accuracy']:.4f}\n"
    )
    (art / "confusion_matrix.txt").write_text(cm_text, encoding="utf-8")
    print("\n" + cm_text)

    payload = {
        "threshold": thr, "overall": overall, "boundary_probe": probe,
        "by_language": by_lang, "tokenizer_probe": tok_probe,
        "model_base": clf.base, "max_len": clf.max_len,
    }
    (art / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: metrics.json, confusion_matrix.txt, sample_predictions.csv "
          f"({len(out)} rows), errors.csv ({len(errors)} rows), trace_example.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
