"""Step 08 - evaluate the trained model on the EXTERNAL datasets (step 07 output).

The external set was never used for training, validation or test.  It contains
almost no profanity (public Kazakh datasets are moderated), so the primary
number is the **false-positive rate on never-seen domains**, plus a manual list
of the comments the model flags.

Usage:
    python src/08_external_eval.py [--model-dir artifacts/model]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProfanityClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "artifacts" / "model"))
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "external" / "external_labeled.csv", encoding="utf-8-sig")
    clf = ProfanityClassifier(args.model_dir)
    probs = clf.predict_proba(df["text"].tolist(), batch_size=16)
    pred = (probs >= clf.threshold).astype(int)
    y = df["label"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n_pos = int(y.sum())
    report = {
        "model": clf.base, "threshold": clf.threshold, "rows": int(len(df)),
        "positives": n_pos, "negatives": int(len(df) - n_pos),
        "flagged": int(pred.sum()),
        "false_positive_rate": round(float(fp / max(len(df) - n_pos, 1)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "by_source": {},
    }
    if n_pos:
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
        report.update({"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4)})

    for src, g in df.assign(pred=pred, prob=probs).groupby("source"):
        report["by_source"][src] = {
            "n": int(len(g)), "flagged": int(g["pred"].sum()),
            "flag_rate": round(float(g["pred"].mean()), 4),
            "mean_prob": round(float(g["prob"].mean()), 4),
            "max_prob": round(float(g["prob"].max()), 4),
            "positives": int(g["label"].sum()),
        }

    out = df.assign(probability=probs.round(6), prediction=pred, correct=(pred == y).astype(int))
    out.to_csv(ROOT / "artifacts" / "external_predictions.csv", index=False, encoding="utf-8-sig")
    (ROOT / "artifacts" / "external_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== flagged comments on the new datasets (manual inspection)")
    flagged = out[out.prediction == 1].sort_values("probability", ascending=False)
    for _, r in flagged.head(15).iterrows():
        print(f"  p={r['probability']:.3f} [{r['source']}] {str(r['text'])[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
