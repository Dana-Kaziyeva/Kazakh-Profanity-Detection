"""Predict Kazakh profanity for a comment from the command line (no retraining).

Usage:
    python -m src.predict "Өнім өте жақсы, рахмет!"
    python -m src.predict "..." --debug
    echo "..." | python -m src.predict
    python -m src.predict --threshold 0.4 "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lexicon_utils import Lexicon  # noqa: E402
from model import ProfanityClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="comment text (reads stdin when empty)")
    ap.add_argument("--model-dir", default=str(ROOT / "artifacts" / "model"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--debug", action="store_true", help="show tokens and lexicon hits")
    args = ap.parse_args()

    text = " ".join(args.text).strip() if args.text else sys.stdin.read().strip()
    if not text:
        print("no text given", file=sys.stderr)
        return 2

    clf = ProfanityClassifier(args.model_dir)
    res = clf.predict(text, threshold=args.threshold)
    print(f"text       : {res['text']}")
    print(f"prediction : {res['label']}  ({res['label_name']})")
    print(f"probability: {res['probability']:.4f}  (threshold {res['threshold']})")

    if args.debug:
        ex = clf.explain(text)
        print(f"model      : {clf.base}  max_len={clf.max_len}")
        print(f"tokens     : {len(ex['tokens'])}  truncated={ex['truncated']}  [UNK]={len(ex['unk_tokens'])}")
        print("token list : " + " | ".join(ex["tokens"][:64]))
        lex_path = ROOT / "data" / "lexicon.json"
        if lex_path.exists():
            matches = Lexicon.load(lex_path).match(text)
            print("lexicon    : " + (", ".join(f"{m.entry_id}({m.lang},{m.mode})" for m in matches) or "no rule-based hits"))
        print("json       : " + json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
