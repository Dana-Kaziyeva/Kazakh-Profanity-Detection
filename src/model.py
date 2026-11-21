"""Load the trained classifier and predict on new Kazakh comments (no retraining).

Example
-------
>>> from model import ProfanityClassifier
>>> clf = ProfanityClassifier("artifacts/model")
>>> clf.predict("Өнім өте жақсы, рахмет!")
{'label': 0, 'probability': 0.98, 'text': 'Өнім өте жақсы, рахмет!'}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# make `text_utils` importable whether this module is imported as `src.model`
# or run from inside `src/`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_utils import model_normalize  # noqa: E402

LABELS = {0: "no Kazakh profanity", 1: "Kazakh profanity present"}


class ProfanityClassifier:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.dir = Path(model_dir)
        meta_path = self.dir / "model_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.threshold = float(meta.get("threshold", 0.5))
        self.max_len = int(meta.get("max_len", 128))
        self.base = meta.get("base", "unknown")
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir)
        self.model.eval()
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = [model_normalize(t) for t in texts[i:i + batch_size]]
            enc = self.tok(batch, truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.array([])

    def predict(self, text: str, threshold: float | None = None) -> dict:
        p = float(self.predict_proba([text])[0])
        t = self.threshold if threshold is None else threshold
        return {"text": text, "label": int(p >= t), "probability": p,
                "label_name": LABELS[int(p >= t)], "threshold": t}

    def explain(self, text: str) -> dict:
        """Tokenisation trace of one comment (for the README / defense)."""
        norm = model_normalize(text)
        enc = self.tok(norm, truncation=True, max_length=self.max_len, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        toks = self.tok.convert_ids_to_tokens(ids)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        return {"text": text, "normalised": norm, "tokens": toks,
                "n_tokens": len(toks), "truncated": len(toks) >= self.max_len,
                "unk_tokens": [t for t in toks if t == self.tok.unk_token],
                "logits": logits.tolist(), "probabilities": probs,
                "p_profanity": probs[1]}
