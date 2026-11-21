"""Step 04 - fine-tune the Kazakh BERT model for binary profanity classification.

Two stages, mirroring the plan:

  grid  : small validation-driven search over {lr} x {max_len} on a training
          subsample (CPU-friendly), results in artifacts/grid_results.json
  final : full fine-tune with the winning settings, early stopping on
          validation F1 of the profanity class, then threshold sweep on the
          validation set.  Saves model + tokenizer + metadata to artifacts/model.

Usage:
    python src/04_train.py --stage grid
    python src/04_train.py --stage final --lr 2e-5 --max-len 128
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class ReviewDataset(Dataset):
    """Tokenises on the fly; padding happens per batch (dynamic padding).

    Most comments are far shorter than ``max_len``, so dynamic padding cuts
    the padded-token waste by roughly 3-4x compared with padding to a fixed
    maximum - decisive on CPU.
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_len)
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": int(self.labels[idx])}


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attention = [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad] * pad)
            attention.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
        }


@torch.no_grad()
def predict_probs(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for batch in loader:
        labels = batch.pop("labels")
        logits = model(**batch).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
        ys.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def metrics_at(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", pos_label=1, zero_division=0)
    macro = f1_score(y_true, pred, average="macro", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "f1_macro": float(macro)}


def build_model(base: str, fallback: str, seed: int):
    """Load the pretrained encoder as a 2-class classifier."""
    try:
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=2, ignore_mismatched_sizes=True
        )
        return tok, model, base
    except Exception as exc:  # noqa: BLE001 - documented fallback
        print(f"!! could not load {base} ({exc}); falling back to {fallback}")
        tok = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForSequenceClassification.from_pretrained(
            fallback, num_labels=2, ignore_mismatched_sizes=True
        )
        return tok, model, fallback


def train_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base: str,
    fallback: str,
    seed: int,
    lr: float,
    max_len: int,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    tag: str,
    patience: int = 1,
    save_dir: Path | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    set_seed(seed)
    tok, model, used_base = build_model(base, fallback, seed)
    model.train()

    train_ds = ReviewDataset(train_df["text"].tolist(), train_df["label"].tolist(), tok, max_len)
    val_ds = ReviewDataset(val_df["text"].tolist(), val_df["label"].tolist(), tok, max_len)
    collate = Collator(tok.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * warmup_ratio), total_steps)

    best = {"f1": -1.0, "epoch": -1, "state": None}
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"  [{tag}] epoch {epoch} step {step}/{len(train_loader)} loss={running/step:.4f}", flush=True)
        probs, ys = predict_probs(model, val_loader)
        m = metrics_at(ys, probs, 0.5)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": running / len(train_loader), **m, "elapsed_s": dt})
        print(f"  [{tag}] epoch {epoch}: val loss={running/len(train_loader):.4f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} ({dt/60:.1f} min)", flush=True)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "epoch": epoch, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= patience:
            print(f"  [{tag}] early stop (no val-F1 improvement for {patience} epoch)")
            break

    model.load_state_dict(best["state"])
    probs, ys = predict_probs(model, val_loader)
    summary = {
        "tag": tag, "base_model": used_base, "lr": lr, "max_len": max_len,
        "epochs_run": history[-1]["epoch"], "best_epoch": best["epoch"],
        "best_val_f1": best["f1"], "history": history,
        "train_size": len(train_df), "val_size": len(val_df),
        "seed": seed, "batch_size": batch_size, "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio, "elapsed_min": (time.time() - t0) / 60,
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        summary["saved_to"] = str(save_dir)
    return summary, probs, ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "final"], required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=800, help="grid stage: training rows to use")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, cpus={os.cpu_count()}")

    proc = ROOT / cfg["paths"]["data_processed"]
    train_df = pd.read_csv(proc / "train.csv")
    val_df = pd.read_csv(proc / "val.csv")
    print(f"train={len(train_df):,} val={len(val_df):,}")

    mc = cfg["model"]
    art = ROOT / cfg["paths"]["artifacts"]

    if args.stage == "grid":
        results = []
        sub = train_df.sample(n=min(args.subsample, len(train_df)), random_state=seed)
        for lr in cfg["grid"]["lr"]:
            for max_len in cfg["grid"]["max_len"]:
                tag = f"grid_lr{lr:g}_len{max_len}"
                print(f"\n=== grid run {tag} (train subsample {len(sub)})")
                summ, _, _ = train_one(
                    sub, val_df, mc["base"], mc["fallback_base"], seed, lr, max_len,
                    epochs=1, batch_size=mc["batch_size"], weight_decay=mc["weight_decay"],
                    warmup_ratio=mc["warmup_ratio"], tag=tag,
                )
                results.append(summ)
        results.sort(key=lambda r: r["best_val_f1"], reverse=True)
        (art / "grid_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("\n=== grid results (best first)")
        for r in results:
            print(f"  lr={r['lr']:g} max_len={r['max_len']:3d} val_f1={r['best_val_f1']:.4f} "
                  f"(epoch {r['best_epoch']}, {r['elapsed_min']:.1f} min)")
        print(f"\nbest: lr={results[0]['lr']:g} max_len={results[0]['max_len']}")
        return 0

    # ---------------- final stage ----------------
    grid_path = art / "grid_results.json"
    lr, max_len = mc["lr"], mc["max_len"]
    if grid_path.exists():
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        lr, max_len = grid[0]["lr"], grid[0]["max_len"]
    if args.lr is not None:
        lr = args.lr
    if args.max_len is not None:
        max_len = args.max_len
    epochs = args.epochs or mc["epochs"]

    model_dir = ROOT / cfg["paths"]["model_dir"]
    max_train_rows = cfg["data"].get("max_train_rows", 0)
    if max_train_rows and len(train_df) > max_train_rows:
        per_class = max_train_rows // 2
        parts = []
        for lab, grp in train_df.groupby("label"):
            n = min(len(grp), per_class)
            parts.append(grp.sample(n=n, random_state=seed))
        train_df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"final training subsample (class-balanced): {len(train_df):,} rows")
    print(f"\n=== final training: lr={lr:g} max_len={max_len} epochs={epochs}")
    summ, probs, ys = train_one(
        train_df, val_df, mc["base"], mc["fallback_base"], seed, lr, max_len,
        epochs=epochs, batch_size=mc["batch_size"], weight_decay=mc["weight_decay"],
        warmup_ratio=mc["warmup_ratio"], tag="final", patience=1, save_dir=model_dir,
    )

    # threshold sweep on the validation set
    sweep = []
    for t in cfg["threshold_grid"]:
        sweep.append({"threshold": t, **metrics_at(ys, probs, t)})
    best_t = max(sweep, key=lambda x: x["f1"])
    summary = {
        "model": {"base": summ["base_model"], "max_len": max_len, "threshold": best_t["threshold"],
                  "labels": {"0": "no Kazakh profanity", "1": "Kazakh profanity present"}},
        "training": summ, "threshold_sweep": sweep, "chosen_threshold": best_t,
        "config": cfg,
    }
    (art / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (model_dir / "model_meta.json").write_text(
        json.dumps({"base": summ["base_model"], "max_len": max_len, "threshold": best_t["threshold"],
                    "labels": {"0": "no Kazakh profanity", "1": "Kazakh profanity present"},
                    "seed": seed}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nchosen threshold={best_t['threshold']} (val F1={best_t['f1']:.4f})")
    print(f"saved to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
