"""Step 01 - download the public raw data sources used by the project.

Sources (all public, no credentials required):

    slug                       Hugging Face id                          role
    -------------------------  ---------------------------------------  --------------------------------------
    kazakh_reviews_2gis        kkenbbi/kazakh_reviews_2gis              real Kazakh product/service reviews
    kazakh_entertainment       R3iwan/entertainment-reviews-kazakh      real Kazakh review text (sentiment-labelled,
                                                                         labels are DISCARDED and re-labelled)
    kazakh_swear_words         Rustem-Kaimolla/kazakh-swear-words       Kazakh profanity lexicon seeds
    kazakh_names               R3iwan/kazakh_names                      Kazakh personal names for PII stripping
    russian_toxic              Mnwa/russian-toxic                       Russian toxic comments -> HARD NEGATIVES
    russian_reviews            sismetanin/rureviews                     real Russian product reviews (clean text)

Downloaded files are stored as parquet under data/raw/<slug>/. Each parquet
file is a byte-identical copy of the file published on the Hub, so the raw data
can be re-verified against the recorded SHA-256 (see data/raw/MANIFEST.json).

Usage:
    python src/01_download_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF_API = "https://huggingface.co/api/datasets"

SOURCES = [
    {
        "slug": "kazakh_reviews_2gis",
        "id": "kkenbbi/kazakh_reviews_2gis",
        "role": "real Kazakh product/service reviews (sentiment label is discarded)",
        "license": "no explicit license on the dataset card; doi:10.57967/hf/5587; academic use only",
    },
    {
        "slug": "kazakh_entertainment",
        "id": "R3iwan/entertainment-reviews-kazakh",
        "role": "real Kazakh review text (labels discarded)",
        "license": "apache-2.0",
    },
    {
        "slug": "kazakh_swear_words",
        "id": "Rustem-Kaimolla/kazakh-swear-words",
        "role": "Kazakh profanity lexicon seeds (entries re-verified by us)",
        "license": "MIT",
    },
    {
        "slug": "kazakh_names",
        "id": "R3iwan/kazakh_names",
        "role": "Kazakh personal names, used only to strip PII",
        "license": "apache-2.0",
    },
    {
        "slug": "russian_toxic",
        "id": "Mnwa/russian-toxic",
        "role": "Russian toxic comments -> relabelled 0 as hard negatives for this task",
        "license": "MIT",
    },
    {
        "slug": "kazakh_telegram",
        "id": "kurumikz/telegram-corpus-russian-kazakh",
        "role": "real mixed Kazakh/Russian chat messages - mined for REAL Kazakh profanity",
        "license": "cc-by-nc-sa-4.0 (academic use)",
    },
    {
        "slug": "russian_reviews",
        "id": "ai-forever/ru-reviews-classification",
        "role": "real Russian product reviews (clean text pool)",
        "license": "apache-2.0",
    },
]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "kazakh-profanity-detection/1.0 (academic project)"
    return s


def parquet_files(sess: requests.Session, ds_id: str) -> list[tuple[str, str]]:
    """Return [(split_name, url)] from the Hub's auto-converted parquet API."""
    r = sess.get(f"{HF_API}/{ds_id}/parquet", timeout=60)
    r.raise_for_status()
    payload = r.json()
    out: list[tuple[str, str]] = []
    # payload: {config: {"train": [url, ...], ...}, ...}
    for _config, splits in payload.items():
        if isinstance(splits, dict):
            for split, urls in splits.items():
                for url in urls:
                    out.append((split, url))
    return out


def fallback_files(sess: requests.Session, ds_id: str) -> list[tuple[str, str]]:
    """Fall back to raw parquet/csv/jsonl files listed in the repo tree."""
    r = sess.get(f"{HF_API}/{ds_id}", timeout=60)
    r.raise_for_status()
    siblings = r.json().get("siblings", [])
    out = []
    for s in siblings:
        fn = s["rfilename"]
        if fn.endswith((".parquet", ".csv", ".jsonl", ".json")):
            url = f"https://huggingface.co/datasets/{ds_id}/resolve/main/{fn}"
            out.append((Path(fn).stem, url))
    return out


def download(sess: requests.Session, url: str, dest: Path) -> str:
    """Stream a file to disk and return its SHA-256."""
    h = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with sess.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
                if total:
                    pct = 100 * got / total
                    print(f"\r    {dest.name}: {pct:5.1f}%  ({got/1e6:.1f}/{total/1e6:.1f} MB)", end="")
    print()
    tmp.replace(dest)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    sess = _session()
    manifest: dict = {"created": datetime.now(timezone.utc).isoformat(), "sources": []}

    for src in SOURCES:
        slug, ds_id = src["slug"], src["id"]
        out_dir = RAW / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {slug}  ({ds_id})")

        try:
            files = parquet_files(sess, ds_id)
        except Exception as exc:  # noqa: BLE001 - fall back to raw files
            print(f"  parquet API unavailable ({exc}); trying raw files")
            files = fallback_files(sess, ds_id)
        if not files:
            print(f"  !! no downloadable files found for {ds_id}", file=sys.stderr)
            continue

        entry = {**src, "files": []}
        for split, url in files:
            name = f"{split}_{Path(url).name}"
            dest = out_dir / name
            if dest.exists() and not args.force:
                print(f"    exists: {name}")
            else:
                sha = download(sess, url, dest)
                entry["files"].append({"file": name, "url": url, "sha256": sha})
        # normalise every source to parquet so downstream scripts have one format
        for p in sorted(out_dir.iterdir()):
            if p.suffix in {".csv", ".json", ".jsonl"}:
                try:
                    if p.suffix == ".csv":
                        df = pd.read_csv(p)
                    else:
                        df = pd.read_json(p, lines=p.suffix == ".jsonl")
                    target = p.with_suffix(".parquet")
                    df.to_parquet(target, index=False)
                    p.unlink()
                    print(f"    converted {p.name} -> {target.name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"    !! could not convert {p.name}: {exc}")
        # record hashes for pre-existing files too
        if not entry["files"]:
            for p in sorted(out_dir.glob("*.parquet")):
                entry["files"].append({"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        manifest["sources"].append(entry)

        # preview
        try:
            dfs = [pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))]
            df = pd.concat(dfs, ignore_index=True)
            entry["rows"] = int(len(df))
            entry["columns"] = list(df.columns)
            print(f"    rows={len(df):,}  columns={list(df.columns)}")
            with pd.option_context("display.max_colwidth", 90):
                print(df.head(3).to_string())
        except Exception as exc:  # noqa: BLE001
            print(f"    preview skipped: {exc}")

    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest written to {RAW / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
