"""Rebuild MVTec AD 2's official folder layout from the pjramg/MVTecAD2_FO mirror.

MVTec's own download servers (mydrive.ch) were returning 404 for everyone as of
2026-10 (see README "Known pitfalls"). This mirror is a FiftyOne export of the
*public* portion of AD 2 -- train, validation, and test_public, across all 8
categories -- rehosted on the Hugging Face Hub. It does not and cannot contain
test_private / test_private_mixed, since MVTec never published ground truth
for those; they only exist behind MVTec's own evaluation server.

The mirror stores every ground-truth mask inline inside samples.json as a
zlib-compressed .npy array (a FiftyOne Segmentation label), not as loose PNG
files. This script decodes those and writes everything into the directory
structure anomalib.data.MVTecAD2 expects:

    <root>/<category>/train/good/*.png
    <root>/<category>/validation/good/*.png
    <root>/<category>/test_public/good/*.png
    <root>/<category>/test_public/bad/*.png
    <root>/<category>/test_public/ground_truth/bad/*_mask.png

Source: https://huggingface.co/datasets/pjramg/MVTecAD2_FO (unofficial mirror).
Original dataset: MVTec AD 2, CC BY-NC-SA 4.0,
https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2

    uv run python scripts/prepare_ad2_mirror.py --root datasets/MVTec_AD_2
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

REPO_ID = "pjramg/MVTecAD2_FO"
FOLDER_FOR = {"train": "train", "validation": "validation", "test_public": "test_public"}


def decode_mask(binary_field: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(binary_field["base64"]))
    return np.load(io.BytesIO(raw), allow_pickle=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--cache-dir", type=Path, default=Path("datasets/.hf_cache"))
    p.add_argument(
        "--copy", action="store_true", help="Copy images instead of hard-linking (use on a different filesystem)."
    )
    args = p.parse_args()

    print(f"Downloading {REPO_ID} snapshot (images + samples.json)...")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            cache_dir=args.cache_dir,
            allow_patterns=["data/*.png", "samples.json"],
        )
    )
    samples = json.loads((snapshot_dir / "samples.json").read_text(encoding="utf-8"))["samples"]

    link = shutil.copy2 if args.copy else (lambda src, dst: Path(dst).hardlink_to(src))
    n_images, n_masks = 0, 0
    for s in tqdm(samples, desc="placing files"):
        category = s["category_label"]["label"]
        folder = FOLDER_FOR[s["folder_type"]]
        defect = s["defect_label"]["label"]
        src = snapshot_dir / s["filepath"]
        stem = Path(s["filepath"]).stem

        if folder == "test_public":
            dest_dir = args.root / category / folder / defect
        else:
            dest_dir = args.root / category / folder / "good"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{stem}.png"
        if not dest.exists():
            link(src, dest)
        n_images += 1

        if defect == "bad":
            mask_dir = args.root / category / "test_public" / "ground_truth" / "bad"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{stem}_mask.png"
            if not mask_path.exists():
                mask = decode_mask(s["segmentation"]["mask"]["$binary"])
                Image.fromarray(mask).save(mask_path)
            n_masks += 1

    print(f"Placed {n_images} images ({n_masks} with ground-truth masks) under {args.root}")
    for category in sorted({s["category_label"]["label"] for s in samples}):
        counts = {
            split: sum(1 for s in samples if s["category_label"]["label"] == category and s["folder_type"] == split)
            for split in FOLDER_FOR
        }
        print(f"  {category:12s} {counts}")

    print(
        "\nNote: test_private / test_private_mixed are not in this mirror "
        "(never publicly released by MVTec). Use test_public for local evaluation."
    )


if __name__ == "__main__":
    main()
