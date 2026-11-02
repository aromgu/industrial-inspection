"""Turn a tile-level score CSV into one score per original image (max over its tiles).

    uv run python scripts/aggregate_tiles.py results/scores_tiles/mhpatchcore_tile__can__seed0.csv \
        --root datasets/MVTec_AD_2

The image keeps its original path so evaluate.py / tag_analysis.py treat it like any other score
file; the label is taken from the original folder (a defective image's defect-free tiles are
still part of a defective image).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--out", type=Path, default=Path("results/scores"))
    args = p.parse_args()
    model, category, seed = args.csv.stem.split("__")
    df = pd.read_csv(args.csv)
    path = df.image_path.str.replace("\\", "/", regex=False)
    folder = path.str.rsplit("/", n=2).str[-2]
    stem = path.str.rsplit("/", n=1).str[-1].str.replace(r"__t\d\d\.png$", "", regex=True)
    df["image_path"] = [str(args.root / category / "test_public" / f / f"{s}.png") for f, s in zip(folder, stem)]
    out = df.groupby("image_path", as_index=False).agg(label=("label", "max"), score=("score", "max"))
    out["label"] = out.image_path.str.replace("\\", "/", regex=False).str.contains("/bad/").astype(int)
    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / f"{model}max__{category}__{seed}.csv"
    out.to_csv(dst, index=False)
    print(f"{len(df)} tiles -> {len(out)} images ({out.label.sum()} defective) -> {dst}")


if __name__ == "__main__":
    main()
