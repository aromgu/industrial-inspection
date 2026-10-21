"""Train one anomalib model on one MVTec AD 2 category and dump per-image scores.

Scores are written raw (post-processing disabled): anomalib's default
min-max normalisation clips to [0, 1], which creates ties at the extremes and
hides where the threshold actually falls. Thresholding happens later in
evaluate.py, on a held-out split.

    uv run python scripts/score.py --model patchcore --category sheet_metal
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from anomalib.data import MVTecAD2
from anomalib.engine import Engine
from anomalib.models import Dinomaly, EfficientAd, Patchcore

MODELS = {
    "patchcore": lambda: Patchcore(post_processor=False, evaluator=False, visualizer=False),
    "efficientad": lambda: EfficientAd(post_processor=False, evaluator=False, visualizer=False),
    "dinomaly": lambda: Dinomaly(post_processor=False, evaluator=False, visualizer=False),
}
# EfficientAD's student-teacher training assumes one image per step.
TRAIN_BATCH = {"patchcore": 16, "efficientad": 1, "dinomaly": 8}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--out", type=Path, default=Path("results/scores"))
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    datamodule = MVTecAD2(
        root=args.root,
        category=args.category,
        train_batch_size=TRAIN_BATCH[args.model],
        eval_batch_size=8,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = MODELS[args.model]()
    trainer_kwargs = {"max_epochs": args.max_epochs} if args.max_epochs else {}
    engine = Engine(
        default_root_dir=Path("results/runs") / args.model / args.category,
        logger=False,
        **trainer_kwargs,
    )

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    train_s = time.perf_counter() - t0

    datamodule.setup()
    batches = engine.predict(model=model, dataloaders=datamodule.test_dataloader())
    rows = []
    for b in batches:
        for path, label, score in zip(b.image_path, b.gt_label, b.pred_score, strict=True):
            rows.append({"image_path": path, "label": int(label), "score": float(score)})

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.model}__{args.category}__seed{args.seed}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"{len(df)} test images ({df.label.sum()} defective) -> {out}  [train {train_s:.0f}s]")


if __name__ == "__main__":
    main()
