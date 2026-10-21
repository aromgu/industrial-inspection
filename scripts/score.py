"""Train one anomalib model on one MVTec AD 2 category and dump per-image scores.

Scores are written raw (post-processing disabled): anomalib's default
min-max normalisation clips to [0, 1], which creates ties at the extremes and
hides where the threshold actually falls. Thresholding happens later in
evaluate.py, on a held-out split.

    uv run python scripts/score.py --model patchcore --category sheet_metal
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from anomalib.data import MVTecAD2
from anomalib.engine import Engine
from anomalib.models import Dinomaly, EfficientAd, MHPatchcore, Patchcore
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

OFF = {"post_processor": False, "evaluator": False, "visualizer": False}
MODELS = {
    "patchcore": lambda: Patchcore(**OFF),
    "mhpatchcore": lambda: MHPatchcore(**OFF),
    "efficientad": lambda: EfficientAd(**OFF),
    "dinomaly": lambda: Dinomaly(**OFF),
    # ViT-L is the strongest encoder in the Dinomaly paper.
    "dinomaly_l": lambda: Dinomaly(encoder_name="vit_large_patch14_reg4_dinov2", **OFF),
}
# EfficientAD's student-teacher training assumes one image per step.
TRAIN_BATCH = {"patchcore": 16, "mhpatchcore": 16, "efficientad": 1, "dinomaly": 8, "dinomaly_l": 4}
# Gradient-trained models need an explicit length, otherwise Lightning runs 1000 epochs.
# Values are the papers' defaults; the memory-bank models fit in a single pass.
MAX_STEPS = {"efficientad": 70_000, "dinomaly": 5_000, "dinomaly_l": 5_000}


def persist_workers(datamodule, num_workers: int) -> None:
    """Rebuild the train/val loaders with persistent workers (anomalib's don't set it)."""
    if num_workers == 0:
        return

    def wrap(make):
        def build():
            dl = make()
            return DataLoader(
                dl.dataset,
                batch_size=dl.batch_size,
                shuffle=isinstance(dl.sampler, RandomSampler),
                num_workers=dl.num_workers,
                collate_fn=dl.collate_fn,
                persistent_workers=True,
            )

        return build

    datamodule.train_dataloader = wrap(datamodule.train_dataloader)
    datamodule.val_dataloader = wrap(datamodule.val_dataloader)


class StepProgress(Callback):
    """tqdm bar that stays readable when stdout is redirected to a log file."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.bar: tqdm | None = None

    def _open(self, total: int | None, desc: str) -> None:
        self.close()
        self.bar = tqdm(total=total, desc=f"{self.label} {desc}", mininterval=15, file=sys.stdout, ascii=True)

    def close(self, *_) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def on_train_start(self, trainer, pl_module) -> None:
        steps = trainer.max_steps if trainer.max_steps > 0 else trainer.max_epochs * trainer.num_training_batches
        self._open(steps, "train")

    def on_train_batch_end(self, trainer, pl_module, *args) -> None:
        if self.bar is not None:
            self.bar.update(1)

    def on_train_end(self, trainer, pl_module) -> None:
        self.close()

    def on_predict_start(self, trainer, pl_module) -> None:
        self._open(sum(trainer.num_predict_batches), "predict")

    def on_predict_batch_end(self, trainer, pl_module, *args) -> None:
        if self.bar is not None:
            self.bar.update(1)

    def on_predict_end(self, trainer, pl_module) -> None:
        self.close()


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
    if args.max_epochs:
        trainer_kwargs = {"max_epochs": args.max_epochs}
    elif args.model in MAX_STEPS:
        # Epochs are ~100 steps on these small train sets. On Windows every epoch would respawn
        # the workers (a full torch/anomalib re-import) and re-run validation, so keep workers
        # alive and validate once, at the last step (EfficientAD needs it for its map quantiles).
        persist_workers(datamodule, args.num_workers)
        trainer_kwargs = {
            "max_epochs": -1,
            "max_steps": MAX_STEPS[args.model],
            "check_val_every_n_epoch": None,
            "val_check_interval": MAX_STEPS[args.model],
        }
    else:
        trainer_kwargs = {}
    engine = Engine(
        default_root_dir=Path("results/runs") / args.model / args.category,
        logger=False,
        callbacks=[StepProgress(args.model)],
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
