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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

from industrial_inspection.plots import test_figure, training_figure

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
SMOKE_STEPS = 20


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


class CurveLogger(Callback):
    """Record training loss and periodic test AUROC, for learning-curve plots.

    The AUROC points are for the plot only: they use the same test_public images the report is
    computed on, so they must not drive any model or threshold choice.
    """

    def __init__(self, out: Path, test_loader: DataLoader, max_steps: int, n_evals: int = 10) -> None:
        self.loss_path = out.with_name(out.name + "__loss.csv")
        self.auroc_path = out.with_name(out.name + "__auroc.csv")
        self.test_loader = test_loader
        self.log_every = max(1, max_steps // 700)
        self.eval_every = max(1, max_steps // n_evals)
        self.loss_rows: list[dict] = []
        self.auroc_rows: list[dict] = []
        out.parent.mkdir(parents=True, exist_ok=True)

    def _losses(self, trainer) -> dict[str, float]:
        # Models log "train_loss" with on_step and/or on_epoch, which Lightning suffixes with _step/_epoch.
        return {
            k: float(v)
            for k, v in trainer.callback_metrics.items()
            if k.startswith("train_") and not k.endswith("_epoch")
        }

    @torch.no_grad()
    def _test_auroc(self, trainer, pl_module) -> float:
        modes = {m: m.training for m in pl_module.modules()}
        pl_module.eval()
        scores, labels = [], []
        try:
            for i, batch in enumerate(self.test_loader):
                batch = trainer.strategy.batch_to_device(batch)
                # Engine.predict resizes/crops in this hook; calling predict_step directly skips it.
                if pl_module.pre_processor is not None:
                    pl_module.pre_processor.on_predict_batch_start(trainer, pl_module, batch, i)
                out = pl_module.predict_step(batch, i)
                scores += out.pred_score.flatten().tolist()
                labels += out.gt_label.flatten().tolist()
        finally:
            for m, was_training in modes.items():
                m.training = was_training
        return float(roc_auc_score(labels, scores))

    def on_train_batch_end(self, trainer, pl_module, *args) -> None:
        step = trainer.global_step
        if step % self.log_every == 0:
            self.loss_rows.append({"step": step, **self._losses(trainer)})
        if step % self.eval_every == 0:
            try:
                self.auroc_rows.append({"step": step, "test_auroc": self._test_auroc(trainer, pl_module)})
            except Exception as e:  # a plotting aid must never kill the training run
                print(f"[curve] test AUROC at step {step} failed: {e!r}", flush=True)
            self.flush()

    def on_train_end(self, trainer, pl_module) -> None:
        self.flush()

    def flush(self) -> None:
        pd.DataFrame(self.loss_rows).to_csv(self.loss_path, index=False)
        pd.DataFrame(self.auroc_rows).to_csv(self.auroc_path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--out", type=Path, default=Path("results/scores"))
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="end-to-end check on 4 test images and ~20 steps; writes only under results/smoke/",
    )
    args = p.parse_args()
    root = Path("results/smoke") if args.smoke else Path("results")
    if args.smoke:
        args.out = root / "scores"
    steps = SMOKE_STEPS if args.smoke else MAX_STEPS.get(args.model)

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
            "max_steps": steps,
            "check_val_every_n_epoch": None,
            "val_check_interval": steps,
        }
    else:
        trainer_kwargs = {}
    if args.smoke:
        trainer_kwargs |= {"limit_val_batches": 1}
        if args.model not in MAX_STEPS:
            trainer_kwargs |= {"limit_train_batches": 2}
    callbacks = [StepProgress(args.model)]
    datamodule.setup()  # test split is needed for the periodic AUROC points and the smoke subset
    # NOT datamodule.test_data: that is only the 90 defective images. The evaluated set (good + defective)
    # is what test_dataloader() serves.
    test = datamodule.test_dataloader().dataset
    if args.smoke:
        # 2 good + 2 defective, so AUROC and every figure are well-defined.
        idx = [i for lab in (0, 1) for i in test.samples.index[test.samples.label_index == lab][:2]]
        idx = [test.samples.index.get_loc(i) for i in idx]
        datamodule.test_dataloader = lambda: DataLoader(
            Subset(test, idx), batch_size=4, num_workers=0, collate_fn=test.collate_fn
        )
    curve_prefix = root / "curves" / f"{args.model}__{args.category}__seed{args.seed}"
    if args.model in MAX_STEPS and not args.max_epochs:
        callbacks.append(
            CurveLogger(
                curve_prefix,
                DataLoader(
                    Subset(test, idx) if args.smoke else test, batch_size=8, num_workers=0, collate_fn=test.collate_fn
                ),
                steps,
                n_evals=2 if args.smoke else 10,
            )
        )
    engine = Engine(
        default_root_dir=root / "runs" / args.model / args.category,
        logger=False,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    train_s = time.perf_counter() - t0

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

    # Figures are part of every run. Scores are already saved, so a plotting error is reported
    # loudly without losing the run.
    try:
        figs = root / "figures"
        print("figure:", test_figure(out, figs / f"{out.stem}__test.png"))
        fig = training_figure(curve_prefix, figs / f"{curve_prefix.name}__training.png")
        print("figure:", fig if fig else "(no training curve: single-pass model)")
    except Exception as e:
        print(f"[FIGURE FAILED] {e!r} - rerun scripts/evaluate.py to redraw", flush=True)
        if args.smoke:
            raise


if __name__ == "__main__":
    main()
