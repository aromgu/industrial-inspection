"""Re-predict test_public with a trained checkpoint and keep the anomaly maps.

Two questions the image scores alone cannot answer:
  * Would another way of turning the map into an image score separate good from defective
    better than the model's own score (usually the map maximum)? Each aggregation is written as
    an extra score file under results/scores_alt/, so evaluate.py can judge it on overkill.
  * How well does the model localise defects? AU-PRO up to a 5 % false positive rate, the
    MVTec AD 2 paper's metric. It is computed at the model's input resolution (after its resize
    and centre crop, downsampled to at most MAP_SIDE), not at full image resolution as the
    official server does, so it is close to but not the same as the paper's number.

    uv run python scripts/export_maps.py --model patchcore --category sheet_metal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anomalib.data import MVTecAD2
from anomalib.engine import Engine
from anomalib.metrics.aupro import _AUPRO
from anomalib.models import Dinomaly, EfficientAd, MHPatchcore, Patchcore
from sklearn.metrics import roc_auc_score

CLASSES = {"patchcore": Patchcore, "mhpatchcore": MHPatchcore, "efficientad": EfficientAd, "dinomaly_l": Dinomaly,
           "patchcore_392": Patchcore, "mhpatchcore_tile": MHPatchcore, "mhpatchcore_tilebig": MHPatchcore}
MAP_SIDE = 256  # maps and masks are stored and scored at most this large
AGGREGATIONS = {
    "max": lambda m: m.max(axis=1),
    "top1": lambda m: np.sort(m, axis=1)[:, -max(1, m.shape[1] // 100):].mean(axis=1),
    "top5": lambda m: np.sort(m, axis=1)[:, -max(1, m.shape[1] // 20):].mean(axis=1),
    "mean": lambda m: m.mean(axis=1),
}


def ckpt_of(model: str, category: str) -> Path:
    found = sorted(Path("results/runs", model, category).rglob("model.ckpt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no checkpoint under results/runs/{model}/{category}")
    return found[-1]


def shrink(x: torch.Tensor, mode: str) -> torch.Tensor:
    """(N, H, W) -> (N, h, w) with max(h, w) <= MAP_SIDE."""
    n, h, w = x.shape
    s = min(1.0, MAP_SIDE / max(h, w))
    if s == 1.0:
        return x
    size = (max(1, round(h * s)), max(1, round(w * s)))
    y = F.interpolate(x[:, None].float(), size=size, mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))
    return y[:, 0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=CLASSES, required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--limit-batches", type=int, default=None, help="smoke test: only the first N test batches")
    p.add_argument("--split", choices=["test", "validation"], default="test",
                   help="validation: only write the model's scores on the good-only validation set "
                        "(results/scores_val/), e.g. to set a normal-only threshold without touching test")
    p.add_argument("--tag", default="", help="suffix for output names, e.g. the tile grid")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    ckpt = ckpt_of(args.model, args.category)
    model = CLASSES[args.model].load_from_checkpoint(ckpt, map_location="cpu")
    dm = MVTecAD2(root=args.root, category=args.category, eval_batch_size=8, num_workers=2)
    dm.setup()
    engine = Engine(logger=False, enable_progress_bar=False, limit_predict_batches=args.limit_batches)
    loader = dm.val_dataloader() if args.split == "validation" else dm.test_dataloader()
    batches = engine.predict(model=model, dataloaders=loader)
    if args.split == "validation":
        rows = [{"image_path": p_, "label": int(l), "score": float(s)}
                for b in batches for p_, l, s in zip(b.image_path, b.gt_label, b.pred_score, strict=True)]
        out = args.out / "scores_val"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / f"{args.model}__{args.category}__seed0.csv", index=False)
        print(f"{len(rows)} validation images -> {out}", flush=True)
        return

    maps, masks, labels, paths, own = [], [], [], [], []
    for b in batches:
        maps.append(shrink(b.anomaly_map.detach().float().cpu().reshape(-1, *b.anomaly_map.shape[-2:]), "bilinear"))
        gt = b.gt_mask.detach().cpu().reshape(-1, *b.gt_mask.shape[-2:]).float()
        masks.append(shrink(gt, "nearest") > 0.5)
        labels += [int(v) for v in b.gt_label]
        paths += list(b.image_path)
        own += [float(v) for v in b.pred_score]
    maps_t, masks_t = torch.cat(maps), torch.cat(masks)
    y = np.array(labels)
    flat = maps_t.reshape(len(maps_t), -1).numpy()

    stem = f"{args.model}{args.tag}__{args.category}"
    out_maps = args.out / "maps"
    out_maps.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_maps / f"{stem}.npz", maps=maps_t.numpy().astype(np.float16), masks=masks_t.numpy(),
                        labels=y, paths=np.array(paths), pred_score=np.array(own))

    row = {"model": args.model + args.tag, "category": args.category, "n_images": len(y), "ckpt": str(ckpt),
           "auroc_model_score": roc_auc_score(y, own)}
    alt_dir = args.out / "scores_alt"
    alt_dir.mkdir(parents=True, exist_ok=True)
    for name, agg in AGGREGATIONS.items():
        s = agg(flat)
        row[f"auroc_{name}"] = roc_auc_score(y, s)
        pd.DataFrame({"image_path": paths, "label": y, "score": s}).to_csv(
            alt_dir / f"{args.model}{args.tag}-{name}__{args.category}__seed0.csv", index=False)

    aupro = _AUPRO(fpr_limit=0.05)
    aupro.update(maps_t, masks_t.int())
    row["aupro_005"] = float(aupro.compute()) if masks_t.any() else float("nan")

    reference = Path("results/scores") / f"{args.model}__{args.category}__seed0.csv"
    if reference.exists() and args.limit_batches is None:
        ref = pd.read_csv(reference)
        row["auroc_reference_csv"] = roc_auc_score(ref.label, ref.score)

    metrics = args.out / ("map_metrics_smoke.csv" if args.limit_batches else "map_metrics.csv")
    old = pd.read_csv(metrics) if metrics.exists() else pd.DataFrame()
    if len(old):
        old = old[~((old.model == row["model"]) & (old.category == args.category))]
    pd.concat([old, pd.DataFrame([row])]).to_csv(metrics, index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "ckpt"}, flush=True)


if __name__ == "__main__":
    main()
