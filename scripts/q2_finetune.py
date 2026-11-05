"""Q2 step 6 (H2-f): briefly fine-tune Dinomaly-L on one good part photographed under the new light.

H2-e showed that for Dinomaly-L a new threshold cannot fix the new light: its good scores under the
new light overlap the defect scores. Here the model itself adapts. Starting from the trained
checkpoint, the bottleneck and decoder (the encoder stays frozen, as in training) get STEPS more
steps at the schedule's final learning rate (2e-4); each batch mixes the new-light photo(s) of the
chosen part with regular training images, so the model does not forget the regular light.

Protocol as q2_fewshot.py (walnuts DEV half): for each shift condition and each choice of k=1 dev
good part, adapt on that part's photos in that condition, then score the other dev parts in that
condition and the dev regular good images; threshold = max regular good score (after adaptation).

    uv run python scripts/q2_finetune.py --category walnuts --steps 200
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anomalib.models import Dinomaly
from anomalib.models.image.dinomaly.components import StableAdamW
from PIL import Image
from torchvision.transforms.v2 import functional as TF

from industrial_inspection.plots import q2_finetune_figure
from industrial_inspection.q2 import half_mask, parts_and_tags
from q2_fewshot import ckpt_of


def load(path: str, transform, device: str) -> torch.Tensor:
    x = TF.to_dtype(TF.to_image(Image.open(path).convert("RGB")), torch.float32, scale=True)[None]
    return transform(x.to(device))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="walnuts")
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=4, help="half new-light photos, half regular training images")
    p.add_argument("--smoke", action="store_true", help="one condition, one choice, 5 steps")
    p.add_argument("--out", type=Path, default=Path("results/q2"))
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    lm = Dinomaly.load_from_checkpoint(ckpt_of("dinomaly_l", args.category), map_location=device).to(device)
    transform = lm.pre_processor.transform
    net = lm.model
    start = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
    train_paths = sorted(str(q) for q in (args.root / args.category / "train" / "good").glob("*.png"))

    df = pd.read_csv(f"results/scores/dinomaly_l__{args.category}__seed0.csv").sort_values("image_path").reset_index(drop=True)
    parts, tag = parts_and_tags(df.image_path)
    y = df.label.to_numpy()
    dev = half_mask(parts, y, "dev")
    df, parts, tag, y = df[dev].reset_index(drop=True), parts[dev], tag[dev], y[dev]
    paths = df.image_path.to_numpy()
    reg_good = (tag == "regular") & (y == 0)
    conditions = sorted({t for t in tag if str(t).startswith("shift")})
    good_parts = np.unique(parts[y == 0])
    cache: dict[str, torch.Tensor] = {}

    def image(q: str) -> torch.Tensor:
        if q not in cache:
            cache[q] = load(q, transform, device).cpu()
        return cache[q].to(device)

    @torch.no_grad()
    def scores(qs) -> np.ndarray:
        net.eval()
        return np.array([float(net(image(q)).pred_score) for q in qs])

    base = scores(paths)
    print(f"max |rescored - saved| / saved = {np.max(np.abs(base - df.score.to_numpy()) / np.abs(df.score.to_numpy())):.2e}",
          flush=True)

    params = [q for q in lm.trainable_modules.parameters() if q.requires_grad]
    rows = []
    jobs = [(c, (gp,)) for c in conditions for gp in good_parts]
    if args.smoke:
        jobs, args.steps = jobs[:1], 5
    for c, chosen in jobs:
        net.load_state_dict({k: v.to(device) for k, v in start.items()})
        # Dinomaly's own optimizer and settings, at the schedule's final learning rate
        opt = StableAdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-8)
        adapt = list(paths[(tag == c) & np.isin(parts, chosen) & (y == 0)])
        t0 = time.perf_counter()
        net.train()
        for step in range(args.steps):
            half = args.batch // 2
            batch = [adapt[i] for i in rng.integers(0, len(adapt), half)] + \
                    [train_paths[i] for i in rng.integers(0, len(train_paths), args.batch - half)]
            x = torch.cat([image(q) for q in batch])
            loss = net(x, global_step=5000)  # end-of-training hard-mining weight
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        keep = (tag == c) & ~np.isin(parts, chosen)
        scored = keep | reg_good
        s = np.full(len(paths), np.nan)
        s[scored] = scores(paths[scored])
        for name, sc in (("none", base), ("finetune", s)):
            thr = sc[reg_good].max()
            g, b = keep & (y == 0), keep & (y == 1)
            rows.append({"model": "dinomaly_l", "method": name, "condition": c, "chosen": "+".join(chosen),
                         "false_alarm": float((sc[g] > thr).mean()), "miss": float((sc[b] <= thr).mean()),
                         "regular_good_max": float(thr), "loss_end": float(loss), "seconds": time.perf_counter() - t0})
        print(f"{c} {chosen}: fa {rows[-2]['false_alarm']:.2f} -> {rows[-1]['false_alarm']:.2f}, "
              f"miss {rows[-2]['miss']:.2f} -> {rows[-1]['miss']:.2f}  [{time.perf_counter() - t0:.0f}s]", flush=True)
    raw = pd.DataFrame(rows)
    if args.smoke:
        print(raw)
        return
    per_choice = raw.groupby(["method", "chosen"])[["false_alarm", "miss"]].mean().reset_index()
    summary = per_choice.groupby("method").agg(false_alarm=("false_alarm", "mean"), fa_min=("false_alarm", "min"),
                                               fa_max=("false_alarm", "max"), miss=("miss", "mean"),
                                               miss_min=("miss", "min"), miss_max=("miss", "max")).reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out / f"finetune__{args.category}__dev_raw.csv", index=False)
    summary.to_csv(args.out / f"finetune__{args.category}__dev.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print("figure:", q2_finetune_figure(raw, args.category))


if __name__ == "__main__":
    main()
