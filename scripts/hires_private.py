"""Run HiResKNN on the official benchmark's private test sets (no ground truth, flat directories), for
the submission to benchmark.mvtec.com. Same model/params as scripts/hires.py (default: the chosen
k=50 setting); only the output goes to the private splits instead of test_public.

Writes results/maps/<name>__<cat>__<split>.npz (maps, paths) for each of test_private, test_private_mixed
-- no labels/refs, since the private splits carry no ground truth locally.

    .venv/Scripts/python.exe scripts/hires_private.py --name hires_k50 --knn 50 --categories can fabric fruit_jelly rice sheet_metal vial walnuts --game-mode
    .venv/Scripts/python.exe scripts/hires_private.py --smoke --game-mode
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from industrial_inspection.hires import HiResKNN
from industrial_inspection.segeval import DOWNSCALE, eval_size
from hires import to_eval

CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
SPLITS = ["test_private", "test_private_mixed"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="hires_k50")
    p.add_argument("--categories", nargs="+", default=CATS)
    p.add_argument("--splits", nargs="+", default=SPLITS)
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--n-ref", type=int, default=16)
    p.add_argument("--short-side", type=int, default=672)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--knn", type=int, default=50)
    p.add_argument("--game-mode", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = Path("results/smoke") if args.smoke else Path("results")
    duty = 1.0
    if args.game_mode:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from score import lower_cpu_priority

        lower_cpu_priority()
        torch.cuda.set_per_process_memory_fraction(0.6, 0)
        duty = 0.6
    if args.smoke:
        args.categories, args.n_ref = ["walnuts"], 2

    model = HiResKNN(n_ref=args.n_ref, short_side=args.short_side, gpu_duty=duty, knn=args.knn)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    for cat in args.categories:
        t0 = time.perf_counter()
        root = args.root / cat
        train = sorted(str(q) for q in (root / "train" / "good").glob("*.png"))
        model.fit(train)
        for split in args.splits:
            paths = sorted(str(q) for q in (root / split).glob("*.png"))
            if args.smoke:
                paths = paths[:3]
            hw = eval_size(paths[0], DOWNSCALE)
            maps = np.stack([np.mean([to_eval(m, hw, args.sigma) for m in model.score_layers(q)], axis=0)
                             for q in paths]).astype(np.float16)
            np.savez_compressed(out / "maps" / f"{args.name}__{cat}__{split}.npz", maps=maps, paths=np.array(paths))
            print(f"{cat}/{split}: {len(paths)} images, {time.perf_counter() - t0:.0f}s so far", flush=True)
        print(f"{cat}: done, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
