"""Run HiResKNN (industrial_inspection.hires) on MVTec AD 2 and save maps for seg_eval.py.

Per category: memory bank from n_ref training images, then anomaly maps for test_public and for
the good validation images (thresholds), at 1/4 of the original resolution, smoothed with a
Gaussian (sigma 4 px at that scale). Written as results/maps/<name>__<cat>.npz and __val.npz in the
layout seg_eval.py reads (view None: the maps already cover the whole image), plus image scores
(map maximum) in results/scores/<name>__<cat>__seed0.csv.

    uv run python scripts/hires.py --categories walnuts --game-mode
    uv run python scripts/hires.py --smoke            # 1 category, 2 references, 4 test images
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from industrial_inspection.hires import HiResKNN
from industrial_inspection.segeval import DOWNSCALE, eval_size

CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def to_eval(m: np.ndarray, hw: tuple[int, int], sigma: float) -> np.ndarray:
    t = torch.from_numpy(m)[None, None]
    up = torch.nn.functional.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()
    return ndimage.gaussian_filter(up, sigma) if sigma > 0 else up


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="hires_dinov2l")
    p.add_argument("--categories", nargs="+", default=CATS)
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--n-ref", type=int, default=16)
    p.add_argument("--short-side", type=int, default=672)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--n-rotations", type=int, default=1,
                   help="bank also gets each reference rotated this many times, evenly over 360 deg "
                        "(e.g. 8 for a category where the object itself lies at any angle, like wallplugs)")
    p.add_argument("--pool", type=int, default=1, help="average each token over a pool x pool neighbourhood before matching (3 = PatchCore's)")
    p.add_argument("--knn", type=int, default=1, help="score = mean distance to the knn nearest bank patches")
    p.add_argument("--bank", choices=["fps", "all"], default="fps",
                   help="fps: n_ref most different images, all patches; all: SuperADD's bank (every training image "
                        "but each 8th, brightness augmentation, distance-based subsampling to --max-bank)")
    p.add_argument("--brightness", type=float, nargs=2, default=[1.0, 1.0], help="random brightness factor range (bank=all)")
    p.add_argument("--max-bank", type=int, default=100_000)
    p.add_argument("--k1-name", default=None,
                   help="also save the nearest-patch (k=1) maps of the same run under this model name, so k=1 and "
                        "k=--knn come from one pass over the photos (same bank)")
    p.add_argument("--save-layers", action="store_true",
                   help="also write each layer's map as its own model <name>_L<layer> (seg_eval.py can score them)")
    p.add_argument("--game-mode", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    rotations = tuple(360.0 * i / args.n_rotations for i in range(args.n_rotations))
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
        args.categories, args.n_ref = args.categories[:1] if args.categories != CATS else ["walnuts"], 2

    model = HiResKNN(n_ref=args.n_ref, short_side=args.short_side, gpu_duty=duty, rotations=rotations,
                     pool=args.pool, knn=args.knn, bank=args.bank, brightness=tuple(args.brightness),
                     max_bank=args.max_bank)
    for cat in args.categories:
        t0 = time.perf_counter()
        root = args.root / cat
        train = sorted(str(q) for q in (root / "train" / "good").glob("*.png"))
        val = sorted(str(q) for q in (root / "validation" / "good").glob("*.png"))
        test = sorted(str(q) for q in (root / "test_public" / "good").glob("*.png")) + \
            sorted(str(q) for q in (root / "test_public" / "bad").glob("*.png"))
        labels = [0 if "/good/" in q.replace("\\", "/") else 1 for q in test]
        if args.smoke:
            test, labels, val, train = test[:2] + test[-2:], labels[:2] + labels[-2:], val[:2], train[:10]
        refs = model.fit(train)
        hw = eval_size(test[0], DOWNSCALE)
        # per image: one map per layer (at eval size); the model's map is their mean
        def maps_for(paths):
            main, nn1 = [], []
            for q in paths:
                main.append(np.stack([to_eval(m, hw, args.sigma) for m in model.score_layers(q)]))
                if args.k1_name:
                    nn1.append(np.stack([to_eval(m, hw, args.sigma) for m in model.last_nn1]))
            return np.stack(main).astype(np.float16), (np.stack(nn1).astype(np.float16) if nn1 else None)

        tl, tl1 = maps_for(test)
        vl, vl1 = maps_for(val)
        (out / "maps").mkdir(parents=True, exist_ok=True)
        (out / "scores").mkdir(parents=True, exist_ok=True)
        outputs = [(args.name, tl.astype(np.float32).mean(1), vl.astype(np.float32).mean(1))]
        if args.k1_name:
            outputs.append((args.k1_name, tl1.astype(np.float32).mean(1), vl1.astype(np.float32).mean(1)))
        if args.save_layers:
            outputs += [(f"{args.name}_L{layer + 1}", tl[:, i], vl[:, i]) for i, layer in enumerate(model.layers)]
        for name, tmaps, vmaps in outputs:
            tmaps, vmaps = tmaps.astype(np.float16), vmaps.astype(np.float16)
            np.savez_compressed(out / "maps" / f"{name}__{cat}.npz", maps=tmaps, labels=np.array(labels), paths=np.array(test),
                                refs=np.array(refs))
            np.savez_compressed(out / "maps" / f"{name}__{cat}__val.npz", maps=vmaps, paths=np.array(val))
            pd.DataFrame({"image_path": test, "label": labels, "score": tmaps.reshape(len(test), -1).max(1).astype(float)}).to_csv(
                out / "scores" / f"{name}__{cat}__seed0.csv", index=False)
        print(f"{cat}: {len(test)} test, {len(val)} val, bank {[len(b) for b in model.bank][0]} patches/layer, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
