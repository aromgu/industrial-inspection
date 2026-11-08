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
    p.add_argument("--backbone", default="vit_large_patch14_reg4_dinov2.lvd142m",
                   help="timm name, or dinov3_vith16plus (SuperADD's; weights/dinov3_vith16plus.pth)")
    p.add_argument("--layers", type=int, nargs="+", default=[5, 11, 17, 23], help="0-based block indices (SuperADD: 7 15 23 31)")
    p.add_argument("--tile", type=int, default=448)
    p.add_argument("--overlap", type=int, default=112)
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
    p.add_argument("--per-image-cap", type=int, default=1500, help="bank=all: random patches kept per training photo before subsampling")
    p.add_argument("--k1-name", default=None,
                   help="also save the nearest-patch (k=1) maps of the same run under this model name, so k=1 and "
                        "k=--knn come from one pass over the photos (same bank)")
    p.add_argument("--save-layers", action="store_true",
                   help="also write each layer's map as its own model <name>_L<layer> (seg_eval.py can score them)")
    p.add_argument("--self-r", type=int, default=0, help="also compute the in-image comparison map (exclusion radius in tokens)")
    p.add_argument("--self-name", default=None, help="model name for the in-image comparison maps (needs --self-r)")
    p.add_argument("--clahe", type=float, default=0, help="CLAHE clip limit on the input photos (0 = off), bank and queries alike")
    p.add_argument("--unsharp", type=float, default=0, help="unsharp-mask amount on the input photos (0 = off)")
    p.add_argument("--proj", default=None, help="learned token projection (scripts/wp_adapt.py) applied to bank and queries")
    p.add_argument("--private", action="store_true", help="also write maps for test_private and test_private_mixed (same bank)")
    p.add_argument("--histmatch", action="store_true",
                   help="match every photo's L-channel histogram to the mean CDF of the category's training photos (bank and queries alike)")
    p.add_argument("--pad-tokens", type=int, default=0, help="reflect-pad the photo by this many patches before tiling")
    p.add_argument("--bank-light", type=float, default=0, help="bank=all: probability of directional relighting per training photo")
    p.add_argument("--tta-flip", action="store_true", help="average with the maps of the left-right mirrored photo")
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

    model = HiResKNN(backbone=args.backbone, layers=args.layers, tile=args.tile, overlap=args.overlap,
                     n_ref=args.n_ref, short_side=args.short_side, gpu_duty=duty, rotations=rotations,
                     pool=args.pool, knn=args.knn, bank=args.bank, brightness=tuple(args.brightness),
                     max_bank=args.max_bank, per_image_cap=args.per_image_cap)
    model.tta_flip, model.self_r, model.clahe = args.tta_flip, args.self_r, args.clahe
    model.unsharp, model.pad_tokens, model.bank_light = args.unsharp, args.pad_tokens, args.bank_light
    if args.proj:
        from industrial_inspection.adapt import load_proj
        model.proj = load_proj(args.proj)
    for cat in args.categories:
        t0 = time.perf_counter()
        root = args.root / cat
        train = sorted(str(q) for q in (root / "train" / "good").glob("*.png"))
        val = sorted(str(q) for q in (root / "validation" / "good").glob("*.png"))
        model.hist_ref = None
        if args.histmatch:
            import cv2
            from PIL import Image
            h = np.zeros(256)
            for q in train[::4]:
                g = cv2.cvtColor(np.asarray(Image.open(q).convert("RGB").reduce(4)), cv2.COLOR_RGB2LAB)[..., 0]
                h += np.bincount(g.ravel(), minlength=256) / g.size
            model.hist_ref = np.cumsum(h) / h.sum()
        test = sorted(str(q) for q in (root / "test_public" / "good").glob("*.png")) + \
            sorted(str(q) for q in (root / "test_public" / "bad").glob("*.png"))
        labels = [0 if "/good/" in q.replace("\\", "/") else 1 for q in test]
        if args.smoke:
            test, labels, val, train = test[:2] + test[-2:], labels[:2] + labels[-2:], val[:2], train[:10]
        refs = model.fit(train)
        hw = eval_size(test[0], DOWNSCALE)
        # per image: one map per layer (at eval size); the model's map is their mean
        def maps_for(paths):
            main, nn1, sf = [], [], []
            for q in paths:
                main.append(np.stack([to_eval(m, hw, args.sigma) for m in model.score_layers(q)]))
                if args.k1_name:
                    nn1.append(np.stack([to_eval(m, hw, args.sigma) for m in model.last_nn1]))
                if args.self_name:
                    sf.append(np.stack([to_eval(m, hw, args.sigma) for m in model.last_self]))
            st = lambda a: np.stack(a).astype(np.float16) if a else None
            return st(main), st(nn1), st(sf)

        tl, tl1, tls = maps_for(test)
        vl, vl1, vls = maps_for(val)
        (out / "maps").mkdir(parents=True, exist_ok=True)
        (out / "scores").mkdir(parents=True, exist_ok=True)
        outputs = [(args.name, tl.astype(np.float32).mean(1), vl.astype(np.float32).mean(1))]
        if args.k1_name:
            outputs.append((args.k1_name, tl1.astype(np.float32).mean(1), vl1.astype(np.float32).mean(1)))
        if args.self_name:
            outputs.append((args.self_name, tls.astype(np.float32).mean(1), vls.astype(np.float32).mean(1)))
        if args.save_layers:
            outputs += [(f"{args.name}_L{layer + 1}", tl[:, i], vl[:, i]) for i, layer in enumerate(model.layers)]
        for name, tmaps, vmaps in outputs:
            tmaps, vmaps = tmaps.astype(np.float16), vmaps.astype(np.float16)
            np.savez_compressed(out / "maps" / f"{name}__{cat}.npz", maps=tmaps, labels=np.array(labels), paths=np.array(test),
                                refs=np.array(refs))
            np.savez_compressed(out / "maps" / f"{name}__{cat}__val.npz", maps=vmaps, paths=np.array(val))
            pd.DataFrame({"image_path": test, "label": labels, "score": tmaps.reshape(len(test), -1).max(1).astype(float)}).to_csv(
                out / "scores" / f"{name}__{cat}__seed0.csv", index=False)
        if args.private:  # benchmark private splits, same bank and settings (thresholds come from the val maps above)
            for split in ("test_private", "test_private_mixed"):
                ps = sorted(str(q) for q in (root / split).glob("*.png"))
                if args.smoke:
                    ps = ps[:2]
                hw = eval_size(ps[0], DOWNSCALE)
                pm, _, _ = maps_for(ps)
                np.savez_compressed(out / "maps" / f"{args.name}__{cat}__{split}.npz",
                                    maps=pm.astype(np.float32).mean(1).astype(np.float16), paths=np.array(ps))
                print(f"{cat}/{split}: {len(ps)} images, {time.perf_counter() - t0:.0f}s so far", flush=True)
        print(f"{cat}: {len(test)} test, {len(val)} val, bank {[len(b) for b in model.bank][0]} patches/layer, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
