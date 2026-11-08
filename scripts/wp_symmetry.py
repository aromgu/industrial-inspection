"""P5 (missing material), symmetry check: a wall plug is (nearly) mirror-symmetric about its long axis, so a strip
broken off one side shows up as silhouette asymmetry -- compared with the plug ITSELF, so tilt and perspective
(which spoiled the template silhouette check, wp_silhouette.py) cancel out.
  1. each SAM plug mask -> canonical frame (long axis horizontal through the crop centre; crop_affine), x2 down
  2. axis refinement: the vertical shift (+-6 cells) that best aligns the mask with its mirror image
  3. missing = mirror & ~mask (material on the other side, none here), on cells valid in BOTH positions (inside
     the photo, not under another plug's mask), outside a +-tol band of the mask outline; opened by one cell
  4. map value at those pixels (warped back to the photo, eval size) = blob area in cells
Writes results/maps/sym__wallplugs(.npz|__val.npz); evaluate with scripts/wp_sil_eval.py sym.
    .venv/Scripts/python.exe scripts/wp_symmetry.py [--tol 1]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.plugs import crop_affine, load_plugs  # noqa: E402
from industrial_inspection.segeval import DOWNSCALE, eval_size  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--name", default="sym")
a.add_argument("--crop", type=int, nargs=2, default=[448, 128])
a.add_argument("--down", type=int, default=2)
a.add_argument("--tol", type=int, default=1)
a.add_argument("--shift", type=int, default=6)
a.add_argument("--free", action="store_true", help="only plugs not touching the photo border or another plug")
args = a.parse_args()
CROP = tuple(args.crop)
cw, ch = CROP[0] // args.down, CROP[1] // args.down
C = Path("results/plugs_cache/wallplugs")
root = Path("datasets/MVTec_AD_2/wallplugs")
lengths = [p.size[0] for f in sorted((C / "train" / "good").glob("*.npz")) for p in load_plugs(f) if p.size[0] / max(p.size[1], 1) >= 3]
plug_length = float(np.median(lengths))


def warp(m, M):
    c = cv2.warpAffine(m.astype(np.uint8), M, CROP, flags=cv2.INTER_NEAREST, borderValue=0)
    return cv2.resize(c, (cw, ch), interpolation=cv2.INTER_AREA) > 0


def residual_map(f: Path, hw):
    pls = load_plugs(f)
    out = np.zeros(hw, np.float32)
    if not pls:
        return out
    shape = pls[0].mask.shape
    allm = np.stack([p.mask for p in pls])
    ev = shape[1] / hw[1]
    k = np.ones((2 * args.tol + 1,) * 2, bool)
    for i, p in enumerate(pls):
        if args.free:
            border = p.mask[0].any() or p.mask[-1].any() or p.mask[:, 0].any() or p.mask[:, -1].any()
            ring = cv2.dilate(p.mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~p.mask
            touch = (ring & allm[np.arange(len(pls)) != i].any(0)).sum() / max(1, ring.sum()) if len(pls) > 1 else 0
            if border or touch > 0.05:
                continue
        M = crop_affine(p, 1.0, plug_length, CROP)
        cm = warp(p.mask, M)
        others = warp(allm[np.arange(len(pls)) != i].any(0), M) if len(pls) > 1 else np.zeros_like(cm)
        inside = cv2.resize(cv2.warpAffine(np.ones(shape, np.uint8), M, CROP, flags=cv2.INTER_NEAREST, borderValue=0),
                            (cw, ch), interpolation=cv2.INTER_AREA) > 0.99
        valid = inside & ~others
        best, bs = None, 0
        for s in range(-args.shift, args.shift + 1):  # mirror about row (ch-1)/2 + s/2
            mir = np.roll(cm[::-1], s, axis=0)
            vmir = np.roll(valid[::-1], s, axis=0)
            v = valid & vmir
            cost = ((mir ^ cm) & v).sum()
            if best is None or cost < best:
                best, bs = cost, s
        mir = np.roll(cm[::-1], bs, axis=0)
        v = valid & np.roll(valid[::-1], bs, axis=0)
        miss = mir & ~ndimage.binary_dilation(cm, k) & v
        miss = ndimage.binary_opening(miss, np.ones((2, 2), bool))
        if not miss.any():
            continue
        lab, n = ndimage.label(miss)
        area = ndimage.sum(miss, lab, np.arange(1, n + 1))
        score = np.zeros((ch, cw), np.float32)
        for c in range(n):
            score[lab == c + 1] = area[c]
        up = cv2.resize(score, CROP, interpolation=cv2.INTER_NEAREST)
        Me = M.copy()
        Me[:, :2] = M[:, :2] * ev
        back = cv2.warpAffine(up, cv2.invertAffineTransform(Me), hw[::-1], flags=cv2.INTER_NEAREST, borderValue=0)
        out = np.maximum(out, back)
    return out


test = sorted(str(q) for q in (root / "test_public/good").glob("*.png")) + sorted(str(q) for q in (root / "test_public/bad").glob("*.png"))
labels = [0 if "/good/" in q.replace("\\", "/") else 1 for q in test]
val = sorted(str(q) for q in (root / "validation/good").glob("*.png"))
hw = eval_size(test[0], DOWNSCALE)
cache = lambda q: C / Path(q).parent.parent.name / Path(q).parent.name / (Path(q).stem + ".npz")
t0 = time.perf_counter()
tm = np.stack([residual_map(cache(q), hw) for q in test]).astype(np.float16)
vm = np.stack([residual_map(cache(q), hw) for q in val]).astype(np.float16)
np.savez_compressed(f"results/maps/{args.name}__wallplugs.npz", maps=tm, labels=np.array(labels), paths=np.array(test))
np.savez_compressed(f"results/maps/{args.name}__wallplugs__val.npz", maps=vm, paths=np.array(val))
print(f"saved {args.name}: {time.perf_counter() - t0:.0f}s")
