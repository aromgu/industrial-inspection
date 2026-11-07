"""wallplugs silhouette check: shape defects (missing material, bent fins) that patch matching cannot see.
  1. every SAM plug mask (cache from plugs_knn.py) is warped into the canonical frame (long axis horizontal, all
     plugs at the same scale; industrial_inspection.plugs.crop_affine) and downsampled x4 (112 x 32 cells)
  2. normal bank: canonical masks of 'free' training plugs (not touching the border, not touching other plugs,
     whole length) in 4 orientations
  3. per test plug: the bank mask and shift (+-15 cells along the axis, +-1 across) with the smallest XOR, counted
     only where the comparison is valid -- inside the photo and not covered by another plug's mask (so plugs cut
     by the border or lying under another plug do not look short). Both terms of the XOR are conv2d's on the GPU.
  4. residual = (template & ~plug) | (plug & ~template) on valid cells, opened by one cell; warped back to the
     photo at the eval size; map value = residual blob area (cells) so bigger shape changes score higher
Writes results/maps/sil__wallplugs(.npz|__val.npz).
    .venv/Scripts/python.exe scripts/wp_silhouette.py [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.plugs import crop_affine, load_plugs  # noqa: E402
from industrial_inspection.segeval import DOWNSCALE, eval_size  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--name", default="sil")
a.add_argument("--crop", type=int, nargs=2, default=[448, 128])
a.add_argument("--down", type=int, default=4)
a.add_argument("--sx", type=int, default=15)
a.add_argument("--sy", type=int, default=1)
a.add_argument("--sx-wide", type=int, default=45, help="axial shift range for plugs at the border or shorter than 80 %% of a whole plug")
a.add_argument("--tol", type=int, default=1, help="ignore differences within this many cells of the template outline")
a.add_argument("--smoke", action="store_true")
args = a.parse_args()
dev = "cuda"
CROP = tuple(args.crop)
cw, ch = CROP[0] // args.down, CROP[1] // args.down
C = Path("results/plugs_cache/wallplugs")
root = Path("datasets/MVTec_AD_2/wallplugs")


def canon(m: np.ndarray, M: np.ndarray) -> np.ndarray:
    c = cv2.warpAffine(m.astype(np.uint8), M, CROP, flags=cv2.INTER_NEAREST)
    return cv2.resize(c, (cw, ch), interpolation=cv2.INTER_AREA) > 0  # any coverage -> cell on (area avg > 0)


def plug_frames(f: Path, plug_length: float):
    """Per plug of a photo: (affine work->crop, canonical mask, canonical valid mask, touches border, touches others)."""
    pls = load_plugs(f)
    if not pls:
        return [], None
    shape = pls[0].mask.shape
    allm = np.stack([p.mask for p in pls])
    out = []
    for k, p in enumerate(pls):
        M = crop_affine(p, 1.0, plug_length, CROP)
        others = allm[np.arange(len(pls)) != k].any(0) if len(pls) > 1 else np.zeros(shape, bool)
        inside = cv2.warpAffine(np.ones(shape, np.uint8), M, CROP, flags=cv2.INTER_NEAREST, borderValue=0)
        inside = cv2.resize(inside, (cw, ch), interpolation=cv2.INTER_AREA) > 0.99
        cov = canon(others, M)
        border = p.mask[0].any() or p.mask[-1].any() or p.mask[:, 0].any() or p.mask[:, -1].any()
        ring = cv2.dilate(p.mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~p.mask
        out.append((M, canon(p.mask, M), inside & ~cov, border, (ring & others).sum() / max(1, ring.sum())))
    return out, shape


# typical plug length (as plugs_knn.py): median long side of elongated training plugs
train_files = sorted((C / "train" / "good").glob("*.npz"))
lengths = [p.size[0] for f in train_files for p in load_plugs(f) if p.size[0] / max(p.size[1], 1) >= 3]
plug_length = float(np.median(lengths))
bank = []
for f in train_files:
    for M, cm, valid, border, touch in plug_frames(f, plug_length)[0]:
        if border or touch > 0.02 or cm.sum() == 0:
            continue
        ys, xs = np.nonzero(cm)
        if np.ptp(xs) + 1 < 0.9 * plug_length / (plug_length * 1.15) * cw:  # shorter than 90 % of a whole plug
            continue
        bank += [cm, cm[::-1], cm[:, ::-1], cm[::-1, ::-1]]
B = torch.from_numpy(np.stack(bank).astype(np.float32))[:, None].to(dev)  # (N, 1, ch, cw)
print(f"plug length {plug_length:.0f}px (work res); silhouette bank {len(B)} masks ({ch}x{cw} cells)", flush=True)


@torch.no_grad()
def best_template(cm: np.ndarray, valid: np.ndarray, sx: int) -> np.ndarray:
    A = torch.from_numpy(cm.astype(np.float32)).to(dev)
    V = torch.from_numpy(valid.astype(np.float32)).to(dev)
    pad = (sx, sx, args.sy, args.sy)
    AV = F.pad((A * V)[None, None], pad)
    Vp = F.pad(V[None, None], pad)
    # XOR on valid cells for template shifted by (dy, dx): sum(A V) + sum(B_s V) - 2 sum(A B_s V)
    sBV = F.conv2d(Vp, B)[0]          # (N, 2sy+1, 2sx+1)
    sABV = F.conv2d(AV, B)[0]
    xor = (A * V).sum() + sBV - 2 * sABV
    n, dy, dx = np.unravel_index(int(torch.argmin(xor)), xor.shape)
    T = np.zeros((ch, cw), bool)
    b = bank[n]
    oy, ox = dy - args.sy, dx - sx  # template placed shifted by -offset relative to the query frame
    ys, xs = np.nonzero(b)
    ys, xs = ys - oy, xs - ox
    ok = (ys >= 0) & (ys < ch) & (xs >= 0) & (xs < cw)
    T[ys[ok], xs[ok]] = True
    return T


def residual_map(f: Path, hw: tuple[int, int], full_w: int) -> np.ndarray:
    frames, shape = plug_frames(f, plug_length)
    out = np.zeros(hw, np.float32)
    if not frames:
        return out
    ev = shape[1] / hw[1]  # eval px -> work px
    for M, cm, valid, border, touch in frames:
        xs = np.nonzero(cm.any(0))[0]
        short = len(xs) == 0 or (xs.max() - xs.min() + 1) < 0.8 * cw / 1.15
        T = best_template(cm, valid, args.sx_wide if (border or short) else args.sx)
        k = np.ones((2 * args.tol + 1,) * 2, bool)
        Ti, To = ndimage.binary_erosion(T, k), ndimage.binary_dilation(T, k)  # tolerance band around the outline
        res = ((Ti & ~cm) | (cm & ~To)) & valid
        res = ndimage.binary_opening(res, np.ones((2, 2), bool))
        if not res.any():
            continue
        lab, n = ndimage.label(res)
        area = ndimage.sum(res, lab, np.arange(1, n + 1))
        score = np.zeros((ch, cw), np.float32)
        for c in range(n):
            score[lab == c + 1] = area[c]
        up = cv2.resize(score, CROP, interpolation=cv2.INTER_NEAREST)
        Me = M.copy()
        Me[:, :2] = M[:, :2] * ev  # eval px -> crop
        back = cv2.warpAffine(up, cv2.invertAffineTransform(Me), hw[::-1], flags=cv2.INTER_NEAREST, borderValue=0)
        out = np.maximum(out, back)
    return out


test = sorted(str(q) for q in (root / "test_public/good").glob("*.png")) + sorted(str(q) for q in (root / "test_public/bad").glob("*.png"))
labels = [0 if "/good/" in q.replace("\\", "/") else 1 for q in test]
val = sorted(str(q) for q in (root / "validation/good").glob("*.png"))
if args.smoke:
    test, labels, val = test[:2] + test[-3:], labels[:2] + labels[-3:], val[:2]
hw = eval_size(test[0], DOWNSCALE)
t0 = time.perf_counter()
cache = lambda q: C / Path(q).parent.parent.name / Path(q).parent.name / (Path(q).stem + ".npz")
tm = np.stack([residual_map(cache(q), hw, 0) for q in test]).astype(np.float16)
vm = np.stack([residual_map(cache(q), hw, 0) for q in val]).astype(np.float16)
out = Path("results/smoke" if args.smoke else "results") / "maps"
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / f"{args.name}__wallplugs.npz", maps=tm, labels=np.array(labels), paths=np.array(test))
np.savez_compressed(out / f"{args.name}__wallplugs__val.npz", maps=vm, paths=np.array(val))
print(f"saved {args.name}: {len(test)} test, {len(val)} val, {time.perf_counter() - t0:.0f}s; "
      f"photos with any residual: test good {np.mean([m.max() > 0 for m, l in zip(tm, labels) if l == 0]) * 100:.0f}%, "
      f"bad {np.mean([m.max() > 0 for m, l in zip(tm, labels) if l == 1]) * 100:.0f}%, val {np.mean([m.max() > 0 for m in vm]) * 100:.0f}%", flush=True)
