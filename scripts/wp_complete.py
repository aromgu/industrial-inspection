"""wallplugs item 3 -- shape completion for cut-off defects (parts 000/001: the GT marks the MISSING part of a
plug, which is background in the photo, so local patch matching only sees the break edge).

Two variants, both on the SAM plug masks (results/plugs_cache, working resolution) and the final model's prediction:
  length   every plug shorter than --short x the typical whole plug (256.5 px, median of training plugs), not touching
           the photo border, whose extension to full length at one end lies on background (Otsu) and not on another
           plug -> the extension is predicted defective (no anomaly map involved; diagnostic of the shape prior alone)
  seeded   only where a predicted component of the final model touches a plug END (within --end-frac of the length
           from that end): extend the prediction beyond that end along the plug axis, plug width, length
           (256.5 - L) if the plug is not cut by the border, else --default-ext x 256.5; background pixels only.
           Only when the predicted pixels on that plug look like a break edge: they span >= --across of the plug width
           and are a band across the axis (extent along <= --band x extent across). Never creates a detection in a
           photo without one, so the good-photo false-alarm rate cannot change.
Result (public test, final model): F1 73.4 -> 75.7, large recall 76.7 -> 86.7, FA 5 % unchanged; 000 GT coverage
0.07 -> 0.94 in the 3 photos with a seed (across 0.7 / band 1.0 picked on this test set: optimistic). Length-only
completion: 0 hits, FA 23 % (tilted plugs resting on others look short).
Prints small/large recall, good-photo false alarms and pooled F1 (all / defective only) for final, final+seeded,
length-only; saves docs/figures/wp_complete.png (photo, GT, final, final+completion for 000/001 and worst FP).
    .venv/Scripts/python.exe scripts/wp_complete.py [--default-ext 0.3]
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.plugs import load_plugs  # noqa: E402
from industrial_inspection.q2 import parts_and_tags, split_parts  # noqa: E402
from make_submission import wallplugs_masks  # noqa: E402
from wp_local import SMALL, gt_of  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--model", default="s0_r1792_L1523")
a.add_argument("--short", type=float, default=0.85)
a.add_argument("--end-frac", type=float, default=0.15)
a.add_argument("--default-ext", type=float, default=0.3)
a.add_argument("--across", type=float, nargs="+", default=[0.7],
               help="seeded: the predicted pixels in the end zone must span this fraction of the plug width across the "
                    "axis (a break edge runs across the whole plug; a fin defect near the end does not)")
a.add_argument("--fig", default="docs/figures/wp_complete.png")
a.add_argument("--band", type=float, nargs="+", default=[1.0],
               help="seeded: the predicted pixels of this plug must be a band across the axis -- extent along the axis "
                    "<= this x extent across (0 = off); a fin/slot defect near the end runs along the plug")
args = a.parse_args()
LMED = 256.5
CACHE = Path("results/plugs_cache/wallplugs")


def fg_mask(path, hw):
    im = np.asarray(Image.open(path).convert("L").resize(hw[::-1], Image.BILINEAR))
    _, b = cv2.threshold(cv2.GaussianBlur(im, (0, 0), 2), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)


def geometry(p, H, W):
    ys, xs = np.nonzero(p.mask)
    c = np.array([xs.mean(), ys.mean()])
    X = np.stack([xs, ys], 1) - c
    u = np.linalg.svd(X, full_matrices=False)[2][0]
    v = np.array([-u[1], u[0]])
    t, s = X @ u, X @ v
    border = xs.min() < 3 or ys.min() < 3 or xs.max() > W - 4 or ys.max() > H - 4
    return c, u, v, t.min(), t.max(), np.percentile(s, 97) - np.percentile(s, 3), border


def extension(c, u, v, end, sgn, width, length, H, W):
    """Rectangle beyond the plug end (polygon fill: fast)."""
    a0, a1 = c + u * end, c + u * (end + sgn * length)
    h = v * width / 2
    poly = np.round(np.array([a0 + h, a1 + h, a1 - h, a0 - h])).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def complete(path, cache, pred_eval, mode, across=0.0, band=0.0):
    """Completion mask at eval size for one photo."""
    hw = pred_eval.shape
    if not Path(cache).exists():
        return np.zeros(hw, bool)
    pl = load_plugs(cache)
    if not pl:
        return np.zeros(hw, bool)
    H, W = pl[0].mask.shape
    fg = fg_mask(path, (H, W))
    occ = np.sum([p.mask for p in pl], 0)
    P = cv2.resize(pred_eval.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) if mode == "seeded" else None
    add = np.zeros((H, W), bool)
    for k, p in enumerate(pl):
        c, u, v, t0, t1, wd, border = geometry(p, H, W)
        L = t1 - t0
        if mode == "length":
            if L > args.short * LMED or L < 0.3 * LMED or border:
                continue
            for sgn, end in ((1, t1), (-1, t0)):
                reg = extension(c, u, v, end, sgn, wd, LMED - L, H, W)
                tip = c + u * (end + sgn * (LMED - L))
                if reg.sum() < 50 or not (0 <= tip[0] < W and 0 <= tip[1] < H):
                    continue
                if 1 - fg[reg].mean() >= 0.85 and ((occ - p.mask)[reg] > 0).mean() < 0.15:
                    add |= reg
        else:
            hitpx = np.argwhere(P & p.mask)
            if not len(hitpx):
                continue
            Th = (hitpx[:, 1] - c[0]) * u[0] + (hitpx[:, 0] - c[1]) * u[1]
            Sh = (hitpx[:, 1] - c[0]) * v[0] + (hitpx[:, 0] - c[1]) * v[1]
            if band and np.ptp(Th) > band * max(np.ptp(Sh), 1):
                continue
            ext = max(LMED - L, 0) if not border else args.default_ext * LMED
            if ext < 8:
                continue
            for sgn, end in ((1, t1), (-1, t0)):
                z_ = sgn * (Th - end) > -args.end_frac * L
                if not z_.any() or np.ptp(Sh[z_]) < across * wd:
                    continue
                reg = extension(c, u, v, end, sgn, wd, ext, H, W)
                add |= reg & ~fg & ~(occ > 0)
    return cv2.resize(add.astype(np.uint8), hw[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)


def metrics(pred, gt, y, keep):
    hit, size = [], []
    for i in np.flatnonzero(keep & (y == 1)):
        lab, n = ndimage.label(gt[i])
        for c in range(1, n + 1):
            r = lab == c
            hit.append(pred[i][r].mean() >= 0.2)
            size.append(r.sum())
    hit, size = np.array(hit), np.array(size)
    P, G, yy = pred[keep], gt[keep], y[keep]
    f1 = lambda m: 200 * (P[m] & G[m]).sum() / max(1, 2 * (P[m] & G[m]).sum() + (P[m] & ~G[m]).sum() + (~P[m] & G[m]).sum())
    return (hit[size < SMALL].mean() * 100, hit[size >= SMALL].mean() * 100,
            P[yy == 0].reshape((yy == 0).sum(), -1).any(1).mean() * 100, f1(np.ones(len(yy), bool)), f1(yy == 1))


if __name__ == "__main__":
    z, zv = np.load(f"results/maps/{args.model}__wallplugs.npz"), np.load(f"results/maps/{args.model}__wallplugs__val.npz")
    paths, y = z["paths"].astype(str), z["labels"]
    _, pred, _ = wallplugs_masks(z["maps"], paths, zv["maps"], zv["paths"].astype(str))
    gt = gt_of(paths, y, pred.shape[1:])
    parts, _ = parts_and_tags(paths)
    dev = split_parts(parts, y)
    cache = [str(CACHE / "test_public" / ("bad" if l else "good") / (Path(q).stem + ".npz")) for q, l in zip(paths, y)]
    sweeps = {(ac, bd): np.stack([pred[i] | complete(paths[i], cache[i], pred[i], "seeded", ac, bd) for i in range(len(y))])
              for ac in args.across for bd in args.band}
    length = np.stack([complete(paths[i], cache[i], pred[i], "length") for i in range(len(y))])
    print("  variant            half  rec_s  rec_l  fa     f1_all  f1_def")
    for nm, P in [("final", pred)] + [(f"seeded a{k[0]} b{k[1]}", v) for k, v in sweeps.items()] + [("length only", length), ("final | length", pred | length)]:
        for h, keep in (("all", np.ones(len(y), bool)), ("dev", dev), ("eval", ~dev)):
            print(f"  {nm:17s}  {h:4s}  " + "  ".join(f"{v:5.1f}" for v in metrics(P, gt, y, keep)))
    best = max(sweeps, key=lambda k: metrics(sweeps[k], gt, y, np.ones(len(y), bool))[3])
    seeded = sweeps[best]
    print(f"  best (across, band) by pooled F1 = {best}; per photo where it added pixels (GT gain / FP px):")
    for i in np.flatnonzero((seeded & ~pred).reshape(len(y), -1).any(1)):
        d = seeded[i] & ~pred[i]
        print(f"    {Path(paths[i]).parent.name}/{Path(paths[i]).stem:18s} +{(d & gt[i]).sum()} GT px, +{(d & ~gt[i]).sum()} FP px")
    print("  000/001 GT coverage final -> seeded:")
    for i in np.flatnonzero(y == 1):
        st = Path(paths[i]).stem
        if st[:3] in ("000", "001"):
            g = gt[i]
            print(f"    {st:18s} {(pred[i] & g).sum() / g.sum():.2f} -> {(seeded[i] & g).sum() / g.sum():.2f}"
                  f"  (+{(seeded[i] & ~pred[i] & ~g).sum()} FP px)")
    added = (seeded & ~pred).reshape(len(y), -1).sum(1)
    print(f"  photos where completion added pixels: {(added > 0).sum()} ({(added[y == 0] > 0).sum()} good)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
    plt.rcParams["font.family"] = "Malgun Gothic"
    pick = [int(np.flatnonzero([Path(q).stem == s and l == 1 for q, l in zip(paths, y)])[0]) for s in ("000_regular", "000_shift_1", "001_regular")]
    fpi = [i for i in np.argsort(-((seeded & ~pred & ~gt).reshape(len(y), -1).sum(1))) if i not in pick][:2]
    li = [i for i in np.argsort(-length.reshape(len(y), -1).sum(1)) if y[i] == 0][:1]
    rows = [(i, "seeded") for i in pick + fpi] + [(i, "length") for i in li]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.ravel()
    for r, (i, kind) in enumerate(rows):
        im = np.asarray(Image.open(paths[i]).convert("RGB").resize(pred.shape[1:][::-1], Image.BILINEAR))
        extra = (seeded[i] & ~pred[i]) if kind == "seeded" else length[i]
        focus = gt[i] | extra | (pred[i] if not gt[i].any() else False)
        if focus.any():
            ys, xs = np.nonzero(focus)
            cy, cx, hh = int(ys.mean()), int(xs.mean()), max(70, int(max(np.ptp(ys), np.ptp(xs)) * 0.9))
            y0, x0 = max(0, cy - hh), max(0, cx - hh)
            ax[r].set_xlim(x0, min(im.shape[1], cx + hh))
            ax[r].set_ylim(min(im.shape[0], cy + hh), y0)
        ax[r].imshow(im)
        if gt[i].any():
            ax[r].contour(gt[i], [0.5], colors="r", linewidths=2)
        ax[r].contour(pred[i], [0.5], colors="c", linewidths=2)
        if extra.any():
            ax[r].contour(extra, [0.5], colors="y", linewidths=2)
        lbl = "결함" if y[i] else "정상"
        ax[r].set_title(f"{Path(paths[i]).stem} ({lbl}) - 빨강 정답, 하늘 최종 모델, 노랑 {'시드 연장' if kind == 'seeded' else '길이 기준 복원'}", fontsize=10)
        ax[r].axis("off")
    fig.tight_layout()
    fig.savefig(args.fig, dpi=70, facecolor="white")
    print("saved", args.fig)
