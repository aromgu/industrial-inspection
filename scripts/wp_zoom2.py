"""wallplugs item 2 -- zoomed second look at candidate spots (small defects: a 2-3 px scratch inside an ~18 px token).

  1. candidates: per photo the --k strongest local maxima (NMS --nms eval px) of the final model's seed map
     N = gauss2(map) - photo p90, above the --qc quantile of the validation per-photo max N (lower than the seed
     threshold, so spots the final rule misses get a second look)
  2. zoom: a crop of the ORIGINAL photo around each candidate, resized so one DINOv3 token covers 2048/--zoom-short x 16
     original pixels (default 3584 -> ~9 px instead of ~18 px), one 640 x 640 window, layers 15 and 23
  3. zoom bank: the same-size crops of the training photos (every 8th held out, as the memory bank), --bank-crops per
     photo on plug pixels, brightness 0.8-1.2, distance-based subsampling to 100k tokens per layer
  4. zoom score = max of the (sigma 1 token) k=1 distance map within --core tokens of the crop centre, mean of layers
  5. decision: candidates whose zoom score beats the --qz quantile of the validation per-photo max zoom score become
     extra seeds; extent = zoom map above the validation zoom pixel q99.5, connected to the centre, pasted back
Prints the final rule vs final + second look (small/large recall, FA, pooled F1, dev/eval halves), how many GT
regions a candidate lands on, and the zoom scores of the hard small defects vs the normal ceiling.
Saves docs/figures/wp_zoom.png (candidate crops: normal-scale map vs zoom map) and the zoom scores to
results/wp_zoom_<name>.npz.
Successor of wp_zoom.py (192 px window x2, no context) with a wider context window (366 px of the original, x1.75).
Result (base features): the 5 candidates/photo never land on the hard small defects 001/002/004/007 (their peaks are
not among the photo's top local maxima); 005 gets zoom score 422-429 vs the normal validation maximum 429; shift-light
good test photos score higher than the (regular-light) validation photos, so FA 5 % -> 18-35 %, F1 73.4 -> 72.8-73.0.
Second look does not separate -> failed again.
    .venv/Scripts/python.exe scripts/wp_zoom2.py [--proj results/adapt_inv.pt --name adapt] [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.hires import HiResKNN, subsample_distance_based  # noqa: E402
from industrial_inspection.q2 import parts_and_tags, split_parts  # noqa: E402
from make_submission import wallplugs_masks  # noqa: E402
from wp_local import SMALL, gt_of  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--model", default="s0_r1792_L1523")
a.add_argument("--name", default="base")
a.add_argument("--zoom-short", type=int, default=3584)
a.add_argument("--k", type=int, default=5)
a.add_argument("--nms", type=int, default=30)
a.add_argument("--qc", type=float, default=0.5)
a.add_argument("--qz", type=float, nargs="+", default=[0.9, 0.95, 1.0])
a.add_argument("--core", type=int, default=6)
a.add_argument("--bank-crops", type=int, default=4)
a.add_argument("--proj", default=None)
a.add_argument("--smoke", action="store_true")
args = a.parse_args()

def metrics(pred, gt, y, keep):
    """small / large region recall (>= 20 % covered), good-photo false alarms, pooled F1 (all / defective only)."""
    hit, size = [], []
    for i in np.flatnonzero(keep & (y == 1)):
        lab, n = ndimage.label(gt[i])
        for c in range(1, n + 1):
            r = lab == c
            hit.append(pred[i][r].mean() >= 0.2)
            size.append(r.sum())
    hit, size = np.array(hit), np.array(size)
    Pk, G, yy = pred[keep], gt[keep], y[keep]
    f1 = lambda m: 200 * (Pk[m] & G[m]).sum() / max(1, 2 * (Pk[m] & G[m]).sum() + (Pk[m] & ~G[m]).sum() + (~Pk[m] & G[m]).sum())
    sm = hit[size < SMALL].mean() * 100 if (size < SMALL).any() else np.nan
    lg = hit[size >= SMALL].mean() * 100 if (size >= SMALL).any() else np.nan
    return sm, lg, Pk[yy == 0].reshape((yy == 0).sum(), -1).any(1).mean() * 100 if (yy == 0).any() else np.nan, f1(np.ones(len(yy), bool)), f1(yy == 1)

torch.manual_seed(0)
rng = np.random.default_rng(0)
root = Path("datasets/MVTec_AD_2/wallplugs")
T = 640
OS = 2048  # original short side
crop_orig = T * OS / args.zoom_short  # crop side in original pixels

model = HiResKNN(backbone="dinov3_vith16plus", layers=[15, 23], short_side=1792, tile=640, overlap=128)
if args.proj:
    from industrial_inspection.adapt import load_proj
    model.proj = load_proj(args.proj)
P = model.patch
g = T // P


def crop(im: Image.Image, cy: float, cx: float) -> torch.Tensor:
    """(3, T, T) in [0, 1]: original-resolution crop of side crop_orig centred at (cy, cx), resized to T."""
    h = crop_orig / 2
    y0 = float(np.clip(cy - h, 0, im.height - crop_orig))
    x0 = float(np.clip(cx - h, 0, im.width - crop_orig))
    c = im.resize((T, T), Image.BICUBIC, box=(x0, y0, x0 + crop_orig, y0 + crop_orig))
    return torch.from_numpy(np.asarray(c, dtype=np.float32) / 255).permute(2, 0, 1), (y0, x0)


@torch.no_grad()
def tokens(x: torch.Tensor) -> list[torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.float16):
        o = model.vit.forward_intermediates(model.normalize(x[None].to(model.device)), indices=model.layers, norm=False,
                                            output_fmt="NCHW", intermediates_only=True)
    fs = [t[0].float() for t in o]
    if getattr(model, "proj", None) is not None:
        fs = [p(f.permute(1, 2, 0)).permute(2, 0, 1).contiguous() for p, f in zip(model.proj, fs)]
    return fs


# ---------- zoom bank ----------
t0 = time.perf_counter()
train = sorted(str(q) for q in (root / "train/good").glob("*.png"))
train = [q for i, q in enumerate(train) if i % 8]
if args.smoke:
    train = train[:4]
pools = [[] for _ in model.layers]
for q in train:
    im = Image.open(q).convert("RGB")
    gr = np.asarray(im.convert("L").reduce(8), dtype=np.float32) / 255
    ys, xs = np.nonzero(gr > 0.35)
    for _ in range(args.bank_crops):
        j = rng.integers(len(ys))
        x, _ = crop(im, ys[j] * 8 + 4, xs[j] * 8 + 4)
        x = torch.clip(x * float(rng.uniform(0.8, 1.2)), 0, 1)
        for pool, f in zip(pools, tokens(x)):
            t = f.flatten(1).T
            pool.append(t[torch.from_numpy(rng.choice(len(t), 600, replace=False)).to(t.device)].half().cpu())
bank = [subsample_distance_based(torch.cat(p), 100_000, device="cuda").cuda().half() for p in pools]
print(f"zoom bank {len(bank[0])} tokens/layer from {len(train)} photos ({time.perf_counter() - t0:.0f}s)", flush=True)


@torch.no_grad()
def zoom_map(x: torch.Tensor) -> np.ndarray:
    out = []
    for f, b in zip(tokens(x), bank):
        q = f.flatten(1).T
        d = torch.cat([torch.cdist(q, b[i:i + 32768].float()).min(1, keepdim=True).values for i in range(0, len(b), 32768)], 1)
        out.append(d.min(1).values.reshape(g, g).cpu().numpy())
    return ndimage.gaussian_filter(np.mean(out, 0), 1.0)


# ---------- candidates from the final model ----------
z, zv = np.load(f"results/maps/{args.model}__wallplugs.npz"), np.load(f"results/maps/{args.model}__wallplugs__val.npz")
paths, y, vpaths = z["paths"].astype(str), z["labels"], zv["paths"].astype(str)
if args.smoke:
    keep = np.r_[0:2, len(y) - 2:len(y)]
    z = {"maps": z["maps"][keep]}
    paths, y, vpaths = paths[keep], y[keep], vpaths[:3]
    zv = {"maps": zv["maps"][:3]}
R, pred, (ts, te) = wallplugs_masks(z["maps"], paths, zv["maps"], vpaths)
gs = lambda arr: np.stack([ndimage.gaussian_filter(m.astype(np.float32), 2.0) for m in arr])
seed = lambda A: A - np.percentile(A.reshape(len(A), -1), 90, axis=1)[:, None, None]
N, Nv = seed(gs(z["maps"])), seed(gs(zv["maps"]))
tc = np.quantile(Nv.reshape(len(Nv), -1).max(1), args.qc)
hw = N.shape[1:]
s_eval = OS / hw[0]  # original px per eval px


def candidates(n):
    b = np.zeros_like(n, bool)
    b[12:-12, 12:-12] = True
    mx = (n == ndimage.maximum_filter(n, size=2 * args.nms + 1)) & (n > tc) & b
    ys, xs = np.nonzero(mx)
    o = np.argsort(-n[ys, xs])[:args.k]
    return [(int(ys[i]), int(xs[i]), float(n[ys[i], xs[i]])) for i in o]


def second_look(path, n):
    im = Image.open(path).convert("RGB")
    res = []
    for cy, cx, nv in candidates(n):
        x, (y0, x0) = crop(im, cy * s_eval, cx * s_eval)
        m = zoom_map(x)
        # candidate position in crop tokens
        ty, tx = (cy * s_eval - y0) / crop_orig * g, (cx * s_eval - x0) / crop_orig * g
        yy, xx = np.mgrid[:g, :g]
        core = (np.abs(yy - ty) <= args.core) & (np.abs(xx - tx) <= args.core)
        res.append(dict(cy=cy, cx=cx, n=nv, score=float(m[core].max()), map=m.astype(np.float16), y0=y0, x0=x0))
    return res


t0 = time.perf_counter()
looks = [second_look(p, n) for p, n in zip(paths, N)]
vlooks = [second_look(p, n) for p, n in zip(vpaths, Nv)]
print(f"second looks: {sum(map(len, looks))} test / {sum(map(len, vlooks))} val candidates ({time.perf_counter() - t0:.0f}s)", flush=True)
vmax = np.array([max([c["score"] for c in L], default=0) for L in vlooks])
vpix = np.concatenate([c["map"].astype(np.float32).ravel() for L in vlooks for c in L]) if any(vlooks) else np.zeros(1)
tpix = np.quantile(vpix, 0.995)

gt = gt_of(paths, y, hw)
parts, tag = parts_and_tags(paths)
dev = split_parts(parts, y)


def paste(L, shape):
    """Zoom-extent masks of accepted candidates at eval size."""
    out = np.zeros(shape, bool)
    for c in L:
        mz = c["map"].astype(np.float32) > tpix
        lab, _ = ndimage.label(mz)
        ty = int(np.clip((c["cy"] * s_eval - c["y0"]) / crop_orig * g, 0, g - 1))
        tx = int(np.clip((c["cx"] * s_eval - c["x0"]) / crop_orig * g, 0, g - 1))
        win = lab[max(0, ty - args.core):ty + args.core + 1, max(0, tx - args.core):tx + args.core + 1]
        ks = np.unique(win[win > 0])
        reg = np.isin(lab, ks) if len(ks) else np.zeros_like(mz)
        reg[ty, tx] = True
        up = np.asarray(Image.fromarray(reg.astype(np.uint8) * 255).resize((int(crop_orig / s_eval),) * 2, Image.NEAREST)) > 0
        oy, ox = int(round(c["y0"] / s_eval)), int(round(c["x0"] / s_eval))
        hh, ww = min(up.shape[0], shape[0] - oy), min(up.shape[1], shape[1] - ox)
        out[oy:oy + hh, ox:ox + ww] |= up[:hh, :ww]
    return out


print(f"== second look ({args.name}): zoom {args.zoom_short}, {args.k} candidates/photo above val-q{args.qc} ({tc:.1f})")
print("  variant              half  rec_s  rec_l  fa     f1_all  f1_def")
halves = (("all", np.ones(len(y), bool)), ("dev", dev), ("eval", ~dev))
for h, keep in halves:
    print(f"  {'final':19s}  {h:4s}  " + "  ".join(f"{v:5.1f}" for v in metrics(pred, gt, y, keep)))
best = None
for qz in args.qz:
    tz = np.quantile(vmax, qz) if qz < 1 else vmax.max() * 1.0001
    P2 = np.stack([pred[i] | paste([c for c in looks[i] if c["score"] > tz], hw) for i in range(len(y))])
    for h, keep in halves:
        mtr = metrics(P2, gt, y, keep)
        print(f"  {f'+ zoom qz{qz}':19s}  {h:4s}  " + "  ".join(f"{v:5.1f}" for v in mtr))
        if h == "all" and (best is None or mtr[3] > best[0]):
            best = (mtr[3], qz, P2)
# which GT regions does a candidate land on, and their zoom score vs the normal ceiling
print(f"  val per-photo max zoom score: median {np.median(vmax):.1f}, q90 {np.quantile(vmax, 0.9):.1f}, max {vmax.max():.1f}")
on_gt, small_on = 0, []
for i in np.flatnonzero(y == 1):
    lab, n = ndimage.label(gt[i])
    for r in range(1, n + 1):
        reg = ndimage.binary_dilation(lab == r, iterations=4)
        cs = [c for c in looks[i] if reg[c["cy"], c["cx"]]]
        if cs:
            on_gt += 1
        if (lab == r).sum() < SMALL:
            small_on.append((Path(paths[i]).stem, max([c["score"] for c in cs], default=np.nan),
                             max([c["n"] for c in cs], default=np.nan), bool(pred[i][lab == r].mean() >= 0.2)))
print(f"  GT regions with a candidate on them: {on_gt}")
print("  small GT regions (photo, zoom score, normal-scale N, already detected):")
for s in small_on:
    print(f"    {s[0]:20s} zoom {s[1]:6.1f}  N {s[2]:6.1f}  {'det' if s[3] else '-'}")
np.savez_compressed(f"results/wp_zoom2_{args.name}.npz", vmax=vmax, test_scores=np.array([[c["score"] for c in L] + [np.nan] * (args.k - len(L)) for L in looks]),
                    test_n=np.array([[c["n"] for c in L] + [np.nan] * (args.k - len(L)) for L in looks]))

# ---------- figure: hard small defects, normal-scale map vs zoom map ----------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
rows = []
for i in np.flatnonzero(y == 1):
    for c in looks[i]:
        if gt[i][max(0, c["cy"] - 4):c["cy"] + 5, max(0, c["cx"] - 4):c["cx"] + 5].any():
            rows.append((i, c))
            break
rows = sorted(rows, key=lambda r: gt[r[0]].sum())[:4]
fp = sorted([(i, c) for i in np.flatnonzero(y == 0) for c in looks[i]], key=lambda r: -r[1]["score"])[:2]
rows += fp
if rows:
    fig, ax = plt.subplots(len(rows), 3, figsize=(10, 3.4 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (i, c) in enumerate(rows):
        im = Image.open(paths[i]).convert("RGB")
        x, _ = crop(im, c["cy"] * s_eval, c["cx"] * s_eval)
        ax[r, 0].imshow(x.permute(1, 2, 0).numpy())
        if gt[i].any():
            gm = np.asarray(Image.fromarray(gt[i].astype(np.uint8) * 255).resize((im.width, im.height), Image.NEAREST))
            gc = np.asarray(Image.fromarray(gm).resize((T, T), Image.NEAREST, box=(c["x0"], c["y0"], c["x0"] + crop_orig, c["y0"] + crop_orig))) > 0
            ax[r, 0].contour(gc, [0.5], colors="r", linewidths=1)
        oy, ox, w = int(round(c["y0"] / s_eval)), int(round(c["x0"] / s_eval)), int(crop_orig / s_eval)
        ax[r, 1].imshow(N[i][oy:oy + w, ox:ox + w], cmap="jet", vmin=0, vmax=max(ts, N[i][oy:oy + w, ox:ox + w].max()))
        ax[r, 2].imshow(c["map"].astype(np.float32), cmap="jet", vmin=np.quantile(vpix, 0.5), vmax=max(np.quantile(vmax, 0.95), float(c["map"].max())))
        lbl = "결함" if y[i] else "정상"
        ax[r, 0].set_title(f"{Path(paths[i]).stem} ({lbl})", fontsize=9)
        ax[r, 1].set_title(f"기존 배율 지도 N={c['n']:.0f} (시드 기준 {ts:.0f})", fontsize=9)
        ax[r, 2].set_title(f"확대 지도 점수 {c['score']:.1f} (정상 q95 {np.quantile(vmax, 0.95):.1f})", fontsize=9)
        for k in range(3):
            ax[r, k].axis("off")
    fig.suptitle(f"확대 재검사 ({args.name}): 위 = 작은 결함 후보, 아래 2줄 = 정상 사진에서 가장 높은 후보", fontsize=11)
    fig.tight_layout()
    out = "results/smoke" if args.smoke else "docs/figures"
    Path(out).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}/wp_zoom2_{args.name}.png", dpi=80, facecolor="white")
    print("saved", f"{out}/wp_zoom2_{args.name}.png")
