"""wallplugs: a small learned discriminator on frozen DINOv3 tokens, trained on SYNTHETIC local defects
(GLASS / SimpleNet / DRAEM family). Distance-to-normal scores cannot tell a rare-but-normal look (plug
overlaps, edges) from a defect; a head trained to separate clean tokens from tokens under small synthetic
defects can learn to ignore the former.
  train: random 640 px windows (input scale --short-side) of normal training photos centred on plug pixels;
         1-3 synthetic defects each -- cut-paste of a small rotated patch from elsewhere in the photo,
         thin scratch lines, brightness blobs, missing material (plug pixels replaced by background) --
         token label = defect covers >= 25 % of the token. Head: small conv net on the --layers tokens.
  apply: whole photo at --short-side via HiResKNN.features (same tiling), head logit per token, upsampled to
         the eval size; written as model <name> for wp_local.py / wp_hyst.py (unsmoothed: smooth when scoring).
Training photos exclude every 8th (as the memory bank does); validation photos stay unseen.
    .venv/Scripts/python.exe scripts/wp_synth.py --name synth_head [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.hires import HiResKNN  # noqa: E402
from industrial_inspection.segeval import DOWNSCALE, eval_size  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--name", default="synth_head")
a.add_argument("--short-side", type=int, default=1792)
a.add_argument("--layers", type=int, nargs="+", default=[15, 23])
a.add_argument("--steps", type=int, default=1500)
a.add_argument("--batch", type=int, default=6)
a.add_argument("--lr", type=float, default=1e-3)
a.add_argument("--kinds", type=int, nargs="+", default=[0, 1, 2, 3], help="0 cut-paste, 1 scratch, 2 blob, 3 missing, 4 local warp (bent fin / deformation), "
                    "5 Cutout at the plug outline (cut-off material: background texture or a flat fill)")
a.add_argument("--label-frac", type=float, default=0.25, help="token label = defect covers >= this fraction")
a.add_argument("--preview", type=int, default=0, help="only save N synthetic examples to docs/figures/<name>_examples.png and exit")
a.add_argument("--smoke", action="store_true")
args = a.parse_args()
if args.smoke:
    args.steps = 10
rng = np.random.default_rng(0)
torch.manual_seed(0)
dev = "cuda"
model = HiResKNN(backbone="dinov3_vith16plus", layers=args.layers, short_side=args.short_side, tile=640, overlap=128)
P, T = model.patch, 640
root = Path("datasets/MVTec_AD_2/wallplugs")
train = sorted(str(q) for q in (root / "train/good").glob("*.png"))
train = [q for i, q in enumerate(train) if i % 8]
if args.smoke:
    train = train[:4]


def load_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    s = args.short_side / min(im.size)
    w, h = round(im.width * s / P) * P, round(im.height * s / P) * P
    return np.asarray(im.resize((w, h), Image.BICUBIC))


if args.preview:
    train = train[:: max(1, len(train) // 12)]
imgs = [load_rgb(q) for q in train]
print(f"{len(imgs)} training photos at {imgs[0].shape[:2]}", flush=True)


def ellipse(shape, cy, cx, r, ratio):
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(r), max(2, int(r * ratio))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return m


def paste(out, m, cy, cx, r, patch):
    y0, x0 = max(0, cy - r), max(0, cx - r)
    y1, x1 = min(out.shape[0], cy + r), min(out.shape[1], cx + r)
    sub = m[y0:y1, x0:x1].astype(bool)
    pp = patch[(y0 - (cy - r)):(y0 - (cy - r)) + (y1 - y0), (x0 - (cx - r)):(x0 - (cx - r)) + (x1 - x0)]
    if pp.shape[:2] == sub.shape:
        out[y0:y1, x0:x1][sub] = pp[sub]


def plug_mask(im: np.ndarray):
    """Plug pixels (Otsu on the blurred grey image, small specks removed, small holes filled) and component labels."""
    g8 = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2GRAY), (0, 0), 2)
    _, b = cv2.threshold(g8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    inv_n, inv = cv2.connectedComponents(1 - b)
    areas = np.bincount(inv.ravel())
    b = b.astype(bool) | ((areas[inv] < 1500) & (inv > 0))
    _, comp = cv2.connectedComponents(b.astype(np.uint8))
    return b, comp


_bgfree = {}


def bg_texture(src: np.ndarray, m: np.ndarray):
    """A window-sized slab of src whose pixels under m are all background (>= 6 px from any plug); None if not found."""
    k = id(src)
    if k not in _bgfree:
        pm, _ = plug_mask(src)
        _bgfree.clear()
        _bgfree[k] = ~cv2.dilate(pm.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    free = _bgfree[k]
    ys, xs = np.nonzero(m)
    y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    H, W = m.shape
    sub = m[y0:y1, x0:x1]
    for _ in range(60):
        sy, sx = int(rng.integers(0, src.shape[0] - (y1 - y0))), int(rng.integers(0, src.shape[1] - (x1 - x0)))
        if free[sy:sy + y1 - y0, sx:sx + x1 - x0][sub].all():
            # whole window-sized slab from src aligned so the removed box lands on (sy, sx); outside m it is unused
            # except for the 1 px feather, so take it from src around the same offset, reflect-padded
            pad = cv2.copyMakeBorder(src, H, H, W, W, cv2.BORDER_REFLECT)
            oy, ox = sy - y0 + H, sx - x0 + W
            return pad[oy:oy + H, ox:ox + W].astype(np.float32)
    return None


def synth(win: np.ndarray, src: np.ndarray):
    """1-3 small defects on plug pixels of win (uint8 HxWx3) -> (image, bool mask)."""
    out, mask = win.copy(), np.zeros(win.shape[:2], np.uint8)
    g = win.mean(2)
    plug = g > np.quantile(g, 0.6)
    ys, xs = np.nonzero(plug[24:-24, 24:-24])
    ys, xs = ys + 24, xs + 24
    for _ in range(int(rng.integers(1, 4))):
        if not len(ys):
            break
        i = rng.integers(len(ys))
        cy, cx = int(ys[i]), int(xs[i])
        r = int(rng.integers(4, 18))
        kind = int(rng.choice(args.kinds))
        if kind == 0:  # cut-paste: a small rotated patch from elsewhere in the photo
            sy, sx = int(rng.integers(r, src.shape[0] - r)), int(rng.integers(r, src.shape[1] - r))
            patch = cv2.warpAffine(np.ascontiguousarray(src[sy - r:sy + r, sx - r:sx + r]),
                                   cv2.getRotationMatrix2D((r, r), float(rng.uniform(0, 360)), 1.0), (2 * r, 2 * r))
            m = ellipse(mask.shape, cy, cx, r, rng.uniform(0.3, 1))
            paste(out, m, cy, cx, r, patch)
        elif kind == 1:  # scratch: a thin darker or brighter line
            m = np.zeros_like(mask)
            ang, L = rng.uniform(0, np.pi), int(rng.integers(10, 40))
            p0 = (int(cx - L * np.cos(ang) / 2), int(cy - L * np.sin(ang) / 2))
            p1 = (int(cx + L * np.cos(ang) / 2), int(cy + L * np.sin(ang) / 2))
            cv2.line(m, p0, p1, 1, int(rng.integers(1, 4)))
            f = float(rng.choice([0.5, 0.7, 1.3, 1.5]))
            sel = m.astype(bool)
            out[sel] = np.clip(out[sel].astype(np.float32) * f, 0, 255).astype(np.uint8)
        elif kind == 2:  # blob: a smooth brightness dent / bump
            m = ellipse(mask.shape, cy, cx, r, rng.uniform(0.4, 1))
            w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), r / 3) * float(rng.choice([-1, 1])) * rng.uniform(0.25, 0.5)
            out = np.clip(out.astype(np.float32) * (1 + w[..., None]), 0, 255).astype(np.uint8)
        elif kind == 4:  # local warp: a small region pushed sideways (bent fin, deformed rim); label = moved pixels
            R2 = 2 * r
            y0, x0 = max(0, cy - R2), max(0, cx - R2)
            y1, x1 = min(out.shape[0], cy + R2), min(out.shape[1], cx + R2)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            wgt = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r * 0.7) ** 2))
            ang = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.35, 0.8) * r
            mx, my = xx - amp * wgt * np.cos(ang), yy - amp * wgt * np.sin(ang)
            patch = cv2.remap(out, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            changed = np.abs(patch.astype(np.int16) - out[y0:y1, x0:x1].astype(np.int16)).max(2) > 25
            out[y0:y1, x0:x1] = patch
            m = np.zeros_like(mask)
            m[y0:y1, x0:x1] = changed
        elif kind == 5:  # Cutout (DeVries & Taylor 2017) at the plug OUTLINE: part of the edge / end erased
            edge = plug & ~cv2.erode(plug.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            ey, ex = np.nonzero(edge[24:-24, 24:-24])
            if not len(ey):
                continue
            j = rng.integers(len(ey))
            cy, cx = int(ey[j]) + 24, int(ex[j]) + 24
            r = int(rng.integers(12, 40))  # 001's missing strip is ~30 px at 1792 px input
            m = np.zeros_like(mask)
            if rng.random() < 0.5:
                cv2.ellipse(m, (cx, cy), (r, max(4, int(r * rng.uniform(0.4, 1)))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
            else:
                rect = ((cx, cy), (2 * r, int(2 * r * rng.uniform(0.3, 1))), float(rng.uniform(0, 180)))
                cv2.fillPoly(m, [cv2.boxPoints(rect).astype(np.int32)], 1)
            m &= plug.astype(np.uint8)
            y0, x0 = max(0, cy - r - 2), max(0, cx - r - 2)
            y1, x1 = min(out.shape[0], cy + r + 2), min(out.shape[1], cx + r + 2)
            sub = m[y0:y1, x0:x1].astype(bool)
            if rng.random() < 0.7:  # fill with background texture from a dark spot of the window
                bg = np.argwhere(g < np.quantile(g, 0.2))
                by, bx = bg[rng.integers(len(bg))]
                by = int(np.clip(by, 0, win.shape[0] - (y1 - y0))); bx = int(np.clip(bx, 0, win.shape[1] - (x1 - x0)))
                src_bg = win[by:by + (y1 - y0), bx:bx + (x1 - x0)]
                if src_bg.shape[:2] == sub.shape:
                    out[y0:y1, x0:x1][sub] = src_bg[sub]
            else:  # classic Cutout: flat fill with the background level
                out[y0:y1, x0:x1][sub] = np.uint8(np.quantile(g, 0.1))
        elif kind == 6:  # straight-line cut, modelled on the real cut-off defects (001: thin strip shaved off a long
            # edge; 000: plug end cut across) -- all plug pixels beyond a straight line near the outline removed, the
            # gap filled with REAL background texture of the same photo (no flat fill, no visible seam), 1 px feather
            pm, comp = plug_mask(win)
            em = pm & ~cv2.erode(pm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ey, ex = np.nonzero(em[32:-32, 32:-32])
            if not len(ey):
                continue
            j = rng.integers(len(ey))
            cy, cx = int(ey[j]) + 32, int(ex[j]) + 32
            sm = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 6)
            gy, gx = np.gradient(sm)
            n = -np.array([gy[cy, cx], gx[cy, cx]])
            if np.linalg.norm(n) < 1e-4:
                continue
            n /= np.linalg.norm(n)  # outward normal
            t = np.array([-n[1], n[0]])
            if rng.random() < 0.6:  # 001-like: thin strip shaved off a long edge
                d, L = float(rng.uniform(4, 14)), float(rng.uniform(60, 220))
            else:  # 000-like: a chunk / the end cut off
                d, L = float(rng.uniform(20, 90)), float(rng.uniform(50, 160))
            ang = rng.normal(0, 0.06)  # cut line nearly parallel to the local outline
            nn_ = np.array([n[0] * np.cos(ang) - n[1] * np.sin(ang), n[0] * np.sin(ang) + n[1] * np.cos(ang)])
            yy, xx = np.mgrid[0:pm.shape[0], 0:pm.shape[1]].astype(np.float32)
            along = (yy - cy) * t[0] + (xx - cx) * t[1]
            jag = 0 if d < 30 else cv2.GaussianBlur(rng.normal(0, 2.5, pm.shape[1]).astype(np.float32)[None], (0, 0), 3)[0][np.clip(xx.astype(int), 0, pm.shape[1] - 1)]
            across = (yy - cy) * nn_[0] + (xx - cx) * nn_[1] + jag
            # only the layer between the cut line and the local outline (+6 px), so the cut never jumps a gap into a
            # neighbouring (touching) plug
            m = (across > -d) & (across < 6) & (np.abs(along) < L / 2) & pm & (comp == comp[cy, cx])
            outer = (across > 8) & (across < 40) & (np.abs(along) < L / 2)
            if m.sum() < 30 or pm[outer].mean() > 0.1:  # outside of the cut line must be open background (outer outline)
                continue
            bgt = bg_texture(src, m)
            if bgt is None:
                continue
            al = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)[..., None]
            al = np.maximum(al, m[..., None])  # removed pixels fully replaced, feather only outside them
            out = (al * bgt + (1 - al) * out.astype(np.float32)).astype(np.uint8)
            m = m.astype(np.uint8)
        else:  # missing material: plug pixels replaced by background texture
            bg = np.argwhere(g < np.quantile(g, 0.2))
            if not len(bg):
                continue
            by, bx = bg[rng.integers(len(bg))]
            by, bx = int(np.clip(by, r, win.shape[0] - r - 1)), int(np.clip(bx, r, win.shape[1] - r - 1))
            m = ellipse(mask.shape, cy, cx, r, rng.uniform(0.4, 1)) & plug.astype(np.uint8)
            paste(out, m, cy, cx, r, win[by - r:by + r, bx - r:bx + r])
        mask |= m
    return out, mask.astype(bool)


if args.preview:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = args.preview
    fig, ax = plt.subplots(2, k, figsize=(3 * k, 6.4))
    done = 0
    while done < k:
        im = imgs[rng.integers(len(imgs))]
        g = im.mean(2)
        pts = np.argwhere(g[T // 2:-T // 2:8, T // 2:-T // 2:8] > np.quantile(g, 0.7))
        cy, cx = pts[rng.integers(len(pts))] * 8 + T // 2
        win = im[cy - T // 2:cy + T // 2, cx - T // 2:cx + T // 2]
        o, m = synth(win, im)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, x0 = int(np.clip(ys.mean() - 100, 0, T - 200)), int(np.clip(xs.mean() - 100, 0, T - 200))
        ax[0, done].imshow(win[y0:y0 + 200, x0:x0 + 200])
        ov = o[y0:y0 + 200, x0:x0 + 200].copy()
        cnt, _ = cv2.findContours(m[y0:y0 + 200, x0:x0 + 200].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ax[1, done].imshow(o[y0:y0 + 200, x0:x0 + 200])
        ax[1, done].contour(m[y0:y0 + 200, x0:x0 + 200], [0.5], colors="r", linewidths=0.6)
        for r in (0, 1):
            ax[r, done].axis("off")
        done += 1
    ax[0, 0].set_title("original (200 px crop)")
    ax[1, 0].set_title("synthetic (red = label)")
    fig.tight_layout()
    fig.savefig(f"docs/figures/{args.name}_examples.png", dpi=90)
    print("saved preview")
    sys.exit(0)


def batch():
    xs, ys = [], []
    for _ in range(args.batch):
        im = imgs[rng.integers(len(imgs))]
        g = im.mean(2)
        grid = g[T // 2:-T // 2:8, T // 2:-T // 2:8] > np.quantile(g, 0.7)
        pts = np.argwhere(grid)
        cy, cx = (pts[rng.integers(len(pts))] * 8 + T // 2) if len(pts) else (im.shape[0] // 2, im.shape[1] // 2)
        win = im[cy - T // 2:cy + T // 2, cx - T // 2:cx + T // 2]
        if rng.random() < 0.2:  # some clean windows too
            w2, m2 = win, np.zeros(win.shape[:2], bool)
        else:
            w2, m2 = synth(win, im)
        w2 = np.clip(w2.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255)
        xs.append(w2 / 255)
        ys.append(F.avg_pool2d(torch.from_numpy(m2.astype(np.float32))[None, None], P)[0, 0] >= args.label_frac)
    x = model.normalize(torch.from_numpy(np.stack(xs).astype(np.float32)).permute(0, 3, 1, 2).to(dev))
    return x, torch.stack(ys).to(dev)


@torch.no_grad()
def feats(x):
    with torch.autocast("cuda", dtype=torch.float16):
        o = model.vit.forward_intermediates(x, indices=args.layers)
    return torch.cat([t.float() for t in o], 1)  # (B, C*L, g, g)


C = 1280 * len(args.layers)
head = nn.Sequential(nn.Conv2d(C, 512, 1), nn.GELU(), nn.Dropout(0.1), nn.Conv2d(512, 256, 3, padding=1), nn.GELU(),
                     nn.Conv2d(256, 1, 1)).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
t0 = time.perf_counter()
hist = []
for step in range(args.steps):
    x, y = batch()
    logit = head(feats(x))[:, 0]
    pos = y.float().mean().clamp_min(1e-3)
    loss = F.binary_cross_entropy_with_logits(logit, y.float(), pos_weight=((1 - pos) / pos).clamp(max=50))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    hist.append(loss.item())
    if step % 100 == 0 or step == args.steps - 1:
        print(f"step {step} loss {np.mean(hist[-100:]):.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
out = Path("results/smoke" if args.smoke else "results")
(out / "maps").mkdir(parents=True, exist_ok=True)
torch.save(head.state_dict(), out / f"{args.name}_head.pt")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.figure(figsize=(6, 3))
plt.plot(np.convolve(hist, np.ones(20) / 20, mode="valid") if len(hist) >= 20 else hist)
plt.xlabel("step")
plt.ylabel("BCE (20-step mean)")
plt.title(f"{args.name} training loss")
plt.tight_layout()
plt.savefig(f"{out if args.smoke else 'docs/figures'}/{args.name}_loss.png", dpi=110)

head.eval()
test = sorted(str(q) for q in (root / "test_public/good").glob("*.png")) + sorted(str(q) for q in (root / "test_public/bad").glob("*.png"))
labels = [0 if "/good/" in q.replace("\\", "/") else 1 for q in test]
val = sorted(str(q) for q in (root / "validation/good").glob("*.png"))
if args.smoke:
    test, labels, val = test[:2] + test[-2:], labels[:2] + labels[-2:], val[:2]
hw = eval_size(test[0], DOWNSCALE)


@torch.no_grad()
def apply(path):
    f = torch.cat(model.features(model.load(path)), 0)[None]  # per layer (C, h, w), tiles stitched
    lg = head(f)[0, 0]
    return F.interpolate(lg[None, None], size=hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


tm = np.stack([apply(q) for q in test]).astype(np.float16)
vm = np.stack([apply(q) for q in val]).astype(np.float16)
np.savez_compressed(out / "maps" / f"{args.name}__wallplugs.npz", maps=tm, labels=np.array(labels), paths=np.array(test))
np.savez_compressed(out / "maps" / f"{args.name}__wallplugs__val.npz", maps=vm, paths=np.array(val))
print(f"saved {args.name}: {len(test)} test, {len(val)} val, {time.perf_counter() - t0:.0f}s", flush=True)
