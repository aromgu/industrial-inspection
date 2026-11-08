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
a.add_argument("--kinds", type=int, nargs="+", default=[0, 1, 2, 3], help="0 cut-paste, 1 scratch, 2 blob, 3 missing, 4 local warp (bent fin / deformation)")
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
        ys.append(F.avg_pool2d(torch.from_numpy(m2.astype(np.float32))[None, None], P)[0, 0] >= 0.25)
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
