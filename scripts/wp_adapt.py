"""wallplugs P3 (local defects) -- feature-space adaptation on NORMAL photos only (CFA / dense-contrastive family).
Diagnosis behind it: every input trick raised defect scores and the normal noise floor together (ratio 0.75-0.97);
the noise is normal-vs-normal variation of the frozen DINOv3 tokens (lighting, highlights, orientation of the
cylindrical plugs). So learn a small per-layer projection g (residual MLP, identity at init) on the frozen tokens that
  * makes the SAME token invariant to photometric changes (brightness 0.5-1.2, directional relighting from
    lighting.py, gamma) and to 90-degree rotations / flips of the window (tokens re-aligned exactly), and
  * keeps DIFFERENT tokens apart -- dense InfoNCE (positives: same position in the other view; negatives: other
    positions of the window, excluding the 8 neighbours), plus a ratio term
    mean ||g(a)-g(b)||^2 / mean ||g(a_i)-g(a_j)||^2 (invariance relative to spread; cannot collapse).
  --cfa adds CFA-style compactness (Lee et al. 2022): pull g(a) towards its nearest neighbour in a memory of adapted
  normal tokens (refreshed every 100 steps) -- not towards everything, so structure is kept.
No defect is ever shown (real or synthetic). The memory bank and test maps are then rebuilt THROUGH g:
    .venv/Scripts/python.exe scripts/wp_adapt.py --name adapt_inv [--cfa 0.1] [--smoke]
    .venv/Scripts/python.exe scripts/hires.py ... --proj results/adapt_inv.pt --name s0_r1792_adapt_inv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.hires import HiResKNN  # noqa: E402
from industrial_inspection.lighting import light_field, relight  # noqa: E402
from industrial_inspection.adapt import Proj  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--name", default="adapt_inv")
a.add_argument("--layers", type=int, nargs="+", default=[15, 23])
a.add_argument("--short-side", type=int, default=1792)
a.add_argument("--steps", type=int, default=1500)
a.add_argument("--batch", type=int, default=2)
a.add_argument("--tokens", type=int, default=384, help="sampled token positions per window for the losses")
a.add_argument("--tau", type=float, default=0.1)
a.add_argument("--ratio", type=float, default=1.0, help="weight of the invariance/spread ratio term")
a.add_argument("--cfa", type=float, default=0.0, help="weight of CFA-style nearest-memory compactness (0 = off)")
a.add_argument("--hidden", type=int, default=512)
a.add_argument("--lr", type=float, default=3e-4)
a.add_argument("--smoke", action="store_true")
args = a.parse_args()
if args.smoke:
    args.steps = 6
torch.manual_seed(0)
rng = np.random.default_rng(0)
dev = "cuda"
model = HiResKNN(backbone="dinov3_vith16plus", layers=args.layers, short_side=args.short_side, tile=640, overlap=128)
P, T = model.patch, 640
root = Path("datasets/MVTec_AD_2/wallplugs")
train = sorted(str(q) for q in (root / "train/good").glob("*.png"))
train = [q for i, q in enumerate(train) if i % 8]  # same hold-out as the memory bank
if args.smoke:
    train = train[:3]
imgs = [model.load_raw(q)[0] for q in train]  # (3, H, W) in [0, 1], CPU
print(f"{len(imgs)} photos at {tuple(imgs[0].shape[1:])}", flush=True)


C = 1280
proj = nn.ModuleList([Proj(C, args.hidden) for _ in args.layers]).to(dev)
opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)


def photometric(x):
    x = torch.clip(x * float(rng.uniform(0.5, 1.2)), 0, 1)
    if rng.random() < 0.7:
        gen = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
        gain, hl, gamma = light_field(*x.shape[-2:], generator=gen)
        x = relight(x, gain, hl, gamma)
    return x


@torch.no_grad()
def tokens(x):
    with torch.autocast("cuda", dtype=torch.float16):
        o = model.vit.forward_intermediates(model.normalize(x[None].to(dev)), indices=args.layers, norm=False,
                                            output_fmt="NCHW", intermediates_only=True)
    return [t[0].float() for t in o]  # per layer (C, g, g)


def window():
    im = imgs[rng.integers(len(imgs))]
    g = im.mean(0)
    H, W = g.shape
    for _ in range(20):
        y0, x0 = int(rng.integers(0, H - T)), int(rng.integers(0, W - T))
        if (g[y0:y0 + T, x0:x0 + T] > 0.35).float().mean() > 0.15:  # enough plug in view
            break
    return im[:, y0:y0 + T, x0:x0 + T], g[y0:y0 + T:P, x0:x0 + T:P] > 0.35


def pairs():
    """Two views of one window: A = photometric(win); B = photometric(rot90^k / flip of win), tokens of B mapped
    back to A's grid. Returns per layer (fa, fb) as (N, C) at sampled positions."""
    win, plug = window()
    k, fl = int(rng.integers(4)), bool(rng.integers(2))
    wb = torch.rot90(win, k, (1, 2))
    if fl:
        wb = torch.flip(wb, (2,))
    fa, fb = tokens(photometric(win)), tokens(photometric(wb))
    out = []
    g = fa[0].shape[-1]
    w = plug.flatten().float() * 3 + 1  # sample plug tokens 4x more often
    idx = torch.multinomial(w, min(args.tokens, g * g), replacement=False)
    yy, xx = idx // g, idx % g
    for A, B in zip(fa, fb):
        if fl:
            B = torch.flip(B, (2,))
        B = torch.rot90(B, -k, (1, 2))  # back to A's orientation
        out.append((A[:, yy.to(dev), xx.to(dev)].T, B[:, yy.to(dev), xx.to(dev)].T, torch.stack([yy, xx], 1).to(dev)))
    return out


def loss_layer(p, fa, fb, pos, mem):
    za, zb = p(fa), p(fb)
    # dense InfoNCE, excluding the 8-neighbourhood from the negatives
    na, nb = F.normalize(za, dim=1), F.normalize(zb, dim=1)
    logits = na @ nb.T / args.tau
    near = (pos[:, None, :] - pos[None, :, :]).abs().max(-1).values <= 1
    near.fill_diagonal_(False)
    logits = logits.masked_fill(near, -1e4)
    tgt = torch.arange(len(za), device=dev)
    nce = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
    inv = (za - zb).pow(2).sum(1).mean()
    perm = torch.randperm(len(za), device=dev)
    spread = (za - za[perm]).pow(2).sum(1).mean()
    loss = nce + args.ratio * inv / (spread + 1e-6)
    cfa = torch.zeros((), device=dev)
    if args.cfa and mem is not None:
        d = torch.cdist(za, mem)  # nearest adapted normal token
        cfa = d.min(1).values.pow(2).mean() / (spread.detach() + 1e-6)
        loss = loss + args.cfa * cfa
    return loss, nce.item(), (inv / spread).item(), cfa.item()


mems = [None for _ in args.layers]
hist = []
t0 = time.perf_counter()
for step in range(args.steps):
    if args.cfa and step % 100 == 0:  # refresh the memory of adapted normal tokens
        with torch.no_grad():
            bufs = [[] for _ in args.layers]
            for _ in range(8):
                for li, (fa, _, _) in enumerate(pairs()):
                    bufs[li].append(proj[li](fa))
            mems = [torch.cat(b) for b in bufs]
    tot, logs = 0, []
    for _ in range(args.batch):
        for li, (fa, fb, pos) in enumerate(pairs()):
            l, *lg = loss_layer(proj[li], fa, fb, pos, mems[li])
            tot = tot + l / args.batch
            logs.append(lg)
    opt.zero_grad()
    tot.backward()
    opt.step()
    sched.step()
    lg = np.mean(logs, 0)
    hist.append([tot.item(), *lg])
    if step % 50 == 0 or step == args.steps - 1:
        h = np.mean(hist[-50:], 0)
        print(f"step {step} loss {h[0]:.3f} nce {h[1]:.3f} inv/spread {h[2]:.4f} cfa {h[3]:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
out = Path("results/smoke" if args.smoke else "results")
out.mkdir(parents=True, exist_ok=True)
torch.save({"layers": args.layers, "hidden": args.hidden, "state": proj.state_dict()}, out / f"{args.name}.pt")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
h = np.array(hist)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.3))
for j, (t, c) in enumerate([("total loss", 0), ("dense InfoNCE", 1), ("invariance / spread", 2)]):
    v = np.convolve(h[:, c], np.ones(20) / 20, mode="valid") if len(h) >= 20 else h[:, c]
    ax[j].plot(v)
    ax[j].set_title(t)
    ax[j].set_xlabel("step")
fig.suptitle(f"{args.name} training (normal photos only)")
fig.tight_layout()
fig.savefig(f"{out if args.smoke else 'docs/figures'}/{args.name}_loss.png", dpi=100)
print("saved", out / f"{args.name}.pt", flush=True)
