"""wallplugs two-stage check, stage 2 = a zoomed second look at stage-1 candidate peaks.
  1. candidates: per photo the --top highest local maxima of the stage-1 map (per-photo offset, wp_cascade
     naming <stage1>_iq90 at eval size), at least --sep eval px apart
  2. each candidate: a --win px window of the ORIGINAL photo around it, resized x--zoom, DINOv3 tokens of
     --layers; stage-2 score = max over the central tokens of the distance to the nearest bank token
  3. bank: the same-zoom windows from training photos, centred on plug pixels, random brightness 0.8-1.2,
     random flips; all their tokens
Validation photos get the same treatment (candidate score distribution of normal photos).
Writes results/seg/wp_zoom_<name>.csv: one row per candidate (photo, y, x, stage-1 value, stage-2 score,
is_defect), and prints candidate-level AUROC of stage 1 vs stage 2 and the share of defective photos whose
top candidate is the defect.
    .venv/Scripts/python.exe scripts/wp_zoom.py --stage1 r1792L_s2_iq90 --name z2 [--smoke]"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image, ImageEnhance
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.hires import _DinoV3, MEAN, STD  # noqa: E402
from industrial_inspection.segeval import ground_truth  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--stage1", default="r1792L_s2_iq90"); a.add_argument("--name", default="z2")
a.add_argument("--top", type=int, default=8); a.add_argument("--sep", type=int, default=12)
a.add_argument("--win", type=int, default=192); a.add_argument("--zoom", type=float, default=2.0)
a.add_argument("--layers", type=int, nargs="+", default=[15, 23]); a.add_argument("--center", type=int, default=4, help="central +-tokens scored")
a.add_argument("--bank-photos", type=int, default=40); a.add_argument("--bank-crops", type=int, default=10)
a.add_argument("--smoke", action="store_true"); args = a.parse_args()
dev = "cuda"; rng = np.random.default_rng(0)
vit = _DinoV3("dinov3_vith16plus", "weights/dinov3_vith16plus.pth").to(dev)
side = int(round(args.win * args.zoom / 16)) * 16
mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)


def window(img: Image.Image, cy: float, cx: float, bright: float = 1.0, flip: int = 0) -> np.ndarray:
    h = args.win / 2
    w = img.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h))).resize((side, side), Image.BICUBIC)
    if bright != 1.0:
        w = ImageEnhance.Brightness(w).enhance(bright)
    a_ = np.asarray(w, dtype=np.float32) / 255
    return a_[:, ::-1] if flip == 1 else (a_[::-1] if flip == 2 else a_)


@torch.no_grad()
def tokens(wins: list[np.ndarray]) -> list[torch.Tensor]:
    x = (torch.from_numpy(np.stack(wins).copy()).permute(0, 3, 1, 2) - mean) / std
    out = [[] for _ in args.layers]
    for i in range(0, len(x), 8):
        with torch.autocast("cuda", dtype=torch.float16):
            o = vit.forward_intermediates(x[i:i + 8].to(dev), indices=args.layers)
        for r, oo in zip(out, o):
            r.append(oo.half())
    return [torch.cat(r) for r in out]  # per layer (n, C, g, g)


root = Path("datasets/MVTec_AD_2/wallplugs")
train = sorted(str(p) for p in (root / "train/good").glob("*.png"))
t0 = time.perf_counter()
bank = [[] for _ in args.layers]
photos = rng.choice(train, min(args.bank_photos, len(train)), replace=False) if not args.smoke else train[:3]
for p in photos:
    img = Image.open(p).convert("RGB"); g = np.asarray(img.convert("L"))
    ys, xs = np.nonzero(g[::8, ::8] > np.quantile(g, 0.7)); ys, xs = ys * 8, xs * 8
    ok = (ys > args.win) & (ys < g.shape[0] - args.win) & (xs > args.win) & (xs < g.shape[1] - args.win)
    sel = rng.choice(np.flatnonzero(ok), args.bank_crops if not args.smoke else 2, replace=False)
    wins = [window(img, ys[i], xs[i], rng.uniform(0.8, 1.2), int(rng.integers(0, 3))) for i in sel]
    for b, t in zip(bank, tokens(wins)):
        b.append(t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]))
bank = [torch.cat(b) for b in bank]
print(f"bank {len(bank[0])} tokens/layer from {len(photos)} photos, window {args.win}px x{args.zoom} -> {side}px, {time.perf_counter()-t0:.0f}s", flush=True)


@torch.no_grad()
def stage2(img: Image.Image, cands) -> list[float]:
    wins = [window(img, cy, cx) for cy, cx in cands]
    if not wins:
        return []
    toks = tokens(wins); g = toks[0].shape[-1]; c0, c1 = g // 2 - args.center, g // 2 + args.center
    total = None
    for t, b in zip(toks, bank):
        q = t[:, :, c0:c1, c0:c1].permute(0, 2, 3, 1).reshape(-1, t.shape[1]).float()
        best = torch.full((len(q),), float("inf"), device=dev)
        for i in range(0, len(b), 65536):
            best = torch.minimum(best, torch.cdist(q, b[i:i + 65536].float()).min(1).values)
        s = best.view(len(wins), -1)
        total = s if total is None else total + s
    return (total / len(bank)).max(1).values.cpu().tolist()


def candidates(m: np.ndarray):
    peaks = (m == ndimage.maximum_filter(m, size=args.sep)) & (m > np.quantile(m, 0.99))
    ys, xs = np.nonzero(peaks); order = np.argsort(-m[ys, xs])[: args.top]
    return [(int(ys[i]), int(xs[i]), float(m[ys[i], xs[i]])) for i in order]


rows = []
for split, f in (("test", f"results/maps/{args.stage1}__wallplugs.npz"), ("val", f"results/maps/{args.stage1}__wallplugs__val.npz")):
    z = np.load(f); maps = z["maps"].astype(np.float32); paths = z["paths"].astype(str)
    labels = z["labels"] if "labels" in z else np.zeros(len(paths), int)
    idx = range(len(paths)) if not args.smoke else [0, len(paths) - 1]
    for n, i in enumerate(idx):
        img = Image.open(paths[i]).convert("RGB"); sy, sx = img.height / maps.shape[1], img.width / maps.shape[2]
        cands = candidates(maps[i])
        gt = ndimage.binary_dilation(ground_truth(paths[i], maps.shape[1:]), iterations=3) if labels[i] else None
        s2 = stage2(img, [((y + 0.5) * sy, (x + 0.5) * sx) for y, x, _ in cands])
        for (y, x, v), s in zip(cands, s2):
            rows.append(dict(split=split, path=paths[i], label=int(labels[i]), y=y, x=x, s1=v, s2=s, defect=int(gt is not None and gt[y, x])))
        if n % 25 == 0:
            print(f"{split} {n}/{len(paths)} ({time.perf_counter()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows); out = Path(f"results/seg/wp_zoom_{args.name}{'_smoke' if args.smoke else ''}.csv"); df.to_csv(out, index=False)
from sklearn.metrics import roc_auc_score
t = df[df.split == "test"]
if t.defect.nunique() > 1:
    print(f"candidate AUROC (defect vs not, test): stage1 {roc_auc_score(t.defect, t.s1):.3f}  stage2 {roc_auc_score(t.defect, t.s2):.3f}")
    for k in ("s1", "s2"):
        top = t[t.label == 1].sort_values(k, ascending=False).groupby("path").head(1)
        print(f"  defective photos whose top-{k} candidate is the defect: {top.defect.mean()*100:.0f}%  ;  any candidate on defect: {t[t.label==1].groupby('path').defect.max().mean()*100:.0f}%")
print("saved", out)
