"""wallplugs transductive mutual scoring (MuSc-style, Li et al. ICLR 2024): the unlabeled TEST photos are each other's
normal reference. A shift-light photo's normal plug looks like the plugs of the other shift-light test photos, while a
defect is rare across the test set.

Per photo and layer (DINOv3-H+, layers 15/23, 1792 px, the final model's tiling): for every token, the distance to the
nearest token of each OTHER test photo (references: plug tokens = Otsu foreground dilated, randomly thinned to --cap, plus --bg random background
tokens). The score of a token = the k-th smallest of those per-photo distances (k > 5: a part's own defect also appears
on its 5 other condition photos, so k must exceed that or the defect is matched to itself). Validation photos are
scored against the same test references (thresholds come from them, as in the final rule).

Writes results/maps/musc_k<K>__wallplugs.npz / __val.npz (eval size, layer mean, same layout as scripts/hires.py) for
every K in --ks, plus musc_xp_k<K> (diagnostic: the photo's own part excluded from its references, test only knowable
on public data). Then evaluate with scripts/wp_eval.py (--model musc_k10, or --model s0_r1792_L1523 --model2 musc_k10).
    .venv/Scripts/python.exe scripts/wp_musc.py --game-mode [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from hires import to_eval  # noqa: E402
from industrial_inspection.hires import HiResKNN  # noqa: E402
from industrial_inspection.q2 import parts_and_tags  # noqa: E402
from industrial_inspection.segeval import DOWNSCALE, eval_size  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--ks", type=int, nargs="+", default=[1, 3, 6, 10, 20, 40])
a.add_argument("--bg", type=int, default=1000, help="random background tokens per reference photo")
a.add_argument("--cap", type=int, default=6000, help="reference tokens per photo (plug tokens randomly thinned)")
a.add_argument("--name", default="musc")
a.add_argument("--game-mode", action="store_true")
a.add_argument("--smoke", action="store_true")
args = a.parse_args()
duty = 1.0
if args.game_mode:
    from score import lower_cpu_priority
    lower_cpu_priority()
    torch.cuda.set_per_process_memory_fraction(0.6, 0)
    duty = 0.6
dev = "cuda"
model = HiResKNN(backbone="dinov3_vith16plus", layers=(15, 23), tile=640, overlap=128, short_side=1792, gpu_duty=duty)
root = Path("datasets/MVTec_AD_2/wallplugs")
test = sorted(str(q) for q in (root / "test_public" / "good").glob("*.png")) + \
    sorted(str(q) for q in (root / "test_public" / "bad").glob("*.png"))
labels = np.array([0 if "/good/" in q.replace("\\", "/") else 1 for q in test])
val = sorted(str(q) for q in (root / "validation" / "good").glob("*.png"))
if args.smoke:
    keep = list(range(0, len(test), 15)) + list(range(len(test) - 12, len(test)))
    test, labels, val = [test[i] for i in keep], labels[keep], val[:2]
parts, _ = parts_and_tags(np.array(test))
rng = np.random.default_rng(0)
t0 = time.perf_counter()


def fg_tokens(path, gh, gw):
    g = np.asarray(Image.open(path).convert("L").resize((gw * 4, gh * 4), Image.BILINEAR))
    _, b = cv2.threshold(cv2.GaussianBlur(g, (0, 0), 2), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.resize(b, (gw, gh), interpolation=cv2.INTER_AREA) > 0
    return cv2.dilate(b.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool).ravel()


# pass 1: reference tokens of every test photo, fp16 in pinned CPU memory (copied to the GPU per comparison)
refs = [[] for _ in model.layers]
for n, p in enumerate(test):
    fs = model.features(model.load(p))
    gh, gw = fs[0].shape[1:]
    m = fg_tokens(p, gh, gw)
    bg = np.flatnonzero(~m)
    fgi = np.flatnonzero(m)
    fgi = rng.choice(fgi, min(args.cap - args.bg, len(fgi)), replace=False)
    idx = np.concatenate([fgi, rng.choice(bg, min(args.bg, len(bg)), replace=False)])
    idx_t = torch.from_numpy(idx).to(dev)
    for r, f in zip(refs, fs):
        r.append(f.flatten(1).T[idx_t].half().cpu().pin_memory())
    if n % 25 == 0:
        print(f"refs {n}/{len(test)}: {len(idx)} tokens ({m.mean()*100:.0f}% plug), {time.perf_counter()-t0:.0f}s", flush=True)
print(f"reference tokens per photo: median {int(np.median([len(r) for r in refs[0]]))}, "
      f"GPU {torch.cuda.memory_allocated()/2**30:.1f} GB, {time.perf_counter()-t0:.0f}s", flush=True)


@torch.no_grad()
def per_ref_dist(path):
    """(n_test, gh*gw) per layer-mean: distance of each token to its nearest token in each test photo."""
    fs = model.features(model.load(path))
    gh, gw = fs[0].shape[1:]
    D = torch.zeros(len(test), gh * gw, device=dev)
    for f, rl in zip(fs, refs):
        q = f.flatten(1).T.float()
        for j, r in enumerate(rl):
            t1 = time.perf_counter()
            r = r.to(dev, non_blocking=True).float()
            best = torch.empty(len(q), device=dev)
            for qi in range(0, len(q), 4096):  # fp32 like HiResKNN._dist (an fp16 matmul overflows: token norms are large)
                best[qi:qi + 4096] = torch.cdist(q[qi:qi + 4096], r).min(1).values
            D[j] += best / len(fs)
            model._throttle(t1)
    return D, (gh, gw)


hw = eval_size(test[0], DOWNSCALE)
ks = [k for k in args.ks if k < len(test) - 1]
out = {f"{args.name}_k{k}": ([], []) for k in ks}
out.update({f"{args.name}_xp_k{k}": ([], None) for k in ks})
for split, paths in (("test", test), ("val", val)):
    for n, p in enumerate(paths):
        D, (gh, gw) = per_ref_dist(p)
        if split == "test":
            i = test.index(p)
            D_self = D.clone(); D_self[i] = float("inf")
            D_xp = D.clone(); D_xp[torch.from_numpy(parts == parts[i]).to(dev)] = float("inf")
        else:
            D_self, D_xp = D, None
        Ds = D_self.sort(0).values
        Dx = D_xp.sort(0).values if D_xp is not None else None
        for k in ks:
            out[f"{args.name}_k{k}"][0 if split == "test" else 1].append(
                to_eval(Ds[k - 1].reshape(gh, gw).cpu().numpy(), hw, 0).astype(np.float16))
            if Dx is not None:
                out[f"{args.name}_xp_k{k}"][0].append(to_eval(Dx[k - 1].reshape(gh, gw).cpu().numpy(), hw, 0).astype(np.float16))
        if n % 10 == 0:
            print(f"{split} {n}/{len(paths)} {time.perf_counter()-t0:.0f}s", flush=True)

dst = Path("results/smoke/maps" if args.smoke else "results/maps")
dst.mkdir(parents=True, exist_ok=True)
for name, (tm, vm) in out.items():
    np.savez_compressed(dst / f"{name}__wallplugs.npz", maps=np.stack(tm), labels=labels, paths=np.array(test))
    np.savez_compressed(dst / f"{name}__wallplugs__val.npz",
                        maps=np.stack(vm if vm is not None else out[name.replace("_xp", "")][1]), paths=np.array(val))
print(f"saved {len(out)} map sets to {dst}, {time.perf_counter()-t0:.0f}s total", flush=True)
