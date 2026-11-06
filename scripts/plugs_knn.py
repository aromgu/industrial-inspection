"""Instance-level anomaly maps for wallplugs: compare each plug, position by position, with normal plugs.

Steps (industrial_inspection.plugs):
  1. SAM segments every plug (cached per image under results/plugs_cache/).
  2. Each plug is cut out of the full-resolution photo, rotated so its long axis is horizontal, scaled
     the same for every plug, and everything outside its mask is blacked out (canonical crop 448x126).
  3. Memory bank: the crops of n_ref training photos, each also rotated 180 deg and flipped both ways
     (head left/right, which side faces the camera), as DINOv2-L token grids (9 x 32).
  4. A query token is compared only with bank tokens at the same place on the plug (+-1 token), and
     scored by its mean distance to the knn nearest of them (k-NN averaging helped HiResKNN on
     wallplugs). The token map is pasted back into the photo under the plug's mask.
  5. Pixels no plug covers (background, plugs SAM missed) take the whole-image HiResKNN map (--fallback);
     both maps are z-scored by their own validation statistics before they are joined.

Writes results/maps/<name>__<cat>.npz (+ __val) for seg_eval.py: <name> is the fused map, <name>_only
leaves uncovered pixels at the normal level, so the instance part can be judged on its own.

    uv run python scripts/plugs_knn.py --smoke --game-mode
    uv run python scripts/plugs_knn.py --game-mode
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from scipy import ndimage

from industrial_inspection.plugs import CROP, canonical_crop, load_plugs, load_sam, read_rgb, save_plugs, segment_plugs

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="wallplugs")
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--name", default="plugs_k50")
    p.add_argument("--fallback", default="hires_k50", help="whole-image maps used where no plug is found")
    p.add_argument("--n-ref", type=int, default=16)
    p.add_argument("--knn", type=int, default=50)
    p.add_argument("--layers", type=int, nargs="+", default=[17, 23])
    p.add_argument("--work", type=int, default=1024, help="long side for SAM")
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--sam", default="results/weights/sam_vit_b_01ec64.pth")
    p.add_argument("--no-mask", action="store_true", help="keep the neighbouring plugs in each crop instead of blacking them out")
    p.add_argument("--pos-tol", type=int, default=1,
                   help="compare a token with bank tokens within this many tokens of the same place (large = anywhere)")
    p.add_argument("--paste", choices=["mask", "rect"], default="mask",
                   help="paste each plug's score back under its mask, or over the whole crop rectangle (also covers "
                        "broken tips and pixels SAM left out of the mask)")
    p.add_argument("--split", default="test_public",
                   help="test_public (has labels/ground truth) or a flat private split (test_private, "
                        "test_private_mixed): reads the fallback maps already exported for that split")
    p.add_argument("--cache-only", action="store_true",
                   help="only run SAM on the validation and test photos and cache the plug masks (no scoring); "
                        "the masks do not depend on the backbone, so any later scoring run reuses them")
    p.add_argument("--game-mode", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    dev = "cuda"
    duty = 1.0
    if args.game_mode:
        sys.path.insert(0, str(Path(__file__).parent))
        from score import lower_cpu_priority

        lower_cpu_priority()
        torch.cuda.set_per_process_memory_fraction(0.6, 0)
        duty = 0.6
    out = Path("results/smoke") if args.smoke else Path("results")
    private = args.split != "test_public"
    tag = f"__{args.split}" if private else ""

    def throttle(t0: float) -> None:
        if duty < 1:
            torch.cuda.synchronize()
            time.sleep((time.perf_counter() - t0) * (1 / duty - 1))

    root = args.root / args.category
    fb = np.load(f"results/maps/{args.fallback}__{args.category}{tag}.npz")
    fbv = np.load(f"results/maps/{args.fallback}__{args.category}__val.npz")
    test = [str(q) for q in fb["paths"]]
    labels = fb["labels"] if not private else np.zeros(len(test), dtype=int)  # private: no ground truth locally
    val = [str(q) for q in fbv["paths"]]
    train_all = sorted(str(q) for q in (root / "train" / "good").glob("*.png"))
    refs = [train_all[i] for i in np.linspace(0, len(train_all) - 1, args.n_ref).round().astype(int)]
    ti = np.arange(len(test))
    vi = np.arange(len(val))
    if args.smoke:
        refs, ti, vi = refs[:2], np.r_[0:2, len(test) - 2:len(test)], vi[:2]

    # 1. segmentation (cached)
    gen = None
    cache = Path("results/plugs_cache") / args.category

    def plugs_of(path: str):
        nonlocal gen
        f = cache / Path(path).parent.parent.name / Path(path).parent.name / (Path(path).stem + ".npz")
        if f.exists():
            return load_plugs(f)
        if gen is None:
            gen = load_sam(args.sam, device=dev)
        work = read_rgb(path, args.work)
        t0 = time.perf_counter()
        pl = segment_plugs(gen, work, plug_area=3287 * (args.work / 612) ** 2, area_tol=(0.25, 1.6), min_elongation=1.5)
        throttle(t0)
        f.parent.mkdir(parents=True, exist_ok=True)
        save_plugs(f, pl, work.shape[:2])
        return pl

    t_start = time.perf_counter()
    ref_plugs = [plugs_of(q) for q in refs]
    whole = [pl.size[0] for pls in ref_plugs for pl in pls if pl.size[0] / max(pl.size[1], 1) >= 3]
    plug_length = float(np.median(whole))
    print(f"typical plug length {plug_length:.0f}px at work res, {sum(map(len, ref_plugs))} plugs in {len(refs)} refs",
          flush=True)

    if args.cache_only:
        todo = [val[i] for i in vi] + [test[i] for i in ti]
        for n, q in enumerate(todo):
            plugs_of(q)
            if n % 25 == 0:
                print(f"sam {n}/{len(todo)} ({time.perf_counter() - t_start:.0f}s)", flush=True)
        print(f"cached plug masks for {len(todo)} photos, {time.perf_counter() - t_start:.0f}s", flush=True)
        return

    vit = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, dynamic_img_size=True).eval().to(dev)

    @torch.no_grad()
    def feats(crops: list[np.ndarray]) -> list[torch.Tensor]:
        """Per layer (n, 9, 32, C) token grids of canonical crops."""
        x = torch.from_numpy(((np.stack(crops) / 255.0 - MEAN) / STD).astype(np.float32)).permute(0, 3, 1, 2)
        res = [[] for _ in args.layers]
        for i in range(0, len(x), 16):
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                o = vit.forward_intermediates(x[i:i + 16].to(dev), indices=args.layers, norm=False, output_fmt="NCHW",
                                              intermediates_only=True)
            for r, oo in zip(res, o):
                r.append(oo.permute(0, 2, 3, 1).half())
            throttle(t0)
        return [torch.cat(r) for r in res]

    def crops_of(path: str, pls):
        full = np.asarray(Image.open(path).convert("RGB"))
        s = full.shape[1] / pls[0].mask.shape[1] if pls else 1.0
        return full, s, [canonical_crop(full, q, s, plug_length, mask_dilate=0 if args.no_mask else 3, use_mask=not args.no_mask)
                         for q in pls]

    # 2-3. memory bank: every ref plug in 4 orientations
    bank_crops = []
    for path, pls in zip(refs, ref_plugs):
        if not pls:
            continue
        _, _, cs = crops_of(path, pls)
        for c, _ in cs:
            bank_crops += [c, c[::-1, ::-1], c[::-1], c[:, ::-1]]
    bank = [b for b in feats([np.ascontiguousarray(c) for c in bank_crops])]  # per layer (N, 9, 32, C)
    gh, gw = bank[0].shape[1:3]
    print(f"bank: {len(bank_crops)} crops, token grid {gh}x{gw}", flush=True)

    @torch.no_grad()
    def score_plugs(q: list[torch.Tensor]) -> np.ndarray:
        """(n, 9, 32) mean over layers of the knn-mean distance to bank tokens at the same place (+-1)."""
        n = q[0].shape[0]
        total = torch.zeros(n, gh, gw, device=dev)
        if args.pos_tol >= max(gh, gw):  # position-free: every query token against every bank token
            for ql, bl in zip(q, bank):
                qf, bf = ql.reshape(-1, ql.shape[-1]).float(), bl.reshape(-1, bl.shape[-1])
                k = min(args.knn, len(bf))
                top = torch.full((len(qf), k), float("inf"), device=dev)
                for bi in range(0, len(bf), 32768):
                    d = torch.cdist(qf, bf[bi:bi + 32768].float())
                    top = torch.cat([top, d.topk(min(k, d.shape[1]), dim=1, largest=False).values], 1).topk(k, dim=1, largest=False).values
                total += top.mean(1).reshape(n, gh, gw)
            return (total / len(bank)).cpu().numpy()
        for ql, bl in zip(q, bank):
            for r in range(gh):
                for c in range(gw):
                    t = args.pos_tol
                    cand = bl[:, max(0, r - t):r + t + 1, max(0, c - t):c + t + 1].reshape(-1, bl.shape[-1]).float()
                    d = torch.cdist(ql[:, r, c].float(), cand)
                    k = min(args.knn, d.shape[1])
                    total[:, r, c] += d.topk(k, dim=1, largest=False).values.mean(dim=1)
        return (total / len(bank)).cpu().numpy()

    def instance_map(path: str, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """(map at eval size with NaN where no plug, coverage mask)."""
        pls = plugs_of(path)
        canvas = np.full(hw, -np.inf, np.float32)
        if not pls:
            return np.where(np.isinf(canvas), np.nan, canvas), np.zeros(hw, bool)
        full, s, cs = crops_of(path, pls)
        sc = score_plugs(feats([c for c, _ in cs]))
        ev = full.shape[1] / hw[1]  # eval pixel -> full-res pixel
        for (c, M), grid, pl in zip(cs, sc, pls):
            up = cv2.resize(grid.astype(np.float32), CROP, interpolation=cv2.INTER_LINEAR)
            Me = M.copy()
            Me[:, :2] = M[:, :2] * ev
            back = cv2.warpAffine(up, cv2.invertAffineTransform(Me), hw[::-1], flags=cv2.INTER_LINEAR,
                                  borderValue=-np.inf)
            if args.paste == "rect":
                m = np.isfinite(back)
            else:
                m = cv2.resize(pl.mask.astype(np.uint8), hw[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            canvas = np.where(m, np.maximum(canvas, back), canvas)
        cov = np.isfinite(canvas)
        return np.where(cov, canvas, np.nan), cov

    hw = fb["maps"].shape[1:]
    vmaps = [instance_map(val[i], hw) for i in vi]
    print(f"validation done ({time.perf_counter() - t_start:.0f}s)", flush=True)
    tmaps = []
    for n, i in enumerate(ti):
        tmaps.append(instance_map(test[i], hw))
        if n % 25 == 0:
            print(f"test {n}/{len(ti)} ({time.perf_counter() - t_start:.0f}s)", flush=True)

    # 5. fuse with the whole-image map, each z-scored by its own validation statistics
    iv = np.concatenate([m[c] for m, c in vmaps])
    gv = fbv["maps"][vi].astype(np.float32).ravel()
    mi, si, mg, sg = iv.mean(), iv.std() + 1e-6, gv.mean(), gv.std() + 1e-6
    coverage = np.mean([c.mean() for _, c in tmaps])
    print(f"plug coverage of test pixels {coverage:.1%}", flush=True)

    def fuse(inst, cov, glob, only: bool) -> np.ndarray:
        z_g = (glob.astype(np.float32) - mg) / sg
        z_i = (np.nan_to_num(inst, nan=0.0) - mi) / si
        base = np.full_like(z_g, (iv.min() - mi) / si) if only else z_g
        m = np.where(cov, z_i, base)
        return ndimage.gaussian_filter(m, args.sigma) if args.sigma > 0 else m

    (out / "maps").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    for name, only in ((args.name, False), (args.name + "_only", True)):
        tm = np.stack([fuse(m, c, fb["maps"][i], only) for (m, c), i in zip(tmaps, ti)]).astype(np.float16)
        vm = np.stack([fuse(m, c, fbv["maps"][i], only) for (m, c), i in zip(vmaps, vi)]).astype(np.float16)
        paths_t = np.array([test[i] for i in ti])
        np.savez_compressed(out / "maps" / f"{name}__{args.category}{tag}.npz", maps=tm, labels=labels[ti], paths=paths_t)
        if not private:  # same validation set regardless of test split; don't rewrite it for private runs
            np.savez_compressed(out / "maps" / f"{name}__{args.category}__val.npz", maps=vm, paths=np.array([val[i] for i in vi]))
        pd.DataFrame({"image_path": paths_t, "label": labels[ti], "score": tm.reshape(len(ti), -1).max(1).astype(float)}).to_csv(
            out / "scores" / f"{name}__{args.category}{tag}__seed0.csv", index=False)
    print(f"{args.category}: {len(ti)} test, {len(vi)} val, {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
