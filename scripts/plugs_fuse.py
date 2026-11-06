"""Fuse the plug-crop map (plugs_rect_only by default) with the whole-image map (hires_k50) for
wallplugs -> plugs_rect_max.

The whole-image map is quantile-matched (on validation) to the plug map's distribution where the plug
map is covered, then the two are combined by a per-pixel max (outside plug coverage only the
whole-image map counts). z-score fusion made every variant worse than either input; this is what
worked. Writes results/maps/<out>__wallplugs[__<split>].npz (+ __val, test_public only).

    .venv/Scripts/python.exe scripts/plugs_fuse.py                                   # test_public
    .venv/Scripts/python.exe scripts/plugs_fuse.py --split test_private
    .venv/Scripts/python.exe scripts/plugs_fuse.py --split test_private_mixed
"""

import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--inst", default="plugs_rect_only")
ap.add_argument("--out", default="plugs_rect_max")
ap.add_argument("--fallback", default="hires_k50")
ap.add_argument("--split", default="test_public")
A = ap.parse_args()
tag = f"__{A.split}" if A.split != "test_public" else ""

I = np.load(f"results/maps/{A.inst}__wallplugs{tag}.npz")
G = np.load(f"results/maps/{A.fallback}__wallplugs{tag}.npz")
Iv = np.load(f"results/maps/{A.inst}__wallplugs__val.npz")
Gv = np.load(f"results/maps/{A.fallback}__wallplugs__val.npz")
paths = I["paths"].astype(str)
i, g, iv, gv = [x["maps"].astype(np.float32) for x in (I, G, Iv, Gv)]

# covered = where the plug map rises above its "no plug" floor (works for mask and rect pasting alike)
floor = float(iv.min())
C = i > floor + 1e-3 * (iv.max() - floor)
Cv = iv > floor + 1e-3 * (iv.max() - floor)
qs = np.linspace(0.5, 0.9999, 60)
src, dst = np.quantile(gv, qs), np.quantile(iv[Cv], qs)
match = lambda x: np.interp(x, src, dst)  # noqa: E731
lo = float(iv[Cv].min())

t = np.maximum(np.where(C, i, lo), match(g)).astype(np.float16)
labels = I["labels"] if "labels" in I else np.zeros(len(paths), dtype=int)
np.savez_compressed(f"results/maps/{A.out}__wallplugs{tag}.npz", maps=t, labels=labels, paths=paths)
if not tag:  # validation maps don't depend on the test split
    v = np.maximum(np.where(Cv, iv, lo), match(gv)).astype(np.float16)
    np.savez_compressed(f"results/maps/{A.out}__wallplugs__val.npz", maps=v, paths=Iv["paths"].astype(str))
pd.DataFrame({"image_path": paths, "label": labels, "score": t.reshape(len(t), -1).max(1).astype(float)}).to_csv(
    f"results/scores/{A.out}__wallplugs{tag}__seed0.csv", index=False)
print(f"ok: {len(paths)} images -> results/maps/{A.out}__wallplugs{tag}.npz")
