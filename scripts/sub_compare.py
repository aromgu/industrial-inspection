"""Pick the per-category model for the submission on test_public: pooled pixel F1 (all photos, good included,
as the benchmark scores) with thresholds from VALIDATION maps only -- p95 x 1.2 and p95 x 1.421 (SuperADD) --
and the submission post-processing (close 6 + fill + drop < 64 px).
    .venv/Scripts/python.exe scripts/sub_compare.py --models hires_k50 hires_dv3_k1
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from seg_post import POSTS, postprocess  # noqa: E402
from industrial_inspection.segeval import ground_truth  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("--models", nargs="+", default=["hires_k50", "hires_dv3_k1"])
a.add_argument("--categories", nargs="+", default=["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "walnuts"])
args = a.parse_args()
post = POSTS["close6+fill+small64"]
print("category      model            p95x1.2  p95x1.421")
for cat in args.categories:
    gt = None
    for m in args.models:
        f = Path(f"results/maps/{m}__{cat}.npz")
        if not f.exists():
            print(f"{cat:12s}  {m:15s}  (no map)")
            continue
        z = np.load(f)
        maps, paths, y = z["maps"].astype(np.float32), z["paths"].astype(str), z["labels"]
        v = np.load(f"results/maps/{m}__{cat}__val.npz")["maps"].astype(np.float32)
        hw = maps.shape[1:]
        gt = np.stack([ground_truth(p, hw) if l else np.zeros(hw, bool) for p, l in zip(paths, y)])
        p95 = np.quantile(v, 0.95)
        out = []
        for k in (1.2, 1.421):
            pred = np.stack([postprocess(mm > p95 * k, *post) for mm in maps])
            tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
            out.append(200 * tp / max(1, 2 * tp + fp + fn))
        print(f"{cat:12s}  {m:15s}  {out[0]:7.1f}  {out[1]:9.1f}", flush=True)
