"""wallplugs: local-background normalisation of saved anomaly maps (test and validation alike), written as
new models <model>_<variant> for wp_tune.py.
  th<s>   white top-hat: m - grey_opening(m, s)   (keeps peaks narrower than s px)
  md<s>   m - median_filter(m, s)
  hyb<s>  0.5 * m + top-hat (keeps some absolute level)
  iq<s>   m - the photo's own s-th percentile (per-photo offset, against whole-photo shifts like 'shift_*')
  iz<s>   (m - photo median) / (photo q90 - median)
    .venv/Scripts/python.exe scripts/wp_localnorm.py --model hires_dv3_k1 --variants th31 th61 md61 hyb31"""
import argparse
import numpy as np
from scipy import ndimage


def apply(m: np.ndarray, v: str) -> np.ndarray:
    if "+" in v:  # chained, left to right: md61+iz10
        for part in v.split("+"):
            m = apply(m, part)
        return m
    kind = v.rstrip("0123456789."); s = float(v[len(kind):])
    if kind == "th":
        return m - ndimage.grey_opening(m, size=(int(s), int(s)))
    if kind == "md":
        return m - ndimage.median_filter(m, size=(int(s), int(s)))
    if kind == "hyb":
        return 0.5 * m + (m - ndimage.grey_opening(m, size=(int(s), int(s))))
    if kind == "iq":  # per-photo offset: subtract the photo's own s-th percentile (s=50: median)
        return m - np.percentile(m, s)
    if kind == "iz":  # per-photo z-score above the median, robust spread (q90 - q50)
        q50, q90 = np.percentile(m, [50, 90])
        return (m - q50) / (q90 - q50 + 1e-6) * (s / 10)
    raise ValueError(v)


def main():
    a = argparse.ArgumentParser(); a.add_argument("--model", required=True); a.add_argument("--variants", nargs="+", required=True)
    args = a.parse_args()
    z = dict(np.load(f"results/maps/{args.model}__wallplugs.npz")); zv = dict(np.load(f"results/maps/{args.model}__wallplugs__val.npz"))
    for v in args.variants:
        t = np.stack([apply(m.astype(np.float32), v) for m in z["maps"]]).astype(np.float16)
        tv = np.stack([apply(m.astype(np.float32), v) for m in zv["maps"]]).astype(np.float16)
        np.savez_compressed(f"results/maps/{args.model}_{v}__wallplugs.npz", **{**z, "maps": t})
        np.savez_compressed(f"results/maps/{args.model}_{v}__wallplugs__val.npz", **{**zv, "maps": tv})
        print("saved", v, flush=True)


if __name__ == "__main__":
    main()
