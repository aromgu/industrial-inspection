"""wallplugs: threshold rule + post-processing chosen on the DEV half of the physical parts of test_public
(q2.split_parts, seed 2026), reported on the EVAL half. Thresholds come from validation (good) maps only;
the dev half only picks WHICH rule. Metrics on each half:
  f1_def   pooled pixel F1 over defective photos (SuperADD's protocol)
  f1_all   pooled pixel F1 over all photos (good photos add false alarms; closer to the benchmark)
  det      share of defective photos with per-photo F1 > 0.1
  pf1      mean per-photo F1 over defective photos
    .venv/Scripts/python.exe scripts/wp_tune.py --models hires_dv3_k1 hires_dv3_k50
Writes results/seg/wp_tune/<model>.csv and docs/figures/wp_tune.png"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from seg_post import postprocess  # noqa: E402
from industrial_inspection.q2 import parts_and_tags, split_parts  # noqa: E402
from industrial_inspection.segeval import eval_size, ground_truth  # noqa: E402

POSTS = {"none": (0, False, 0), "fill": (0, True, 0), "fill+s64": (0, True, 64), "fill+s256": (0, True, 256),
         "c3+fill": (3, True, 0), "c6+fill+s64": (6, True, 64)}


def rules(v: np.ndarray) -> dict[str, float]:
    flat = v.reshape(-1); mx = v.reshape(len(v), -1).max(1)
    r = {f"q{q}": float(np.quantile(flat, q)) for q in (0.99, 0.995, 0.999, 0.9995, 0.9999)}
    r.update({f"p95x{g}": float(np.quantile(flat, 0.95) * g) for g in (1.2, 1.3, 1.421, 1.5, 1.6, 1.8)})
    r.update({f"vmax_mean x{f}": float(mx.mean() * f) for f in (0.6, 0.7, 0.8, 0.9, 1.0)})
    r.update({f"vmax_q{int(q*100)}": float(np.quantile(mx, q)) for q in (0.25, 0.5, 0.75)})
    return r


def stats(pred, gt, y):
    tp = (pred & gt).sum((1, 2)); fp = (pred & ~gt).sum((1, 2)); fn = (~pred & gt).sum((1, 2))
    d = y == 1
    f = lambda a, b, c: 2 * a.sum() / max(1, 2 * a.sum() + b.sum() + c.sum())
    pf = 2 * tp[d] / np.maximum(1, 2 * tp[d] + fp[d] + fn[d])
    return dict(f1_def=f(tp[d], fp[d], fn[d]) * 100, f1_all=f(tp, fp, fn) * 100, det=(pf > 0.1).mean() * 100, pf1=pf.mean() * 100)


def main():
    a = argparse.ArgumentParser(); a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--select", default="f1_all"); a.add_argument("--posts", nargs="+", default=list(POSTS))
    a.add_argument("--fig", default="docs/figures/wp_tune.png"); args = a.parse_args()
    out = Path("results/seg/wp_tune"); out.mkdir(parents=True, exist_ok=True)
    best = []
    for model in args.models:
        z = np.load(f"results/maps/{model}__wallplugs.npz")
        v = np.load(f"results/maps/{model}__wallplugs__val.npz")["maps"].astype(np.float32)
        paths, y = z["paths"].astype(str), z["labels"]; maps = z["maps"].astype(np.float32)
        hw = maps.shape[1:]
        gt = np.stack([ground_truth(p, hw) if l else np.zeros(hw, bool) for p, l in zip(paths, y)])
        parts, _ = parts_and_tags(paths); dev = split_parts(parts, y)
        rows = []
        for rn, t in rules(v).items():
            raw = maps > t
            for pn, pp in ((k, POSTS[k]) for k in args.posts):
                pred = np.stack([postprocess(b, *pp) for b in raw]) if any(pp) else raw
                for half, m in (("dev", dev), ("eval", ~dev), ("all", np.ones_like(dev))):
                    rows.append(dict(model=model, rule=rn, post=pn, half=half, thr=t, **stats(pred[m], gt[m], y[m])))
        df = pd.DataFrame(rows); df.to_csv(out / f"{model}.csv", index=False)
        piv = df.pivot_table(index=["rule", "post"], columns="half", values=["f1_def", "f1_all", "det", "pf1"])
        pick = piv[(args.select, "dev")].idxmax()
        print(f"\n== {model}: chosen on dev by {args.select}: {pick}")
        print(piv.loc[[pick]].round(1).T.unstack().to_string())
        top = piv.sort_values((args.select, "dev"), ascending=False).head(8)
        print("top on dev:\n", top[[(args.select, "dev"), (args.select, "eval"), ("det", "eval"), ("f1_def", "all")]].round(1).to_string())
        best.append((model, pick, piv.loc[pick]))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ms = ["f1_def", "f1_all", "det", "pf1"]
    for i, (m, pick, s) in enumerate(best):
        ax.bar(np.arange(4) + i * 0.8 / len(best), [s[(k, "eval")] for k in ms], 0.8 / len(best), label=f"{m} {pick}")
    ax.set_xticks(np.arange(4) + 0.4 - 0.4 / len(best), ["pooled F1 (defect)", "pooled F1 (all)", "photos detected %", "mean photo F1"])
    ax.set_title("wallplugs, rule chosen on dev half, scored on eval half"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.fig, dpi=120)


if __name__ == "__main__":
    main()
