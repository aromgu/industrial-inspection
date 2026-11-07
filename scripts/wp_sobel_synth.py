"""P3: Sobel re-ranker trained WITHOUT real defects. Positives = synthetic small defects (scratch lines, dents,
cut-paste patches) drawn at random plug pixels of normal training photos (original resolution); negatives = the
same windows without a defect + the stage-1 candidate peaks of the normal VALIDATION photos (hard normal peaks).
Same window features as wp_sobel.py. The classifier is then applied to the TEST candidates (labels only used to
report), combined with the stage-1 score as z(stage 1) + w * z(logit).
    .venv/Scripts/python.exe scripts/wp_sobel_synth.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = 32
rng = np.random.default_rng(0)


def grads(im):
    gx, gy = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy), np.arctan2(gy, gx), np.abs(cv2.Laplacian(im, cv2.CV_32F, ksize=3))


def stats(m, t):
    w = m.ravel() + 1e-6
    c, s = (w * np.cos(2 * t.ravel())).sum() / w.sum(), (w * np.sin(2 * t.ravel())).sum() / w.sum()
    return [m.mean(), np.percentile(m, 95), m.max(), (m > 40).mean(), np.hypot(c, s)]


def feat(im, mag, ang, lap, cy, cx):
    inner = (slice(max(0, cy - W), cy + W), slice(max(0, cx - W), cx + W))
    outer = (slice(max(0, cy - 3 * W), cy + 3 * W), slice(max(0, cx - 3 * W), cx + 3 * W))
    fi, fo = stats(mag[inner], ang[inner]), stats(mag[outer], ang[outer])
    li, lo = lap[inner].mean(), lap[outer].mean()
    return fi + [li] + [a / (b + 1e-6) for a, b in zip(fi[:4] + [li], fo[:4] + [lo])] + [im[inner].mean(), im[inner].std()]


def add_defect(im, cy, cx):
    out = im.copy()
    kind = rng.integers(3)
    r = int(rng.integers(8, 30))
    if kind == 0:  # scratch
        ang, L = rng.uniform(0, np.pi), int(rng.integers(20, 70))
        p0 = (int(cx - L * np.cos(ang) / 2), int(cy - L * np.sin(ang) / 2))
        p1 = (int(cx + L * np.cos(ang) / 2), int(cy + L * np.sin(ang) / 2))
        m = np.zeros(im.shape, np.uint8)
        cv2.line(m, p0, p1, 1, int(rng.integers(2, 6)))
        out[m.astype(bool)] *= float(rng.choice([0.55, 0.75, 1.3, 1.5]))
    elif kind == 1:  # dent / bump
        m = np.zeros(im.shape, np.float32)
        cv2.ellipse(m, (cx, cy), (r, max(3, int(r * rng.uniform(0.4, 1)))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        out *= 1 + cv2.GaussianBlur(m, (0, 0), r / 3) * float(rng.choice([-1, 1])) * rng.uniform(0.25, 0.5)
    else:  # cut-paste of a rotated patch from elsewhere
        sy, sx = int(rng.integers(r, im.shape[0] - r)), int(rng.integers(r, im.shape[1] - r))
        patch = cv2.warpAffine(np.ascontiguousarray(im[sy - r:sy + r, sx - r:sx + r]), cv2.getRotationMatrix2D((r, r), float(rng.uniform(0, 360)), 1.0), (2 * r, 2 * r))
        m = np.zeros((2 * r, 2 * r), np.uint8)
        cv2.ellipse(m, (r, r), (r, max(3, int(r * rng.uniform(0.3, 1)))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        sub = out[cy - r:cy + r, cx - r:cx + r]
        if sub.shape == patch.shape:
            sub[m.astype(bool)] = patch[m.astype(bool)]
    return np.clip(out, 0, 255)


root = Path("datasets/MVTec_AD_2/wallplugs")


def synthetic_set():
    """(features, labels) of synthetic-defect windows and the same windows clean, from normal training photos."""
    train = sorted(str(q) for q in (root / "train/good").glob("*.png"))
    train = [q for i, q in enumerate(train) if i % 8][::4]  # ~37 photos, never the bank's held-out ones
    X, Y = [], []
    for p in train:
        im = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        ys, xs = np.nonzero(im[::4, ::4] > np.quantile(im, 0.7))
        ys, xs = ys * 4, xs * 4
        ok = (ys > 4 * W) & (ys < im.shape[0] - 4 * W) & (xs > 4 * W) & (xs < im.shape[1] - 4 * W)
        sel = rng.choice(np.flatnonzero(ok), 12, replace=False)
        base = grads(im)
        for i in sel:
            cy, cx = int(ys[i]), int(xs[i])
            X.append(feat(im, *base, cy, cx)); Y.append(0)
            lo = (slice(cy - 4 * W, cy + 4 * W), slice(cx - 4 * W, cx + 4 * W))
            win = add_defect(im[lo].copy(), 4 * W, 4 * W)
            X.append(feat(win, *grads(win), 4 * W, 4 * W)); Y.append(1)
    return np.array(X), np.array(Y)


def main():
    X, Y = synthetic_set()
    df = pd.read_csv("results/seg/wp_zoom_z2.csv")

    df["part"] = df.path.str.replace("\\", "/").str.extract(r"/(?:good|bad)/(\d+)_")[0]
    cache = {}

    def cand_feats(rows):
        out = []
        for r in rows.itertuples():
            if r.path not in cache:
                cache.clear()
                im = np.asarray(Image.open(r.path).convert("L"), dtype=np.float32)
                cache[r.path] = (im, *grads(im))
            im = cache[r.path][0]
            cy, cx = int((r.y + 0.5) * im.shape[0] / 512), int((r.x + 0.5) * im.shape[1] / 612)
            out.append(feat(*cache[r.path], cy, cx))
        return np.array(out)


    val = df[df.split == "val"]
    Fv = cand_feats(val)
    X = np.r_[np.array(X), Fv]
    Y = np.r_[np.array(Y), np.zeros(len(Fv), int)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X, Y)
    print(f"trained on {int(Y.sum())} synthetic defects, {int((Y == 0).sum())} normal windows ({len(Fv)} validation candidate peaks)")
    test = df[df.split == "test"].copy()
    Ft = cand_feats(test)
    lt, lv = clf.decision_function(Ft), clf.decision_function(Fv)
    s1v = val.s1.values
    for wgt in (0.0, 0.25, 0.5, 1.0):
        s = (test.s1.values - s1v.mean()) / s1v.std() + wgt * (lt - lv.mean()) / lv.std()
        test["s"] = s
        top = test[test.label == 1].sort_values("s", ascending=False).groupby("path").head(1)
        hard = top[top.part.isin(["002", "004", "005", "007"])]
        print(f"  stage-1 + {wgt} x synthetic-trained Sobel: AUROC {roc_auc_score(test.defect, s):.3f} | top is defect {top.defect.mean()*100:.0f}%, hard {int(hard.defect.sum())}/{len(hard)}")
    print(f"  synthetic-trained Sobel alone: AUROC {roc_auc_score(test.defect, lt):.3f}")


if __name__ == "__main__":
    main()
