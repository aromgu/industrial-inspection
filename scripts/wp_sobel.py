"""P3 idea (user): look at the high-heat candidates with an edge filter. At every stage-1 candidate peak
(results/seg/wp_zoom_z2.csv: 8 per photo, test + validation, labelled defect / not) hand-crafted gradient features
are measured on the ORIGINAL photo in a window around the peak:
  Sobel magnitude mean / p95 / max, share of strong-edge pixels, orientation coherence (structure tensor),
  Laplacian energy, each also relative to the same statistics in a ring around the window (local contrast).
Then candidate-level AUROC (defect vs not) of: stage-1 score alone, Sobel features alone, both -- logistic
regression with part-grouped cross-validation (a physical part never in train and test fold). Also the share of
defective photos whose top candidate is the defect, and the same for the never-detected parts 002 004 005 007.
    .venv/Scripts/python.exe scripts/wp_sobel.py [--win 24]
"""
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

a = argparse.ArgumentParser(); a.add_argument("--win", type=int, default=24, help="half window, original px")
a.add_argument("--csv", default="results/seg/wp_zoom_z2.csv"); args = a.parse_args()
df = pd.read_csv(args.csv)
df["part"] = df.path.str.replace("\\", "/").str.extract(r"/(?:good|bad)/(\d+)_")[0].fillna("val")
df["grp"] = np.where(df.split == "val", "val_" + df.path.str[-12:], df.label.astype(str) + "_" + df.part)


def stats(g, ang, sl):
    m = g[sl]; t = ang[sl]
    w = m.ravel() + 1e-6
    c, s = (w * np.cos(2 * t.ravel())).sum() / w.sum(), (w * np.sin(2 * t.ravel())).sum() / w.sum()
    return [m.mean(), np.percentile(m, 95), m.max(), (m > 40).mean(), np.hypot(c, s)]


cache = {}
feats = []
for r in df.itertuples():
    if r.path not in cache:
        cache.clear()
        im = np.asarray(Image.open(r.path).convert("L"), dtype=np.float32)
        gx, gy = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
        cache[r.path] = (im, np.hypot(gx, gy), np.arctan2(gy, gx), np.abs(cv2.Laplacian(im, cv2.CV_32F, ksize=3)))
    im, mag, ang, lap = cache[r.path]
    sy, sx = im.shape[0] / 512, im.shape[1] / 612
    cy, cx = int((r.y + 0.5) * sy), int((r.x + 0.5) * sx)
    w, W = args.win, args.win * 3
    inner = (slice(max(0, cy - w), cy + w), slice(max(0, cx - w), cx + w))
    outer = (slice(max(0, cy - W), cy + W), slice(max(0, cx - W), cx + W))
    fi, fo = stats(mag, ang, inner), stats(mag, ang, outer)
    li, lo = lap[inner].mean(), lap[outer].mean()
    feats.append(fi + [li] + [a_ / (b_ + 1e-6) for a_, b_ in zip(fi[:4] + [li], fo[:4] + [lo])] + [im[inner].mean(), im[inner].std()])
F = np.array(feats)
names = ["mag_mean", "mag_p95", "mag_max", "strong", "coherence", "lap", "r_mean", "r_p95", "r_max", "r_strong", "r_lap", "int_mean", "int_std"]
t = df.split == "test"
X, Xs1, y, g = F[t], df.loc[t, ["s1"]].values, df.loc[t, "defect"].values, df.loc[t, "grp"].values


def cv_scores(Xm):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xm, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(Xm[tr], y[tr]); out[te] = clf.predict_proba(Xm[te])[:, 1]
    return out


res = {"stage-1 score": df.loc[t, "s1"].values, "Sobel only (CV)": cv_scores(X), "stage-1 + Sobel (CV)": cv_scores(np.c_[Xs1, X])}
sub = df[t].copy()
print(f"candidates: {t.sum()} test ({int(y.sum())} on a defect); window +-{args.win}px")
for k, s in res.items():
    sub["s"] = s
    top = sub[sub.label == 1].sort_values("s", ascending=False).groupby("path").head(1)
    hard = top[top.part.isin(["002", "004", "005", "007"])]
    print(f"  {k:22s} AUROC {roc_auc_score(y, s):.3f} | top candidate is the defect: all defective photos {top.defect.mean()*100:.0f}%, "
          f"hard parts 002/004/005/007 {hard.defect.mean()*100:.0f}% ({int(hard.defect.sum())}/{len(hard)})")
print("single-feature AUROC:", {n: round(roc_auc_score(y, X[:, i]), 3) for i, n in enumerate(names)})

# ---- unsupervised: no test labels. Gaussian of the Sobel features of VALIDATION candidates (normal photos only);
# a test candidate's score = Mahalanobis distance to it; combined with stage 1 as a sum of validation z-scores.
v = (df.split == "val").values
from sklearn.covariance import LedoitWolf
for cols, nm in ((slice(None), "all Sobel features"), (slice(0, 11), "edge features only (no intensity)")):
    Fv, Ft = F[v][:, cols], F[t][:, cols]
    mu, sd = Fv.mean(0), Fv.std(0) + 1e-6
    lw = LedoitWolf().fit((Fv - mu) / sd)
    mah = lw.mahalanobis((Ft - mu) / sd)
    mv = lw.mahalanobis((Fv - mu) / sd)
    s1v = df.loc[v, "s1"].values
    for wgt in (0.5, 1.0):
        comb = (df.loc[t, "s1"].values - s1v.mean()) / s1v.std() + wgt * (np.sqrt(mah) - np.sqrt(mv).mean()) / np.sqrt(mv).std()
        for k, s in ((f"unsup Sobel ({nm})", np.sqrt(mah)), (f"stage-1 + {wgt} x unsup Sobel ({nm})", comb)):
            if k.startswith("unsup") and wgt != 0.5:
                continue
            sub["s"] = s
            top = sub[sub.label == 1].sort_values("s", ascending=False).groupby("path").head(1)
            hard = top[top.part.isin(["002", "004", "005", "007"])]
            print(f"  {k:55s} AUROC {roc_auc_score(y, s):.3f} | top is defect {top.defect.mean()*100:.0f}%, hard {int(hard.defect.sum())}/{len(hard)}")
