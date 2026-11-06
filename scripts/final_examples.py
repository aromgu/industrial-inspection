"""Qualitative results with ground truth, 3 defective test_public photos per category, for the model we would
submit (make_submission.CONFIG: its own threshold rule and post-processing) next to the SuperADD-bank
variant (hires_sabank_k50, the common rule), everything at 1/4 of the original resolution.

Columns: photo | ground truth (red outline) | anomaly map of the submission model + ground truth (blue) |
submission model, binary prediction vs truth | SuperADD-bank k=50, binary prediction vs truth.
Prediction colours: green = correctly flagged, red = false alarm, blue = missed.

Writes docs/figures/qualitative/final/<category>.png

    .venv/Scripts/python.exe scripts/final_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from make_submission import CONFIG, threshold_of  # noqa: E402
from seg_post import postprocess  # noqa: E402

from industrial_inspection.segeval import eval_size, ground_truth  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"

OTHER = ("hires_sabank_k50", "p95x1.2", (6, True, 64))
OUT = Path("docs/figures/qualitative/final")


def pick(paths: np.ndarray, labels: np.ndarray, n: int = 3) -> list[int]:
    """n defective regular-light photos spread over the sorted list."""
    cand = [i for i in np.argsort(paths) if labels[i] == 1 and "_regular" in Path(paths[i]).name]
    if len(cand) < n:
        cand = [i for i in np.argsort(paths) if labels[i] == 1]
    return [cand[round(k * (len(cand) - 1) / (n - 1))] for k in range(n)]


def overlay(img: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    o = (img * 0.45).astype(np.uint8)
    o[pred & gt] = (27, 175, 122)
    o[pred & ~gt] = (227, 73, 72)
    o[~pred & gt] = (42, 120, 214)
    return o


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.sum(pred & gt)
    d = 2 * tp + np.sum(pred & ~gt) + np.sum(~pred & gt)
    return float(2 * tp / d) if d else 0.0


def load(model: str, cat: str, rule: str, post: tuple[int, bool, int]):
    z = np.load(f"results/maps/{model}__{cat}.npz")
    v = np.load(f"results/maps/{model}__{cat}__val.npz")["maps"].astype(np.float32)
    return z, threshold_of(rule, v), post


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cat, (model, rule, post) in CONFIG.items():
        zA, tA, pA = load(model, cat, rule, post)
        zB, tB, pB = load(OTHER[0], cat, OTHER[1], OTHER[2])
        paths, labels = zA["paths"].astype(str), zA["labels"]
        pathsB = {p: i for i, p in enumerate(zB["paths"].astype(str))}
        fig, axes = plt.subplots(3, 5, figsize=(19, 9.6), facecolor="white")
        for r, i in enumerate(pick(paths, labels)):
            path = paths[i]
            hw = eval_size(path)
            img = np.asarray(Image.open(path).convert("RGB").resize(hw[::-1], Image.BILINEAR))
            gt = ground_truth(path, hw)
            mA = zA["maps"][i].astype(np.float32)
            mB = zB["maps"][pathsB[path]].astype(np.float32)
            predA = postprocess(mA > tA, *pA)
            predB = postprocess(mB > tB, *pB)
            good = zA["maps"][labels == 0][:20].astype(np.float32)
            lo, hi = np.quantile(good, [0.5, 0.999])
            axes[r, 0].imshow(img)
            axes[r, 1].imshow(img)
            axes[r, 1].contour(gt, levels=[0.5], colors="#e34948", linewidths=1.4)
            axes[r, 2].imshow(mA, cmap="magma", vmin=lo, vmax=max(hi, float(mA.max())))
            axes[r, 2].contour(gt, levels=[0.5], colors="#4aa3ff", linewidths=1.1)
            axes[r, 3].imshow(overlay(img, predA, gt))
            axes[r, 4].imshow(overlay(img, predB, gt))
            axes[r, 0].set_ylabel(Path(path).name, fontsize=10)
            axes[r, 3].set_xlabel(f"이 사진 F1 {f1(predA, gt):.2f}", fontsize=10)
            axes[r, 4].set_xlabel(f"이 사진 F1 {f1(predB, gt):.2f}", fontsize=10)
        titles = ["사진", "정답(GT, 빨간 윤곽)", f"이상 맵 + GT(파란 윤곽)\n[{model}]",
                  f"제출 후보 예측 vs 정답\n[{model}]", "SuperADD식 뱅크 k=50 예측 vs 정답\n[hires_sabank_k50]"]
        for ax, t in zip(axes[0], titles):
            ax.set_title(t, fontsize=11)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{cat}  —  초록: 맞게 잡음, 빨강: 오경보, 파랑: 놓침 (공개 test 불량 사진, 원본 1/4 해상도)",
                     fontsize=14, y=0.995)
        fig.tight_layout()
        fig.savefig(OUT / f"{cat}.png", dpi=95, facecolor="white")
        plt.close(fig)
        print("saved", cat, flush=True)


if __name__ == "__main__":
    main()
