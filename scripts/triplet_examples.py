"""Original | ground truth | prediction, 4 defective test_public photos per category (submission model).
Prediction: binary mask of make_submission.CONFIG (own threshold + post), drawn on the photo.
Writes docs/figures/qualitative/triplet/<cat>.png"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from final_examples import f1, load, pick  # noqa: E402
from make_submission import CONFIG  # noqa: E402
from seg_post import postprocess  # noqa: E402
from industrial_inspection.segeval import eval_size, ground_truth  # noqa: E402

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
OUT = Path("docs/figures/qualitative/triplet"); OUT.mkdir(parents=True, exist_ok=True)
for cat, (model, rule, post) in CONFIG.items():
    z, t, p = load(model, cat, rule, post)
    paths, labels = z["paths"].astype(str), z["labels"]
    fig, axes = plt.subplots(4, 3, figsize=(12, 15.5), facecolor="white")
    for r, i in enumerate(pick(paths, labels, 4)):
        hw = eval_size(paths[i])
        img = np.asarray(Image.open(paths[i]).convert("RGB").resize(hw[::-1], Image.BILINEAR))
        gt = ground_truth(paths[i], hw)
        pred = postprocess(z["maps"][i].astype(np.float32) > t, *p)
        gtv = (img * 0.45).astype(np.uint8); gtv[gt] = (227, 73, 72)
        pv = (img * 0.45).astype(np.uint8); pv[pred] = (42, 120, 214)
        for ax, im in zip(axes[r], (img, gtv, pv)):
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(Path(paths[i]).name, fontsize=9)
        axes[r, 2].set_xlabel(f"이 사진 F1 {f1(pred, gt):.2f}", fontsize=11)
    for ax, ti in zip(axes[0], ("원본", "정답 (GT, 빨강)", "예측 (파랑)")):
        ax.set_title(ti, fontsize=13)
    fig.suptitle(f"{cat} (공개 test 불량 사진, 제출 후보 모델)", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / f"{cat}.png", dpi=80, facecolor="white"); plt.close(fig)
    print("saved", cat, flush=True)
