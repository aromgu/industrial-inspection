"""Q2 step 5 (H2-e): after the lighting changes, photograph k good parts under the new light and adapt.

The line scenario: the model was set up under "regular" light; then the light changes to one of the
shift_* conditions. Before production resumes, the operator photographs k good parts under the new
light (no defective parts needed). Two ways to use them, neither retrains anything:

  thr   recalibrate the normal-only threshold: max of the regular good scores and the k new photos
  bank  (PatchCore only) add the patch features of the k new photos to the memory bank, so the new
        lighting counts as normal; threshold = max regular good score, as before. (Recalibrating on
        top would be circular: photos in the bank score ~0 against themselves.)

Protocol: walnuts DEV half only (q2.split_parts). For each shift condition and every choice of k of
the dev good parts, those parts' photos in that condition adapt the model, and the metrics are taken
on the other dev parts in that condition: false alarms on good images (threshold = see above), and
misses on defective images. k = 0 is no adaptation. Results are averaged over conditions and
choices; the spread over choices is reported too.

    uv run python scripts/q2_fewshot.py --category walnuts --ks 0 1 2
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anomalib.models import Patchcore
from PIL import Image
from torchvision.transforms.v2 import functional as TF

from industrial_inspection.plots import q2_fewshot_figure
from industrial_inspection.q2 import half_mask, parts_and_tags


def ckpt_of(model: str, category: str) -> Path:
    found = sorted(Path("results/runs", model, category).rglob("model.ckpt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no checkpoint under results/runs/{model}/{category}")
    return found[-1]


class BankScorer:
    """PatchCore image scores against its memory bank, optionally extended with extra images' patches."""

    def __init__(self, category: str, device: str) -> None:
        self.model = Patchcore.load_from_checkpoint(ckpt_of("patchcore", category), map_location=device).to(device)
        self.net = self.model.model
        self.base_bank = self.net.memory_bank.clone()
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}

    def image(self, path: str) -> torch.Tensor:
        if path not in self.cache:
            x = TF.to_dtype(TF.to_image(Image.open(path).convert("RGB")), torch.float32, scale=True)[None]
            self.cache[path] = self.model.pre_processor.transform(x.to(self.device))
        return self.cache[path]

    @torch.no_grad()
    def patches(self, paths: list[str]) -> torch.Tensor:
        self.net.train()
        emb = torch.cat([self.net(self.image(p)) for p in paths])
        self.net.embedding_store.clear()
        self.net.eval()
        return emb

    @torch.no_grad()
    def scores(self, paths: list[str], extra: torch.Tensor | None = None) -> np.ndarray:
        self.net.memory_bank = self.base_bank if extra is None else torch.cat([self.base_bank, extra])
        self.net.eval()
        out = [float(self.net(self.image(p)).pred_score) for p in paths]
        self.net.memory_bank = self.base_bank
        return np.array(out)


def run(model: str, category: str, ks: list[int], device: str) -> pd.DataFrame:
    df = pd.read_csv(f"results/scores/{model}__{category}__seed0.csv").sort_values("image_path").reset_index(drop=True)
    parts, tag = parts_and_tags(df.image_path)
    y = df.label.to_numpy()
    dev = half_mask(parts, y, "dev")
    df, parts, tag, y = df[dev].reset_index(drop=True), parts[dev], tag[dev], y[dev]
    paths = df.image_path.to_numpy()
    good_parts = np.unique(parts[y == 0])
    reg_good = (tag == "regular") & (y == 0)
    conditions = sorted({t for t in tag if str(t).startswith("shift")})
    bank = BankScorer(category, device) if model == "patchcore" else None
    base = df.score.to_numpy() if bank is None else bank.scores(list(paths))
    if bank is not None:  # the re-scored base must match the training-time scores
        print(f"{model}: max |rescored - saved| = {np.abs(base - df.score.to_numpy()).max():.2e}", flush=True)

    rows = []
    for k in ks:
        for chosen in itertools.combinations(good_parts, k):
            for c in conditions:
                adapt = (tag == c) & np.isin(parts, chosen) & (y == 0)
                keep = (tag == c) & ~np.isin(parts, chosen)
                methods = {"none": (base, base[reg_good].max())} if k == 0 else {
                    "thr": (base, max(base[reg_good].max(), base[adapt].max()))}
                if bank is not None and k > 0:
                    scored = keep | reg_good
                    s = np.full(len(paths), np.nan)
                    s[scored] = bank.scores(list(paths[scored]), bank.patches(list(paths[adapt])))
                    methods["bank"] = (s, s[reg_good].max())
                for name, (s, thr) in methods.items():
                    g, b = keep & (y == 0), keep & (y == 1)
                    rows.append({"model": model, "k": k, "method": name, "condition": c, "chosen": "+".join(chosen),
                                 "false_alarm": float((s[g] > thr).mean()), "miss": float((s[b] <= thr).mean()),
                                 "n_good": int(g.sum()), "n_bad": int(b.sum())})
        print(f"{model} k={k}: done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="walnuts")
    p.add_argument("--models", nargs="+", default=["patchcore", "dinomaly_l"])
    p.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--smoke", action="store_true", help="k = 1 only, first choice, to check the code path")
    p.add_argument("--out", type=Path, default=Path("results/q2"))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = pd.concat([run(m, args.category, [0, 1] if args.smoke else args.ks, device) for m in args.models])
    if args.smoke:
        print(raw.groupby(["model", "k", "method"])[["false_alarm", "miss"]].mean())
        return
    per_choice = raw.groupby(["model", "k", "method", "chosen"])[["false_alarm", "miss"]].mean().reset_index()
    summary = per_choice.groupby(["model", "k", "method"]).agg(
        false_alarm=("false_alarm", "mean"), fa_min=("false_alarm", "min"), fa_max=("false_alarm", "max"),
        miss=("miss", "mean"), miss_min=("miss", "min"), miss_max=("miss", "max"), choices=("chosen", "count")).reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out / f"fewshot__{args.category}__dev_raw.csv", index=False)
    summary.to_csv(args.out / f"fewshot__{args.category}__dev.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print("figure:", q2_fewshot_figure(summary, args.category))


if __name__ == "__main__":
    main()
