"""Per-image inference latency: PyTorch eager vs ONNX Runtime (CUDA), batch size 1.

Each trained checkpoint is exported with anomalib's to_onnx (pre-processing included), then the
same random input is timed in both runtimes after a warm-up. Image decoding and the initial resize
from the camera resolution are not included. Run it on an otherwise idle GPU: a game or another
training job in the background inflates and scatters the numbers.

    uv run python scripts/latency.py --category sheet_metal
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anomalib.models import Dinomaly, EfficientAd, MHPatchcore, Patchcore

from industrial_inspection.plots import latency_figure

# model key -> (class, input size fed to the exported graph)
MODELS = {
    "patchcore": (Patchcore, (256, 256)),
    "mhpatchcore": (MHPatchcore, (256, 256)),
    "efficientad": (EfficientAd, (256, 256)),
    "dinomaly_l": (Dinomaly, (448, 448)),
}


def ckpt_of(model: str, category: str) -> Path | None:
    found = sorted(Path("results/runs", model, category).rglob("model.ckpt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def scores_of(out) -> np.ndarray:
    if isinstance(out, (list, tuple)):
        out = out[0]
    t = getattr(out, "pred_score", out)
    return t.detach().float().cpu().numpy().ravel() if torch.is_tensor(t) else np.asarray(t).ravel()


def timed(fn, warmup: int, iters: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    return np.array(ms)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="sheet_metal")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("results/latency.csv"))
    args = p.parse_args()

    import onnxruntime as ort  # after torch, so ORT reuses torch's CUDA/cuDNN DLLs

    rows = []
    for key in args.models:
        cls, size = MODELS[key]
        ckpt = ckpt_of(key, args.category)
        if ckpt is None:
            print(f"{key}: no checkpoint, skipped")
            continue
        model = cls.load_from_checkpoint(ckpt, map_location="cpu").eval().cuda()
        x = torch.rand(1, 3, *size, device="cuda")

        def torch_run():
            with torch.no_grad():
                out = model(x)
            torch.cuda.synchronize()
            return out

        torch_ms = timed(torch_run, args.warmup, args.iters)
        ref = scores_of(torch_run())

        onnx_dir = Path("results/onnx") / f"{key}__{args.category}"
        # Static batch-1 graph: with the default dynamic batch axis the dynamo exporter specialises
        # these models to its batch-2 example input.
        onnx_path = model.to_onnx(export_root=onnx_dir, input_size=size, dynamic_shapes=None)
        sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        feed = {sess.get_inputs()[0].name: x.cpu().numpy()}
        ort_ms = timed(lambda: sess.run(None, feed), args.warmup, args.iters)
        names = [o.name for o in sess.get_outputs()]
        outs = dict(zip(names, sess.run(None, feed)))
        got = np.asarray(outs.get("pred_score", next(iter(outs.values())))).ravel()
        for runtime, ms in (("pytorch", torch_ms), ("onnxruntime-cuda", ort_ms)):
            rows.append({
                "model": key, "category": args.category, "runtime": runtime, "input": f"{size[0]}x{size[1]}",
                "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
                "provider": sess.get_providers()[0] if runtime != "pytorch" else "cuda",
                "score_abs_diff": float(np.abs(got[: ref.size] - ref).max()) if got.size else float("nan"),
            })
        print(f"{key}: pytorch {np.median(torch_ms):.1f} ms, onnxruntime {np.median(ort_ms):.1f} ms "
              f"({sess.get_providers()[0]}), score diff {rows[-1]['score_abs_diff']:.2e}", flush=True)
        del model, sess
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print("figure:", latency_figure(df))


if __name__ == "__main__":
    main()
