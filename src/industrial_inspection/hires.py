"""High-resolution, training-free anomaly maps from frozen DINOv2 features (our SOTA track).

Segmentation F1 on test_public showed that our 256-pixel models are 30+ points below SuperADD and
that the map itself, not the threshold, is the gap. This model follows the recipe the strongest
training-free entries share (SuperAD, SuperADD): a frozen self-supervised ViT, the image seen at
high resolution in overlapping tiles, several layers, and nearest-neighbour distance to a small
memory bank of normal patches.

  * the image is resized so its shorter side is `short_side` (aspect ratio kept), cut into
    overlapping `tile` x `tile` windows; the patch tokens of the windows are stitched back into
    one token grid (overlaps averaged)
  * for each of `layers`, a patch's score is its L2 distance to the nearest patch of the memory
    bank; the map is the mean over layers
  * memory bank: the patches of `n_ref` training images chosen to be as different from each other
    as possible (greedy farthest-point selection on their global descriptors)
"""

from __future__ import annotations

import math
import time

import numpy as np
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional import rotate as tv_rotate

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def subsample_distance_based(x: torch.Tensor, target: int, iterations: int = 100, knn: int = 100,
                             seed: int = 0, device: str = "cuda") -> torch.Tensor:
    """Port of SuperADD's subsampling_distance_based_fast (github.com/LukasRoom/SuperADD,
    tracks/industrial/src/industrial/nearest_neighbor.py): keep a sample with probability 1 / (number of
    neighbours closer than r), raising r until about target/iterations samples per random subset survive,
    so dense regions of feature space are thinned and rare patches are kept. x: (N, C) on the GPU."""
    if len(x) <= target:
        return x
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(x), bool)
    size, per = max(1, len(x) // iterations), max(1, target // iterations)
    for _ in range(iterations):
        cand = np.flatnonzero(~keep)
        idx = rng.choice(cand, size=min(size, len(cand)), replace=False)
        sub = x[torch.from_numpy(idx)].to(device).float()  # x may live on the CPU (large pools), only subsets go to the GPU
        d = torch.cdist(sub, sub).topk(min(knn, len(idx)), dim=1, largest=False).values.cpu().numpy()
        r = float(d.mean()) / 10
        u = rng.random(len(idx))
        while True:
            m = u < 1.0 / ((d < r).sum(1) + 1)
            if m.sum() <= per:
                break
            r *= 1.1
        keep[idx] = m
    short = target - keep.sum()
    if short > 0:
        rest = np.flatnonzero(~keep)
        keep[rng.choice(rest, size=min(short, len(rest)), replace=False)] = True
    return x[torch.from_numpy(np.flatnonzero(keep)).to(x.device)]


class _DinoV3:
    """DINOv3 (torch.hub, local weights) behind the slice of the timm interface HiResKNN uses."""

    HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main"

    def __init__(self, name: str, weights: str) -> None:
        self.m = torch.hub.load(str(self.HUB), name, source="local", weights=weights).eval()
        self.patch_embed = type("PE", (), {"patch_size": self.m.patch_embed.patch_size})()

    def to(self, device):
        self.m = self.m.to(device)
        return self

    def eval(self):
        return self

    def forward_intermediates(self, x, indices, norm=False, output_fmt="NCHW", intermediates_only=True):
        return list(self.m.get_intermediate_layers(x, n=list(indices), reshape=True, norm=norm))


class HiResKNN:
    def __init__(self, backbone: str = "vit_large_patch14_reg4_dinov2.lvd142m", layers=(5, 11, 17, 23),
                 short_side: int = 672, tile: int = 448, overlap: int = 112, n_ref: int = 16,
                 device: str = "cuda", gpu_duty: float = 1.0, rotations: tuple[float, ...] = (0.0,),
                 pool: int = 1, knn: int = 1, bank: str = "fps", brightness: tuple[float, float] = (1.0, 1.0),
                 max_bank: int = 100_000, per_image_cap: int = 1500, holdout_every: int = 8) -> None:
        """pool: average each layer's token grid over a pool x pool neighbourhood (stride 1) before
        matching, for bank and queries alike -- PatchCore's "locally aware patch features", so a
        patch is compared together with its neighbours instead of alone. knn: score a patch by its
        mean distance to the knn nearest bank patches instead of the single nearest one."""
        if backbone.startswith("dinov3"):  # "dinov3_vith16plus" (weights/<name>.pth) -- SuperADD's backbone
            self.vit = _DinoV3(backbone, f"weights/{backbone}.pth").to(device)
        else:
            self.vit = timm.create_model(backbone, pretrained=True, dynamic_img_size=True).eval().to(device)
        self.patch = self.vit.patch_embed.patch_size[0]
        self.layers, self.short_side, self.tile, self.overlap, self.n_ref = list(layers), short_side, tile, overlap, n_ref
        self.device, self.gpu_duty, self.rotations = device, gpu_duty, list(rotations)
        self.pool, self.knn = pool, knn
        # bank="all": SuperADD's memory bank -- every training image except each holdout_every-th (SuperADD
        # keeps those for its threshold), each with a random brightness factor in `brightness`, patches pooled
        # (at most per_image_cap random patches per image, to bound RAM) and thinned to max_bank per layer by
        # distance-based subsampling. bank="fps": the n_ref most mutually different images, all patches.
        self.bank_mode, self.brightness = bank, brightness
        self.max_bank, self.per_image_cap, self.holdout_every = max_bank, per_image_cap, holdout_every
        self.bank: list[torch.Tensor] = []

    def _local(self, f: torch.Tensor) -> torch.Tensor:
        """(C, h, w) -> same shape, averaged over the pool x pool neighbourhood (edges: only real cells)."""
        if self.pool <= 1:
            return f
        return F.avg_pool2d(f[None], self.pool, stride=1, padding=self.pool // 2, count_include_pad=False)[0]

    # ---------- input ----------
    def load_raw(self, path: str) -> torch.Tensor:
        """(1, 3, H, W) in [0, 1], before ImageNet normalisation (so rotation can pad with black)."""
        im = Image.open(path).convert("RGB")
        s = self.short_side / min(im.size)
        w, h = (round(im.width * s / self.patch) * self.patch, round(im.height * s / self.patch) * self.patch)
        a = np.asarray(im.resize((w, h), Image.BICUBIC))
        if getattr(self, "clahe", 0):  # local contrast (CLAHE on the L channel), same for bank and queries
            import cv2
            lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
            lab[..., 0] = cv2.createCLAHE(clipLimit=float(self.clahe), tileGridSize=(16, 16)).apply(lab[..., 0])
            a = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if getattr(self, "unsharp", 0):  # high-frequency boost (unsharp masking): a + k * (a - blur(a)), sigma 2 px
            import cv2
            f = a.astype(np.float32)
            a = np.clip(f + float(self.unsharp) * (f - cv2.GaussianBlur(f, (0, 0), 2.0)), 0, 255).astype(np.uint8)
        x = torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)
        return x[None]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - torch.tensor(MEAN, device=x.device)[:, None, None]) / torch.tensor(STD, device=x.device)[:, None, None]

    def load(self, path: str) -> torch.Tensor:
        return self.normalize(self.load_raw(path))

    def _starts(self, length: int) -> list[int]:
        if length <= self.tile:
            return [0]
        n = math.ceil((length - self.overlap) / (self.tile - self.overlap))
        return [round(i * (length - self.tile) / (n - 1) / self.patch) * self.patch for i in range(n)]

    def _throttle(self, t0: float) -> None:
        if self.gpu_duty < 1:
            torch.cuda.synchronize()
            time.sleep((time.perf_counter() - t0) * (1 / self.gpu_duty - 1))

    # ---------- features ----------
    @torch.no_grad()
    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Per layer, a (C, H/patch, W/patch) token grid for the whole image. With self.pad_tokens > 0 the photo is
        reflect-padded by that many patches on every side first (border patches then see context instead of the
        image edge) and the padded tokens are cropped away again."""
        pt = int(getattr(self, "pad_tokens", 0))
        if pt:
            x = F.pad(x, (pt * self.patch,) * 4, mode="reflect")
            return [f[:, pt:-pt, pt:-pt] for f in self._features(x)]
        return self._features(x)

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        _, _, H, W = x.shape
        gh, gw = H // self.patch, W // self.patch
        acc, cnt = None, torch.zeros(1, gh, gw, device=self.device)
        for y0 in self._starts(H):
            for x0 in self._starts(W):
                t0 = time.perf_counter()
                win = x[:, :, y0:y0 + self.tile, x0:x0 + self.tile].to(self.device)
                with torch.autocast("cuda", dtype=torch.float16):
                    outs = self.vit.forward_intermediates(win, indices=self.layers, norm=False, output_fmt="NCHW",
                                                         intermediates_only=True)
                if acc is None:
                    acc = [torch.zeros(o.shape[1], gh, gw, device=self.device) for o in outs]
                ty, tx = y0 // self.patch, x0 // self.patch
                for a, o in zip(acc, outs):
                    a[:, ty:ty + o.shape[2], tx:tx + o.shape[3]] += o[0].float()
                cnt[:, ty:ty + outs[0].shape[2], tx:tx + outs[0].shape[3]] += 1
                self._throttle(t0)
        return [a / cnt for a in acc]

    @torch.no_grad()
    def descriptor(self, path: str) -> torch.Tensor:
        """Global descriptor for choosing references: mean last-layer token of a 224-px view."""
        x = F.interpolate(self.load(path), size=(224, 224), mode="bilinear", align_corners=False).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            o = self.vit.forward_intermediates(x, indices=[self.layers[-1]], output_fmt="NCHW", intermediates_only=True)[0]
        return o.float().mean(dim=(2, 3))[0]

    # ---------- memory bank ----------
    def fit(self, train_paths: list[str]) -> list[str]:
        """Build the bank from the n_ref most mutually different training images (greedy farthest-point
        selection on a global descriptor). If self.rotations has more than one angle, each reference
        image is also rotated by those angles (around its centre, black fill) and its rotated patches
        are added too -- for a category like wallplugs where the object itself lies at any angle on a
        uniform background, so a bank built from upright photos alone treats most orientations as novel."""
        if self.bank_mode == "all":
            return self._fit_all(train_paths)
        d = torch.stack([self.descriptor(p) for p in train_paths])
        chosen = [int(torch.linalg.norm(d - d.mean(0), dim=1).argmin())]  # start at the most typical image
        dist = torch.linalg.norm(d - d[chosen[0]], dim=1)
        while len(chosen) < min(self.n_ref, len(train_paths)):
            nxt = int(dist.argmax())
            chosen.append(nxt)
            dist = torch.minimum(dist, torch.linalg.norm(d - d[nxt], dim=1))
        refs = [train_paths[i] for i in chosen]
        banks = [[] for _ in self.layers]
        for p in refs:
            raw = self.load_raw(p)
            for angle in self.rotations:
                x = self.normalize(tv_rotate(raw, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0) if angle else raw)
                for b, f in zip(banks, self.features(x)):
                    b.append(self._local(f).flatten(1).T.half())
        self.bank = [torch.cat(b) for b in banks]
        return refs

    def _fit_all(self, train_paths: list[str]) -> list[str]:
        refs = [p for i, p in enumerate(train_paths) if self.holdout_every <= 1 or i % self.holdout_every != 0]
        rng = np.random.default_rng(0)
        pools = [[] for _ in self.layers]
        for p in refs:
            raw = self.load_raw(p)
            raw = torch.clip(raw * float(rng.uniform(*self.brightness)), 0, 1)
            for pool, f in zip(pools, self.features(self.normalize(raw))):
                t = self._local(f).flatten(1).T
                if len(t) > self.per_image_cap:
                    t = t[torch.from_numpy(rng.choice(len(t), self.per_image_cap, replace=False)).to(t.device)]
                pool.append(t.half().cpu())
        self.bank = []
        for pool in pools:
            x = torch.cat(pool)  # stays on the CPU: a large pool (per_image_cap x photos) need not fit in VRAM
            self.bank.append(subsample_distance_based(x, self.max_bank, device=self.device).to(self.device).half())
            del x
            torch.cuda.empty_cache()
        return refs

    @torch.no_grad()
    def score_layers(self, path: str, q_chunk: int = 2048, b_chunk: int = 32768) -> list[np.ndarray]:
        """Per layer, an (H/patch, W/patch) map of the mean distance to the knn nearest bank patches.

        Both the query patches and the bank are chunked, so the (q_chunk, b_chunk) distance matrix
        stays small and memory does not grow with the bank size -- a rotation-augmented bank
        (fit(), rotations=...) can be 8x the default and still score without running out of VRAM."""
        x = self.load(path)
        feats = self.features(x)
        out, out1 = self._dist(feats, q_chunk, b_chunk)
        if getattr(self, "self_r", 0):
            self.last_self = [self._self_dist(f, self.self_r) for f in feats]
        if getattr(self, "tta_flip", False):  # also the mirrored photo, its maps mirrored back, averaged
            o2, o21 = self._dist(self.features(x.flip(-1)), q_chunk, b_chunk)
            out = [np.ascontiguousarray((a + b[:, ::-1]) / 2) for a, b in zip(out, o2)]
            out1 = [np.ascontiguousarray((a + b[:, ::-1]) / 2) for a, b in zip(out1, o21)]
        self.last_nn1 = out1  # the k=1 maps of the same call, for free (used by scripts/hires.py --k1-name)
        return out

    @torch.no_grad()
    def _self_dist(self, f: torch.Tensor, r: int, q_chunk: int = 2048) -> np.ndarray:
        """In-image comparison: per patch, the distance to the nearest patch of the SAME photo that lies more
        than r tokens away (Chebyshev). A defect is rare within its photo; a normal part of a plug has look-alikes
        on the photo's other plugs, under the same exposure."""
        gh, gw = f.shape[1:]
        q = f.flatten(1).T
        yy, xx = torch.meshgrid(torch.arange(gh, device=self.device), torch.arange(gw, device=self.device), indexing="ij")
        yy, xx = yy.flatten(), xx.flatten()
        best = torch.empty(len(q), device=self.device)
        for qi in range(0, len(q), q_chunk):
            d = torch.cdist(q[qi:qi + q_chunk], q)
            near = ((yy[qi:qi + q_chunk, None] - yy[None]).abs() <= r) & ((xx[qi:qi + q_chunk, None] - xx[None]).abs() <= r)
            best[qi:qi + q_chunk] = d.masked_fill(near, float("inf")).min(1).values
        return best.reshape(gh, gw).cpu().numpy()

    def _dist(self, feats, q_chunk: int, b_chunk: int):
        gh, gw = feats[0].shape[1:]
        out, out1 = [], []
        for f, bank in zip(feats, self.bank):
            q = self._local(f).flatten(1).T
            k = min(self.knn, len(bank))
            best = torch.empty(len(q), device=self.device)
            best1 = torch.empty(len(q), device=self.device)  # distance to the single nearest patch (k=1)
            for qi in range(0, len(q), q_chunk):
                qc = q[qi:qi + q_chunk]
                top = torch.full((len(qc), k), float("inf"), device=self.device)
                for bi in range(0, len(bank), b_chunk):
                    d = torch.cdist(qc, bank[bi:bi + b_chunk].float())
                    kk = min(k, d.shape[1])
                    top = torch.cat([top, d.topk(kk, dim=1, largest=False).values], dim=1).topk(k, dim=1, largest=False).values
                best[qi:qi + q_chunk] = top.mean(dim=1)
                best1[qi:qi + q_chunk] = top[:, 0]  # topk returns ascending values, so column 0 is the nearest
            out.append(best.reshape(gh, gw).cpu().numpy())
            out1.append(best1.reshape(gh, gw).cpu().numpy())
        return out, out1

    def score(self, path: str) -> np.ndarray:
        """(H/patch, W/patch) anomaly map: the mean of score_layers over layers."""
        return np.mean(self.score_layers(path), axis=0)
