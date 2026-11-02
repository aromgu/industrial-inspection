"""Cut an MVTec AD 2 category into tiles, to test whether small defects become detectable.

Each image is split into a cols x rows grid. Every tile window is enlarged by 1/KEEP around its
core and clamped inside the image, so the models' centre crop (224/256) removes only the margin
and still sees the whole core. Masks are tiled the same way. The result has the normal
MVTec AD 2 folder layout, so score.py runs on it unchanged:

    uv run python scripts/make_tiles.py --category can --grid 4 2
    uv run python scripts/score.py --model mhpatchcore_tile --category can --root datasets/tiles/can_4x2

Tile names are <image stem>__t<row><col>.png; scripts/aggregate_tiles.py maps tile scores back
to one score per image (max over its tiles).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

KEEP = 0.875  # fraction of each side the centre crop keeps


def windows(w: int, h: int, cols: int, rows: int) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    out = []
    cw, ch = w / cols, h / rows
    tw, th = min(w, round(cw / KEEP)), min(h, round(ch / KEEP))
    for r in range(rows):
        for c in range(cols):
            cx, cy = (c + 0.5) * cw, (r + 0.5) * ch
            x0 = int(min(max(cx - tw / 2, 0), w - tw))
            y0 = int(min(max(cy - th / 2, 0), h - th))
            out.append((r, c, (x0, y0, x0 + tw, y0 + th)))
    return out


def tile_dir(src: Path, dst: Path, cols: int, rows: int, masks: bool = False) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    pattern = "*_mask.png" if masks else "*.png"
    for p in sorted(src.glob(pattern)):
        stem = p.name[: -len("_mask.png")] if masks else p.stem
        im = Image.open(p)
        for r, c, box in windows(*im.size, cols, rows):
            name = f"{stem}__t{r}{c}" + ("_mask.png" if masks else ".png")
            im.crop(box).save(dst / name)
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/MVTec_AD_2"))
    p.add_argument("--category", required=True)
    p.add_argument("--grid", type=int, nargs=2, metavar=("COLS", "ROWS"), required=True)
    p.add_argument("--out", type=Path, default=Path("datasets/tiles"))
    args = p.parse_args()
    cols, rows = args.grid
    src = args.root / args.category
    dst = args.out / f"{args.category}_{cols}x{rows}" / args.category
    for sub in ["train/good", "validation/good", "test_public/good", "test_public/bad"]:
        print(sub, tile_dir(src / sub, dst / sub, cols, rows), flush=True)
    gt = "test_public/ground_truth/bad"
    print(gt, tile_dir(src / gt, dst / gt, cols, rows, masks=True), flush=True)
    print("->", dst.parent)


if __name__ == "__main__":
    main()
