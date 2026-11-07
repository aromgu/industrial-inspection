"""P3 fix: seeds only where false alarms are rare. Gate = SAM plug masks dilated by --dil eval px, minus a
--border px band at the photo border (normal high peaks sit 39 % at the border, 32 % on background; defect peaks
85 % on plug bodies -- scripts/wp_fp_where.py). Outside the gate the seed map is lowered by a large constant, so
hysteresis seeds can only start on plugs; extent still grows from the raw map. Writes model <out>.
    .venv/Scripts/python.exe scripts/wp_gate.py --seed r1792L_s2_iq90 --out gate_iq90 [--dil 3 --border 12]"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
import sys
sys.path.insert(0, str(Path(__file__).parent))
from industrial_inspection.plugs import load_plugs  # noqa: E402
a = argparse.ArgumentParser(); a.add_argument("--seed", required=True); a.add_argument("--out", required=True)
a.add_argument("--dil", type=int, default=3); a.add_argument("--border", type=int, default=12); a.add_argument("--no-plug-gate", action="store_true")
args = a.parse_args()
C = Path("results/plugs_cache/wallplugs")


def gate(p, hw):
    g = np.ones(hw, bool)
    if not args.no_plug_gate:
        pls = load_plugs(C / Path(p).parent.parent.name / Path(p).parent.name / (Path(p).stem + ".npz"))
        g = np.zeros(hw, bool)
        for q in pls:
            g |= cv2.resize(q.mask.astype(np.uint8), hw[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        g = ndimage.binary_dilation(g, iterations=args.dil) if args.dil else g
    b = args.border
    if b:
        g[:b] = g[-b:] = False; g[:, :b] = g[:, -b:] = False
    return g


for suf in ("", "__val"):
    z = dict(np.load(f"results/maps/{args.seed}__wallplugs{suf}.npz"))
    m = z["maps"].astype(np.float32); hw = m.shape[1:]
    low = m.min() - 1000
    out = np.stack([np.where(gate(p, hw), mm, low) for mm, p in zip(m, z["paths"].astype(str))]).astype(np.float16)
    np.savez_compressed(f"results/maps/{args.out}__wallplugs{suf}.npz", **{**z, "maps": out})
print("saved", args.out)
