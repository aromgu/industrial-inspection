"""Fuse the plug-crop map (plugs_free_only) with the whole-image map (hires_k50) for wallplugs -> plugs_max.

The whole-image map is quantile-matched (on validation) to the plug map's distribution inside plug masks,
then the two are combined by a per-pixel max (outside plugs only the whole-image map counts). z-score
fusion made every variant worse than either input; this is what worked. Writes results/maps/plugs_max__*.

    .venv/Scripts/python.exe scripts/plugs_fuse.py
"""
import argparse
import numpy as np, cv2, pandas as pd
from pathlib import Path
from industrial_inspection.segeval import eval_size
from industrial_inspection.plugs import load_plugs
def cov_of(p,hw):
    pl=load_plugs(Path("results/plugs_cache/wallplugs")/Path(p).parent.parent.name/Path(p).parent.name/(Path(p).stem+".npz"))
    c=np.zeros(hw,bool)
    for q in pl: c|=cv2.resize(q.mask.astype(np.uint8),hw[::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
    return cv2.dilate(c.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
ap=argparse.ArgumentParser(); ap.add_argument("--inst",default="plugs_free_only"); ap.add_argument("--out",default="plugs_max"); A=ap.parse_args()
I=np.load(f"results/maps/{A.inst}__wallplugs.npz"); G=np.load("results/maps/hires_k50__wallplugs.npz")
Iv=np.load(f"results/maps/{A.inst}__wallplugs__val.npz"); Gv=np.load("results/maps/hires_k50__wallplugs__val.npz")
paths=I["paths"].astype(str); vpaths=Iv["paths"].astype(str); hw=eval_size(paths[0])
i,g,iv,gv=[x["maps"].astype(np.float32) for x in (I,G,Iv,Gv)]
# covered = where the plug map rises above its "no plug" floor (works for mask and rect pasting alike)
floor=float(iv.min()); C=i>floor+1e-3*(iv.max()-floor); Cv=iv>floor+1e-3*(iv.max()-floor)
qs=np.linspace(0.5,0.9999,60); src,dst=np.quantile(gv,qs),np.quantile(iv[Cv],qs)
m=lambda x: np.interp(x,src,dst)
lo=float(iv[Cv].min())
t=np.maximum(np.where(C,i,lo),m(g)).astype(np.float16); v=np.maximum(np.where(Cv,iv,lo),m(gv)).astype(np.float16)
np.savez_compressed(f"results/maps/{A.out}__wallplugs.npz",maps=t,labels=I["labels"],paths=paths)
np.savez_compressed(f"results/maps/{A.out}__wallplugs__val.npz",maps=v,paths=vpaths)
pd.DataFrame({"image_path":paths,"label":I["labels"],"score":t.reshape(len(t),-1).max(1).astype(float)}).to_csv(f"results/scores/{A.out}__wallplugs__seed0.csv",index=False)
print("ok")
