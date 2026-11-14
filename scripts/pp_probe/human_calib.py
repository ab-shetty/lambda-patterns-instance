"""Calibration renders for human.py: whole sheet at fit-to-screen size, label-only pixels red, prediction-only blue, the
reference box orange, titled with the largest visible patch as a share of the region. Edit `want` / MAPS; -> data/diag/human_calib_*.jpg"""
import sys; sys.path[:0]=["scripts/pp_probe","scripts","."]
import numpy as np, cv2, common, product, human
from pathlib import Path
from polygonize import rasterize
common.WITH_BOX[0]=True
want={"valfix4":[(13,3),(4,1)],"hf14fix4":[(12,0),(0,0),(11,0),(25,1)]}
common.MAPS=Path("data/floz-diag/fgab_fg")
for s,qq in want.items():
  for job in common.jobs(s):
    if job[0] not in {a for a,_ in qq}: continue
    idx,img,_,qs=common.load(job); h,w=img.shape[:2]
    for ref,cat,tgt,p4,box in qs:
      if (idx,ref) not in qq: continue
      mp={z:(p4 if z==4096 else np.load(common.MAPS/f"{common.map_split(s)}_{z}/probs/{idx:03d}_{ref:03d}.npz")["prob"]) for z in product.SCALES}
      P=product.product_polygons(mp,img,box); pred=rasterize(P,(h,w)); ign=common.IGNORE.get((idx,ref))
      if ign is not None: pred&=~ign
      r=human.judge(P,tgt,ign,(h,w))
      v=img.copy(); v[pred]=(0.6*v[pred]+[0,100,0]).astype(np.uint8)
      v[tgt&~pred]=(0,0,255); v[pred&~tgt]=(255,0,0)
      for p in P:
        for rr in [p["outer"]]+p["holes"]: cv2.polylines(v,[np.array(rr,np.int32)],True,(0,120,0),max(1,int(max(h,w)/1500)))
      x,y,bw,bh=box; cv2.rectangle(v,(x,y),(x+bw,y+bh),(0,200,255),max(2,int(max(h,w)/800)))
      sc=2000/max(h,w); v=cv2.resize(v,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
      big=max(r["miss"]+r["extra"]+[0])
      cv2.putText(v,f"{s} {idx} q{ref}  IoU {r['iou']:.3f}  biggest patch {100*big:.1f}% of region",(10,40),0,1.1,(0,0,0),3)
      cv2.imwrite(f"data/diag/human_calib_{100*big:04.1f}pct_{s}_{idx}_{ref}.jpg",v,[cv2.IMWRITE_JPEG_QUALITY,88])
