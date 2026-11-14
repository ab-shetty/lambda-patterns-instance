"""Montage: python3 scripts/probes/fg_montage.py POOL OUT.jpg id1,id2,... -> image | label overlay (ref_exclude in black)."""
import json, sys, os
import numpy as np, cv2
D, out, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
cols = [(230,25,75),(60,180,75),(0,130,200),(245,130,48),(145,30,180),(70,240,240),(240,50,230),(210,245,60)]
tiles = []
for n in names:
    a = json.load(open(f"{D}/annotations/{n}.json")); img = cv2.imread(f"{D}/images/{n}.png")
    H, W = img.shape[:2]; ov = img.copy()
    for an in a["annotations"]:
        m = np.zeros((H, W), np.uint8); segs = an["segmentation"]
        cv2.fillPoly(m, [np.round(np.array(segs[0]).reshape(-1,2)).astype(np.int32)], 1)
        for h in segs[1:]: cv2.fillPoly(m, [np.round(np.array(h).reshape(-1,2)).astype(np.int32)], 0)
        ov[m > 0] = cols[(an["category_id"]-1) % len(cols)]
    lab = cv2.addWeighted(img, 0.5, ov, 0.5, 0)
    for r in a.get("ref_exclude", []):
        cv2.polylines(lab, [np.round(np.array(r).reshape(-1,2)).astype(np.int32)], True, (0,0,0), 3)
    s = 1400 / W; img = cv2.resize(img, (1400, round(H*s)), interpolation=cv2.INTER_AREA); lab = cv2.resize(lab, (1400, round(H*s)), interpolation=cv2.INTER_AREA)
    pair = np.hstack([img, np.full((img.shape[0], 10, 3), 255, np.uint8),
                      lab])
    cv2.putText(pair, n, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    tiles.append(pair)
w = max(t.shape[1] for t in tiles)
tiles = [np.pad(t, ((0, 12), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
cv2.imwrite(out, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out)
