"""Cache scripts/text_mask.py masks per eval sheet (Tesseract is slow: minutes per sheet on a few CPU cores), 4 sheets at a
time, one Tesseract thread each, 600 s per orientation (a timeout leaves that orientation empty and is printed).
    python3 scripts/pp_probe/textcache.py gemini30 val hf14   -> data/textmask/{split}_{idx:03d}.npz"""
import os
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, "scripts/pp_probe"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, cv2, pytesseract
import common
import text_mask as tm

OUT = Path("data/textmask")


def words(gray, timeout):
    try:
        d = pytesseract.image_to_data(gray, config="--psm 11", output_type=pytesseract.Output.DICT, timeout=timeout)
    except RuntimeError:
        return None
    return [(x, y, w, h) for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"])
            if float(c) >= 60 and len(tm.WORD.findall(t.strip())) >= 3 and 4 <= h <= 200 and w <= 40 * h]


def one(job_split):
    job, split = job_split
    f = OUT / f"{split}_{job[0]:03d}.npz"
    if f.exists():
        return f"{f} cached"
    idx, img, _, _ = common.load(job)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); H, W = gray.shape
    a = words(gray, 600); b = words(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), 600)
    boxes = (a or []) + [(y, H - x - w, h, w) for x, y, w, h in (b or [])]
    m = np.zeros((H, W), bool)
    for x, y, w, h in boxes:
        p = int(round(0.35 * min(w, h)))
        m[max(0, y - p):y + h + p, max(0, x - p):x + w + p] = True
    np.savez_compressed(f, mask=np.packbits(m), shape=np.array([H, W]))
    return f"{f} words {len(boxes)}" + (" TIMEOUT upright" if a is None else "") + (" TIMEOUT rotated" if b is None else "")


def load(split, idx, hw):
    z = np.load(OUT / f"{split}_{idx:03d}.npz")
    return np.unpackbits(z["mask"], count=int(np.prod(z["shape"]))).reshape(z["shape"]).astype(bool)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    common.MAPS = Path("data/floz-diag/gem2s_x")
    jobs = [(j, s) for s in sys.argv[1:] for j in common.jobs(s)]
    with Pool(4) as p:
        for r in p.imap_unordered(one, jobs):
            print(r, flush=True)
    print("TEXTCACHE DONE", flush=True)
