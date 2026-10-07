"""Invariance probe sheets: an elevation-like sheet with two wall panels, A and B, labelled as two families that
differ in ONE attribute (direction / spacing / fill colour / line colour / wash / texture). One question per panel,
a clean 220 px box in its middle. Leak = share of the OTHER panel the model selects."""
import json, os, sys, numpy as np, cv2
OUT = sys.argv[1]; os.makedirs(f'{OUT}/images', exist_ok=True); os.makedirs(f'{OUT}/annotations', exist_ok=True)
W, H = 2400, 1300
PA = (180, 420, 1080, 1180)    # x0, y0, x1, y1 panel A
PB = (1320, 420, 2220, 1180)
S = 14                          # base line spacing (px at 2400 wide)
INK = (40, 40, 40)

def lines(img, rect, angle, sp, col, th=2, fill=None, ticks=None, pairs=False):
    x0, y0, x1, y1 = rect
    tile = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    if fill is not None: tile[:] = fill
    h, w = tile.shape[:2]; D = int(np.hypot(h, w)) + 4
    big = np.full((D, D, 3), 255, np.uint8) if fill is None else np.tile(np.array(fill, np.uint8), (D, D, 1))
    for k, y in enumerate(range(0, D, sp)):
        cv2.line(big, (0, y), (D, y), col, th, cv2.LINE_AA)
        if pairs: cv2.line(big, (0, y + 4), (D, y + 4), col, th, cv2.LINE_AA)
        if ticks:  # shingle courses: staggered vertical joints between this line and the next
            off = (k % 2) * ticks // 2
            for x in range(off, D, ticks): cv2.line(big, (x, y), (x, min(D, y + sp)), col, 1, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((D / 2, D / 2), angle, 1.0)
    big = cv2.warpAffine(big, M, (D, D), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255) if fill is None else tuple(int(v) for v in fill))
    oy, ox = (D - h) // 2, (D - w) // 2
    img[y0:y1, x0:x1] = big[oy:oy + h, ox:ox + w]

def brick(img, rect, sp, col):
    x0, y0, x1, y1 = rect
    for k, y in enumerate(range(y0, y1, sp)):
        cv2.line(img, (x0, y), (x1, y), col, 2, cv2.LINE_AA)
        off = (k % 2) * sp * 2
        for x in range(x0 + off, x1, sp * 4): cv2.line(img, (x, y), (x, min(y1, y + sp)), col, 2, cv2.LINE_AA)

TAN, OLIVE, BLUE, WASH = (150, 190, 215), (110, 160, 150), (190, 160, 130), (215, 215, 215)   # BGR
P = {  # name: (A spec, B spec, attribute)
 'identical':        (dict(angle=0, sp=S), dict(angle=0, sp=S), 'control'),
 'dir_H_vs_V':       (dict(angle=0, sp=S), dict(angle=90, sp=S), 'direction'),
 'dir_H_vs_45':      (dict(angle=0, sp=S), dict(angle=45, sp=S), 'direction'),
 'dir_V_vs_45':      (dict(angle=90, sp=S), dict(angle=45, sp=S), 'direction'),
 'sp_x1.5':          (dict(angle=0, sp=S), dict(angle=0, sp=int(S * 1.5)), 'spacing'),
 'sp_x2':            (dict(angle=0, sp=S), dict(angle=0, sp=S * 2), 'spacing'),
 'fill_tan_vs_olive':(dict(angle=0, sp=S, fill=TAN), dict(angle=0, sp=S, fill=OLIVE), 'fill colour'),
 'fill_tan_vs_blue': (dict(angle=0, sp=S, fill=TAN), dict(angle=0, sp=S, fill=BLUE), 'fill colour'),
 'line_black_vs_red':(dict(angle=0, sp=S), dict(angle=0, sp=S, col=(40, 40, 200)), 'line colour'),
 'wash_vs_bare':     (dict(angle=0, sp=S, fill=WASH), dict(angle=0, sp=S), 'tone'),
 'lap_vs_shingle':   (dict(angle=0, sp=S), dict(angle=0, sp=S, ticks=S * 3), 'texture'),
 'board_vs_batten':  (dict(angle=90, sp=S * 2), dict(angle=90, sp=S * 2, pairs=True), 'texture'),
 'lap_vs_brick':     (dict(angle=0, sp=S), 'brick', 'texture (easy)'),
}
boxes = {'_meta': {'what': 'invariance probe boxes', 'pool': OUT}}
for i, (name, (a, b, attr)) in enumerate(P.items()):
    img = np.full((H, W, 3), 255, np.uint8)
    for rect, spec in ((PA, a), (PB, b)):
        if spec == 'brick': brick(img, rect, S * 2, INK)
        else: lines(img, rect, spec.get('angle', 0), spec['sp'], spec.get('col', INK), fill=spec.get('fill'), ticks=spec.get('ticks'), pairs=spec.get('pairs', False))
        cv2.rectangle(img, rect[:2], rect[2:], (20, 20, 20), 4)
    # gable roof outline (unlabelled) + ground line, so it reads as an elevation
    cv2.polylines(img, [np.array([[140, 420], [1200, 120], [2260, 420]])], False, (20, 20, 20), 4)
    cv2.line(img, (60, 1180), (2340, 1180), (20, 20, 20), 5)
    fn = f'{i:02d}_{name}'
    cv2.imwrite(f'{OUT}/images/{fn}.png', img)
    poly = lambda r: [[float(r[0]), float(r[1]), float(r[2]), float(r[1]), float(r[2]), float(r[3]), float(r[0]), float(r[3])]]
    anns = [{'category_name': 'pattern1', 'segmentation': poly(PA)}, {'category_name': 'pattern2', 'segmentation': poly(PB)}]
    json.dump({'image': {'file_name': f'{fn}.png', 'width': W, 'height': H}, 'mode': 'probe', 'attr': attr, 'annotations': anns},
              open(f'{OUT}/annotations/{fn}.json', 'w'))
    c = lambda r: [(r[0] + r[2]) // 2 - 110, (r[1] + r[3]) // 2 - 110, 220, 220]
    boxes[str(i)] = {'0': c(PA), '1': c(PB)}
json.dump(boxes, open(f'{OUT}/boxes.json', 'w'))
print(len(P), 'sheets')
