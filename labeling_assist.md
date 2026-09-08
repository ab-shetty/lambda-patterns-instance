# Labelling assist — box prompt to polygon

**A different track from the product model.** `startup.md` owns the product
metric (reference-conditioned union IoU on HF14). This file owns a tool that
makes *labelling* cheaper, so it is scored differently: polygon IoU against
human polygons on a held-out set of labelled plans. Do not mix the two numbers.

Why it matters: source count is the binding constraint on the product metric
(28 -> 86 real sources bought +0.073), and generated sources are the best-value
supply. Anything that lowers the cost of labelling a sheet multiplies that lever.

## What it does

Box-drawing is the fast gesture in the Roboflow UI; polygon tracing is the slow
one. So: draw rough boxes, and the model turns each into a polygon on the region
that box encloses.

Roboflow's own SAM 3 Label Assist was tried and rejected in practice — its
polygons carry hundreds of stair-stepped vertices, which costs more time than it
saves. That is the problem this track solves, and it is a *post-processing*
problem, not a mask-quality one:

| | median vertices | median IoU |
|---|---:|---:|
| SAM 3 raw contour (what Label Assist gives) | 284 | 0.848 |
| + Douglas-Peucker (`regularize_polygon.py`) | 7 | 0.831 |

40x fewer vertices for 1.7 points of IoU. GT polygons are median 5 vertices.

## Result (2026-09-08)

Fine-tuned SAM 3 mask decoder, evaluated on 29 held-out plans / 98 instances:

| | mask IoU mean/med | polygon IoU mean/med | >= 0.8 |
|---|---|---|---:|
| zero-shot | 0.820 / 0.879 | 0.798 / 0.856 | 71.4% |
| **fine-tuned (epoch 10)** | **0.881 / 0.914** | **0.853 / 0.897** | **84.7%** |

`>= 0.8` is the number that matters operationally: roughly "accept with a nudge"
versus "redraw". Checkpoint `data/runs/sam3_ft/best.pth` — 4.2 MB, decoder only,
loads on top of stock `facebook/sam3`.

**Not yet replicated.** One split, one seed. The gain is monotone across 16
epochs, which is decent evidence, but by this repo's own standard it needs a
second seed before it is quoted.

## Reproduce

```bash
pip install --user "transformers>=4.50"          # 5.16.1 verified
python3 scripts/build_sam3_finetune_data.py --out data/sam3_ft
#   train 165 images / 767 instances | val 29 images / 98 instances
PYTHONPATH=. python3 scripts/train_sam3_boxseg.py --epochs 16 --out data/runs/sam3_ft
```

`facebook/sam3` is a gated HF repo; the account behind `HF_TOKEN` must have
accepted its terms. Loading reports 0 missing / 0 unexpected / 0 mismatched keys
— the `sam3_video`/`sam3_tracker` type warning it prints is cosmetic.

## Use it

```python
from scripts.sam3_region_model import RegionModel
rm = RegionModel(checkpoint="data/runs/sam3_ft/best.pth")   # None = zero-shot
polys = rm.polygons(image, [[x0, y0, x1, y1], ...])         # COCO-ready
```

Batch path, before a UI exists — label a sheet with boxes, export, upgrade,
re-import:

```bash
python3 scripts/sam3_box_to_polygon.py \
  --coco data/roboflow/<batch>/train/_annotations.coco.json \
  --img-dir data/roboflow/<batch>/train \
  --out data/proposals/<batch>.coco.json --preview data/proposals/<batch>_preview
```

## Constraints and measured negatives

- **1008x1008 is the accuracy ceiling.** SAM 3's vision encoder is fixed at
  1008 square, so a 3168px sheet is squashed and fine boundaries are limited by
  it. Decoder training does not change this. It is also why SAM 3 is a poor
  candidate as a *backbone* for the product model, whose largest measured win is
  training at 2048 (+0.05 to +0.07).
- **Orientation snapping is negative**, monotone in the tolerance (0.831 -> 0.814
  -> 0.812 -> 0.801 at 8/12/20 degrees). Rakes, gables and eaves are not on a
  rectilinear lattice. Kept as `snap_tol_deg`, defaulting to 0. Do not re-open.
- **Holes are not predicted.** A window punched out of a wall stays filled; it is
  cut afterwards as a separate `remove` polygon, which is what the Roboflow
  pipeline already does. Training targets the outer ring deliberately.
- **CPU inference is slow** — the frozen 454M encoder still runs. Cache the image
  embedding per sheet: it is computed once and every subsequent box on that sheet
  is then nearly free, which suits labelling many regions per sheet.
- **SAM 3 cannot do the product task.** Zero-shot PCS with the reference box as
  an exemplar scores **0.308** on the 52 HF14 selections (median 0.122, 8.6 masks
  returned per prompt) against RefUNet's 0.769: it finds *objects* on the sheet,
  not material regions. Exemplar prompts are box-only — points are PVS, one
  object. Do not re-open without fine-tuning it on the product task.

## Failure mode to watch

Where two adjacent regions share the same siding and are divided only by a corner
board or a change of plane, there is no visual edge and the boundary wanders into
the wall interior. Fine-tuning improves this markedly on held-out sheets (audit:
`item_000030.png`), but it is the residual error, and it is exactly the case a
generic segmenter cannot know about.

## Files

- `scripts/regularize_polygon.py` — mask -> few-vertex polygon; the measured
  simplification/snapping table lives in its docstring.
- `scripts/build_sam3_finetune_data.py` — manifest builder; splits by SOURCE
  IMAGE (instance-level splits leak drawing style). HF14 is not in these pools.
- `scripts/train_sam3_boxseg.py` — freezes the 454M encoder, trains the 4.2M
  mask decoder, jitters boxes each epoch (the deployed prompt is hand-drawn).
- `scripts/sam3_region_model.py` — inference entry point for the web app.
- `scripts/sam3_box_to_polygon.py` — batch box-COCO -> polygon-COCO.
