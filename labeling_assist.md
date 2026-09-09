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
versus "redraw".

**Where the model lives: `abshetty/floz-sam3-labelassist` on the HF Hub**
(private). Decoder only — 4.22M parameters, 16.9 MB of fp32 — published as
safetensors and loaded on top of stock `facebook/sam3`, whose frozen 454M
encoder is byte-identical to upstream and so is not republished. `RegionModel()`
with no arguments resolves it, downloads it, and caches it, so a fresh VM needs
no local artifacts.

> The 2026-09-08 run wrote only to git-ignored `data/runs/sam3_ft/best.pth` and
> existed on one machine; when that machine went, the model went with it. The
> Hub repo is the fix. **Publish every checkpoint worth keeping** —
> `scripts/publish_sam3.py` — or it is not persisted.

(An earlier version of this file called the checkpoint "4.2 MB". That was 4.2M
*parameters* misread as megabytes; the file is 16.9 MB.)

**Not yet replicated.** One split, one seed. The gain is monotone across 16
epochs, which is decent evidence, but by this repo's own standard it needs a
second seed before it is quoted.

**Rebuilt and reproduced 2026-09-09** from a clean clone on a fresh GH200: the
data rebuild matched every recorded count (86/282, 28/107, 36/231, 44/247;
165 train / 767 instances, 29 val / 98), zero-shot reproduced to four decimals
(0.8203/0.8794 mask, 0.7978/0.8564 poly, 71.4%), and the run peaked at **0.8971**
val polygon median against the recorded 0.897.

## Chaining to RefUNet: one box -> a pattern family (2026-09-09)

76% of held-out instances sit in a family of >1 and one sheet carries a family
of 15, so the BOXES are the labelling cost, not the tracing. The chain:
user box -> SAM 3 mask -> reference crop sampled inside it -> RefUNet union ->
connected components -> SAM 3 polygon per component.
`scripts/sam3_refunet_chain.py`, `--oracle` for the ceiling.

Scored as gestures on the 29 held-out sheets (468 regions, so the hand-drawn
baseline is 468 boxes):

| matcher | family recall | FP / box | regions from 98 boxes | deletions |
|---|---:|---:|---:|---:|
| ORACLE (union IoU 1.0) | 0.913 | 1.32 | 435 | 129 |
| RefUNet real-165 @1280 (0.63), tuned | 0.612 | 2.11 | 212 | 207 |
| same, unfiltered | 0.648 | 6.06 | 232 | 594 |

**Not adopted.** At 0.63 the chain buys 212 of 468 regions for 98 boxes but
costs 207 deletions -- break-even at best. The ceiling is a 44% gesture saving
(260 vs 468), so the idea is sound and the matcher is the binding constraint.

**Two findings worth keeping.**

- **The size filter is the whole lever.** Requiring a component to be >= 25% of
  the prompted region's area cuts false proposals 4.7 -> 1.0 for ~0.03 of
  recall; RefUNet's spurious fragments are small. The probability threshold
  barely matters (0.35 to 0.80 changes little). Defaults are `--min-frac 0.25
  --topk 8`.
- **The oracle's 9% miss is SAM 3, not the matcher.** SAM 3's standalone accept
  rate is 90.8%; oracle family recall is 91.3%. Converting a perfect union into
  boxes loses almost no regions. What it does lose is precision -- the 1.32
  false proposals -- and that is irreducible here, because **a union mask cannot
  represent the boundary between two touching same-family regions.** No
  threshold recovers a split that is not in the representation.

So more matcher quality pays up to the ceiling and nothing past it. Going
further means changing the decomposition -- SAM 3 proposes regions, the matcher
scores them -- which drops the union->components step and, unlike a union mask,
lets a proposal carry holes.

**Beware the cheap proxy.** Tuning on component masks at IoU >= 0.5 predicted
0.90 FP/box; the full chain at >= 0.8 on polygons gave 2.11. SAM 3 tracing a
fragment box, rather than a region box, is what the proxy misses. Tune on the
proxy, but confirm on the chain.

## Inference-time cropping (2026-09-09) — the largest lever found on this track

No retraining. Run the encoder on a window around the prompt box instead of on
the whole sheet, and paste the mask back. `RegionModel(..., crop_zoom=2)`, or
per call `rm.polygons(im, boxes, crop_zoom=2)`.

Fine-tuned decoder (`nodihedral`, seed 7), 29 val sheets / 98 instances:

| crop_zoom | mask mean | poly mean | >= 0.8 | sheets >=1500px | cost |
|---|---|---|---:|---|---|
| off (whole sheet) | 0.8891 | 0.8678 | 86.7% | 0.8691 | 1 forward / **sheet** |
| 1.5x | **0.9059** | **0.8851** | 88.8% | **0.8907** | 1 forward / **box** |
| 2x | 0.9049 | 0.8819 | 90.8% | 0.8880 | 1 forward / box |
| 3x | 0.8993 | 0.8754 | **91.8%** | 0.8790 | 1 forward / box |

Mean IoU peaks at 1.5x while `>=0.8` keeps climbing to 3x: a wide window pulls
the hard cases over the bar while costing a little on the ones already easy.
`>=0.8` is the operational metric, so prefer 2-3x; 2x is the documented default.

**It is not a free win, and it is not a property of cropping.** The same sweep
on other decoders:

| decoder | off | 1.5x | 3x |
|---|---:|---:|---:|
| `nodihedral` (crop-augmented) | 86.7% | 88.8% | **91.8%** |
| `none` (no crop augmentation) | 82.7% | 86.7% | 86.7% |
| stock SAM 3, zero-shot | 71.4% | **56.1%** | 65.3% |

Cropping a stock decoder is **strongly negative**. What pays is train/test
consistency: the decoder has to have been fine-tuned, and it gains most when it
was trained with the zoom-crop augmentation. `RegionModel` therefore refuses
`crop_zoom` without a fine-tuned checkpoint rather than silently degrading.

The cost is real: cropping forfeits the per-sheet embedding cache (embed once,
every later box on that sheet nearly free), which is what makes the CPU
deployment viable. 0.065 -> 0.174 s/instance on a GH200. So `crop_zoom` defaults
to **off**, and the labelling app should turn it on when it has a GPU.

**One split, one seed.** Consistent across three decoders and monotone in zoom,
which is decent evidence, but the val set is 29 sheets.

### Polygon IoU is noisier than mask IoU — a floor for reading any comparison

Re-evaluating **identical published weights** in a fresh process, deterministic
within a process but not across:

| | training log (ep 8) | fresh re-eval |
|---|---|---|
| mask IoU mean/med | 0.8831 / 0.9152 | 0.8831 / 0.9153 |
| poly IoU mean/med | 0.8455 / 0.8971 | 0.8471 / 0.8916 |
| >= 0.8 | 76.5% | 78.6% |

Mask IoU reproduces to four decimals; polygon IoU does not, because
Douglas-Peucker turns a one-pixel boundary shift into a *moved vertex*. So
**differences under ~0.006 polygon median, or ~2 points of `>=0.8`, are noise**
and must not be read as an effect. Prefer mask IoU when comparing arms; quote
polygon IoU because it is what the labeller actually receives.

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
rm = RegionModel()                     # the published decoder, from the Hub
polys = rm.polygons(image, [[x0, y0, x1, y1], ...])         # COCO-ready
```

`checkpoint=` also takes any Hub repo id or a local `.pth`/`.safetensors`;
`checkpoint=None` is stock zero-shot SAM 3. Needs `HF_TOKEN` on an account that
has accepted `facebook/sam3`'s terms, because the encoder comes from there.

To publish a new checkpoint (do this for anything worth keeping):

```bash
python3 scripts/publish_sam3.py \
  --checkpoint data/runs/<run>/best.pth --history data/runs/<run>/history.json \
  --repo abshetty/floz-sam3-labelassist
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

- **1008x1008 bounds what one forward can see — but it is a ceiling on the
  SHEET, not on the region.** SAM 3's vision encoder is fixed at 1008 square, so
  a 3168px sheet is squashed 3x and fine boundaries are limited by it. Decoder
  training does not change this, and it is why SAM 3 is a poor candidate as a
  *backbone* for the product model, whose largest measured win is training at
  2048 (+0.05 to +0.07). **It does not bound inference**: cropping to a window
  around the prompt box spends the 1008 on the region instead of the sheet and
  is worth more than anything on the training side — see "Inference-time
  cropping" above. This bullet previously read "Decoder training does not change
  this" full stop, which was true and was then misread as "nothing changes it".
- **Orientation snapping is negative**, monotone in the tolerance (0.831 -> 0.814
  -> 0.812 -> 0.801 at 8/12/20 degrees). Rakes, gables and eaves are not on a
  rectilinear lattice. Kept as `snap_tol_deg`, defaulting to 0. Do not re-open.
- **Synthetic sheets are NEGATIVE on this task** (2026-09-09). 800 sheets from
  `toparea1600_balanced` added to train (val held byte-identical, so the two are
  comparable), same `nodihedral` augmentation, compared at matched OPTIMIZER
  STEPS because 965 sheets/epoch against 165 makes epoch counts meaningless:

  | steps | real-only 165 | +800 synthetic |
  |---|---|---|
  | 965 | 0.8853 / 80% | 0.8656 / 76% |
  | 2895 | 0.8925 / 87% | 0.8792 / 78% |
  | 4825 | 0.9000 / 87% | 0.8924 / 83% |

  (polygon IoU median / `>=0.8`.) Behind at every point, and still behind after
  22% more steps than the real-only run ever got. Best 0.8924/82.7% against
  0.9037/86.7%. Killed at epoch 4.

  This does **not** contradict synthetic's value for the product model. That
  task is reference matching, where synthetic teaches pattern similarity. This
  task is boundary precision, and the generator draws region edges as clean
  geometric fills — the wrong boundary prior — while 800 synthetic against 165
  real puts 83% of the training signal on that prior. Do not re-open without a
  generator whose edges look like drawn ones.

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
- `scripts/sam3_region_model.py` — inference entry point for the web app;
  resolves a local file or a Hub repo id, and owns `crop_zoom`.
- `scripts/sam3_box_to_polygon.py` — batch box-COCO -> polygon-COCO. **Until
  2026-09-09 this built its own stock `facebook/sam3` and never loaded the
  fine-tuned decoder**, so the batch path was labelling zero-shot (71.4%) while
  the tuned model sat unused. It now goes through `RegionModel` with
  `--crop-zoom 2` by default.
- `scripts/sam3_augment.py` — sheet augmentation. Every transform is verified
  against a rasterized reference (identity 1.0000, hflip/vflip/rot90 0.999+);
  the flip convention is `W - x`, not `W - 1 - x`, because
  `render_instance_mask` truncates vertices to int32.
- `scripts/publish_sam3.py` — publish a checkpoint to the Hub. Run it for
  anything worth keeping.
- `scripts/sam3_status.sh` — one-screen progress for the `sam3_*` runs.
- `scripts/sam3_crop_probe.py` — prices `crop_zoom` on a checkpoint.
- `scripts/sam3_aug_report.py`, `run_sam3_aug.sh` — the augmentation ablation.
