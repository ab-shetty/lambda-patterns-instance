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

## Holes: a second decoder, subtracted (2026-09-09) — +11.2 points end to end

**The reported numbers were scored against the wrong target.** The model
predicts the outer ring with holes filled and was evaluated against that same
filled ring, but the labeller needs the holed polygon. Against the TRUE
annotation the single model is much weaker, and on a holed region it is
unusable:

| scored against | mean IoU | >= 0.8 |
|---|---|---:|
| outer ring (what this file reported) | 0.8818 | 90.8% |
| **true holed annotation** | 0.8449 | **77.6%** |
| ... the 21 holed instances only | 0.6782 | **14.3%** |

21% of held-out instances have holes, averaging 21% of the region's area (max
37%) and 3.3 holes each (max 8). A perfect outer-ring model caps at 86.7%
against the true annotation, purely from unfilled holes.

**The fix is a second decoder on the same prompt.** Same frozen encoder, same
4.2M mask-decoder architecture, same box the labeller already drew -- only the
target changes, to the union of that region's openings. Then
`region = outer - holes`. **No extra gesture.** Polygon level, vs the true
annotation:

| | mean | >= 0.8 |
|---|---|---:|
| outer only | 0.8449 | 77.6% |
| **outer - holes** | **0.8698** | **88.8%** |
| the 21 holed regions, outer only | 0.6781 | 14.3% |
| **the 21 holed regions, subtracted** | **0.8059** | **71.4%** |
| the 77 solid regions, outer only | 0.8904 | 94.8% |
| the 77 solid regions, subtracted | 0.8873 | 93.5% |

+11.2 points end to end, recovering 85% of the 13.2-point hole gap, for 1.3
points given back on solid walls. The hole decoder alone scores 0.774 mask IoU
on holed regions and leaves 96.1% of solid regions untouched; it emits 87 rings
against a true 69, and that over-prediction is exactly the solid-wall cost.

```python
rm = RegionModel(crop_zoom=2, hole_checkpoint=DEFAULT_HOLE_CHECKPOINT)
segs = rm.segmentation(image, boxes)   # [[outer, hole1, ...], ...] COCO-ready
```

Published: `abshetty/floz-sam3-labelassist-holes`. The two decoders share one
loaded encoder and swap the 4.2M decoder between passes, so holes cost a second
forward, not a second model in memory.

### Openings are rectangles; region boundaries are not

The usability criterion for this whole track is vertex count, and the first
version failed it on holes: predicted hole rings carried a median of **9**
vertices against a human's **4**, because `eps_frac` is relative to contour
perimeter -- a small window gets a tiny absolute tolerance and keeps its wobble.

| hole treatment | mean IoU | >= 0.8 | median hole vertices |
|---|---|---:|---:|
| Douglas-Peucker eps 0.010 | 0.8823 | 87.8% | 9.0 |
| Douglas-Peucker eps 0.035 | 0.8819 | 87.8% | 4.0 |
| **minAreaRect (adopted)** | **0.8827** | **88.8%** | **4.0** |

A rotated-rectangle fit is best on *both* axes and lands on the human median.
**This is the opposite of the outer ring**, where snapping to a rectilinear
lattice is negative and monotone in the tolerance (`regularize_polygon.py`):
rakes, gables and eaves sit at many angles, but a window is a rectangle. The
rectilinear prior is right for openings and wrong for region boundaries -- do
not generalise either result to the other. `hole_shape="dp"` restores
Douglas-Peucker.

Final output: outer ring median 5.5 vertices (human 6.0), hole ring 4.0 (human
4.0), 8.0 total per instance (human 7.5), from a raw contour of 284.

**Two things to know before changing this.**

- **Filter hole components by RELATIVE area, not a pixel count.** The decoder's
  soft edges fragment into specks. A fixed 16px floor returned 11 rings on a
  region with 2 real holes; `min_hole_frac` (0.005 of the region) returns 82
  across the val set against a true 69. The measured numbers above use the
  relative floor.
- **Score the checkpoint on both failure modes.** 79% of instances have no holes,
  so a plain mean is dominated by "predict nothing" and would hide hallucinated
  openings. The selection score is `0.5 * hole IoU + 0.5 * solid-clean rate`.

Still open: a box-per-hole fallback for the ones it misses. That needs no new
model -- `RegionModel` is already box -> mask -- so it is a UI affordance.

## Clicks vs boxes (2026-09-09): negatives are what make a click work

A box is a drag that must enclose the region; a click is one event that only has
to land inside. `scripts/sam3_point_probe.py` prices them, sampling clicks from
the region interior by distance transform (and from the region with holes
removed -- a labeller selecting a wall clicks on wall, never through a window).
Corrective clicks go in the largest area the current prediction gets wrong,
labelled 1 to add a missed area or 0 to remove a false one.

Polygon IoU mean, 29 held-out sheets, `--prompt` is what the decoder trained on:

| decoder | 1 click | 2 | 3 | 5 | box |
|---|---:|---:|---:|---:|---:|
| box only (shipped) | 0.6435 | 0.8087 | 0.8141 | 0.8332 | **0.8678** |
| mixed, 1 positive | 0.7361 | 0.8067 | 0.8067 | 0.8388 | 0.8648 |
| points, 1-3 positive | 0.7531 | 0.8036 | 0.8367 | 0.8559 | 0.8554 |
| **points + 1-2 NEGATIVE** | 0.7383 | 0.8166 | 0.8610 | **0.8850** | 0.8473 |

At `>= 0.8`, points+negatives with 5 clicks reaches **91.8%** against the box's
86.7%.

**The result is in the scaling, not the level.** Every arm without negative
training saturates -- the box-trained decoder goes 0.809 -> 0.814 -> 0.833 over
2 to 5 clicks and buys almost nothing. The negatives arm climbs the whole way,
0.817 -> 0.861 -> 0.885. Training on negatives is what makes a corrective click
corrective: without it the model is handed a signal at inference that it never
learned to read, which was a straight train/test mismatch in the first two arms.

Why it should work: a single positive click cannot say where a region *ends*. On
repeating siding it is equally consistent with one course, one panel or the
whole wall -- the same ambiguity that gives SAM 3's PCS 0.308 on the product
task. A negative click just past the boundary supplies exactly that missing
extent. The sampler puts half the negatives in a band 8-40px outside the region
and half inside other labelled regions on the sheet (the confusable-material
case).

**One decoder for both gestures does not dominate two specialists.** Adding
negatives to the mixed arm (`--prompt mixed --max-neg 2`) keeps the box almost
exactly and lifts clicks well above the box-only decoder, but loses to each
specialist on that specialist's own gesture:

| decoder | box | 5 clicks |
|---|---|---|
| box-only (`floz-sam3-labelassist`) | **0.8678 / 86.7%** | 0.8332 / 77.6% |
| points+negatives (`...-clicks`) | 0.8473 / 78.6% | **0.8850 / 91.8%** |
| mixed + negatives | 0.8673 / 84.7% | 0.8764 / 86.7% |

So the shipped setup is the **pair**, not the all-rounder. The usual argument for
one model is memory and it does not apply: the decoders are 4.2M of a 458M
model and `RegionModel._swap()` already hot-swaps them over one shared frozen
encoder, which is how the holes path works. Two specialists cost one extra 17 MB
download and a few ms per swap, and buy +2 points on boxes and +5 on clicks.
Reach for `--prompt mixed --max-neg 2` only if a single decoder is a hard
requirement; it is the best all-rounder by a clear margin.

**Mixed prompting is free on boxes.** Training on box AND click alternating per
sheet scores 85.7% on boxes against the box-only decoder's 86.7% -- inside the
noise. So supporting clicks costs nothing on the gesture already relied upon.

**What this does and does not buy.** On gesture count the box still wins for an
ordinary region: 5 clicks is 5 gestures against one drag. What changed is the
achievable ceiling -- clicks can now exceed what a box reaches, which makes them
the right tool for the ~13% of regions a box gets wrong, not a replacement for
it. Note also that the corrective clicks here are placed using ground truth to
find the worst error, so these are an upper bound on a real user.

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

**Prerequisite: the Roboflow pools.** `build_sam3_finetune_data.py` reads
`data/roboflow/*-clean`, which a clone does not have. Build them first with the
"Real data from Roboflow" section of `startup.md` (four projects, then
`roboflow_to_local.py`, then merge the two Gemini rounds). Verified 2026-09-09:
86/282, 28/107, 36/231, 44/247 instances, and the manifest below then lands on
165 train / 767 instances and 29 val / 98 exactly.

```bash
pip install --user "transformers>=4.50"          # 5.16.1 verified
set -a; . ~/.env; set +a                         # HF_TOKEN, ROBOFLOW_API_KEY
PYTHONPATH=. python3 scripts/build_sam3_finetune_data.py --out data/sam3_ft
#   train 165 images / 767 instances | val 29 images / 98 instances
```

The three published decoders, ~20 min each on one GH200:

```bash
# 1. outer ring -> abshetty/floz-sam3-labelassist
PYTHONPATH=. python3 scripts/train_sam3_boxseg.py   --aug nodihedral --seed 7 --epochs 24 --out data/runs/sam3_nodihedral_s7

# 2. openings -> abshetty/floz-sam3-labelassist-holes
PYTHONPATH=. python3 scripts/train_sam3_boxseg.py --target holes   --aug nodihedral --seed 7 --epochs 24 --out data/runs/sam3_holes_s7

# 3. click-prompted -> abshetty/floz-sam3-labelassist-clicks
PYTHONPATH=. python3 scripts/train_sam3_boxseg.py   --prompt point --max-pos 3 --max-neg 2   --aug nodihedral --seed 7 --epochs 24 --out data/runs/sam3_pointneg_s7
```

Then publish anything worth keeping, or it is not persisted:

```bash
python3 scripts/publish_sam3.py --checkpoint data/runs/<run>/best.pth \
  --history data/runs/<run>/history.json --repo abshetty/<name>
```

Evaluate:

```bash
PYTHONPATH=. python3 scripts/sam3_crop_probe.py --checkpoint <ck> --zooms 0 1.5 2 3
PYTHONPATH=. python3 scripts/sam3_holes_eval.py            # outer vs outer-holes
PYTHONPATH=. python3 scripts/sam3_point_probe.py --checkpoint <ck> --clicks 1 2 3 5
PYTHONPATH=. python3 scripts/sam3_aug_report.py            # augmentation arms
bash scripts/sam3_status.sh -w                             # live progress
```

The RefUNet matcher for the chain (not published; ~35 min, and the chain is not
adopted -- see above):

```bash
bash scripts/build_refunet_matcher.sh
```

`facebook/sam3` is a gated HF repo; the account behind `HF_TOKEN` must have
accepted its terms. Loading reports 0 missing / 0 unexpected / 0 mismatched keys
— the `sam3_video`/`sam3_tracker` type warning it prints is cosmetic.

### What a clone does and does not have

Source, docs and every command above are committed. **Model weights live on the
HF Hub**, so inference needs no local artifacts. Everything under `data/` is
git-ignored and is a provenance record, not a file you have: the Roboflow pools,
the manifests, and every `data/runs/*` checkpoint rebuild from the commands
above. Experimental arms that were measured and not adopted (`none`, `default`,
`mixedprompt`, `pointpos`, `mixedneg`, `synth800`) are documented with their
numbers and rebuild from the same script with different flags; their weights
were deliberately not published.

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

  This does **not** contradict synthetic's value for the product model: that
  task is reference matching, where synthetic teaches pattern similarity, and
  this one is boundary precision.

  **The measured gap is colour** (2026-09-09). Mean HSV saturation over
  non-white pixels, and the share of ink that is saturated enough not to be
  grey:

  | pool | mean saturation | coloured ink | sheets >20% coloured |
  |---|---:|---:|---:|
  | **synth 1600** | **0.107** | **29.8%** | **53%** |
  | real 86 | 0.023 | 3.6% | 9% |
  | generated 28 | 0.001 | 0.0% | 0% |
  | gemini 80 | 0.040 | 8.5% | 18% |

  Synthetic is 4.6x more saturated than the real pool and carries 8x the
  coloured ink, and 800 synthetic sheets against 165 real puts 83% of the
  training signal on that distribution.

  **But measure HF14 before concluding synthetic is the odd one out.** The
  product acceptance set is far more colourful than any Roboflow pool, and
  synthetic is the pool CLOSEST to it:

  | set | mean saturation | coloured ink | distance from HF14 |
  |---|---:|---:|---:|
  | **HF14 (acceptance)** | **0.085** | **19.7%** | — |
  | synth 1600 | 0.107 | 29.8% | +10.1 pt |
  | gemini 80 | 0.040 | 8.5% | -11.2 pt |
  | real 86 | 0.023 | 3.6% | -16.1 pt |

  (HF14's 14-image validation complement reads the same, 17.6%, so this is the
  eval distribution and not a quirk of the acceptance split. Three HF14 sheets
  are 59-72% coloured ink.)

  So "synthetic is too colourful" holds only against **this track's val set**,
  which is 29 Roboflow sheets at 3.6% coloured ink -- and that val set is itself
  unrepresentative of the drawings the product is scored on. Two consequences,
  and the second is the one that matters:

  1. It is consistent with synthetic helping the product model, which is scored
     on HF14, while hurting here, where it is scored on near-monochrome sheets.
  2. **The labelling numbers in this file are measured on a near-monochrome
     val set.** Around a fifth of real sheets are colourised; none of the 88.8%,
     91.8% or 90.8% figures test that case. Before trusting them on a colourised
     drawing, build a val split that contains some. Note this is a *different* comparison
  from the colour work in `synth_progress.md` and
  `image_generation/README.md`, which measured the GENERATED pools against the
  real 28 eval set; nobody had compared the v5 synthetic pool against the
  Roboflow pools.

  An earlier version of this entry blamed the generator drawing "region edges as
  clean geometric fills — the wrong boundary prior". **That was speculation and
  was never measured**; the colour gap above is measured and is far larger. The
  edge-shape hypothesis may still hold, but it is not the recorded reason.
  Before re-opening synthetic for this task, desaturate the pool and re-run —
  that is a one-line change and tests the measured gap directly.

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
