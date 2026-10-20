# Synthetic Dataset Progress

Working summary. Every dated entry (2026-08-06 to 2026-09-26) is kept verbatim
in `synth_progress_archive.md`, Part 1, under its original heading; other docs'
references to "`synth_progress.md` (DATE)" resolve there. Append new results
below "Log".

## Current state (2026-10-01)
- **Defaults changed 2026-10-04 (with `--cut-standing 0` below): `--plan-v2 2`, `--downspouts 0.05`,
  `--ref-exclude 1`.** (a) Floor plans get furniture in every room (solid on rendered sheets),
  patio furniture, framing, and the floor labels that leave out wall-fixed pieces -- until now
  those labels and all furniture applied only to `--plan-v2 2` / `--plan-source` pools, i.e. no
  trained pool. (b) Downspouts on ~5% of wall ends. (c) Each annotation file carries
  `ref_exclude` (pixel rings): posts, railing panels, exterior stairs, downspouts, solid furniture,
  rugs -- inside the labels (the material continues behind them), but training
  (`refmask2former/dataset.py`) and the local-pool evaluator / mining draw REFERENCE boxes from
  the label minus them (`reference_region`). A piece wholly hidden behind them is never a
  reference: training asks its family from another visible piece (or another family), the
  local-pool evaluator / mining skip that question; it stays in every target. Real / Gemini
  records have no field: unchanged (same RNG draws). Measured: floor-plan boxes > 50% on
  furniture 10.1% -> 0%; elevation boxes > 10% on posts / railings / downspouts 11.7% -> 0%
  (18 / 486 elevation and 2 / 507 floor-plan pieces skipped as hidden; no family lost). Mining scores on new pools are not comparable with old ones. Earlier pools:
  `--plan-v2 1 --cut-standing 1 --downspouts 0 --ref-exclude 0` (byte-identical, 40 elevations +
  150 floor plans checked).
- **Default changed 2026-10-04: `generate_synthetic_fc.py --cut-standing 0` (Revit elevations).**
  Porch posts, railing / deck posts and newel posts are drawn as before but stay inside the wall
  label -- the wall continues behind them, as every hand-labelled sheet draws it (Gemini, real 02 /
  27). Every pool before this (since `--trim-cut` on 2026-09-29, incl. `revitfail_hard2000`, the
  best model's pretraining pool) cut them out. Images are byte-identical; `--cut-standing 1`
  reproduces earlier pools exactly (40/40 images and labels). Audit below ("Label cuts vs the hand
  labels"). New opt-in `--downspouts P` (default 0 = byte-identical).
- **Default changed 2026-10-01: `generate_synthetic_fc.py --cedar-shingle 0.5`.** Half of shingle
  styles are drawn as cedar shingles (vertical joints dominant, like HF14 25/27) instead of
  brick-like courses. Pools generated before this commit reproduce only with `--cedar-shingle 0`.
- **Default changed 2026-10-01: `--masonry-base 0.35`.** The Revit base band (0.6-2.6 ft, on 65%
  of houses) is a brick / stone / block wainscot 35% of the time instead of concrete or flat,
  in a muted masonry palette, labelled ~80% (HF14 18 / val 17: grey running-bond roof vs a teal
  running-bond brick band, dE 16). In a 120-sheet sample: 18 of 75 elevations get a masonry base,
  2 colour sheets pair a labelled one with a labelled roof (dE 19, 30). `--masonry-base 0`
  reproduces earlier pools.
- **Default changed 2026-10-01: `--plan-v2 1` (floor plans, `--revit-plans`).** (a) A ceiling /
  electrical sheet type (15% of floor plans; HF14 12): line-only, dense MEP clutter, the eave
  soffit band (1-2.5 ft outside the walls, whole ring or some sides) plus porch ceilings
  labelled `soffit`. (b) Rendered sheets (HF14 0): half get a covered patio in a footprint notch
  walled on two sides and butting the interior floors; half pave in the interior floors' grey;
  interior plank / tile lines stronger. In 40 floor plans: 8 ceiling sheets, 9 rendered.
  `--plan-v2 0` reproduces earlier pools (40 sheets checked).
- **New, opt-in 2026-10-01: `--plan-source data/reference/swiss_dwellings/layouts.pkl.gz`
  (`--plan-source-p P`, default 1).** Floor plans use real apartment layouts from Swiss Dwellings
  v3 (Archilyse AG, Zenodo 7788422, **CC BY 4.0: commercial use OK, credit required** in any
  released data or model card) instead of `plan_layout`. Each apartment comes with real walls and
  thicknesses, doors, windows, balconies and railings, and fitted kitchens, sinks, toilets, tubs,
  showers and stairs. Build the file with `python3 scripts/swiss_plans.py` from the zip
  (`curl -L -o data/reference/swiss_dwellings/sd.zip
  https://zenodo.org/api/records/7788422/files/swiss-dwellings-v3.0.0.zip/content`, 932 MB;
  about 20 min on CPU). Of 45,176 apartments, 19,046 are kept (75 MB): residential, 4+ rooms,
  45-260 m2, walls within 2 degrees of square, one footprint, exact repeats dropped. The judges'
  last tells were layout (wall stubs, door collisions, odd rooms); real layouts remove those.
  Nothing trained on it yet. Off = byte-identical (checked at plan-v2 1 and 2).
  Ruled out on licence (non-commercial): CubiCasa5K and FloorPlanCAD.
- **Floor labels follow the Gemini finish plans (2026-10-02; Swiss layouts and `--plan-v2 2` only).**
  Checked against r4 008 / 038 / 039 / 062: wall-fixed pieces (counters, fridge, vanities, tubs,
  showers, built-ins, washers, stairs) are left out of the floor label, as notches, with any
  strip under 0.4 ft between them and the wall; toilets, islands and loose furniture stay in.
  Swiss areas with no wall between them (open plan, ~0.05 ft apart) now grow into the gap, so
  one finish is one polygon through the open side and the doorways (653040: 5 floor polygons -> 1);
  doors find the room on each side of the wall (openings 5 -> 7 of 10). A ROOM open to the
  kitchen / a corridor is the living room; open corridors take that room's finish. Carpet stipple
  6x denser, 2 px dots (was 1-3 one-pixel dots per sq ft, read as a blank floor). Plan-v2 0 / 1
  byte-identical (12 renders checked).
  Kept on purpose (2026-10-02): on rendered sheets the paving and the unlabelled interior can be
  the same pattern at near-equal spacing, told apart only by tone (653038: white deck boards vs
  grey planks, base 255 vs 223). The v2 grey-paving rule runs after `_separate` and can bring the
  tones closer still. These are hard cases the model should learn; don't "fix" them.

- **Shipped model unchanged: HF14 0.8440** (below). The Revit source in the real
  mix did NOT beat it (2026-09-30: 0.8367, -0.007 p=0.58; val 0.784 vs 0.815).
- **Best synthetic-only: 10k Revit + ROI** (`abshetty/floz-refunet-swint-revit10k-roi-e5`,
  private): val 0.8150 / **0.8861**, HF14 0.7868 / 0.8126 @2048 / 4096 -- the best
  validation of any model, real-trained included. Fresh unseen Revit plans 0.872.
- **Steps per sheet matter as much as sheet count** (2026-10-01): 500 Revit sheets (>= 3
  families) trained 1+60 epochs then a 30-epoch restart (~11k steps, swin_t, ROI add) reach
  **HF14 0.822-0.830 @4096** in all three arms (no DR / `--ref-sample instance` /
  `--dr-scale-min 0.7`), above `revit10k-roi-e5`'s 0.8126 -- but val 0.809-0.834 against its
  0.886. Not a clear win; untried: the long schedule on more sheets.
- **Hand-placed reference boxes** (`--boxes eval_boxes/hand_v1.json`) change no model by more
  than +-0.016 paired on HF14 (none significant); the shipped model reads 0.8461 (51 q).
- **Reference jitter fix is the default since 728def3** (reference gets the sheet's
  `--domain-random` brightness/contrast); `--legacy-ref-jitter` = every earlier run.
- **Screens now use ROI** (`EXTRA="--roi-ref --roi-ref-mode add" ./run_revit_2k.sh 7 <arm>`);
  control: `abshetty/floz-refunet-swint-revit2k-roi-e8` (private, HF14 0.7855 / 0.7970).

- **Best model:** `abshetty/floz-refunet-swint-mixr4-roiadd-swa15`, the
  restart recipe (`run_restart_swa.sh` from `mixr4-e8`) plus `--roi-ref
  --roi-ref-mode add`, inference at 4096: **HF14 0.8440** (+0.027 paired vs
  0.8170, p=0.017; one seed). Previous: `...-mixr4-restart-e13`, 0.8170.
  TTA x4 (+0.0015) and multi-scale inference (0.8123) not adopted.
- **Seeds: one per arm from here** (user's call, 2026-09-28): at ~0.85, judge
  a change by the paired per-question test on HF14 against its same-recipe
  control; no seed replicates.
- **Synthetic source: Revit-style FreeCAD sheets beat v6d** (2026-09-27, one
  seed, synth-only, 2k @2048): `scripts/generate_synthetic_fc.py --revit
  --mode-weights 66,16,18` gives **HF14 0.7564 / val 0.7568** against v6d's
  0.7066 / 0.6702 (`abshetty/floz-refunet-swint-revit2k-e8`, round-trip
  verified). The gain is the Revit RENDERER, not the 3D geometry (FreeCAD
  geometry drawn by v6: 0.7126 / 0.6248). At 4096 inference the lead shrinks
  to +0.016 on both splits. Not yet tested in the real mix. Every v6 generator
  change before it lost or tied: v6e look-alike pairs (negative), v7 (-0.023),
  `--hardscape-plan` / `--same-fill-subtle` (null, 3 seeds), `--mottle`
  (null), v5 confusable pairs (-0.104), **r8 (-0.076 HF14, 2026-09-27)**.
- **Best synthetic-only:** Revit 2k + `--refs-per-image 6`, swin_t @2048,
  only 4 epochs (VM deadline): **val 0.7816, HF14 0.7607**
  (`abshetty/floz-refunet-swint-revit2k-sixq-e4`, one seed) -- vs one-question
  Revit (9 epochs) +0.025 val / +0.004 HF14, both noise, at under half the
  epochs; fresh Revit plans (HF14 protocol) 0.789 vs 0.758. One-question Revit 2k: HF14 0.7564
  (`abshetty/floz-refunet-swint-revit2k-e8`). Previous: v6d 100k single pass on RefUNet 0.7130; swin_t v6d 1,995
  0.7066 (`abshetty/floz-refunet-swint-v6d-e8`).
- **Realism probe** (vs val 14): r8 0.909 < v6d 0.948 < FreeCAD 0.954 <
  Revit 0.960 -- and on HF14 the order is REVERSED (Revit best, r8 worst).
  The probe does not predict training value; don't use it to pick pools.

## How to test a synthetic change

- **Screen at 2048, not 1024** (2026-09-27, `run_res_calibration.sh`): the
  same r8-vs-v6d A/B trained at 1024 and at 2048 (1,600 sheets, seed 7).
  1024 said r8 HELPS on validation (+0.028) where 2048 says it HURTS (-0.046);
  on HF14 both say hurts (-0.055 / -0.037; per-selection delta correlation
  0.61 on HF14, -0.03 on validation). Fine-texture changes alias away at 1024
  (x0.4 `--domain-random` shrink on top). Use `run_revit_2k.sh`'s recipe:
  swin_t, ~2k sheets @2048, 1+8 epochs, one seed, ~20 min per arm alone;
  control = the published v6d 2k (`abshetty/floz-refunet-swint-v6d-e8`).
  Single-epoch HF14 swings 0.02-0.04 between adjacent epochs, so a one-seed
  delta under ~0.04 is noise; a winner still needs the real mix.
- The old 1024 screen (`run_synth_only_screen.sh`): swin_t at 1024, three runs
  concurrently ~20 min, v6d control spans 0.640-0.694 on HF14 across seeds.
- **Don't** use `run_synth_ft_ab.sh`: a gentle fine-tune of a converged model
  leaves 48/52 selections tied and cannot detect a pixel-only change. A full
  retrain (`run_nogray_mix.sh`-style) takes ~80 min.
- Choose epochs and inference size on validation
  (`select_epoch_on_val.py`); read HF14 once. HF14 is 52 fixed questions
  (reference boxes identical across runs), so compare **paired** per
  selection, and only against a same-recipe control: two base runs differing
  only by seed pair at p = 0.0001.
- Checkpoint averaging reads ~0.04 low when a run is still climbing at the last
  epoch; check that it turned over first.
- Transfer: use `scripts/question_difficulty.py`, not `fresh_synth_iou.py`
  ratios (different question mix). `fresh_synth_iou.py` decides "seen" by
  filename, and merged pools are all `item_XXXXXX`. An empty `--trained-pool`
  crashes (FileNotFoundError); for known-disjoint splits (fresh ids 100000+)
  pass the model's own training pool.
- **Realism probe** (`scripts/synth_realism_probe.py`): frozen DINOv2 plus
  logistic regression on crops inside labelled regions, reference val 14 at
  3168 px (indistinguishable from HF14: AUC 0.423), mean of 3 seeds, 320 synth
  sheets (seed 6, ids 200000+). Differences under ~0.04 are noise. It measures
  "looks like these ~12 documents", not "looks like a real drawing" (real
  Boise Revit sheets score 0.998), so it ranks changes and its crop sheets
  show what's off, but it is not a training-value metric.

## Findings that still bind

- **Volume:** v6d-only at 2048: 1,600 -> 0.640, 12,000 -> 0.693, 100,000
  seen once -> 0.713 (RefUNet). Gains shrink ~60% per 8x. With enough unique
  data, one pass beats many epochs; a second pass already lowers
  fresh-synthetic IoU. swin_t (2026-09-27, `run_v6d_scaling.sh`, nested
  pools, fresh v6d scored with the HF14 protocol): 2k 0.660, 8k 0.716, 16k
  0.732 -- a 3-point fit flattens at ~0.77. HF14 0.707 / 0.737 / 0.729.
- **swin_t underfits family grouping, not data** (2026-09-27): the 16k model
  scores 0.736 on its OWN training plans (HF14 protocol) against a
  resolution ceiling of 0.904 (`label_ceiling.py`, same 1,798 questions).
  Single-family sheets 0.957; 4-6 family sheets 0.673 (65% of the error);
  non-look-alike targets hold 60% of it. Asking all families per step
  (`--refs-per-image 6`) matches 16k with 2k plans (fresh 0.727, seen 0.741)
  but hits the same ~0.74 wall; real plans +0.017 val / +0.012 HF14 (noise).
- **Real-plan penalty:** net of question difficulty, every source (Gemini
  included) pays ~-0.14 to -0.17 on real plans. At matched 148 sources Gemini
  and v6d tie on HF14.
- **Where the error is:** target thickness at input resolution (thin < 74 px at
  2048: 0.437, mid 0.716, thick 0.825; the thin third is about half the missing
  IoU). Region identification, not boundary placement: boundary methods and
  threshold sweeps (flat, peak 0.35) give ~0. HF14 image 7's reference box sits
  on printed text (worth ~+0.011; don't chase).
- **Resolution:** `--domain-random` shrinks training sheets 0.4-1.0x, so
  inference above the nominal size helps (+0.009 on v6d-only, +0.028 on the
  real mix at 4096). Training at 2560 beat 2048 once (0.7834,
  `abshetty/floz-refunet-res2560-e4`); a second seed was never run.
- **Distinct labelled real sources** are the binding data constraint:
  re-augmenting the same 114 sources 18x/36x/72x, synthetic volume and mix
  ratio at mix scale are all null.
- **Closed levers** (all null or negative, detail in the archive):
  conditioning mechanisms (`--corr-grid`, `--scale-matched-ref`, ranking loss,
  self/dynamic prototypes, anchor; the anchor-plane bug is fixed, default
  `--anchor-ref-plane 0.0`), decoder width, confusable/look-alike synthetic
  pairs, same-fill recolouring, grayscale-free augmentation (the model does use
  colour), contrast normalisation, two-pass inference, small-reference
  training.
- **Backbone:** swin transfers best and fits worst, stable across LRs;
  material separability of frozen features does not predict transfer.

- **Default changed 2026-10-02: `--openings-per-face 1`** (`generate_synthetic_fc.py`). Windows
  and doors were placed against the union of a block's wall faces, so on `--shaped` houses they
  ran straight across the corners of a projecting bay or wing: 34 of 150 Revit elevations (23%)
  had one, the 10k pool included. Now each opening sits inside one planar face (0.3 ft clear);
  a front / garage door that would straddle slides to the nearest free spot on a face that fits
  it; a narrow face (3.8-12 ft) left empty gets one centred window per storey. 46 / 150 sheets
  change. `--openings-per-face 0` reproduces earlier pools (12 sheets checked).

- **HF14 label fix v1 (2026-10-02, opt-in `--label-fixes eval_labels/hf14_fixes_v1.json`):** sheet
  14's lap panel under window 09 added to pattern1's target (missed in labelling; boxes and
  questions unchanged). Re-scored on CPU, hand boxes, 4096 (without the flag both reproduce the
  Model Results artifact's sheet-14 means exactly): shipped sheet 14 0.9250 -> 0.9357, HF14
  0.8466 -> 0.8483; `revit10k-roi-e5` 0.8403 -> 0.8505, HF14 0.8118 -> 0.8134. Six pattern1
  questions gain ~+0.014 each in both models; the gap between them is unchanged (-0.035).

## Synthetic-only failures on HF14 (2026-10-02) and the flags aimed at them

From the Model Results artifact (`revit10k-roi-e5` vs shipped, hand boxes, 4096): HF14 0.8118 vs
0.8466, while val is 0.898 vs 0.819. The loss sits on five sheets: 27 (-1.06 summed IoU), 14
(-0.68), 25 (-0.47), 2 (-0.31), 12 (-0.30); synth gains on 0 (+0.63) and 18 (+0.46). Four failures,
each now with an opt-in flag (own RNG each; all 0 = byte-identical, 12 sheets checked):

| failure (real sheet) | what the generator lacked | flag |
|---|---|---|
| tight horizontal roof lines merged with looser vertical siding, both unpainted (25 q01/q02, 27 q02-q04: 0.94-0.99 shipped -> 0.39-0.72) | roofs were rows with joints or standing seam (vertical lines); a plain horizontal-lines roof never met vertical boards (lap + vertical walls do pair: 51 / 300 sheets) | `--roof-lines P` |
| railing / stair in front of siding (27: deck balusters over lap read as a grid; stair + rail over vertical boards) | Revit elevations have porch posts but no railings or exterior stairs | `--railings P`: balcony (2nd floor) or raised deck across part of a wall, pickets or cables, half with a stair to grade; rim / stringer are trims (cut), posts are drawn as trims but stay in the wall label (since 2026-10-04, `--cut-standing 0`), rails, balusters, treads and handrail are drawn over and the wall label runs on behind; on shaded sheets the deck + railing shadow is cast `depth` ft out by the same sun as `wall_shadows` |
| faint line-only texture taken for blank paper (14 q06 brick tile 0.987 -> 0.555 picks blank stucco) | line-only pattern lines were 110-175 grey; 14's are ~240 | `--faint-lines P` (line patterns only: faint sparse stucco dots read as blank paper, so dot kinds keep their weight) |
| siding in porch shade missed (2 q00/q01 0.88 -> 0.73) | shade was one x0.55-0.75 multiply keeping the hue; 02's is x0.81 and neutral grey (246,242,230 -> 199,200,195) with a x0.68 overlap level | `--soft-shadows P` (two suns through `wall_shadows`, the 3D massing's projection) |

Railings are a 2D overlay with a depth, not FreeCAD solids: in the fused massing they would hide
the wall and cut its label, which 27's labels do not. FreeCAD 2026.09 (`/opt/fc`, BIM) also has
`ArchStairs` (stairs + railings), `ArchFence`, `ArchCovering` (cladding / tile patterns on faces)
and `ArchTruss` / `ArchFrame`. **ArchCovering tried and dropped (2026-10-02):** headless it lays
real 3D tiles (10x8 ft wall: lap 1 s, brick 480 solids 11 s), but every tile is a flat box --
no tilted overlapping lap boards, no tapered shingles -- so line work equals our 2D patterns and
the only gain is a few-px joint shadow; ~1 min more FreeCAD per brick sheet. If shadow relief is
ever needed, build tilted lap boards with Part directly. The rest untried; `Shape.makeParallelProjection` would give exact shadows
of any solid we add.
Screen values (judgment, not tuned; measured on 300 Revit elevations): `--roof-lines 0.3` (30%
of sheets), `--railings 0.15` (per view: 15% of views, 27% of sheets -- 0.3 gave 48%, more than
real sheets show), `--faint-lines 0.2` (line-only sheets only: 18% of them, 10% of all),
`--soft-shadows 0.5` (48%; the two-level grey shade on shaded sheets: 19%). Arms
`revit10kcurroi` / `revit10kfailroi` in `run_revit_2k.sh`.
12's thin soffit band is the ceiling sheet type added 2026-10-01 (`--plan-v2 1`), after this model.
Correction to 2026-09-30 below: the 25 / 27 failures were read there as "not paint". The direct
cause is the horizontal-roof / vertical-siding pair above; paint is still untested at the
2048 GPU recipe. None of these flags is trained yet.

## HF14 25 / 27: shingle roof vs unpainted vertical boards (2026-10-04, CPU)

The best model (`synpre-realmix-lr5e5-e22`, hand boxes) trails shipped only on 25 / 27, and only on
three questions: 27 q3 0.48 vs 0.94, 25 q3 0.73 vs 0.91, 27 q4 0.90 vs 0.98 (~+0.014 HF14 if fixed).
In each it over-selects: with the reference on the shingle-roof band it also takes the unpainted
vertical boards.
- Not colour: re-scored in greyscale, shipped keeps 0.96 / 0.95 and the best model is unchanged.
- ~~Inherited from synthetic pretraining~~ **Wrong (corrected 2026-10-06): the real-mix fine-tune
  causes it.** The best model's own synthetic start (`longhardfail-swa16-20`, hand boxes, 4096) gets
  27 q3 0.906, 25 q3 0.949, 25 q1 0.899 (best: 0.479 / 0.725 / 0.538; shipped 0.944 / 0.907 /
  0.514). The 10k synthetic model (0.51) and v6d-only (0.307) fail, the long mined one does not;
  both fine-tunes (5e-5 and 2e-4 `synpre-realmix-swa24-26`, 0.455) lose it. Same as the GH200 note
  "the fine-tune trades synthetic wins on line-only markup for real ones".
- The generators make similar-looking pairs as often as the real sheets (look-alike descriptor from
  `question_difficulty.py`, share of questions >= 0.95: real 28 12%, v6d 18%, Revit 14%; >= 0.98:
  8% / 7% / 2%, Revit's `distinct_looks`). The best model scores 0.92 on the 18 real questions >= 0.95;
  the four in 0.90-0.95 are exactly this pair (0.59). What is missing is the drafting convention
  (staggered shakes next to uneven-spaced boards, both uncoloured): cedar shingle + vertical/bb on
  the same sheet in 2 / 80 Revit sheets, both coloured; 0 / 80 on line-only sheets.
  (So the pair is rare in the pools, yet the long mined synthetic model learned it; the lever is
  keeping it through the fine-tune, not -- or not only -- more synthetic.)

## Tiny reference boxes in training and in the mined pool (2026-10-04, CPU)

Training draws its reference like the evaluator: a family, then any one of its instances, then a box
fully inside that instance (`--ref-min-side` 0 = slivers allowed). Mining (`build_hard_pool.py`)
ranks sheets by automatic-box IoU and filters only on target resolution, never on the box. Measured
on 399 regenerated fresh sheets (ids 100000-100399, the `revitfail` flags; 98 of them mined):

| | questions / sheet | eval box side @2048 median | < 16 px | < 32 px | training draw, largest box that fits < 32 px |
|---|---:|---:|---:|---:|---:|
| fresh, mined | 14.2 | 42 | 17% | 40% | 24% |
| fresh, not mined | 7.9 | 57 | 10% | 27% | 12% |
| real 28, automatic boxes | 4.6 | 85 | 0% | 5% | |
| real 28, hand boxes | 4.4 | 76 | 0% | 5% | |
| Gemini 20, automatic boxes | 7.8 | 107 | 1% | 13% | |

Revit sheets ask about slivers far more than real ones (window-cut pieces), and mining doubled
that: part of what it picked as "hard" is unanswerable tiny references. Untested whether it hurt
(mining +0.037 at 2k / 9 epochs, null at 30). Fix candidates: a box-size filter in mining
(drop questions whose box < 32 px @2048 from hardness), and `--ref-min-side` in training.

## Best model on 20 Gemini plans it trained on (2026-10-04, CPU)

20 random plans from Gemini r2-r4 (seed 20261004), automatic boxes (no hand boxes exist), 4096:
**mean 0.809 over 156 questions** (sheet-mean 0.797); 44% of questions >= 0.90, 13% < 0.60; 6 of
20 sheets >= 0.90. Worst questions are about half unrealistic automatic boxes (on a porch railing,
a post, a sliver) and half real confusions: same texture in two colours that the labels treat as two
families (17), roof vs wall (0), a plan floor vs hatching (4). Not near 0.90, even on training data.

## Label cuts vs the hand labels (2026-10-04, CPU)

Every element the Revit renderer draws over a wall, checked against the Gemini labels (~10 plans)
and real 02 / 03 / 05 / 08-10 / 13 / 14 / 27 at full resolution. The hand-label rule: a wall label
runs on behind anything standing in front of it with no pattern of its own; flat trim bands and
openings are left out.

| element | Revit before | hand labels | now |
|---|---|---|---|
| porch posts | cut (`--trim-cut`) | wall continues (Gemini 15, #60; real 02) | **kept** |
| railing / deck posts, newel | cut (`--railings`) | wall continues (real 27) | **kept** |
| stair stringer | cut | wall continues (real 27) | cut (edge case, left) |
| windows / doors / garage + casing, head | cut | cut | cut |
| belts, porch beam, eave fascia, rake, deck rim | cut | cut (Gemini #10 / #45 / #60, real 27) | cut |
| porch roof | its own roof label | same | same |
| pickets, handrails, treads, shadows, leaders / text, jutting sills | kept | kept | kept |
| downspouts / gutters, porch furniture, wall lights | not drawn | wall label runs over them (real 02 / 03 / 05) | `--downspouts P` (opt-in) |

Nothing found that Revit keeps and the hand labels cut. 40-sheet check (`--railings 0.5`): 28 / 40
sheets gain label (319 tall strips, median 5:1); the only removed pixels are <= 3 px polygon-
simplification jitter on merged shapes. Where a post stands in front of an opening or belt that part
stays unlabelled. `--downspouts P`: per visible block wall end, with probability P, a pipe from just
under the wall top to grade with a kick-out shoe, skipped within 1 ft of an opening, drawn under the
shadows and the railing (so it shades with the wall), never cut from the label. Not checked: roof
plans (ridges, hips, vents, skylights) and Revit floor plans.

## Roof and floor plan label cuts vs the hand labels (2026-10-04, CPU)

Rule (a contractor's takeoff): label where the material is actually installed -- holes where it
is not (openings, skylights, chimneys, fixed cabinets / tubs), continuous behind things standing
in front of it (posts, railings, downspouts, freestanding furniture, rugs).
- Roof plans: grid lines, text, slope arrows, gutters, walls-below stay inside the label (both).
  Skylights: synthetic and Gemini cut them (Gemini also cuts vents); real 1 / 16 (HF14) and 11
  (val) keep them inside the roof label -- by the takeoff rule the real labels are the outliers
  (small areas; not changed). Flat roofs: synthetic labels 60%; real 15 / 16 leave them out,
  Gemini sometimes labels them as their own family.
- Floor plans: real 0 labels patio paving with the patio furniture inside; 12 the soffit band;
  7 the under-floor area with its text. Synthetic: wall-fixed pieces cut, freestanding inside (as
  Gemini). Furniture boxes: see defaults above (`ref_exclude`).

## Do the real-mix labels teach the 25 / 27 loss? (2026-10-06, CPU)

Checked every label overlay of the fine-tune's real 86, Gemini r2-r4 168 and generated 28.
The generated 28 are commercial elevations, wall sections, details and floor plans: no 25 / 27-type
sheet, no shingle / board pair; many labels are thin slivers or detail layers.
- No label teaches it: different materials are separate families wherever shingle siding and
  vertical boards share a sheet (Gemini #151, #161, #165, #97, #129); same-looking shingles (roof +
  siding) share one (#151), consistent with the rest.
- No 25 / 27-type sheet exists in either set (colour markup with unpainted line textures). The
  fine-tune never shows the case, so the loss is most likely drift away from the synthetic start,
  not a taught error. Levers: probability-average the synthetic start with the fine-tune, or a
  gentler fine-tune (larger synthetic share, fewer epochs).
- Partial labelling is common: ~half of real 86 label only the roof and leave patterned walls
  unlabelled (#33, #35, #39, #49, #57-66, #73-79); Gemini does it too. Those walls become negatives.
- **Reference boxes in the real / Gemini training data are often unrealistic** (user's call): the
  sampler takes any box inside a labelled piece, and these hand labels include slivers, coarse
  polygons over doors / posts / railings (Gemini 8 q2, 15 q1) and partial labels. Unlike synthetic
  pools, they have no `ref_exclude`, so none of the synthetic fixes reach them. Not yet measured
  or fixed; candidates: `--ref-min-side` for real / Gemini records, or hand-reviewed reference
  regions for the 254 training sheets.

## Generator flags (`generate_synthetic_v6.py`, all default-off, v6d byte-identical when off)

| flag | status |
|---|---|
| `--hardscape-plan`, `--same-fill-subtle` | null, 3 seeds |
| `--same-fill-new-colour` | negative on the real mix (band worse) |
| `--mottle` | null, one seed |
| `--vocab2` | untested (`run_vocab2_screen.sh` ready) |
| `--gemini-colour`, `--muted-palette` | probe only; r8 uses `--neutral-palette` instead |
| **r8 set:** `--val-details --material-mix --val-fills --mode-weights 86,7,7 --res-degrade 0.15 --neutral-palette --fill-scale --real-labelling` | **negative**: HF14 -0.076, val -0.008 (2k @2048; trained WITHOUT `--mode-weights`, i.e. the `R8` dict in `generate_synthetic_fc.py`) |
| `--real-labelling` alone (on Revit sheets) | **negative on validation**: -0.187 vs Revit, 89% of it val 17 (unlabels the foundation band that val asks about); HF14 -0.008 |
| `--tight-crop` | framing fix (ink box + pad); untested |

FreeCAD (`scripts/generate_synthetic_fc.py`, `scripts/fc_massing.py`,
`scripts/revit_render.py`), all 2k @2048, seed 7, epoch 8, `--mode-weights
66,16,18` (v6's mix):

| pool | val | HF14 |
|---|---:|---:|
| v6d (control) | 0.6702 | 0.7066 |
| `--revit` | **0.7568** | **0.7564** |
| no renderer (v6 drawing on FreeCAD houses) | 0.6248 | 0.7126 |
| `--revit --real-labelling` | 0.5702 | 0.7485 |
| `--r8 --revit` | 0.6891 | 0.6713 |

Revit model on its own fresh plans (HF14 protocol) 0.758 = its HF14 0.756; it
scores 0.507 on fresh v6d, the v6d model 0.724 on fresh Revit.

`--revit --revit-plans` (2026-09-28, `scripts/revit_plans.py`; untrained): roof
and floor plans drawn Revit-style too, replacing the v6 plan sheets that are 34%
of the Revit pool (the only sheet types where Revit lost to v6d). Roof plans
take the roof faces of the FreeCAD model seen from above: all pitched surfaces
one family (flat roofs labelled only on all-flat houses or sometimes when
patterned), hatch parallel to each eave / global fine shingle / diagonal
crosshatch / dark fill / sun-shaded greys, grid bubbles over the roof, section
heads, slope arrows, ridge text, trellis slats, 1-3 building copies. Floor plans
label exterior hardscape (flagstone / ashlar / deck boards / pavers / basketweave;
20% two families) and 30% an interior finish region, over poche walls, drop
shadows, textured floors, rugs, room tags, tick dimension chains, MEP overlay.
Off = byte-identical `--revit` pool (checked, 32/32 files). 200 sheets in 56 s on
4 CPUs, 0 failures. Pool for the screen: `--n 2000 --seed 6 --start 0
--mode-weights 66,16,18 --revit --revit-plans` (same ids as `revit_train2000`,
so only the plan sheets differ).

2026-09-29 plan rework (all `--revit-plans`, untrained): roofs always textured
(from the 14 Gemini + 4 real roof plans), skylights / piers always holes;
floor plans rewritten from ~40 Gemini + 3 real (finish plans by room group
through doorways, rendered hardscape plans, under-floor plans), shaped
footprints and rooms (notches, chamfers, bays, U, corridors, L-rooms, carved
closets); FreeCAD TechDraw hatch library as tiles (`scripts/hatch_tiles/`);
rule: two families of one pattern type differ in spacing (>= 1.4x, effective
line spacing for tiles) or tone, never by direction alone (0 violations on
300 floor + 300 roof sheets).

`--shaped` (2026-09-29, untrained; needs `--revit`): 3D houses beyond boxes. The main
block becomes an L (notch, split into two fused rect blocks: real valleys), a U
(rear courtyard, three blocks) or a chamfered polygon, plus 25% a 45-degree bay.
Convex polygon blocks are roofed as the lower envelope of the edges' slope planes
(prism intersected with half-space boxes, `fc_massing.py poly_block_solids`); FreeCAD's
ArchRoof was tried and rejected (needs a correct per-edge run; hips on L / bay
outlines fail). Pieces keep `parent` so elevation zoning uses the right v6 block.
Mix on 400 houses: 136 chamfer, 89 L, 85 bay, 16 U, 124 plain; 0 FreeCAD failures,
0 roof-plan gaps / overlaps; off = byte-identical (--revit and --revit-plans, 80/80
files each). Floor plans still use their own 2D shaping, not the 3D spec.

Window holes (2026-09-29): wall labels now ALWAYS exclude windows / doors
(`WINDOW_HOLE_PROB` 0.8 -> 1.0 in v6, inherited by the FreeCAD / Revit generator;
`--window-hole-prob`). All 7 real eval sheets that label walls cut every opening;
Gemini / real-pool labels cut them with `remove` polygons (106 of 165 elevation-type
images use them; the rest trace around openings or don't label walls). The old 0.8
left windows inside the wall label on 1 elevation sheet in 5 (~210 sheets, ~3% of the
shipped mix) with no evidence behind it. Recorded pools pin `--window-hole-prob 0.8`
(run_best_mix.sh, run_fit_100k.sh, run_synth_v6.sh) and reproduce byte-identically;
with 1.0, 16/120 v6 label files change (2/120 images: highlight / cloud markup picks
a labelled piece), revit 5/60 labels, 0 images.
The hole also covers each window/door casing and head as one rectangle since
2026-09-29 (`--casing-holes 1`, default; the sill/head ears that jut past the casing
stay in the wall, as in the hand-labelled Gemini sheets). Before, casing, head and sill
(painted in trim colour) sat inside the wall label: ~15% of revit main-wall label area,
on 91% of those labels; real eval sheets leave casings out. `--casing-holes 0` +
`--window-hole-prob 0.8` reproduce the old pools byte-identically (60/60 revit, 60/60 v6);
the new default changes 49/60 revit labels, no images.
Trim cut (2026-09-29, `--trim-cut 1` default): white trim (eave fascia, belts, corner
boards, chimney cap, porch fascia) is cut out of every surface label and slivers < ~3" are
dropped; the porch roof hides the wall belt/fascia behind it (they were painted across it);
a "gable" accent on an all-hip/flat house becomes "upper"/"wainscot" (it only zoned the strip
under the eave, which the fascia covered: a label of pure white trim). Revit, 80 sheets: labels
with >5% of their area on trim 173 -> 42 (the rest are 1-2 px edges on thin labels), trim share
of label pixels 1.9% -> 0.65%. v6: labelled trim also inside the main-wall label on 81/81
sheets (median 27% of the trim label) -> 42/81 (median 3%). Images change on 22/60 revit and
69/300 v6 sheets (porch belt, accent remap, markup/crop RNG after label count changes).
`--trim-cut 0` (with the two flags above) reproduces the old pools byte-identically.
Distinct looks (2026-09-29, `--distinct-looks 1` default, Revit elevations): a family with no
pattern (flat colour: TPO roofs, plain walls/chimneys on colour sheets) is drawn but never
labelled (130 sheets: 13 labelled plain families -> 0); two families may not share a look class
(vertical boards = standing seam) with spacing within 1.4x and colour within CIELAB dE 8 -- the
lower-priority one is re-drawn with another kind from its own RNG (look-alike labelled pairs
2 -> 0; a light grey vs cream lap pair, dE ~10, is a real difference and stays). 18/150 sheets
change. Slivers are left in the labels on purpose (selecting the wall must still segment
them); training can keep them out of the REFERENCE instead: `train_refunet.py --ref-min-side 24`
draws references only from instances that fit a 24 px square when the family has one (Revit
pool: sliver references 4.0% -> 2.0%, the rest are families with only thin pieces). Off by
default. Gemini labels trace the middle of the outline (half the line inside), as Revit does.

## Open

- **ROI replace mode on the 2k screen** (`EXTRA="--roi-ref --roi-ref-mode replace"`,
  KEEP `--domain-random`), paired vs `revit2k-roi-e8`: ROI add beat no-ROI on
  line-only look-alikes (0.606 -> 0.695); replace, or feeding the box's true size,
  is the next step on that axis.
- **Line-only look-alikes are the synthetic weak spot** (fresh 10k: 0.77, 17% fail;
  colour look-alikes with dE >= 10 are solved, 0.94). On real, the shipped model is
  0.96-0.99 on line-only sheets 13/14 where synthetic-only is 0.53-0.95.
- **Hard negatives:** `--distinct-looks` (default) redraws look-alike pairs; real
  failures are look-alike regions next to the target (sheet 0 patio vs interior
  floors, 25/27 roof vs lap siding, 18 grey roof vs grey foundation). Untested arm:
  the new pool with `--distinct-looks 0`.
- **Revit mix, other ratios / ROI from stage 1:** only the 1,600-sheet swap with a
  no-ROI stage 1 was run; the 10k pool and ROI-from-the-start are untried in the mix.
- **The ~0.74 grouping wall is not mainly step budget** (2026-09-27): one
  fresh 9-epoch cosine on the six-question model (2x steps, epoch 17,
  `abshetty/floz-refunet-swint-v6d2k-sixq-restart-e17`) moved own-plan fit
  0.741 -> 0.765 and fresh 0.727 -> 0.745, against a 0.904 ceiling; val
  0.688 -> 0.705, HF14 0.719 -> 0.707 (noise). Next suspect: conditioning /
  capacity. Untested: `--swin-decoder selfattn` (code in, CPU-checked; in the
  frozen screen it only TIED baseline at the best LR). Revit + six-question
  ran for 4 epochs only (see "Best synthetic-only"); a full 1+8 run is
  `./run_multiref.sh 7 revit`.
- `--vocab2` screen.

## Log

### 2026-09-26 — realism probe, r8, FreeCAD, Revit (detail: archive Part 1)

- Probe v6d vs Gemini: 0.999. Candidate references are all far apart (eval 28
  vs scraped 86: 0.965 at 640; eval vs Gemini 0.954), so val 14 became the
  reference. It looks like modern CAD/BIM exports with colour markup.
- Flag rounds against val 14 (3-seed means): v6d 0.948, r2 0.919, r4 0.932,
  r6 0.964, r7 0.945, **r8 0.909**. The biggest single gain came from labelling
  like real annotators (`--real-labelling`). Blurring crops to 28 px only
  takes r4 from 0.93 to 0.86, so the gap is coarse content.
- `--tight-crop` came from the user spotting white space: val drawings fill
  92% of the image, v6 62%.
- Boise public-domain ADU sets (`data/reference/boise_adu/`, 6 bid sets): the
  six A201 sheets (elevations plus sections, one firm) score 0.998 against val
  14. Another real document family is trivially separable, so AUC < 0.7 would
  mean overfitting to the eval documents.
- FreeCAD geometry on r8's houses: 0.954. Geometry is correct but the probe
  looks at local rendering. Revit-style renderer (cast shadows from the solids,
  line-weight hierarchy, level datums, view titles): 0.960. Looks like a Revit
  set, but val 14's most typical crops are hand-drafted shingle and colour
  markup, not Revit. The user prefers the Revit look; train on it as a mix arm
  and a Revit-only arm.
- FreeCAD install (container-local, ~5 min): micromamba from
  `conda.anaconda.org/conda-forge`, then
  `micromamba create -p /opt/fc -c conda-forge freecad`.

### 2026-09-27 — Revit beats v6d; 1024 screens mislead; the grouping wall

One seed throughout, synth-only, swin_t, 2048, 1+8 epochs, epoch 8 scored.
Numbers are in "Current state", "How to test" and "Generator flags" above;
drivers: `run_res_calibration.sh` (+ `scripts/res_calibration_report.py`),
`run_revit_2k.sh [seed] [arms]`, `run_v6d_scaling.sh`, `run_multiref.sh [seed]
[v6d|revit]` (env `DEC`, `RESTART_FROM`, `TAG`, `BS`).

- Batch size: swin_t @2048 saturates the GH200 at batch 4 (17.7 / 18.0 /
  18.1 img/s at 4 / 8 / 10; 8 peaks 67 GiB on a full 2048 canvas, 16 OOMs).
  `--compile` never finished compiling in 15 min (`--domain-random` shapes).
- `--refs-per-image K` (dataset + `RefSwinUNet.forward_multi`): backbone once
  per plan, decoder per question; 26 questions/s vs 18; 3 plans x 6 questions
  ~66 GiB worst case. Default 0 is byte-identical (checked against HEAD).
- `--swin-decoder selfattn`: `CrossAttnCondition` (moved to
  `refmask2former/attn_condition.py`) at 1/16 and 1/32; published baseline
  checkpoints still load strictly.


### 2026-09-28 — the grouping wall was the reference and the shrink; image-box reference on real: 0.8440

One seed throughout (seed 7), swin_t, 2048 training. Drivers: `run_overfit64.sh`,
`run_multiref.sh` (env `NODR`, `EXTRA`), `run_restart_swa.sh` (env `EXTRA`,
`NODR`, `CMP`, `SEED`); scorers `scripts/fit_diagnose.py` (per-question error
decomposition), `scripts/clean_loss.py` (un-augmented loss, comparable across
DR / no-DR runs), `scripts/export_reference_boxes.py` (the 28 real sheets with
the evaluator's exact boxes).

**Not data, not steps.** The six-question Revit model fits its own plans no
better than fresh ones (0.785 vs 0.789, ceiling 0.960). Memorising 64 Revit
plans (60 epochs, 2 plans x 6 questions) reaches only 0.799 (ceiling 0.957).
Backbone LR x10: 0.719. `--swin-decoder selfattn`: 0.762 (worse; it had never
run at 2048 because `attn_condition.py` fed non-contiguous tensors to SDPA,
forcing the 27 GiB math kernel -- fixed with `.contiguous()`, 2 GiB).

**Two causes, 64-plan overfit (IoU / thin<40px / wrong regions per q / missed per q):**

| arm | IoU | thin | wrong | missed |
|---|---:|---:|---:|---:|
| base | 0.799 | 0.715 | 0.35 | 0.35 |
| `--roi-ref` (prototype pooled from the IMAGE's features in the box) | 0.822 | 0.764 | 0.12 | 0.54 |
| `--roi-ref-mode add` (crop prototype + zero-init ROI projection) | 0.841 | 0.784 | 0.17 | 0.33 |
| no `--domain-random` | 0.865 | 0.805 | 0.27 | 0.31 |
| roi replace + no DR | **0.901** | 0.859 | 0.09 | 0.24 |
| roi add + no DR | 0.895 | 0.842 | 0.14 | 0.18 |

1. The 224 px crop is magnified ~5x relative to the plan and encoded
   separately, so reference and targets meet at different scales; colour/tone
   is the only scale-free cue and the model matches on it (dark tile ->
   dark walls). Pooling the prototype from the image's own features fixes
   wrong-family picks.
2. `--domain-random`'s 0.4-1.0x shrink makes thin targets thinner in training.

**At 2k (six-question, ROI add, no DR, 1+8 epochs):** own plans 0.786 ->
**0.887**, fresh 0.789 -> 0.872; clean loss 0.268 -> 0.108 (train), 0.255 ->
0.071 (held-out). But val 0.7816 -> 0.718 (paired -0.063, p=0.013; 55% one
sheet) and HF14 0.7607 -> 0.743 (noise). **All of it is small user boxes**
(side at 2048 input): <32 px -0.44, 32-64 -0.088, >=64 +0.01. A sub-cell box
pools background at 1/16-1/32, and without DR synthetic training has no small
boxes. `--roi-min-cells` (coverage-gated ROI term) is written, untested.

**Small boxes are answerable -- do not reject them in the UI.** The shipped
0.8170 model scores 0.827 on the 21 questions under 50 px (val17 q13, a flat
lavender 34 px patch: 0.810); only val17 q00 fails every model. Its remaining
loss is over-selection instead: 8 HF14 questions predicting >1.3x the target
hold 48% of it (val: 9 questions, 51%; val 17 alone is 72% of val's loss).

**On the real mix: HF14 0.8440** (+0.0269 paired vs the shipped 0.8170,
p=0.017, 16 better / 6 worse / 30 tied; val 0.8150 vs 0.808). Recipe = the
shipped restart (`run_restart_swa.sh` from `mixr4-e8`, DR on, one question per
plan) + `--roi-ref --roi-ref-mode add`; validation chose swa_12-15 @4096. No
small-box regression (<50 px 0.900 -> 0.897); gains on 50-100 px (+0.030) and
>=100 px (+0.030) boxes; the shipped model's over-selected questions 0.425 ->
0.504. Val 17 untouched. ONE SEED -- the seed-31 replicate queued in
`logs/queue6.sh` was dropped (no multi-seed runs at ~0.85, user's call); the
no-DR mix queued before it ran (below).

Control reproduced exactly on this machine: `mixr4-restart-e13` @4096 = 0.8170.
Rebuilt `v6dmix_plus_r4` = 6,676 records (recorded 6,671; augmentation is
chunked, not byte-identical).

**Domain-random on real (mix, ROI add, restart from `mixr4-e8`, seed 7):** without
DR the val-selected checkpoint scores **HF14 0.8583** (e10 @4096, val 0.8547;
+0.041 paired vs 0.8170, p=0.060, 19 better / 14 worse) -- but e10 is a val-17
spike (sheet 17: 0.51 -> 0.79 -> 0.49 over e9-e11; 27 of 77 val questions).
Without sheet 17, DR beats no-DR at all 12 candidates (+0.01 to +0.04) and
no-DR degrades over the restart (0.913 e8 -> 0.875 e14) while DR holds ~0.92.
**Keep `--domain-random`.** Record 0.8583 as a val-17 lottery, not a recipe.

**Box-size probe (`scripts/box_size_probe.py`, 0.8440 model, @4096, CPU):**
every question it scores < 0.6 (9 val + 6 HF14: val 17's band x10, HF14 0
paving vs floor tile, HF14 18 roof vs band, HF14 12, 25), re-asked with
larger boxes inside the SAME instance. Mean IoU: evaluator box 0.277, largest
square <= 512 px 0.278, largest uncapped 0.276, square crop + ROI box
elongated along the region (<= 4x, the thin rectangle a user draws on a band)
0.278. Over-selection stays 2-8x the target. **The remaining failures are not
reference-information limited: the model cannot separate these look-alike
materials even from the largest box a user could draw.** No inference-time box
trick or UI guidance reaches 0.90; the fix has to be learned (data/training on
look-alike pairs).

**Six-question training on the real mix: negative.** Restart from the 0.8440
checkpoint (same mix, DR, ROI add, `--refs-per-image 6 --batch-size 3`, 8
epochs, `data/runs/ck_mix_sixq_from844`): val swings 0.753-0.843 per epoch
with no trend; val-selected e18 @4096 -> **HF14 0.8165**, -0.027 paired vs
0.8440 (p=0.050), 7 better / 24 worse (sign p=0.003) -- broad, not one sheet.
The six-question fit gain on synthetic does not carry to real. Best stays
0.8440 (`data/runs/ck_mix_roiadd_dr/swa_12-15.pth`).

### 2026-09-28 — where Revit beats v6d, per question (CPU re-score)

Published `swint-revit2k-e8` vs `swint-v6d-e8`, 2048, re-scored on CPU
(reproduces 0.7566 / 0.7078 HF14, 0.7574 / 0.6719 val).
`scripts/per_question_breakdown.py` with hand-tagged sheet types.

| sheet type (HF14 + val questions) | n | v6d | Revit | delta | better / worse |
|---|---:|---:|---:|---:|---|
| line-only CAD elevations (8-10, 13, 14) | 34 | 0.57 | 0.70 | **+0.13** | 30 / 3 |
| colour-markup elevations (17-27) | 64 | 0.71 | 0.78 | +0.07 | 36 / 7 |
| shaded Revit elevations (2-6) | 17 | 0.80 | 0.85 | +0.05 | 12 / 2 |
| floor plans (0, 7, 11, 12) | 8 | 0.56 | 0.56 | 0.00 | 3 / 3 |
| roof plans (1, 15, 16) | 6 | 0.92 | 0.81 | **-0.11** | 0 / 4 |

- The gain is in questions v6d OVER-selects (>1.3x the target): +0.07 HF14,
  +0.19 val (32 better / 6 worse), about half and 70% of the delta. Revit
  reduces the look-alike over-selection that is the shipped model's main loss.
- Small boxes gain too: 32-64 px +0.10 HF14 / +0.12 val.
- Multi-family sheets gain; single-family sheets are flat (-0.01 / -0.02).
- Val 17 is 53% of the val gain (0.570 -> 0.699, 23 / 0 better/worse).
- Revit loses only on plans: roof plans (4 of 6 worse) and floor plan HF14
  12 (0.44 -> 0.36). Those pool sheets are still v6-drawn (34% of the pool,
  same as v6d), so the loss is elevation-heavy training, not worse plan data.
  Generator next: Revit-style roof and floor plans from the same houses.


### 2026-09-29/30 -- Revit plans, ref-jitter fix, ROI screens, 10k, Revit in the mix

One seed (7) throughout, swin_t @2048, `run_revit_2k.sh` (env `EXTRA`, `P2`) unless
noted. Pools: `generate_synthetic_fc.py --revit --revit-plans --shaped --mode-weights
66,16,18 --seed 6` (ids 0-1999 / 0-9999; fresh 200000+), default 09-29 label rules.

**The new pool (plans + shaped + label fixes) vs old Revit, synth-only, no ROI:** HF14
0.7721 vs 0.7564 published (0.7425 retrained here); plans recover (roof 0.81 -> 0.96,
floor 0.56 -> 0.68 on real, 5/0 and 7/0). Ablation (plans only / shaped+labels only)
puts the plan gain on `--revit-plans`. The apparent colour-markup "regression" was
val 17 noise: the old pool retrained HERE scores val 17 brick 0.08 at e4-e6 (paints
the roof) and 0.39 at e7-e8; the published 0.699 was a lucky draw. Read val 17 over
epochs 4-8, never one epoch.

**Reference jitter (728def3, bbf5259 for --realism-aug):** new pool, fix vs no fix:
HF14 +0.020 / +0.035 @2048 / 4096 (n.s.), line-only real elevations 0.761 -> 0.864
@4096 (21 better / 3 worse).

**ROI on the screen (add mode):** new pool + fix + ROI = `revit2k-roi-e8`, HF14 0.7855 /
0.7970. **Colour pairs** (`--colour-pairs`, off by default): 0.5 gave HF14 +0.032
@4096 (p=0.01) and val 17 brick 0.46, but 1.0 did not reproduce it -- run noise,
not adopted.

**10k pool + ROI, 1+5 epochs** (`revit10k-roi-e5`): vs 2k ROI val +0.069 / +0.128
(p<0.001, val 17 brick 0.10 -> 0.89 @4096), HF14 +0.001 / +0.016 (n.s.). Fresh unseen
plans 0.810 -> 0.872 (1,901 better / 404 worse). Volume transfers to val's sheets,
not HF14's (as v6d 8k/16k).

**Where the remaining error is (fresh plans, 10k @4096):** the top-similarity quarter
(greyscale descriptor >= 0.908, `question_difficulty.py`) scores 0.78 (15% < 0.5).
Split by sheet appearance: line-only 0.766 (684 q), colour with pair dE < 10 0.657
(39), dE >= 10 0.942 (98, no failures). Line-only by pair type (styles from
`record_family_styles.py`): different kinds in one look class (lap vs asphalt rows)
309 q, no-ROI 0.606 / ROI 0.695 / 10k ROI 0.837; same kind near 1.4x spacing: only 2 q.

**Real HF14 error budget (shipped 0.8440 @4096):** 37 questions at 0.944 hold 25% of
the loss; 15 below 0.85 hold 75%; five sheets (18, 0, 12, 27, 25) ~75%. Over-selection
47%. Failure types: grey roof vs grey foundation (18), paver patio vs interior floors
(0), roof lines vs lap siding (25, 27), under-selected thin L band (12). The greyscale
look-alike score does not flag them: they are confusions with UNLABELLED regions or
colour differences.

**Revit in the real mix** (`run_revit_mix.sh` then `run_restart_swa.sh` with START=9:
the restart from a local epoch_8 numbers its epochs 9-16): v6d_1600 -> first 1,600 of
the Revit pool, 6,676 records, jitter fix on, stage 1 without ROI. Val chose restart
e10 @4096 (0.7843 vs shipped 0.8150); **HF14 0.8367 vs 0.8440** (-0.007, 12 better /
18 worse). Floor plans gain (HF14 12 +0.141, 0 +0.095); roof plans (val 15 -0.124,
HF14 1 -0.108), line-only (val 9/10/13) and rendered elevations lose. Published
`abshetty/floz-refunet-swint-revitmix-roiadd-e10` (private); not adopted.


### 2026-09-30 (CPU session, no GPU) -- error budget, markup mode, inference levers

The GPU VM was gone; everything below ran on a 4-core CPU container. Models pulled from
the Hub (`scripts/hf_ckpt_to_pth.py`, private repos download through the proxy) and
re-scored on all 28 real sheets with `--save-probs`, so ensembles, multi-scale and
thresholds are recomputed offline (`scripts/ensemble_selection.py`, reproduces the
evaluator to 4e-5). CPU fp32 reproduces the GPU numbers: shipped HF14 0.8444 / val
0.8156 @4096 (published 0.8440 / 0.8150); `revit10k-roi-e5` 0.8146 / 0.8870 (0.8126 /
0.8861). ~25 s per question at 4096, ~5 s at 2048.

**Where the best synthetic-only model (10k ROI @4096) loses HF14** (`per_question_breakdown`
sheet types; `scripts/failure_panels.py` draws them):

| HF14 sheet type | n | shipped | 10k | share of 10k's loss |
|---|---:|---:|---:|---:|
| colour-markup elevations (18, 23, 24, 25, 27) | 24 | 0.837 | 0.796 | 51% |
| floor plans (0, 7, 11, 12) | 8 | 0.661 | 0.741 | 21% |
| line-only elevation (14) | 9 | 0.919 | 0.816 | 17% |
| rendered elevations (2, 3) | 8 | 0.916 | 0.885 | 9% |
| roof plans (1, 16) | 3 | 0.977 | 0.963 | 1% |

The markup row is roughly proportional (24 of 52 questions = 46%), not an outsized
share, and it is three sheets: 18 (19% of the loss), 27 (18%), 25 (12%); 24 and 23 score
0.96 / 0.94. Line-only 14 (17%) and plan 12 (14%) lose as much as 27 or 25.
Over-selection is 43% of the loss. Worst questions: 25/27 merge the UNPAINTED vertical
siding with an unpainted roof drawn in horizontal lines (and the reverse) while the
green-painted lap siding stays separate; 18 q08/q09 small dark-grey roof pieces pick up
the red brick band; 14 (faint line-only) thin eave band and small chimney; 12 thin L band
under-selected (0.5x); 0 patio vs interior floors (10k 0.66 vs shipped 0.32).
**The markup-sheet failures are look-alike textures, not paint:** on 25/27 both confused
regions are unpainted and the painted siding is handled; on 18 two coursed textures of
similar darkness are confused, and the shipped model fails the same way (0.16 / 0.29).

**Markup sheets have no synthetic counterpart** (option built, NOT recommended -- see the
correction below). HF14 18-27 and val 17-26 are line
drawings with flat see-through colour painted on top. Their labels follow the TEXTURE,
not the paint (painted plain stucco walls are not questions on 20/23; unpainted vertical
siding is on 25/27; one colour can span two families). New `--markup P`
(`generate_synthetic_fc.py`, `revit_render.py`): P of the line-only Revit elevations get
paint (own RNG: family painted 75% main / 50% others, 15% shared colours, partial faces
30%, openings/trims usually skipped, multiply blend), plus grid paper (35%) and dashed
work-area boxes (40%). Labels unchanged (40/40 sheets); P=0 byte-identical to HEAD.

**Inference levers (10k ROI), chosen on val, not adopted:** threshold 0.25-0.55 flat
(val 0.885-0.887); multi-scale 2048+4096 best +0.0015 val (w=0.3), not read on HF14;
shipped + 10k ensemble (w=0.7 on val, 0.892 because val 17 flips) reads HF14 **0.8258**
vs 0.8444 (-0.019, 20/20, p=0.40): val and HF14 disagree.

Also: `generate_synthetic_v6.py --trim-label-prob` (v6 asks about blank trim on ~35% of
elevations; default 1.0 byte-identical); `run_revit_mix.sh SYN=v6fix` (the v6d slot
rebuilt with the label fixes and no trim questions); `record_family_styles.py`;
`evaluate_refunet_selection.py --local-pool` (HF14 protocol on a synthetic pool).

**Markup breaks the best synthetic-only model (fresh sheets, HF14 protocol, 2048).** 32
fresh Revit elevations (ids 300000+) rendered twice, identical drawings and labels, with
and without `--markup 1.0` paint: `revit10k-roi-e5` 0.881 -> **0.730** (366 questions,
187 worse / 32 better, p=3e-33; share below 0.5 3.3% -> 18.9%). So synthetic paint throws
the model off. **Correction (same day):** this was first written up as "the model groups by
paint, the 25/27 failure mode" and markup training was put at the top of the GPU runbook.
That does not follow: the HF14 markup-sheet failures are between UNPAINTED look-alike
textures (above), so there is no evidence that paint costs anything on HF14. Markup
training is dropped from the runbook; `--markup` stays in the code, off by default.

**CPU fine-tunes cannot answer it (inconclusive, not negative).** From `revit10k-roi-e5`,
one epoch of 156 steps, batch 2, 1024, ROI add, domain-random: (A) 317 mixed sheets,
`--markup 0.5`, lr 5e-5: HF14 @4096 0.8146 -> 0.8108 (13/16, p=0.15), markup sheets
0.796 -> 0.794, fresh painted 0.780 -> 0.777. (C/D) 317 elevation-only sheets, markup
1.0 vs the same unpainted, lr 1e-4: fresh painted (147 q) 0.773 vs 0.778 (p=0.63). Loss
fell ~1.0 -> ~0.1 in both, but nothing moved at 2048 inference: too few steps at the
wrong resolution. Needs the 2048 GPU recipe.

**The 10k model underfits multi-family sheets (seen vs fresh, 2048, HF14 protocol).** 32
sheets from its own training pool (ids 0-399) vs 32 fresh (ids 600000+), 8 per labelled-
family count: all 0.807 seen vs 0.868 fresh; 1 family 0.933 / 0.984, 2: 0.889 / 0.895,
3: 0.757 / 0.834, 4+: 0.794 / 0.864 (seen 4+: 17% of questions over-selected). Training
sheets score no better than fresh ones, so the limit is fit, not data: grouping several
families on one sheet. Width and `selfattn` lost before; swin_b only ever lost at 744
steps at 1024 (v6d). Untested at this scale: longer 10k training and swin_s / swin_b.
**Correction (2026-10-01, GPU check):** this is an evaluation artifact, not underfitting. On
the same 40 seen sheets, questions drawn the way TRAINING draws them (one family, a random
instance's box) score 0.948 (1.9% below 0.5; loss 0.076 vs logged 0.069). The evaluator
asks once per INSTANCE with that instance's box: Revit families split into many pieces
(>= 5 pieces: 64% of questions, 0.836), slivers give tiny boxes (29% under 32 px at input
vs 18% in training; under 12 px 0.56), and native-res scoring caps at 0.950. One question
per family gives 0.886. HF14 is not like that: median 1 piece per family, 7% of families
>= 5 pieces, no box under 12 px, 77% of boxes >= 64 px at 2048. So the synthetic per-
instance score is not a proxy for HF14, and the fit sweep's premise is gone.


### 2026-10-01 -- fit sweep Stage A, training to convergence, hand-placed boxes

One seed (7), swin_t unless noted, 2048, ROI add, batch 4. Pool: `generate_synthetic_fc.py
--revit --revit-plans --shaped --mode-weights 66,16,18 --seed 6`, ids 700000+ (3,000 drawn,
2,985 ok), the 500 sheets with >= 3 labelled families (`fitA_train`); fresh = 40 such sheets
from ids 800000+. Seen / fresh are per-instance questions at 2048 (`--local-pool`), so they
carry the sliver artifact above; read them as relative, not as HF14 proxies.

**Stage A (`run_fit_sweep.sh A`, 1+15 epochs, DR on).** Batch 8 OOMs swin_s at 2048, so
all three ran at batch 4 (`BS=4`). Seen / fresh: swin_t 0.739 / 0.786, swin_s 0.742 /
0.803, swin_b 0.754 / 0.799; per-epoch HF14 @2048 0.730 / 0.744 / 0.725. Train and val loss
were still falling at the last epoch (~2k steps): undertrained, so the backbone comparison
says nothing about capacity. swin_t at batch 8 (half the steps): 0.718 / 0.736.

**Training to convergence (`run_fit_converge.sh`, swin_t, DR off).** 1+60 epochs: seen 0.764
/ 0.820 / 0.853 / 0.862 at e15 / 30 / 45 / 60, fresh 0.797 / 0.847 / 0.864 / 0.878 -- fresh
climbs with seen, so this is learning, not memorising. Fresh-cosine restart from e60 for 30
epochs (`INIT=...`): **seen 0.880, fresh 0.890** at e90; over-selection ~20% -> ~4%. Ceilings
(`label_ceiling.py`, opt stride 4 @2048): seen 0.950, fresh 0.954. On 4+ family seen sheets
(0.873) half the remaining gap is that ceiling and most of the rest is boxes under 12 px at
input (0.55); boxes >= 32 px score 0.929.

**Restart arms from e60, 30 epochs each, paired:**

| arm | seen | fresh | val @4096 | HF14 @4096 (val-chosen size) |
|---|---:|---:|---:|---:|
| no DR (control) | 0.880 | 0.890 | 0.809 | 0.8220 |
| `--ref-sample instance` | 0.889 | 0.891 | 0.823 | 0.8302 (+0.008, 12/9, p=0.62) |
| `--domain-random --dr-scale-min 0.7` | 0.854 | 0.881 | **0.834** | 0.8258 (+0.004, 10/17, p=0.87) |

`--ref-sample instance` (new, default `family` byte-identical, checked on 30 hashed samples):
draws any labelled instance uniformly and asks for its family, with sheets weighted by
instance count -- the evaluator's question mix (boxes < 12 px 12% vs evaluator 10%; targets
in >= 5-piece families 66% vs 64%). It closes the seen/fresh gap but not HF14's, which has
~1 piece per family. All three arms beat `revit10k-roi-e5` on HF14 (0.8126) and lose to it
on val (0.886): 500 sheets at ~11k steps roughly match 10k sheets at ~7.5k. Real-sheet
loss of the no-DR model is on normal boxes (>= 64 px: 65%), 40% under-selection.

**Hand-placed boxes (`run_hand_box_evals.sh`, `eval_boxes/hand_v1.json`, @4096).**
Automatic boxes reproduce every published number exactly. Paired hand - auto on HF14 (51
shared questions): shipped +0.003 (0.8461), restart-e13 +0.016 (p=0.11), revitmix -0.007,
revit10k -0.005, revit2k +0.013, mixr4-e8 -0.003, the three fit arms -0.003 to +0.002. On
val (72 shared of 77): only revit2k moves (+0.009, p=0.033); the dropped questions alone move
some means (revit10k 0.886 -> 0.898 with paired +0.0003). Compare paired, not means.

Next: the long schedule with `--dr-scale-min 0.7` on more sheets (2k from scratch ~4.3 h,
or continue the 500-sheet e90 on 2k for ~30 epochs, ~1.5 h).

### 2026-10-02/03 -- failure options, hard mining, long schedule, synthetic pretraining: HF14 0.8733

One seed (7) throughout, swin_t, 2048 training, ROI add, HF14 / val at 4096 unless noted. Pools:
`generate_synthetic_fc.py --revit --revit-plans --shaped --mode-weights 66,16,18 --seed 6`.

**2k screen (`run_revit_2k.sh`, 1+8 epochs), paired vs `revitcurroi` (today's defaults, 0.7801):**
the four failure options (`revitfailroi`, `--roof-lines 0.3 --railings 0.15 --soft-shadows 0.5
--faint-lines 0.2`) 0.8051 (+0.025, p=0.09; targeted sheets 14/27/2 up, 18 down); hard mining
(`revithardroi`) 0.8335 (+0.053, p=0.0005 at e8; +0.037 averaged over e5-8); both
(`revithardfailroi`) 0.8156. The three treatments are indistinguishable from each other.
Every val loss in this table is val 17 alone (-4.7 summed IoU); without 17 all move with HF14.

**Hard mining (`scripts/build_hard_pool.py`, `run_hard_mine.sh`).** The screen model scores a
fresh pool (ids 100000+) with the HF14 protocol at 2048; questions under the naive resolution
ceiling (0.85) are dropped (0.4-2% of them on Revit); sheet hardness = mean over families of
(1 - family IoU); 1,000 sheets drawn from the hardness top 30% minus the hardest 1%, added to the
control pool's first 1,000. Mined sheets: 75% line-only elevations, failures 2/3 over-selection.
The mined model fits its own mined sheets no better than unseen band sheets (0.778 vs 0.779 one
question per family): underfit, not memorised.

**Long schedule (`run_hard_long.sh`: 30-epoch fresh cosine from the 2k e8, `--dr-scale-min 0.7`).**
Fit climbs with steps (seen-mined 0.770 -> 0.831, fresh 0.830 -> 0.860, e18 -> e38). HF14
single checkpoints 0.80-0.85. **At long schedule mining is null:** late-epoch-averaged (e29-38,
@2048 per-epoch diagnostic) hard - control = -0.005 +- 0.0075.

**Noise at this level (measured, 2048, last 10 epochs, LR ~0):** one checkpoint's HF14 sd 0.009-
0.024; difference of two single checkpoints sd 0.024, driven by ~5 flip-prone questions (sd
0.11-0.24 each, median question 0.034). Averaging 10 epochs' per-question scores brings the
difference se to 0.0075. HF14's 52-question sampling se (~0.017) is separate and irreducible.
Compare late-epoch averages or SWA weights, not single checkpoints.

**SWA of the long runs (14 candidates, 4096):** `ck_long_hardfail` swa16-20 HF14 **0.8668**
synthetic-only (best val without 17: 0.904); plain val prefers `ck_long_fail` swa33-38 (val 0.879,
HF14 0.8405). Published: `abshetty/floz-refunet-swint-longhardfail-swa16-20`, `...-longhardfail-e18`.

**Synthetic pretraining + real-mix fine-tune: HF14 0.8733.** `run_restart_swa.sh` from the
swa16-20 model on `data/mixed/realmix_revithard` (`run_build_realmix.sh`: revitfail_hard2000 + real
86 x18 + generated 28 x18 + Gemini r2-r4 168 x18 = 7,076). Validation is flat over restart e24-27
(0.882-0.890); SWA 24-26 is best on val under both rules (0.8964; 0.9228 without 17), HF14 read
once: **0.8733**, +0.029 paired vs shipped 0.8440 (re-scored here exactly; p=0.19), +0.007 vs
its synthetic init. Published: `abshetty/floz-refunet-swint-synpre-realmix-swa24-26`.
The fine-tune loses synthetic wins on line-only markup (27 q3 0.94 -> 0.51, 25 q1 0.90 -> 0.52)
while gaining 18 / 12 / 0; per-question best of {new, synthetic, shipped} would be 0.9125.

**Failure groups (best synthetic model, panels via `failure_panels.py`):** masonry base vs grey
shingle roof (val 17 brick, HF 18 roof; the shipped model fails it too); railing pickets over lap
(27 q1, only `--railings` moves it); vertical boards vs horizontal-line roof (25 q1, 27 q2; every
model 0.5-0.6); basketweave floor (7); thin soffit band (12). Steps fixed basketweave and val 17
brick (0.89-0.90 at long schedule); 18's roof and 27's boards survive every model.

**Also:** `generate_synthetic_fc.py` runs the FreeCAD massing on `--workers` processes (byte-
identical to serial on 48 sheets; massing 7 s -> 1 s on 16, 6,000 houses ~14 min -> seconds).
64-sheet memorisation with capacity arms (width 256, swin_s, swin_b) was queued and stopped for
the deadline: never run (`run_overfit64.sh` arms `roiaddnodr_*`, pool `revitfail_mined64`).
Only one 2048 training fits on the GPU; four 64-sheet no-DR arms together OOM.

**Fine-tune LR (same synthetic start and mix, validation-chosen, 4096):** 2e-4 SWA 24-26 0.8733,
**5e-5 e22 0.8816** (val 0.9009), 2e-5 e25 0.8739. Soup of the 2e-4 and 5e-5 models: val 0.9051
(best), HF14 0.8753. WiSE (synthetic <-> fine-tuned, alpha 0.5 / 0.7 / 0.85): val 0.895-0.898,
not read on HF14. Published: `abshetty/floz-refunet-swint-synpre-realmix-lr5e5-e22`.

**64 mined sheets memorised for 240 passes (`ck_overfit64_roiaddnodr_mined64_e240`, ROI add, no
DR):** IoU **0.906** vs ceiling 0.950 (60 passes: 0.843). Wrong-family selection 0.058 -> 0.006,
wrong regions / q 0.52 -> 0.00, missed regions / q 1.19 -> 0.17; what remains is boundary (fp_bg
0.028, fn 0.028). The architecture can learn the hard look-alike discrimination; grouping is a
step / coverage problem, the residual is edge precision (resolution). Width / selfattn /
corr-grid stay closed (all tested before, negative or parity).

**Why 18's teal brick base and grey roof merge (measured):** dE76 14.7 between their mean colours,
the same size as each region's own lightness sd (13.8-15.7); the roof is darker (L 39 vs 50) and
less saturated -- what a Revit shadow does to one family (`wall_shadows`, `--soft-shadows`'s
neutral grey), which training shows on about half of the shaded sheets. The 2026-09-24 recolour
probe already showed the model uses colour. Untested cleanly: the offline `--strong`
augmentation still grayscales 15% of every real / generated / Gemini copy (the one no-gray run
also changed the synthetic data). Next: `--gray-prob 0` mix + the 5e-5 fine-tune, paired vs 0.8816.

### 2026-10-06 (GH200) -- new-default pools, Swiss mining, best synthetic 0.8779, Gemini eval set

One seed (7), swin_t, 2048, ROI add; hand boxes @4096 unless noted. Pools `generate_synthetic_fc.py --revit
--revit-plans --shaped --mode-weights 66,16,18 --seed 6 --roof-lines 0.3 --railings 0.15 --soft-shadows 0.5
--faint-lines 0.2` at the 2026-10-04 defaults (`pool_manifests/*_v1004.json` pin the mined sheets).

| model | val | HF14 | Gemini30 hand |
|---|---:|---:|---:|
| 2k `revitfailroi` (new defaults), e8 | 0.8177 | 0.8060 | 0.7309 @2048 |
| 2k + Swiss floor plans (`--plan-source`, p 1) | 0.7611 | 0.8251 | 0.7162 @2048 |
| `longhardfail-swa16-20` (start, old pool) | 0.8787 | 0.8612 | -- |
| + 20 ep on new-default mined pool, SWA 27-31 | 0.8800 | 0.8564 | 0.7972 |
| + 20 ep on Swiss mined pool, SWA 37-41 | 0.8766 | **0.8779** | **0.8152** |
| its 5e-5 real-mix fine-tune (text-excluded, no gray), e54 | 0.9004 | 0.8578 | 0.8592* |

*Gemini sheets are in the fine-tune's training mix.
- Swiss vs no-Swiss 2k (pools differ only in floor plans): HF14 floor-plan questions 0.764 vs 0.697 @2048, the
  rest is run noise (val 17, unchanged elevations). Gemini30 (auto boxes) -0.012 to -0.018 on unchanged sheets:
  noise. Swiss layouts are not shown to hurt; adopted in the long pool.
- Gemini-like flags (`--tight-crop --pale-ink 0.25 --revit-view-weights 50,45,5`, measured: piece side @2048
  190 -> 245 px, Gemini 259; labelled share 0.22 -> 0.27, Gemini 0.28) on the Swiss 2k pool, SWA 5-8 both arms:
  Gemini -0.013 (p~0.08), HF14 +0.010, val +0.014 -- framing does not move Gemini.
- Fine-tune drift reproduced: loses 23/25/27; WiSE 30/50/70% toward the fine-tune: HF14 0.8843 / 0.8827 / 0.8715,
  val 0.8906 / 0.8943 / 0.8979 (val monotonic toward the fine-tune, can't select).
- Unrealistic reference boxes on Gemini (automatic sampler): 33 / 201 by Claude, user changed 42 (35 moved, 7
  dropped; 29 of Claude's 33). Text detection (`scripts/text_mask.py`, conf >= 60, >= 3 chars) catches 17 / 33
  with 1% of labelled area excluded; fixtures / windows / door swings / railings / ridges have no automatic fix
  (backbone-feature consistency AUC 0.58, model-failure AUC 0.68).
- Gemini failure modes of the best synthetic model (Gemini Failure Review artifact): r4 #078 tan vs olive lap
  (14% of loss), r3 #048 (11%), r4 #016 (10%), r4 #041 + #091 single-family sheets (12%; #091 was a label error,
  fixed upstream). The model matches line texture and under-uses fill tone / colour.
- **Same texture, different colour is rare in the pools (measured, 400 fresh sheets, long-pool flags + Swiss):**
  of 253 elevations, 24 (9.5%) have two labelled families of the same texture kind, and only 13 (5.1%) with a
  colour difference (CIELAB dE >= 10) -- and those are far apart (dE 32-114: roof vs wall seams, chimney vs
  wall stone). Subtle pairs like Gemini r4 #078 (tan vs olive lap, one sheet) are essentially absent; tone
  pairs on line-only sheets (grey-washed fill vs bare lines, r4 #041) were not measured. Next generator change:
  plant same-kind pairs at dE ~12-35 on colour sheets and wash-vs-bare tone pairs on line-only plans and
  elevations, both labelled as separate families; screen with Gemini30 hand boxes + HF14 + val.
- **Roboflow `floz-gen-gemini-r4` version 2** (generated 2026-10-06, no preprocessing / augmentation): only r4
  #091 changed (the user's fix: roof pieces `pattern2`, wall `pattern1`; was one family). Image byte-identical.
  `run_build_realmix.sh` now downloads v2 (`floz-gen-gemini-r4v2`, merged as `r234v2`, so v1 builds are never
  reused). The Gemini30 eval pool's #091 annotation was replaced by v2 (no hand boxes on it): best synthetic
  model 0.8152 -> **0.8255** on Gemini30 (#091's three questions 0.09-0.50 -> 0.94-0.95). Every Gemini30 number
  above is on v1 labels.


### 2026-10-07 (CPU) -- the user's notes on the Gemini30 failures

21 notes on the best synthetic model's Gemini30 questions (`eval_boxes/gemini30_notes_v1.json`; the review page is
claude.ai/artifact/W5FqgEoTubFgJZwwiGDHPb). What they say, grouped:
- **Over-selects patterns that are plainly different** (most of the loss): different colour (r4 #078 q4/q7, 0.21/0.26),
  different spacing (#078 pattern1; r3 #048 q5 roof in another colour and spacing), different texture (r3 #019 q5 shingles
  vs vertical lines, 0.27; r4 #016 q11; r4 #041 q0; r4 #098 q3 piping vs stripes), different direction at different
  spacing (r2 #029 q2 diagonal vs horizontal, 0.34).
- **Roof-plan habits leaking onto other sheets** (user's hypothesis, #029 / #046 / #016). Supported by the generator:
  `revit_plans.py` makes one roof family's lines turn with each facet's eave and treats eave lines vs seams
  (90 deg apart) as the same look; same-look roof families must differ in spacing or tone. Roof plans therefore teach
  that line direction is not a cue. Unchecked: whether any elevation / floor-plan pool has two families that differ
  only in direction (which would teach the opposite).
- **Isolated splotches** in unrelated places (r2 #060 door, r3 #023, dark brick). Dropping prediction blobs below 1%
  of the prediction's area: Gemini30 0.8244 -> 0.8298 (+0.005, 138 better / 45 worse; CPU probs, threshold picked on
  the same set). Visible but cheap in IoU: not where the gap is.
- **Narrow strips between regions selected, as if posts** (#098). **Door behind a balcony railing** (#098): check the
  railing exclusion rule against doors.
- **Box quality:** a box over text selects text (r2 #002 q1, product question: stop users boxing over text?); tiny
  or poor automatic boxes (r2 #028 q4, r2 #053 q10, r3 #019 q5); a box over an edge (r4 #050 q6).
- **Labels / Gemini:** r3 #048 q10 bottom right is a labelling error; r4 #016 / #041 Gemini may be inconsistent in
  spacing. Floor-plan furniture still confuses the model (r4 #060), and r4 #031 q5 half-selects a bathroom.

### 2026-10-07 (CPU) -- investigating the user's Gemini30 notes: why, and does the synth do it?

Best synthetic model `longswiss-swa37-41` @4096 throughout. Scripts: `scripts/probes/` (invariance probe sheets,
family-pair and within-family descriptors: line direction + spacing from FFT peaks, fill / ink CIELAB, a
direction-aligned gradient histogram for texture; validated on the probe sheets, blind to board vs board-and-batten).

**1. The model ignores direction completely, and spacing / subtle colour mostly** (`make_invariance_probe.py`: an
elevation with two wall panels labelled as two families that differ in ONE attribute; one box per panel; leak = share
of the other panel selected):

| differs only in | leak (ask A / ask B) |
|---|---|
| nothing (control) | 0.99 / 0.99 |
| direction: H vs V, H vs 45, V vs 45 | 1.00 / 1.00 (all three) |
| spacing 1.5x | 0.54 / 1.00 |
| spacing 2x | 0.00 / 0.99 (a sparse reference takes the dense panel too) |
| fill tan vs olive (dE 21) | 0.80 / 0.46 |
| fill tan vs blue (dE 45) | 0.00 / 0.00 |
| ink black vs red | 0.86 / 0.95 |
| grey wash vs bare (dE 14) | 1.00 / 0.98 |
| lap vs shingle courses / vs brick | 0.00-0.10 |
| board vs board-and-batten | 0.99 / 1.00 |

**2. Why direction: the roof plans teach it, by design, and nothing teaches the opposite.** Measured on 300 fresh
sheets (long-pool flags + Swiss, `within_family.py --one-dir`, one-direction patterns only): the pieces of one
labelled family turn in 65% of roof-plan families (facets turn with the eave) vs 7% of elevation families (the
measurement's noise floor). [Corrected 2026-10-07: an earlier count said 71% of floor-plan families turn; that was
tile grids, whose two equal directions flip the FFT peak. Synthetic floors keep one direction per material.] The
floor-plan rule ("Direction never counts", `LOOK` puts hatch / hatch45 / plank / joists in one class) means no two
floor families ever differ by direction alone, and no synthetic sheet of any kind had such a pair (0 of 147). The training flips / rotations are shared by image and reference,
so augmentation is not the cause. Gemini families turn too (26-36%; sheet types unknown), so turning roof facets
are right -- what is missing is elevations / plans where direction is the only difference (H lap vs V board, H vs
diagonal hatch), e.g. r4 #098 (V board-and-batten selects the H roof lines) and r2 #029 (diagonal vs horizontal).

**3. Near-miss families are rare in the synth, rarer than in Gemini** (`family_pairs.py`, pairs of labelled families
on one sheet): same texture class 19% of Gemini pairs (128 pairs, 78 sheets) vs 10% synthetic (147 pairs; elevations
15%, floor plans 2%, roof plans 0%). `distinct_looks` (elevations) and `_distinct` (plans) re-draw any same-look pair
(same class, spacing < 1.4x, dE < 8), and the GPU session found colour-only pairs are 5% of elevations at dE 32-114.
So the model is never asked to separate tan from olive lap or grey-washed from bare lines.

**4. Where the excess goes (Gemini30, 171 questions, sheet 16 left out):** half lands in other labelled families
(5.2% of the union), half in unlabelled area (5.4%). Merges > 2% of a question by what separates the two families:
spacing + colour (same texture, same direction) 12 -- all r4 #078 (pattern2 vs pattern1: spacing 1.8x, fill dE 17);
direction + spacing (+texture) 11 (r2 #029, r2 #060, r2 #010); texture-only 5 (r4 #031); 4 unmeasurable (tiny).
Excess within 16 px of the target's edge (narrow strips, trim, rakes) costs 0.024 IoU, misses within 16 px 0.015,
far excess 0.068: the strips the user saw on #098 are real but a third of the over-selection cost.

**5. Splotches are low-confidence:** 8,580 small blobs wholly outside the target, mean probability 0.41 (90% < 0.5)
vs 0.96 on correct pixels. Threshold 0.35 -> 0.5: +0.001 (126 better / 66 worse); dropping blobs < 1% of the
prediction: +0.005. Visible, nearly free in IoU.

**6. Door behind a balcony railing (r4 #098): the synth never shows one.** `generate_synthetic_fc.py` puts doors
only at grade (front door, garage); upper floors get windows with sills ~3 ft up, i.e. above the railing. 55% of
two-storey railings are balconies, so every synthetic railing stands in front of siding and the label always runs on
behind it. The model learned "railing => siding behind it" and selects the balcony door.

**7. Boxes (counterfactual: same question, box on the family's best piece):** r2 #028 q4 (49 px automatic) 0.849 ->
0.944; r4 #050 q6 (box over an edge) 0.770 -> 0.865; r4 #098 q3 0.707 -> 0.852; r4 #078 q4 / q7 0.21 / 0.26 -> 0.25
(not the box: a real merge). r3 #019 q5 and r2 #053 q10 cannot get a better box: every piece of those families is
<= 50 px. r2 #002 q1 (the user's box over "85 SF"): a box on the other bath, which holds "GUEST BATHROOM", scores
0.59 vs 0.69 and selects the room-name text all over the sheet (kitchen, living room, hallway). Text in the
reference makes the model select text. Synthetic floor plans put a bold room tag at every room's centre, inside the
label, and the box sampler favours deep (central) points, so training references often hold a tag: fix (b) in
codex_doc (generator text -> ref_exclude) addresses it.

**8. Gemini inconsistency (r4 #016 / #041):** within-family spacing jumps > 1.3x in 32-42% of Gemini families vs
19% of synthetic elevation families (the descriptor's noise floor); some of it is Gemini, some FFT harmonics.
r4 #016 pattern1 pieces measure 8.7-19 px, r4 #041 pattern1 turns 0 / 90 / 45 deg at a constant 24 px.

**Label:** r3 #048 fixed by the user (Roboflow r3 v3; also adds newly labelled r3 #016): sheet 0.830 -> 0.862,
Gemini30 0.8255 -> **0.8295**.

**What to change in the generator, in order:** (a) elevations and floor plans with two labelled families that differ
ONLY in direction (H lap vs V board at the same spacing; hatch vs hatch45), keeping roof facets turning; (b) same
texture, different spacing (1.3-2x) or colour / tone (dE 10-30) as separate families -- extends the planned colour
pairs; (c) balcony doors behind railings, door cut from the label; (d) room tags out of references (fix (b)). Re-run
`make_invariance_probe.py` on every candidate: the probe takes ~20 min on CPU and shows directly whether a pool fixed
the invariance.

### 2026-10-07 (CPU) -- generator: near-miss pairs and balcony doors (new defaults, GEN_VERSION 2026-10-07)

From the Gemini30 investigation above. All three are DEFAULTS (own RNG streams; `--near-pairs 0 --near-pairs-plan 0
--balcony-doors 0` reproduces the 2026-10-04 generator byte for byte, checked on 60 sheets: elevations, roof and floor
plans, railings 0.6). Every pool built before is stale for `scripts/pool_guard.sh`.

- **`--near-pairs 0.35` (elevations, `revit_render.plant_near_pair`):** two labelled families that differ in ONE
  attribute: direction (lap vs vertical boards at the same spacing, colour, pen; 40%), spacing (same kind 1.4-2x
  apart; 30%), tone (fill dE 10-30 on colour sheets, grey wash vs bare paper on line-only; 30%). A = the main wall
  (a plain one gets lines), B = an upper-storey / wing accent, else the roof drawn as plain lines (a gable triangle
  or wainscot band is too small to teach anything). Annotation files carry `near_pair`.
- **`--near-pairs-plan 0.6` (finish floor plans, `revit_plans.plant_near_pair_plan`):** two zones whose floors differ
  in ONE attribute: orthogonal lines vs diagonal hatch at the same spacing / pen / fill, spacing 1.4-2x, or tone (fill
  25-45 grey levels apart, or a wash vs bare). Planted after `_separate_all`, which would push them apart. Roof facets
  still turn inside one family.
- **`--balcony-doors 0.7` (with `--railings`):** an upper-floor balcony gets a door (or slider) behind its railing,
  cut from the wall label like every door, the railing drawn over it. Before, doors existed only at grade.

**Measured on 300 sheets, same ids as the 2026-10-04 pool above (long-pool flags + Swiss, railings 0.15):** planted
pairs on 58 / 201 elevations (23 direction, 14 spacing, 21 tone; both families visible and labelled on 45) and 8 / 54
floor plans at the 0.35 setting used for the count (raised to 0.6 since: finish sheets are ~40% of plans). Same-texture
pairs 10% -> 15% of all pairs (elevations 15% -> 24%; Gemini 19%); colour-only pairs 0.7% -> 2.5%, spacing-only
1.4% -> 3.2%. Rendered tone gaps dE 10.6-32.2 (17 pairs). The descriptor cannot measure most direction pairs (B
often too fragmented for 96 px patches), so the planted counts are the record there.

**Next (GPU):** a 2k screen of this pool vs the `revitfailswissroi` control (same flags, ids, seed), scored on
Gemini30 hand boxes @2048 (`eval_baselines/revitfailswiss2k-e8_gemini30_2048.json`; sheet 16 changed, compare on the
other 29 or rescore), HF14, val -- AND `scripts/probes/make_invariance_probe.py` on both: the probe is the direct
test (direction leak 1.00 today). Then a long restart of longswiss-swa37-41 on a mined pool from these defaults.

### 2026-10-08 (CPU) -- slivers: reference floor 32 px at training scale, mining ignores sliver questions

The GPU session's unfair mined questions (gable / porch slivers, blank band edges; pieces < 32 px @2048: 27% base vs
36% mined) and the user's boxes set the number. Box sides at 2048 (eval_baselines + hand boxes):

| | n | min | p5 | median | < 24 | < 32 |
|---|---:|---:|---:|---:|---:|---:|
| user's boxes, real | 88 | 22 | 35 | 66 | 2 | 4 |
| user's boxes, Gemini30 | 35 | 54 | 68 | 118 | 0 | 0 |
| automatic boxes the user dropped, real / Gemini | 6 / 7 | 17 / 16 | | 34 / 34 | 2 / 1 | 2 / 1 |

- **`train_refunet.py --ref-min-side 32` is the DEFAULT** (was 0 = off): a training reference comes only from a piece
  that fits a 32 px square at the training scale (measured on the native mask at 32 x long side / image-max-size; the
  old flag compared native pixels, ~15 px at 2048 on a synthetic sheet) and its visible region (minus ref_exclude),
  when the family has such a piece. Targets unchanged. On 300 new-default sheets: P(reference piece < 32 px) 0.18 ->
  0.10; the rest are the 10% of families with no bigger piece, still asked from their best piece. `--ref-min-side 0`
  = every run before.
- **`build_hard_pool.py --min-ref-side 32` is the DEFAULT:** questions whose reference box is < 32 px at
  --image-max-size are left out of sheet hardness (on a 120-sheet test: 321 of 1,155 automatic questions). The
  manifest records `min_ref_side` and `questions_dropped_small_ref`. 0 = every pool mined before.
- **Generator text in references: not excluded.** Proposed (text -> ref_exclude) and dropped on the user's call: users
  do box over text (their own r2 #002 box holds "85 SF"); a model that never sees text in a reference cannot learn to
  look past it. Open option: record the generator's text boxes and reject only references that are MOSTLY text
  (e.g. > 30% of the box), which is what the mined "BATH 2" case was.

### 2026-10-08 (GH200, 80-min VM) -- the near-miss pool fails the screen; the 32 px reference floor is neutral

The runbook's 2k screen, both arms, seed 7, swin_t, ROI add, 1+8 epochs, epoch 8, hand boxes @2048 (no 4096: no time).
Pools rebuilt here (Swiss layouts: 19,046 kept, as documented): `revitnear_train2000` 1,992 sheets, control
`revitfailswissR_train2000` 1,988. Hub (private): `abshetty/floz-refunet-swint-revitnear2k-e8`,
`...-revitfailswissr32-2k-e8`, each with `evaluations/` (per-question JSONs, probe and Gemini30 `--save-probs`).
Per-question files also in `eval_baselines/` (`revitnear2k-e8_*`, `revitfailswissr32-2k-e8_*`).

| model | HF14 | val | Gemini30 |
|---|---:|---:|---:|
| `revitfailswiss2k-e8` (published control, ref floor off) | 0.8090 | 0.7559 | 0.7239 |
| `revitfailswissr32roi` (same pool, `--ref-min-side 32`) | 0.8049 | 0.7580 | 0.7393 |
| `revitnearroi` (2026-10-07 generator defaults + ref floor) | 0.7779 | **0.6578** | 0.7518 |

- **Reference floor (r32 vs control): neutral.** HF14 -0.004 (p=0.76), val +0.002 (p=0.87), Gemini30 +0.015 (p=0.04,
  sign test 72/59 p=0.29). Keep the default.
- **Near-miss pool (near vs r32): val -0.100 (8 better / 50 worse, p=2e-8), HF14 -0.027 (12 / 24, p=0.12), Gemini30
  +0.013 (p=0.24).** The val loss is broad (sheets 5, 6, 9, 13, 17, 26), not val 17 alone. HF14 25 (shingle roof vs
  vertical boards) 0.73 -> 0.90; line-only 14 0.81 -> 0.68 and 2 0.83 -> 0.67 fall.
- **The probe did not move: direction leak 0.91-0.98 in every arm** (control 0.81-0.96). Spacing 2x and tan vs olive
  unchanged. So the planted pairs do not teach direction / spacing / tone at this scale, and they cost val.
  Probe rows are themselves noisy across runs: tan vs blue reads 0.13/0.07 (control), 0.83/0.11 (r32),
  0.35/0.81 (near) -- read only rows that move together across all probes of one attribute.
- **Per the runbook gate: do not mine / long-restart on the near pool.** Next: look at the near pool (are the pairs
  visible and labelled as intended at training scale? does B as "the roof drawn as plain lines" corrupt roofs?) before
  any GPU; candidates are direction pairs alone at a lower rate, or near-pairs without the roof-as-B fallback.
- **Where the near pairs went (pool `annotations/*.json` `near_pair`, 574 / 1,992 sheets):** elevations 488 -- of them
  **299 (61%) are `main+roof`, i.e. the fallback that redraws the ROOF as plain lines** in the wall's look (direction
  130, tone 88, spacing 81) -- ~23% of all elevations get a roof that looks like siding; `main+accent` 189; floor plans
  86 (tone 35, direction 34, spacing 17). The fallback is the majority case, not an exception: the prime suspect for
  the val loss. Ablation run the same session: `--near-pairs 0` (plan pairs + balcony doors kept), below.
- **Ablation `revitnearnoelevroi` (`--near-pairs 0`; plan pairs + balcony doors kept; `...-revitnearnoelev2k-e8`):**
  HF14 0.7856, val 0.6961, Gemini30 0.7248. vs near: val **+0.038** (44 / 16, p=7e-4), HF14 +0.008, Gemini30 -0.027
  (p=0.02). vs r32 control: val **-0.062** (19 / 40, p=8e-5), HF14 -0.019, Gemini30 -0.015. Direction leak 0.91-0.97.
  So the elevation pairs carry ~40% of the val loss AND the whole Gemini30 gain; the rest of the loss comes from the plan
  pairs and / or balcony doors. Neither part teaches direction. Next (CPU first): drop the main+roof fallback; split
  plan pairs from balcony doors.

### 2026-10-08 (CPU) -- why the near-miss pool failed

Inputs: the three 2k arms' per-question files (`eval_baselines/revit{failswissr32,nearnoelev,near}*`), their Gemini30
probabilities from the Hub (`evaluations/*gemini30_probs`), and val 5 / 6 / 9 / 17 rescored here with `--save-probs`
(reproduces the GPU numbers to 0.006). Scripts: `scripts/probes/piece_recall.py`, `scripts/probes/drop_tone.py`.

1. **The new pools trade merging for splitting, everywhere.** Share of the loss that is under-selection (pred/target
   < 0.77): val 0.23 (r32) -> 0.41 (noelev) -> 0.47 (near); HF14 0.25 -> 0.29 -> 0.40; Gemini30 0.17 -> 0.37 -> 0.37,
   while over-selection falls (Gemini30 0.40 -> 0.19). Of the questions that lost > 0.1 (near vs r32), 16 / 25 on val,
   8 / 10 on HF14, 18 / 24 on Gemini30 under-select. Gemini30 nets a gain because its losses were mostly merging; the
   real sheets net a loss. Not calibration: the best threshold per arm gains <= 0.004.
2. **What gets dropped: parts of the family that look a little different from the reference.** Pixels the near model
   drops and the control kept (interior pixels, > 12 px from the target edge) sit farther from the reference box's
   colour than pixels both keep: Gemini30 median dE 2.3 vs 1.1 (68% of questions); val 5 / 6 (rendered) +11 / +12 dE.
   In the overlays the reference is on a chimney face, a dormer gable or a gable corner, and the near model drops
   blotchy swaths of the main wall it belongs with (same siding, other light / paint). It drops 27% of the target
   pixels the control kept on val 5 / 6 / 9 / 17. By piece: recall of the family's other pieces falls at every size
   (0.03-0.09); pieces measurably different from the reference piece in spacing or direction do NOT fall more.
3. **Why: the planted differences sit inside real within-family variation -- on the sheets that lost.** [Corrected
   2026-10-08: an earlier line here said "25% of same-family piece pairs differ by dE 10-20"; all 72 such pairs are
   val 17's, one sheet.] Measured per pixel instead (15 px local mean, interior, distance from the family's median
   colour): family pixels > 10 dE off are 11-33% on the rendered elevations 2, 3, 5, 6 (shadow, porch recesses, eave
   bands) and 26% on markup 17 (lavender board-and-batten L 71 on the wall, L 56 in the dormers), 8-10% on markup 18 /
   19, 0-7% elsewhere (mean 4%). The rendered sheets and 17 are where the near model lost most (-0.13 to -0.20);
   18 / 19 did not move, line sheets 9 / 13 and plan 12 lost without colour variation, so colour is the main but not
   the only route. On val 5 / 6 the reference sits on the chimney (a small, lighter piece of the same siding) and the
   near model drops blotches of the main wall, shaded and sunlit alike. Gemini families jump > 1.3x in spacing
   within the family 14-33% of the time (one-direction patterns), against spacing pairs at 1.4-2x. So the pool told the
   model "this much colour / spacing difference = another family", which real labels contradict a quarter of the time,
   and the model became cautious about every piece that is not a close match of the reference. The floor-plan pairs
   (noelev) teach the same lesson (a near-identical floor in another room is another family) and under-select too.
4. **Why the probe did not move:** the planted pairs are separable without looking at the attribute. B is the roof
   (61% of elevation pairs: above the eave, sloped outline) or an upper storey (above a belt trim), or another room;
   the model can split them by region and boundaries, so it never needs line direction. The probe's two identical
   rectangles leave direction as the only cue, and leak stays 0.91-0.98.
5. **Caveat on size:** val 17 (23 of 72 val questions, documented to swing 0.2-0.5 between epochs) is 0.063 of the
   0.100 val loss; one seed, epoch 8. The direction of the result is solid (under-selection on every split, val
   w/o 17 -0.055, HF14 2 / 3 / 12 fall), the magnitude is not.

**Recommendation:** revert the tone and spacing pairs (elevations and plans) -- they contradict real labels. Keep the code.
Direction pairs only if they cannot be solved by region: two side-by-side walls of the same storey (a wing, `block`
accent), never the roof, never another storey; test with the probe at 2k before any pool. Balcony doors are confounded
with the plan pairs in the noelev arm (no clean read); they are a labelling correction, rare (railings 0.15 x upper
floor), and can stay or be screened alone.

### 2026-10-13 (CPU) -- the near pairs fixed Gemini's merges; the lightness-only tone pairs contradicted shadows

The 2026-10-08 recommendation (revert the tone and spacing pairs) was wrong for the goal: Gemini's worst merges ARE those
pairs. Measured instead of reverted.

1. **Where the best model (longswiss @4096) merges on Gemini30** (151 question x other-family rows, leak = share of the
   other family selected): fill dE 0-10: leak 0.06; 10-35: 0.23; spacing ratio 1.6-2x: **0.41** (2-3x: 0.01); clearly
   different colour (fill > 35 or ink > 40): 0.02. Worst: r4 #078 (sheet 26) -- tan / olive / red lap, all horizontal,
   tan vs olive |dL| 4, d(a,b) 16 -- leak 0.5-1.0 both ways; r2 #029 (sheet 4) 1.65x + 45 deg, leak 0.93.
2. **The near pool fixed sheet 26:** mean IoU r32 0.47 -> near 0.63 (worst question 0.22 -> 0.57); most of near's
   Gemini30 gain (0.7518 vs 0.7393).
3. **What it broke is lightness.** On val 5 / 6 the pixels near drops and r32 keeps differ from the reference by
   |dL| +12 / +13.5 but d(a,b) +1.8 / +0.0: shade, not hue.
4. **Shadows already exist** (`wall_shadows`: cast from the massing on 85% of elevations; `--soft-shadows 0.5` in every
   run). Same 400 sheets (seed 6, run_revit_2k near flags) rendered with and without shading, 15 px local CIELAB, 25-75th
   percentiles:

   | | \|dL\| | d(a,b) |
   |---|---|---|
   | shadow on one family, line-only sheets | 16-29 | 0 |
   | tone pair, line-only sheets (grey wash vs paper; 13 of 22 pairs) | 15-20 | 0 |
   | shadow on one family, colour sheets | 16-25 | 4-14 |
   | tone pair, colour sheets | 2-5 | 14-26 |
   | spacing pair | 1-2 | 0 |

   On line-only sheets the tone pair IS a shadow: one change labelled "same family" (shadow) and "another family"
   (pair). The plan tone pairs were the same grey-wash step. The colour pairs were mostly hue and do not overlap.
5. **Change (GEN_VERSION 2026-10-13):** tone pairs only on colour sheets (elevations: `shaded`; plans: `colour`), and as
   a hue change: fill and pen moved together 15-30 in (a, b), lightness within +-3 (`revit_render._shift_colour`,
   `revit_plans._hue_shift`). Line-only sheets / mono plans draw direction or spacing instead. Spacing and direction
   pairs unchanged. `--near-pairs 0 --near-pairs-plan 0 --balcony-doors 0` still reproduce older pools.
   `--soft-shadows` defaults to 0.5 since this date (every run already passed it; pools built without it: add `--soft-shadows 0`).
6. **Verified** (300 new-default sheets, seed 11, same with / without shading): tone pairs |dL| 1-2, d(a,b) 15-24
   (8 pairs, colour sheets only; shadows on colour sheets |dL| 13-23, d(a,b) 4-16); no lightness-only pair left.
   Pair mix 43 direction / 31 spacing / 8 tone. `--near-pairs 0 --near-pairs-plan 0 --balcony-doors 0`: 24 sheets
   byte-identical to the previous commit.

### 2026-10-13 (GH200) -- the hue-pair pool fails the screen: Gemini30 loses near's gain, val / HF14 still below r32

The 2026-10-13 runbook's one arm, `revithueroi` (GEN_VERSION 2026-10-13, same flags / ids / seed 6 / training as
`revitnearroi`), seed 7, swin_t, ROI add, 1+8 epochs, epoch 8, hand boxes. Fresh box: FreeCAD 26.3 (rev 20260916) into
`/opt/fc`, Swiss layouts rebuilt (19,046), Gemini30 rebuilt (r3 v3, #048 fixed). Pool 1,992 sheets (same as near). Old
v6d / revit2k controls not re-scored (`NOCTRL=1`). Per-question files: `eval_baselines/revithue2k-e8_*`; not on the Hub.

| model @2048 | HF14 | val | val w/o 17 | Gemini30 |
|---|---:|---:|---:|---:|
| r32 (no pairs) | 0.8049 | 0.7580 | 0.8448 | 0.7393 |
| near (2026-10-07 pairs) | 0.7779 | 0.6578 | 0.7895 | 0.7518 |
| **hue (2026-10-13 pairs)** | 0.7806 | 0.7161 | 0.7988 | **0.7068** |

@4096: HF14 0.7966, val 0.7815.

- **vs r32: val -0.042 (12 / 40, p=2e-4), HF14 -0.024 (13 / 22, p=0.24), Gemini30 -0.033 (69 / 84, p=0.005).** Gate
  (Gemini30 >= r32 + 0.01, HF14 / val w/o 17 within ~0.01) fails on all three.
- **vs near: val +0.058 (40 / 11, p=1e-4), HF14 +0.003, Gemini30 -0.045 (44 / 101, p=2e-7).** Half of near's val loss came
  back; all of near's Gemini30 gain went.
- **Shaded rendered sheets recover only part way:** val 5 / 6 0.790 / 0.831 (near 0.714 / 0.767, r32 0.876 / 0.897);
  HF14 2 / 3 0.712 / 0.830 (near 0.671 / 0.791, r32 0.805 / 0.928).
- **Gemini30 26 (r4 #078, tan / olive / red lap): 0.489** (r32 0.417, near 0.634) -- most of the gain near got from
  its tone pairs is gone with them. Gemini30 also lost broadly on sheets near did not hurt: 5 (0.907 / 0.710 / 0.583 for
  r32 / near / hue), 9 (0.907 / 0.899 / 0.730), 28 (0.820 / 0.723 / 0.635), 14, 17, 3.
- **Splitting is down on val / HF14, UP on Gemini30.** Share of the IoU loss that is under-selection (pred/target < 0.77):
  val 0.23 / 0.47 / 0.33 (r32 / near / hue), HF14 0.25 / 0.40 / 0.29, Gemini30 0.17 / 0.37 / **0.46**.
- **Line sheets (the runbook's stop condition):** val 9 0.783 (near 0.774, r32 0.877), val 13 0.539 (0.512, 0.590) did not
  recover; HF14 12 0.748 (0.636, 0.719) did; HF14 14 +0.085 vs r32. Val 26 (one question) 0.954 -> 0.489.
- **Probe unchanged:** direction leak 0.91-0.97; spacing x2 0.32 / 0.99; tan vs olive 0.98 / 0.86.
- **Pair mix** (`near_pair` in annotations, 574 sheets, same sheets as near): elevations main+roof 299 (direction 153,
  spacing 111, tone 35), main+accent 189 (103 / 62 / 24); plans 86 (40 / 33 / 13). The tone pairs removed on line-only
  sheets became direction / spacing pairs, and **the main+roof fallback (61% of elevation pairs) is still there** --
  the 2026-10-08 prime suspect, never removed.
- Caveat: one seed; the controls were trained on an earlier box (FreeCAD / library versions may differ).

**Reading:** with the lightness pairs gone, the remaining pairs (mostly direction / spacing, mostly roof-as-B) still
cost val / HF14 against r32 and buy nothing on Gemini30. Per the runbook: stop, do not mine. Next suspects, in order:
the main+roof fallback (61% of elevation pairs; separable by region, so it teaches nothing the probe measures and
redraws roofs as siding), then spacing pairs (inside real within-family spacing variation, 2026-10-08 item 3).

### 2026-10-13 (CPU) -- reading the hue screen: not seed noise; the pairs teach "another zone = another family"

- **Not noise.** Per-sheet |difference| between two near-identical runs (r32 vs the control: same pool, ref floor
  only) on Gemini30: mean 0.037, max 0.136, no sheet > 0.15. Hue vs r32: mean 0.068, 4 sheets > 0.15. The sheets hue
  lost (5, 14, 17, 28) were already falling under near (5: 0.91 / 0.71 / 0.58 for r32 / near / hue; 14: 0.87 / 0.69 /
  0.69; 28: 0.82 / 0.72 / 0.64).
- **It splits single families along zone boundaries.** Gemini30 5 and 14 have ONE family, yet hue selects 30-70% of
  it. Near's saved probabilities (same failure, smaller): 5 q1 (dot tile, reference in the laundry) drops the whole
  garage; 14 q1 (board-and-batten, reference on the main wall) drops the gables; 28 q12 (reference in the upper gable)
  drops the lower wing; 9 q5 drops shingle blotches. These are exactly the boundaries the pairs are planted across:
  B = the roof (61% of elevation pairs) or an upper-storey / wing accent; floor pairs = another room. The pairs teach
  "a near-identical texture across that boundary is another family", and the model applies it to real sheets where the
  family continues across it.
- **Why hue is worse than near on Gemini30:** the line-sheet tone pairs became direction / spacing pairs on the same
  boundaries, so the pool has MORE positional pairs and fewer whose difference is visible in appearance alone; the
  only Gemini gain the pairs ever bought (sheet 26, tone) shrank with the colour-sheet-only tone pairs. Gemini30
  under-selection share 0.17 / 0.37 / 0.46 (r32 / near / hue).

### 2026-10-14 (CPU) -- colour pairs only, on side wings, colour gap in CIEDE2000 (GEN_VERSION 2026-10-14)

- **The colour gap was measured wrong.** CIELAB dE76 overrates chroma steps in saturated colours. Rescored with
  CIEDE2000, the 2026-10-13 hue pairs: sage vs lime (000204, same hue, more saturated) dE76 16.0 -> 5.9; green vs green
  (000033) 17.2 -> 8.2; teal / grey-teal 9.3; red / brown, brown / mauve 14; grey / salmon 20.8. Gemini r4 #078 tan vs
  olive: dE76 16.5 -> 8.9 -- and it also differs in spacing (1.8x) and line density, which is why it reads as far more
  distinct than 000204 to the eye: real "other family" pairs differ in several attributes at once.
- **Change:** pairs are colour pairs only -- "hue" (CIEDE2000 10-25, lightness +-3, hue angle >= 20 deg or, from a
  near-grey, chroma >= 15) or "hue+spacing" (CIEDE2000 7-20 plus spacing 1.4-2x) -- on colour elevations whose house
  has a side wing; B is the wing (decided before the geometry, which makes the wing the accent zone), never the roof,
  an upper storey or another room. P 0.6 (`--near-pairs`). Direction / spacing-only pairs and floor-plan pairs off
  (`--near-pairs-plan` 0). Runbook: `codex_doc.md`, arm `revitwingroi`.
- **Verified** (300 sheets, seed 11, new defaults): 22 pairs on 182 elevations (12%; 11 hue, 11 hue+spacing; 69 colour
  elevations), B a side wing on every one; measured fill CIEDE2000 9.4-22.4 (median 17.6). No floor-plan pairs.
  `--near-pairs 0 --near-pairs-plan 0 --balcony-doors 0 --soft-shadows 0`: the previous commit's code and this one
  give byte-identical 24-sheet pools on this box (one pixel of one image differs from the same pool built before the
  container restarted: environment, not code).
- **How often is one real family two colours? Almost never, piece to piece.** Per piece (>= 4000 px at 2048 after a
  7 px erosion; fill = median CIELAB of its lighter 40%), max pairwise CIEDE2000 inside a family: real 28 (26 families
  with 2+ measurable pieces) 85% < 3, 8% 3-6, 4% 6-10, 4% 10-15, none above; Gemini30 (45 families) 98% < 3, one at 16.1.
  The two above 10: val 17 (lavender markup paint, 12.4) and Gemini r2 #053 (a brick piece under the rear porch roof,
  in shadow, 16.1). Between families: real 36% < 3 (same colour, told apart by texture), 57% >= 15; Gemini 64% < 3. So the
  within-family colour variation found on 2026-10-08 is shading INSIDE pieces, not differently coloured pieces: a
  piece-to-piece hue gap of CIEDE2000 >= 10 at the same lightness is another family on real sheets.

### 2026-10-14 (GH200) -- the wing colour-pair pool passes the screen; mined + long restart started

`revitwingroi` (GEN_VERSION 2026-10-14), same recipe as every 2k arm (seed 7, swin_t, ROI add, 1+8, epoch 8, hand
boxes). Pool 1,989 sheets; pairs on 195 (all elevation main+accent: 94 hue, 101 hue+spacing). Per-question files
`eval_baselines/revitwing2k-e8_*`; not on the Hub.

| model @2048 | HF14 | val | val w/o 17 | Gemini30 |
|---|---:|---:|---:|---:|
| r32 (no pairs) | 0.8049 | 0.7580 | 0.8448 | 0.7393 |
| near | 0.7779 | 0.6578 | 0.7895 | 0.7518 |
| hue | 0.7806 | 0.7161 | 0.7988 | 0.7068 |
| **wing** | 0.7920 | 0.7527 | 0.8365 | **0.7647** |

@4096: HF14 0.8291, val 0.8010.

- **vs r32: Gemini30 +0.025 (85 / 53, p=0.015; sign p=0.008), val w/o 17 -0.008 (se 0.009), val -0.005, HF14 -0.013
  (11 / 19, p=0.27).** Gate passes (HF14 at the edge of "~0.01", not significant). vs hue: Gemini30 +0.058 (p=8e-7).
- **The splitting is gone:** Gemini30 5 / 14 / 28 = 0.939 / 0.879 / 0.825 (r32 0.907 / 0.873 / 0.820; hue 0.583 / 0.689 /
  0.635). Gemini30 16 0.749 (r32 0.458; +0.29, beyond the 0.14 noise band); 26 0.499 (r32 0.417; within noise).
  Losses > 0.1 vs r32: Gemini30 4, 10, 25 (-0.11 to -0.14, inside the noise band).
- **Under / over-selection share of the loss** (r32 -> wing): Gemini30 0.17 / 0.40 -> 0.28 / 0.24; HF14 0.25 / 0.32 ->
  0.21 / 0.45; val 0.23 / 0.36 -> 0.59 / 0.03 (val is 23 / 72 sheet 17, which reads 0.574 in both).
- **The probe moves for the first time, on colour:** fill tan vs olive leak 0.98 / 0.86 (hue) -> 0.61 / 0.10; tan vs blue
  0.66 / 0.51 -> 0.01 / 0.00; spacing x2 0.32 -> 0.12 (A->B). Direction stays blind (0.75-0.98), as expected (no
  direction pairs).
- Next (runbook step 3): mining on the wing pool and the 20-epoch restart of longswiss-swa37-41, below.

### 2026-10-14 (GH200) -- long restart on the mined wing pool: HF14 0.8805, level with longswiss; val -0.06

Runbook step 3. Mining: miner `revitwingroi` e8, 4,000 fresh sheets (ids 100000+), 3,966 ranked, mined mean IoU 0.704
(fresh kept 0.858) -> `revitwing_hard2000`. Restart: `run_hard_long.sh` EP=20 from the Hub `longswiss-swa37-41`
(epochs numbered 0-19: Hub checkpoints carry no epoch), ROI add, peak LR 2e-4 cosine to 1e-5, 137 s/epoch. Train loss
0.19 -> 0.10. SWA windows @4096, hand boxes (val / HF14): e15-19 0.8126 / 0.8781, **e16-19 0.8161 / 0.8805 (val pick)**,
e17-19 0.8141 / 0.8842, e14-19 0.8117 / 0.8770. Checkpoint `data/runs/ck_long_hardwing/avg_e16-19.pth` (not on the Hub);
per-question files `eval_baselines/longwing-swa16-19_*_4096.json`.

| @4096, hand boxes | HF14 | val | val w/o 17 | Gemini30 |
|---|---:|---:|---:|---:|
| longswiss-swa37-41 (start) | 0.8779 | 0.8766 | 0.9093 | 0.8295 |
| longwing-swa16-19 | 0.8805 | 0.8161 | 0.8846 | 0.8343 |

- **Paired vs the start: HF14 +0.003 (18 / 13, p=0.88), Gemini30 +0.005 (p=0.59), val -0.061 (p=7e-4; 72% is val 17,
  0.807 -> 0.670), val w/o 17 -0.025 (se 0.018).** No gain toward 0.92; by the val rule the start stays the best model.
- **The pair did its job on its target:** Gemini30 26 (tan / olive / red lap) 0.380 -> 0.665. Also 10 0.77 -> 0.85, 4
  0.66 -> 0.72. Losses: Gemini30 1 0.833 -> 0.432, 0 0.736 -> 0.592, 6, 17; HF14 14 0.882 -> 0.803 (12 0.714 -> 0.794);
  val 9 (line sheet) 0.877 -> 0.745.

### 2026-10-14 (GH200) -- the wing long run by epoch; synthetic vs real; diagnostics on the Hub

CPU-session requests (`codex_doc.md`). All @4096, hand boxes; per-question files in `eval_baselines/` (`longwing-eN_*`,
`*_synth300_4096`); probability maps on the private Hub dataset `abshetty/floz-diag` (`<model tag>/<split>_<res>/probs`
+ `metrics.json`): longswiss-swa37-41 and longwing-swa16-19 (HF14 / val / Gemini30 / synth300 @4096), revitwing2k-e8 and
revitfailswissr32-2k-e8 (HF14 / val / Gemini30 @2048); epochs 4 / 9 / 14 / 19 metrics only. The SWA checkpoint is on the
Hub: `abshetty/floz-refunet-swint-longwing-swa16-19` (private, round-trip identical).

**By epoch** (IoU, then share of the loss from under- / over-selection, pred/target < 0.77 / > 1.3):

| | HF14 | val | val w/o 17 | Gemini30 |
|---|---|---|---:|---|
| start (longswiss) | 0.8779 .17/.20 | 0.8766 .23/.00 | 0.9093 | 0.8295 .13/.32 |
| e4 | 0.8536 .52/.04 | 0.8457 .33/.00 | 0.8870 | 0.8096 .20/.25 |
| e9 | 0.8710 .27/.16 | **0.8959** .04/.00 | **0.9195** | 0.8325 .13/.28 |
| e14 | 0.8728 .41/.08 | 0.7966 .61/.00 | 0.8797 | 0.8115 .28/.20 |
| e19 | 0.8811 .36/.15 | 0.7905 .65/.00 | 0.8830 | 0.8344 .19/.23 |
| SWA 16-19 | 0.8805 .36/.15 | 0.8161 .52/.00 | 0.8846 | 0.8343 .19/.21 |

- **Splitting starts at once and is not monotonic.** e4 (LR near its 2e-4 peak) under-selects everywhere (HF14 .52);
  e9 is back to the start's balance and has the best val of any model (0.8959; w/o 17 0.9195) at HF14 0.8710; by e14
  val splits again (.61, mostly sheet 17) while HF14 and Gemini30 recover by e19. The SWA window (last epochs only)
  could not pick e9. Fit (run_hard_long): mined-half sheets 0.862 -> 0.877 from e9 to e19, fresh flat 0.886.
- **Synthetic vs real.** On 300 fresh wing-generator sheets (ids 500000+, 2,718 questions): longswiss 0.8739, longwing
  0.9000 (+0.026, 821 / 246, p=2e-33); pair elevations 0.832 -> 0.920, other elevations 0.885 -> 0.898, floor plans
  0.773 -> 0.819, roofs 0.939 -> 0.976; over-selection share 0.32 -> 0.16. On real: HF14 +0.003, Gemini30 +0.005. The run
  learned what the synthetic sheets teach; almost none of it transfers.
- **Mined pool** (`pool_manifests/revitwing_hard2000.json`): pair sheets 14.8% of the mined half (20.8% of its
  elevations) vs 9.8% of the 2k pool, 10.2% of the base half, 9.5% of the fresh 4,000. Not because pair sheets are harder
  for the miner (pair elevations 0.797 vs other elevations 0.795): they fall in the 17-51% hardness band more often.

### 2026-10-14 (CPU) -- gable-base trim on one-family gables (GEN_VERSION 2026-10-14b)

The wing long run splits Gemini30 1 (r2 #004: one board-and-batten family in three bands -- gable, upper, lower --
divided by white trim boards) at the bands: reference in the gable 0.85 -> 0.23, in the lower storey 0.87 -> 0.24, in
the upper storey 0.78 -> 0.83 (pred/target 0.2 on the first two). In the generator a trim board across a gable base
existed ONLY under accent gables (`zone_for("gable")`), i.e. 100% of synthetic gable-base trims start a new family
(5,000 sampled houses: 1,095 accent gables, 0 same-family gables with a band); storey belts are 65% same family. The
hypothesis (unconfirmed until the long run's maps arrive): once the wing pairs made pattern alone insufficient, the
model began using that perfectly predictive band. Change: `--gable-trim P` (default 0.5): a house whose gables are not
an accent gets the band across every gable base (one decision per house, own RNG), the main family continuing above.
40-sheet check: 7 / 40 elevation sheets change; `--gable-trim 0` byte-identical to the previous commit.

### 2026-10-14 (GH200) -- are the mined reference boxes too hard? No

Every question of the wing mining run (34,802 on 3,982 fresh sheets, miner `revitwing2k-e8`, automatic boxes) against
real HF14 + val (123, hand boxes) and Gemini30 (195). Box side at the 2048 training scale; the reference piece's share of
its family; crop content on the 224 px crop the model sees (Canny edge density; "ink" = pixels > 25 grey levels off
the crop median).

| | box p10 / p50 / p90 | box < 32 | box >= 128 | piece share < 10% | largest piece | crop edges < 1% |
|---|---|---:|---:|---:|---:|---:|
| real HF14 + val | 40 / 76 / 168 | 5% | ~30% | 27% | 41% | 9% |
| Gemini30 | 41 / 100 / 191 | 6% | ~30% | 34% | 29% | 0% |
| synth, not mined, all | 16 / 56 / 129 | 29% | 10% | 31% | 29% | -- |
| synth, mined, all | 17 / 55 / 119 | 27% | 10% | 32% | 28% | -- |
| synth, not mined, kept (>= 32) | 39 / 72 / 151 | 0% | | 19% | 36% | 8% |
| synth, mined, kept (>= 32) | 38 / 68 / 135 | 0% | | 21% | 35% | 7% |

- **Mining does not select for small or odd boxes.** Mined and not-mined questions have the same box sizes, piece shares
  and crop content. The 32 px floor already removes the slivers (27% of synthetic questions vs 5-6% of real): below
  32 px the miner scores 0.615, real 0.855 (n=18).
- **Crop content does not predict failure.** Near-empty crops (edges < 1%; a box one pattern period wide on sparse
  battens / stipple, seen on a contact sheet of 16 hard mined questions) are 7-8% of synthetic and 9% of real
  questions, and score no worse (synthetic 0.854 vs 0.84 overall); 7% of the mined IoU < 0.6 questions, same as all mined.
- **What makes a mined sheet hard is over-selection:** of mined kept questions under IoU 0.6, 75% over-select (pred /
  target > 1.3), 17% under-select. Enriched reference roles: floor-plan floors x4.6-7.3 (miner 0.66), chimney x1.6,
  accent / accent2 x1.1-1.4; roof plans almost absent (3% of mined sheets vs 21%). That is look-alike families, the
  intended hardness.
- **One real gap: synthetic boxes are smaller than the user's.** After the floor, median 68-72 px vs 76 (real) / 100
  (Gemini); boxes >= 128 px are 10% of synthetic vs ~30% of real. Synthetic pieces are thinner (largest inscribed
  square median 56-57 px vs 99 / 127). Training draws boxes the same way (`sample_reference_box`, 128-512 native
  preference, capped by the piece), so large user boxes are under-represented in training; real IoU by box size is
  flat at 0.82-0.86 (n = 8 above 256 px), so the cost is not measurable here.

### 2026-10-14 (GH200) -- gable-trim screen: best 2k arm so far, but Gemini30 1 does not split at the bands at 2k

`revitgableroi` = the wing arm's flags / ids / seed with GEN_VERSION 2026-10-14b (`--gable-trim 0.5` default), same
2k recipe (seed 7, 1+8, epoch 8, hand boxes). Pool 1,989 sheets, 195 pair sheets (= wing). Per-question files
`eval_baselines/revitgable2k-e8_*`; not on the Hub.

| @2048 | HF14 | val | val w/o 17 | Gemini30 |
|---|---:|---:|---:|---:|
| r32 | 0.8049 | 0.7580 | 0.8448 | 0.7393 |
| wing | 0.7920 | 0.7527 | 0.8365 | 0.7646 |
| **gable** | **0.8264** | 0.7786 | 0.8525 | 0.7753 |

@4096: HF14 0.8302, val 0.8253.

- **vs wing: HF14 +0.034 (23 / 9, p=0.003), val +0.026 (p=0.07), Gemini30 +0.011 (p=0.13). vs r32: HF14 +0.022 (p=0.06),
  val +0.021, Gemini30 +0.036 (p=3e-4).** Best 2k screen on HF14. HF14 under-selection share 0.21 -> 0.00 (over 0.45 ->
  0.40). Sheets moved > 0.08 vs wing: HF14 25 0.72 -> 0.83, 27 0.69 -> 0.81 (shingle roof vs boards), 0 0.81 -> 0.66;
  Gemini30 17 0.72 -> 0.87, **1 0.82 -> 0.54**.
- **The test it was built for does not apply at 2k.** The band split on Gemini30 1 (r2 #004) appeared only in the long
  run (longwing q0 / q2 0.23 / 0.24); the wing 2k arm reads 0.82 / 0.90 / 0.73 there. The gable arm drops to 0.48 /
  0.77 / 0.38, but its maps show blotchy misses INSIDE every band, the reference's own included -- low confidence on
  one faint painted board-and-batten family, not a boundary rule. (Wing 2k shows the same blotches, milder, on q2.)
  So the trim did not fix a band split here (there was none) and did not create one.
- **Probe:** colour leak back up (tan vs olive 0.95 / 0.78, wing 0.61 / 0.10; tan vs blue 0.32 / 0.00); lap vs brick
  0.08 / 0.83 (wing 0.00 / 0.04) -- one-direction rows are noisy (2026-10-08 note), read with care.
- The direct test of the fix is the long-run version (mine on this pool, 20-epoch restart of longswiss) against longwing
  on Gemini30 1.

### 2026-10-14 (GH200) -- gable long run; longwing epoch 9 picked on val: neither beats longswiss; the trim does not fix Gemini30 1

**Gable long run.** `run_hard_mine.sh SRC=gable` (miner `revitgableroi` e8, 3,981 fresh sheets, GEN 2026-10-14b; mined
mean IoU 0.687, fresh kept 0.850) -> `revitgable_hard2000`; then the longwing recipe exactly (20-epoch restart of
longswiss-swa37-41, ROI add, peak LR 2e-4 cosine). Checkpoints `data/runs/ck_long_hardgable/` (not on the Hub). The old
fit-diagnostic pool `fit_fresh150_13c3cc06` (GEN 2026-10-14) was renamed `..._gen20261014` so the guard could rebuild it.

| @4096, hand boxes | HF14 | val | val w/o 17 | Gemini30 | Gemini30 1 (q0 / q1 / q2) |
|---|---:|---:|---:|---:|---|
| longswiss-swa37-41 (start) | 0.8779 | 0.8766 | 0.9093 | 0.8295 | 0.85 / 0.78 / 0.87 |
| longwing swa16-19 | 0.8805 | 0.8161 | 0.8846 | 0.8343 | 0.23 / 0.83 / 0.24 |
| **longwing e9** (val pick of e5-13, swa7-11, swa8-10) | 0.8710 | 0.8959 | 0.9195 | 0.8325 | -- |
| gable swa14-19 (val pick, last windows) | 0.8674 | 0.8863 | 0.9006 | 0.8404 | 0.37 / 0.49 / 0.28 |
| **gable swa7-11** (val pick, any window) | 0.8806 | 0.8903 | 0.9054 | 0.8282 | 0.24 / 0.63 / 0.21 |
| gable e4 / e9 / e14 / e19 | 0.8787 / 0.8736 / 0.8937 / 0.8550 | 0.8739 / 0.8428 / 0.8880 / 0.8803 | | 0.8438 / 0.8053 / 0.8491 / 0.8332 | e9 0.10 / 0.08 / 0.08 |

- **Paired vs longswiss, all noise:** longwing e9 HF14 -0.007 (p=0.70), val +0.019 (p=0.07), Gemini30 +0.003; gable swa7-11
  HF14 +0.003 (p=0.84), val +0.014, Gemini30 -0.001 (sign test 42 / 70, p=0.01); gable swa14-19 HF14 -0.011, Gemini30
  +0.011. **longswiss-swa37-41 stays the best model (HF14 0.8779).** Gable e14 reads HF14 0.8937 but val does not pick it.
- **The gable trim does not fix Gemini30 1.** Every gable checkpoint still splits it (e9: 0.10 / 0.08 / 0.08). The
  hypothesis that the perfectly predictive trim band caused the split is refuted.
- **longwing epochs 5-13 on val** (w/o 17): 0.884 / 0.883 / 0.883 / 0.895 / **0.920** / 0.890 / 0.868 / 0.890 / 0.885;
  swa7-11 0.896, swa8-10 0.912. Epoch 9 is a val peak that does not carry to HF14 or Gemini30: choosing among 11 points
  on 72 questions finds the lucky one.
- **Both long runs show the same shape:** a mid-run val bump and HF14 / Gemini30 flat at 0.87-0.88 / 0.83-0.85 throughout,
  while fit on synthetic keeps rising. Three restarts on three mined pools (failswiss, wing, gable) have all landed
  within +-0.01 of 0.878 on HF14.

### 2026-10-14 (CPU) -- HF14 18's two roofs are two patterns (label fix v2)

The user (2026-10-14): on HF14 18 the left and right roof pieces, both labelled pattern3, are different patterns; either
answer is acceptable, two families is the correct one. `eval_labels/hf14_fixes_v2.json` = v1 + a "reassign" fix moving
annotation 9 (right roof) to its own category (`evaluate_refunet_selection.py --label-fixes` now supports "reassign";
boxes and questions unchanged); `run_revit_2k.sh`'s hf14fix split uses v2. Rescored from the saved maps (sheet 18 only):
longswiss q8 / q9 0.533 / 0.236 -> 0.321 / 0.298, HF14 0.8779 -> 0.8749; longwing 0.523 / 0.636 -> 0.883 / 0.741, HF14
0.8805 -> 0.8896. Longwing separates the two roofs; longswiss merges them.

### 2026-10-14 (CPU) -- checkpoint selection on val is per sheet

`select_epoch_on_val.val_score`: the mean over val SHEETS of each sheet's mean IoU, used by `select_epoch_on_val.py` and
`average_checkpoints.py` to pick (the per-question mean is still printed); `run_hard_long.sh` prints a `val_sheet`
column. Per question, sheet 17 (23 of 72 questions) decided every pick. On the saved longwing files the per-sheet score
still ranks e9 first (0.9287), then swa8-10 (0.9257) and longswiss (0.9254); swa16-19 0.9011.

### 2026-10-14 (CPU) -- verifying "val 17's dormers are lighter" and "roof sections turn": wrong / not the cause

- **val 17 (q15, the worst longwing loss):** the reference piece (a dormer cheek, 73x103 px) has fill L72 a5 b-13 --
  the SAME as the main-wall pieces 9 / 10 (L72). Some dormer faces of the family are darker (L57, same hue; CIEDE2000
  12.4). Longwing keeps the dormers and drops the main wall, which matches the reference's colour exactly: colour does
  not explain it. The earlier "dormers are a lighter shade" line was wrong.
- **Gemini30 17 (q1, hip roof plan):** the family's sections do turn (line direction 0 / 90 deg) and also change spacing
  2x (9.6-10.7 vs 19.2 px; Gemini inconsistency). But longwing drops same-direction sections as much as turned ones
  (dir 0: 0.08-0.30 selected; dir 90: 0.06-0.28): direction does not explain it.
- **What does fit: pieces much bigger than the reference piece.** Over every question x other same-family piece (HF14 +
  val + Gemini30; `scratchpad piece_size.py`), share selected by piece area / reference-piece area:
  0-0.25x 0.83 -> 0.83, 0.25-0.8x 0.93 -> 0.91, 0.8-1.25x 0.95 -> 0.94, 1.25-4x 0.91 -> 0.90, 4-16x 0.87 -> 0.85,
  **> 16x 0.87 -> 0.69** (longswiss -> longwing; n 64, of which val 17 24, Gemini30 9 6, val 13 5, Gemini30 2 / 16 / 17
  4 each). Longwing skips family pieces far larger than the piece the box sits on -- a small dormer / valley-facet
  reference loses the main wall / the big roof sections. Small n, concentrated on a few sheets.

### 2026-10-14 (CPU) -- the size check, inference levers, and where 0.92 has to come from

- **Small-reference / big-family questions are NOT missing from synthetic data.** Share of questions whose family has a
  piece > 16x the reference piece: synthetic (wing generator, 2,824) 9.5%, Gemini30 10.4%, HF14 5.8%, val w/o 17 22%;
  > 4x: 28% / 24% / 15% / 26%. The longwing size effect (previous section) is not a data gap.
- **Inference levers on the two long models' saved maps (prob-map resolution, within +0.002-0.006 of official):** a
  0.5 / 0.5 probability average of longswiss + longwing: HF14 0.888 (0.885 with fix v2), Gemini30 0.844, val per sheet
  0.927 (longswiss alone 0.929); box-blurring the probabilities before the 0.35 threshold: 9 px +0.002 HF14 / -0.003
  Gemini30, wider loses. Ceiling of what we hold: ~0.89 HF14.

### 2026-10-14 (GH200) -- the best model on hard-mined synthetic; 18,000 fresh sheets seen once, with and without DR

**How good is longswiss on hard synthetic?** Scored @4096 (never trained on these) on the 1,000 mined sheets of
`revitgable_hard2000` (10,720 questions): **0.812** (boxes >= 32 px: 0.825; family-weighted per sheet, as mining: 0.790);
elevations 0.838, floor plans 0.733, pair sheets 0.804; loss 45% over-selection, 9% under. Fresh synthetic (300) 0.874,
HF14 0.878, Gemini30 0.830. So hard synthetic is not solved for the best model -- the limit is transfer, not supply.
File `eval_baselines/longswiss-swa37-41_gablemined1000_4096.json` (and `_synth300_`).

**18k single pass.** `single18k_s600000`: 18,000 sheets at GEN 2026-10-14b defaults (seed 6, ids 600000+; 17,907 ok,
7.7 min). Restart of longswiss-swa37-41 with the restart recipe for ONE epoch (2,214 steps, peak LR 2e-4 cosine to
1e-5, ROI add, `--train-split 0.99`, seed 7), arm `dr` with `--domain-random --dr-scale-min 0.7`, arm `nodr` without
(flips / rot90 are always on). ~20 min each. Checkpoints `data/runs/ck_single18k_{dr,nodr}/epoch_0.pth` (not on the Hub).

| @4096, hand boxes | HF14 | val | val w/o 17 | Gemini30 | mined 1000 | fresh 300 |
|---|---:|---:|---:|---:|---:|---:|
| longswiss-swa37-41 | 0.8779 | 0.8766 | 0.9093 | 0.8295 | 0.8123 | 0.8739 |
| 18k once, DR | 0.8849 | 0.8860 | 0.9010 | 0.8211 | +0.020 | +0.010 |
| **18k once, no DR** | 0.8573 | 0.8802 | 0.8910 | **0.8622** | **+0.052** | **+0.029** |

- **DR arm: real flat, synthetic up** -- HF14 +0.007 (p=0.60), val +0.010 (sign 20 / 4), Gemini30 -0.008; mined +0.020
  (p=1e-42), fresh +0.010. Same picture as the three mined restarts, with no repetition: repetition was not why they
  stalled.
- **No-DR arm: Gemini30 +0.033 vs longswiss (p=0.001; +0.041 vs DR, p=6e-8) -- the best Gemini30 of any model -- but
  HF14 -0.021 (p=0.18; -0.028 vs DR, p=0.11).** Synthetic gains double (mined +0.052, fresh +0.029). Gemini30 4 0.66 ->
  0.86, 8 0.76 -> 0.84, 11, 17, 26 0.38 -> 0.55 rise. HF14 loss is under-selection (share 0.17 -> 0.58, over 0.20 -> 0.00)
  on 14 (0.88 -> 0.71) and 12 (0.71 -> 0.64); 0 rises 0.86 -> 0.93. Not small boxes: real questions with native box side
  < 150 px move -0.006, larger -0.008 (unlike the from-scratch no-DR 2k run of 2026-09-28, where < 32 px lost 0.44).
  Gemini sheets are consistent renders (one scale, clean tone), which DR's shrink / jitter does not help; HF14 is not.
- **Gemini30 1 breaks in every model trained on the 2026-10-14 generator** (wing colour pairs): longwing 0.23 / 0.83 /
  0.24, longgable 0.24 / 0.63 / 0.21, 18k DR 0.258, 18k no-DR 0.236 (sheet means), against longswiss 0.833. The trim
  was not the cause (2026-10-14 gable entry); the wing pairs are the common factor.

### 2026-10-14 (GH200) -- 50/50 blend of the two 18k arms: new best, HF14 0.8877

`data/runs/ck_single18k_blend50.pth` = the float mean of `ck_single18k_dr/epoch_0.pth` and `ck_single18k_nodr/epoch_0.pth`
(no training). Hub: `abshetty/floz-refunet-swint-single18k-blend50` (private; round-trip identical, HF14 re-read
0.8877). Per-question files `eval_baselines/single18k-blend50_*`; maps on `abshetty/floz-diag` `single18k-blend50/`.

| @4096, hand boxes | HF14 | val | val per-sheet | val w/o 17 | Gemini30 | Gemini30 1 | fresh 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| longswiss-swa37-41 | 0.8779 | 0.8766 | 0.9254 | 0.9093 | 0.8295 | 0.833 | 0.8739 |
| 18k DR | 0.8849 | 0.8860 | 0.9252 | 0.9010 | 0.8211 | 0.258 | +0.010 |
| 18k no DR | 0.8573 | 0.8802 | 0.9194 | 0.8910 | 0.8622 | 0.236 | +0.029 |
| **blend 50/50** | **0.8877** | **0.8964** | **0.9339** | **0.9149** | 0.8497 | 0.297 | +0.026 |

- **Paired vs longswiss: HF14 +0.010 (17 / 7, p=0.35; sign p=0.06), val +0.020 (22 / 2, p=0.036), Gemini30 +0.020
  (81 / 41, p=0.024), fresh synthetic +0.026 (p=2e-38).** Best HF14, val (both the question and the per-sheet mean)
  and second-best Gemini30 of any model. One candidate, not a selection among many, so val is not optimistic.
- The blend is better on HF14 than either parent (DR 0.8849, no-DR 0.8573): the two arms err differently (DR over-,
  no-DR under-selects on HF14), and averaging the weights lands between them.
- Gemini30 1 is still broken (0.297): the wing-pair failure stays.

### 2026-10-14 (GH200) -- counterfactual probe cf_v1 (17 worst sheets, one change per variant)

`make_counterfactual_pool.py --out data/eval_pools/cf_v1` (17 sheets x 6 image variants, 164 questions); every variant scored @4096,
mask 0.35, `--boxes` from the pool (orig boxes, or `boxes_boxlarge` / `boxes_box2x` on the orig images). Per-question JSONs
`eval_baselines/cf_v1/<tag>_<variant>.json`. `counterfactual_summary.py` output:

```
== cf summary single18k-blend50
orig: 0.8306 over 164 questions (under-selecting 10, over 12, ok 142)
autocontrast  0.8280  (-0.0026) | under -0.018 over +0.016 ok -0.003 | gain>0.1: 2  loss>0.1: 1
box2x         0.4675  (-0.3631) | under +0.062 over -0.191 ok -0.408 | gain>0.1: 7  loss>0.1: 100
boxlarge      0.8282  (-0.0024) | under +0.189 over +0.008 ok -0.017 | gain>0.1: 10  loss>0.1: 13
flatbg        0.7991  (-0.0315) | under +0.084 over +0.025 ok -0.044 | gain>0.1: 11  loss>0.1: 36
gray          0.7114  (-0.1192) | under +0.124 over -0.118 ok -0.136 | gain>0.1: 5  loss>0.1: 49
sharpen       0.8044  (-0.0261) | under +0.060 over -0.015 ok -0.033 | gain>0.1: 7  loss>0.1: 15
soften        0.8488  (+0.0182) | under +0.191 over +0.174 ok -0.007 | gain>0.1: 12  loss>0.1: 8

== cf summary longswiss-swa37-41
orig: 0.8049 over 164 questions (under-selecting 14, over 16, ok 134)
autocontrast  0.7979  (-0.0070) | under -0.030 over +0.004 ok -0.006 | gain>0.1: 2  loss>0.1: 6
box2x         0.5376  (-0.2673) | under +0.079 over -0.076 ok -0.326 | gain>0.1: 11  loss>0.1: 89
boxlarge      0.8115  (+0.0066) | under +0.259 over +0.027 ok -0.022 | gain>0.1: 19  loss>0.1: 15
flatbg        0.7648  (-0.0401) | under -0.056 over +0.031 ok -0.047 | gain>0.1: 7  loss>0.1: 32
gray          0.6693  (-0.1356) | under -0.060 over -0.027 ok -0.156 | gain>0.1: 1  loss>0.1: 53
sharpen       0.7530  (-0.0519) | under -0.041 over -0.044 ok -0.054 | gain>0.1: 5  loss>0.1: 38
soften        0.8184  (+0.0135) | under +0.131 over +0.155 ok -0.016 | gain>0.1: 20  loss>0.1: 18
```

Change against orig by source (mean over questions; HF14 n=30, val n=54, Gemini30 n=80):

| variant | blend50 HF14 / val / Gemini30 | longswiss HF14 / val / Gemini30 |
|---|---|---|
| soften | -0.046 / -0.009 / **+0.061** | -0.029 / -0.005 / **+0.042** |
| boxlarge | -0.024 / +0.014 / -0.005 | +0.019 / +0.032 / -0.016 |
| flatbg | -0.037 / -0.062 / -0.009 | -0.033 / -0.105 / +0.001 |
| gray | -0.050 / -0.164 / -0.115 | -0.032 / -0.240 / -0.104 |

- **Both models agree on every direction.** No single image change rescues the real sheets: every variant loses on
  HF14 except boxlarge for longswiss (+0.019).
- **soften is the only net gain, and it is all Gemini30** (+0.06 / +0.04); on HF14 it costs 0.03-0.05. Gemini sheets are
  clean high-resolution renders; halving their sharpness brings them toward the training scale. Same direction as the
  no-DR 18k arm, whose gain was also Gemini30-only. Real scans are not "too sharp".
- **Colour carries the real sheets:** gray costs 0.03-0.05 on HF14, 0.16-0.24 on val, 0.10-0.12 on Gemini30.
- **flatbg (paper tone removed) hurts, most on val** (-0.06 / -0.11): the paper / scan tone is information, not noise.
- **boxlarge fixes under-selection** (under-selecting questions +0.19 / +0.26) but loses elsewhere; net about zero. The
  model's misses on real sheets are partly the reference sitting on a small or atypical piece.
- **box2x is not a fair variant** (-0.36 / -0.27): doubling the box usually spills it over other families, so it asks
  a different question.

### 2026-10-14 (GH200) -- wing pairs off in the best recipe: no net effect; they trade Gemini30 1 for 26

The single18k experiment repeated exactly (ids 600000+, seed 6, GEN 2026-10-14b, one-epoch restarts of
longswiss-swa37-41 with / without `--domain-random --dr-scale-min 0.7`, 50/50 weight blend), with the pool generated
`--near-pairs 0` (`single18k_np0_s600000`, 17,907 ok). Checkpoints `data/runs/ck_single18knp0_{dr,nodr}/epoch_0.pth`,
blend `data/runs/ck_single18knp0_blend50.pth` (not on the Hub). Per-question files
`eval_baselines/single18knp0-blend50_*`; maps in `data/diag/single18knp0-blend50/`.

| @4096, hand boxes | HF14 | val | val per-sheet (`val_score`) | Gemini30 | Gemini30 1 | Gemini30 26 |
|---|---:|---:|---:|---:|---:|---:|
| longswiss-swa37-41 | 0.8779 | 0.8766 | 0.9254 | 0.8295 | 0.833 | 0.380 |
| single18k-blend50 (wing pairs) | 0.8877 | 0.8964 | 0.9339 | 0.8497 | 0.297 | 0.514 |
| **single18k-np0-blend50 (no pairs)** | **0.8894** | 0.8961 | **0.9345** | **0.8538** | **0.606** | 0.419 |

- **Paired, no pairs vs pairs: HF14 +0.002 (11 / 13, p=0.82), val -0.0002 (7 / 5), Gemini30 +0.004 (51 / 48, p=0.36).**
  No sheet of HF14 or val moves by more than 0.07. The pairs neither help nor hurt on the real sheets.
- **They trade two Gemini sheets:** without them Gemini30 1 recovers 0.30 -> 0.61 (and 0 0.68 -> 0.76), and 26 gives
  back 0.51 -> 0.42. Both effects are about the size of the per-sheet noise band (0.14) on 26, beyond it on 1.
- **Answer: the wing pairs do not pay for themselves.** They fix the merge they were built for (26 stays above
  longswiss's 0.38 either way, so most of 26's gain came from the 18k recipe, not the pairs) and cost the one-family
  sheet 1 about the same, net zero.
- Gemini30 1 is still well below longswiss (0.61 vs 0.83) without the pairs, so the 18k recipe itself (or the 2026-10-14b
  generator's other changes) also costs it.
- The no-pairs blend is a tie with single18k-blend50 for best model (val per-sheet 0.9345 vs 0.9339, HF14 0.8894 vs
  0.8877, all differences noise); not published.

### 2026-10-14 (CPU) -- wing pairs off by default (GEN_VERSION 2026-10-14c)

`--near-pairs` default 0.6 -> 0 after the single18k no-pairs arm (previous GH200 entry): real-sheet scores unchanged,
Gemini30 1 recovers. The wing / gable arms in `run_revit_2k.sh` / `run_hard_mine.sh` now pass `--near-pairs 0.6`
explicitly. Pair code stays.

### 2026-10-14 (GH200) -- 18k checkpoints on the Hub

Published (private, safetensors; each round-trip identical to the local file): `abshetty/floz-refunet-swint-single18knp0-dr`,
`-single18knp0-nodr`, `-single18knp0-blend50`, `-single18k-dr`, `-single18k-nodr` (and earlier `-single18k-blend50`).
The 54k run was not started (VM shutdown).

### 2026-10-14 (GH200) -- eval-only weight blends around single18knp0-blend50: all ties

@4096, hand boxes, vs `single18knp0-blend50` (val per-sheet 0.9345, val 0.8961, HF14 0.8894, Gemini30 0.8538), paired. (a)
single18knp0-blend50 averaged 50/50 with longswiss-swa37-41: val per-sheet 0.9326 (val -0.004, p=0.18), HF14 0.8935
(+0.004, p=0.38), Gemini30 0.8497 (-0.004, p=0.31). (b) DR 0.7 / no-DR 0.3: val per-sheet 0.9349 (val +0.0002), HF14 0.8855
(-0.004, p=0.11), Gemini30 0.8477 (-0.006, p=0.001). (c) DR 0.3 / no-DR 0.7: val per-sheet 0.9339 (val -0.0004), HF14
0.8892 (-0.0002), Gemini30 0.8594 (+0.006, p=0.04). By `val_score` (b) is highest, by 0.0004 -- far inside noise -- and its
HF14 is 0.004 lower, so nothing replaces single18knp0-blend50. The only consistent signal is the DR share on Gemini30
(more no-DR, more Gemini30: 0.8477 / 0.8538 / 0.8594 at DR 0.7 / 0.5 / 0.3), matching the no-DR arm and the counterfactual
soften result; HF14 and val are flat across 0.3-0.7. Per-question files `eval_baselines/single18knp0-blend{70,30,50-longswiss50}_*`;
blend checkpoints `data/runs/ck_single18knp0_blend{30,70,50_longswiss50}.pth` (not on the Hub).

### 2026-10-14 (CPU) -- loss breakdown of single18k-blend50 (maps from floz-diag)

| IoU lost to | HF14 longswiss -> blend50 | val | Gemini30 |
|---|---|---|---|
| merge with another family | 0.029 -> 0.005 | 0.002 -> 0.001 | 0.049 -> 0.039 |
| unlabelled drawing | 0.010 -> 0.008 | 0.035 -> 0.033 | 0.027 -> 0.025 |
| missed inside the family | 0.046 -> 0.061 | 0.036 -> 0.022 | 0.050 -> 0.042 |
| missed whole large pieces | 0.022 -> 0.032 | 0.002 -> 0.013 | 0.003 -> 0.003 |
| total loss | 0.122 -> 0.112 | 0.123 -> 0.104 | 0.171 -> 0.150 |

HF14 is now almost all misses (FN 0.085 of 0.112; 45% of the loss on under-selecting questions), and three sheets hold
them: 12 (floor plan; 0.020 of whole pieces missed), 14 and 18 -- 0.062 of the 0.112. 0.92 on HF14 means fixing
those sheets' misses; merges are done.

### 2026-10-14 (GH200) -- what else is on the Hub before shutdown

- **Swiss layouts:** `data/reference/swiss_dwellings/layouts.pkl.gz` (the `scripts/swiss_plans.py` output, 19,046 layouts)
  is in the private dataset `abshetty/floz-assets` as `swiss_dwellings/layouts.pkl.gz` (sha256 round-trip identical).
  Fetch instead of the 932 MB Zenodo download + ~20 min build:
  `python3 -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('abshetty/floz-assets', 'swiss_dwellings/layouts.pkl.gz', repo_type='dataset'), 'data/reference/swiss_dwellings/layouts.pkl.gz')"`
  (CC BY 4.0, Archilyse AG, Zenodo 7788422: credit required in any released data or model).
- **Miners:** `abshetty/floz-refunet-swint-revitgable2k-e8` (miner of `revitgable_hard2000`) and `-revitwing2k-e8` (miner of
  `revitwing_hard2000`), round-trip identical; re-mining no longer needs the 2k screen retrained.
- **Maps of the current best** `single18knp0-blend50` (HF14 / val / Gemini30 @4096) on `abshetty/floz-diag`,
  `single18knp0-blend50/`.
- **Gable mining scores** (per-question IoU on 3,981 fresh sheets, miner revitgable2k-e8 @2048):
  `eval_baselines/hardmine_gable/`.

### 2026-10-14 (CPU) -- HF14 12: one deck family whose boards turn; generator --turn-boards (GEN_VERSION 2026-10-14d)

The best model's HF14 loss sits on 12 / 14 / 18. **HF14 12** (floor plan; one family = deck boards around the house): the
best model (single18k-blend50) selects only the pieces whose boards run like the reference's -- box on a horizontal-board
deck: horizontal pieces 0.97, vertical pieces 0.06-0.12; box on a vertical deck: the horizontal piece 0.06.
**Causal test** (CPU, @4096, the model's own boxes): redraw the two vertical decks with the horizontal deck's boards ->
sheet mean 0.680 -> **0.923** (q1 0.71 -> 0.94, q3 0.32 -> 0.91); control, redrawn with their own vertical boards ->
0.736. The direction change itself is what the model reads as a family boundary.
**The gap:** `revit_plans.floor_material` gives a floor family ONE board direction for all its rooms. Real rooms and
decks turn their boards. **Change:** `--turn-boards P` (default 0.6): on a plan chosen with P, every room (and doorway)
of a directional floor family (planks, plank lines, joists, 45-degree hatch) gets its own direction; labels unchanged;
own RNG, and the v3 per-room stream (p3) is consumed exactly as before. 60-plan check: 9 / 59 plans change, labels
identical on all 59, `--turn-boards 0` byte-identical to the previous commit. `run_revit_2k.sh`'s r32 arm passes 0.

### 2026-10-14 (CPU) -- HF14 14: a thin band boxed in heavy lines; generator --heavy-pieces (GEN_VERSION 2026-10-14e)

**HF14 14** (one lap-siding family: four panels under the windows + two thin bands, 76 px tall, 4 lines, boxed in heavy
lines). The bands' lines match the panels exactly (spacing 15 px, darkness 35, both). Best model, @4096, own boxes;
selected share of panels 0-3 / band 4:

| variant | q4 (box on band, 64 px) | q0 (box on panel) |
|---|---|---|
| original | 0.55 (panels 0.00-0.35) | 0.66 (band 0.12) |
| heavy band outlines whitened, texture kept | **0.90** | **0.75** (band 0.55) |
| panel texture inside the band, outlines kept | 0.77 | 0.69 (band 0.13) |
| control: 64 px box on a panel (q1) | 0.94 (bands 0.94-0.98) | |

The heavy outlines are the cause; box size is not. Synthetic elevations drew heavy lines only around the whole
building (the profile); every piece inside got the thin outline, so a band boxed in heavy lines was never one family
with large panels. **Change:** `--heavy-pieces P` (default 0.4): on a sheet chosen with P, surface pieces are re-outlined
at 0.6-1.0x the heavy profile weight, thin pieces (< 1.5 ft) with probability 0.6, others 0.15; labels unchanged; own
RNG. 40-elevation check: 14 / 40 change, labels identical, `--heavy-pieces 0` byte-identical; r32 arm passes 0.

### 2026-10-14 (CPU) -- remaining HF14 / val failures; railings 0.35 and the run flags as defaults (GEN_VERSION 2026-10-14f)

Per-piece census (best model single18k-blend50, HF14 + val + Gemini30, 1,509 same-family pieces): only 37 pieces are
missed as a whole (< 30% selected); most of the loss is partial. Among the 37, "line darkness differs from the reference
piece by > 30" is 57% (kept pieces 12%) -- HF14 12 (turned boards, fixed), 14 (heavy outlines, fixed), 18 (label, fix
v2) and val 13. Direction > 30 deg 5% / 9%, spacing > 1.3x 0% / 7%, fill CIEDE2000 > 5 0% / 9%.
- **val 13 q7 (0.17):** the box sits on a strip under a glazed door whose lap lines are 25 px apart and darker (the
  family elsewhere: 15 px). Erasing the heavy ground line (0.07), the dashed level line (0.16), both (0.16), or
  lightening the strip's lines (0.16) changes nothing. The strip looks genuinely different; not a generator gap.
- **Gemini30 1 (0.24-0.33):** the cf_v1 probe already holds it -- gray 0.69-0.86, flatbg 0.56-0.70, soften 0.45-0.83
  rescue it for the 18k blend, longswiss needs none. The family's colour is uniform (CIEDE2000 <= 2 across the sheet):
  pale blue-grey boards close to warm paper. Unexplained; one sheet.
- **Heavy lines in the box are not a general failure:** boxes with > 2% heavy stroke pixels: HF14 4% (IoU 0.59 vs 0.90:
  the two sheet-14 band boxes), val 25% (0.88 vs 0.90), Gemini30 0.5%, synthetic 27% (0.92 vs 0.90).
- **HF14 27 (railing):** the family continues behind a balcony railing and is missed there; replacing the railing with
  the siding above it: q1 0.865 -> 0.928, q0 0.920 -> 0.928. Synthetic railings (see-through pickets 0.3-0.5 ft, posts,
  rails) look like the real one, but only on 15% of views and only when a run passes `--railings`.
**Change:** generator defaults `--railings 0.35` (was 0; runs passed 0.15), `--roof-lines 0.3`, `--faint-lines 0.2`
(the values every run passed). New defaults == the old code with those three flags (byte-identical, 20 sheets);
`--roof-lines 0 --railings 0 --faint-lines 0` == the old defaults; the cur / mk50 arms pass the zeros.

### 2026-10-15 (GH200) -- single18k-np0 repeated at GEN 2026-10-14f: HF14 0.9157 (+0.026, p=0.006), val per-sheet a tie

The single18knp0 experiment repeated exactly on a pool at the 2026-10-14f defaults (turn-boards 0.6, heavy-pieces 0.4,
railings 0.35, roof-lines 0.3, faint-lines 0.2, wing pairs off) with only `--plan-source`: `single18kf_s600000` (ids 600000+,
seed 6, 17,907 ok -- same count as np0, 469 s), one-epoch restarts of longswiss-swa37-41 with / without `--domain-random
--dr-scale-min 0.7` (2,215 steps, ROI add, seed 7, one arm after the other: both at once OOM the 94.5 GB GH200), float 50/50
blend. Now scripted: `TAG=single18kf ./run_single_pass.sh`. Checkpoints `data/runs/ck_single18kf_{dr,nodr}/epoch_0.pth`,
`data/runs/ck_single18kf_blend50.pth` (local only). FreeCAD 2026.09.16 (conda-forge) on this VM. Baseline re-read first:
single18knp0-blend50 HF14 0.8894 exactly. Per-question files `eval_baselines/single18kf-blend50_{hf14,hf14fix2,val,gemini30}_4096.json`
and the baseline's `eval_baselines/single18knp0-blend50_hf14fix2_4096.json`.

| @4096, hand boxes | HF14 | HF14 fix v2 | val | val per-sheet (`val_score`) | Gemini30 |
|---|---:|---:|---:|---:|---:|
| single18knp0-blend50 | 0.8894 | 0.9038 | 0.8961 | **0.9345** | 0.8538 |
| **single18kf-blend50** | **0.9157** | **0.9224** | 0.8971 | 0.9342 | 0.8526 |
| paired | +0.026 (17 / 4, p=0.006) | +0.019 (16 / 5, p=0.08) | +0.001 (5 / 3, p=0.46) | -0.0003 | -0.001 (38 / 47, p=0.73) |

- **HF14: the targeted sheets moved.** 12 (turned boards) 0.709 -> **0.893** (q0 0.92 -> 0.94, q1 0.72 -> 0.93, q2 0.80 -> 0.94,
  q3 0.40 -> 0.77): 78% of the gain, close to the 0.923 the causal redraw promised. 14 (heavy outlines) 0.867 -> 0.882 (q4, the
  box on the band, 0.84 -> 0.91; q0 0.62 -> 0.61). 27 (railing) 0.846 -> 0.886 (q0 0.91 -> 0.93, q1 0.84 -> 0.85, q2 0.69 -> 0.77,
  q4 0.87 -> 0.94) -- q1, the behind-railing question, barely moves. No HF14 sheet loses more than 0.003 on the original labels.
- **Fix v2 gain is smaller because of 18:** with 18's two roofs as two patterns, 18 loses 0.030 (q8 0.80 -> 0.67, q9 0.97 -> 0.79):
  the new model merges the two roofs more. On the original labels (one family) 18 rises 0.875 -> 0.884.
- **val: flat** (no sheet moves > 0.05). **Gemini30: flat overall, sheets trade:** 8 0.77 -> 0.84, 19 0.40 -> 0.46, 26 0.42 -> 0.47
  up; 1 0.61 -> 0.46, 0 0.76 -> 0.68, 4 0.74 -> 0.67 down. Gemini30 1 is lower again (it was 0.30 with the wing pairs, 0.61 without).
- **Not published:** the gate was val per-sheet above single18knp0-blend50, and it is a tie (0.9342 vs 0.9345). The HF14 gain is
  read once and is the targeted-sheet effect the CPU diagnoses predicted, but those diagnoses were made on HF14 sheets, so HF14
  is no longer a clean holdout for these three changes; val does not confirm them (val has no turned-board / heavy-band sheet
  of that kind). No maps saved.
- **Published anyway (the user's call, same day): new best.** `abshetty/floz-refunet-swint-single18kf-blend50` (private,
  safetensors, round-trip identical: 243 / 243 tensors). val / Gemini30 re-scored @4096 with `--save-probs`, per-question IoU
  identical to the files above; maps + metrics on `abshetty/floz-diag` `single18kf-blend50/{val,gemini30}_4096/` (72 + 195).

### 2026-10-15 (CPU) -- Gemini30 drops of single18kf-blend50: diagonal vs orthogonal floor lines merged; plan direction pairs back (GEN 2026-10-15)

Maps `single18kf-blend50/gemini30_4096`. The new model loses > 0.08 on 7 Gemini30 questions (none on HF14 / val), on 3 sheets:
- **Gemini30 4 (crawlspace plan), q4 0.95 -> 0.72, q2 0.41 -> 0.32, q5 0.80 -> 0.71:** a room of 45-degree hatch and a room of
  horizontal lines at the same spacing and pen are two families; the new model selects both (pred/tgt 1.04 -> 1.38). The
  older single18k-blend50 already leaked into the other room (q2 pred/tgt 2.77), turn-boards made it worse. Cause in the
  generator: `_distinct` puts `hatch45` and `plank_lines` in one look class and never lets direction separate two families,
  so in synthetic data diagonal next to orthogonal at one spacing never was two families; `--turn-boards` then added
  "direction changes across rooms, same family" with no counter-example. The plan near pairs that supplied the
  counter-example were off since 2026-10-14 (all three kinds taught "another room = another family", Gemini30 5).
- **Gemini30 0 (coloured plan), q1 0.77 -> 0.62, q3 0.66 -> 0.55:** the tile reference also takes the sofas / chairs; the
  older single18k-blend50 does the same (pred/tgt 1.21 / 1.38). Not new, no fix.
- **Gemini30 1 (board-and-batten, orange marker), q0 0.35 -> 0.21, q2 0.64 -> 0.34:** the model takes the battens only, not
  the boards between; single18k-blend50 does the same (0.24 / 0.32) -- np0's 0.64 was the outlier. Unexplained (2026-10-14:
  grey rescues it), no fix.

**Change:** `--near-pairs-plan` default 0.3 with new `--near-pairs-plan-kinds` default `direction` (orthogonal plank lines
vs 45-degree hatch, same spacing, pen and fill, two zones). Turned boards turn 0 <-> 90 or 45 <-> -45, never
orthogonal <-> diagonal, so the two rules are consistent: "a 90-degree turn is the same family, diagonal vs orthogonal is
another". 9 of 40 test plan sheets get a pair (finish sheets only). Byte-identical (40 plan sheets): `--near-pairs-plan 0`
== the 14f code's defaults; `--near-pairs-plan 0.6 --near-pairs-plan-kinds direction,spacing,tone` == the 14f code with
`--near-pairs-plan 0.6` (2026-10-07..13 behaviour). Not measured yet: needs a training run (GPU).

### 2026-10-15 (GH200) -- plan direction pairs on (GEN 2026-10-15): Gemini30 4 fixed, HF14 12 loses half its turn-boards gain

`TAG=single18kp ./run_single_pass.sh`: same ids / seed as single18kf (600000+, seed 6, default GEN = 2026-10-15, only
`--plan-source`), so against single18kf-blend50 only the plan direction pairs differ (`--near-pairs-plan 0.3`, kinds
`direction`). Pool 17,906 ok (one plan sheet fewer than kf). Same training / blend. Per-question files
`eval_baselines/single18kp-blend50_{hf14,hf14fix2,val,gemini30}_4096.json`.

| @4096, hand boxes | HF14 | HF14 fix v2 | val | val per-sheet (`val_score`) | Gemini30 |
|---|---:|---:|---:|---:|---:|
| single18kf-blend50 | **0.9157** | **0.9224** | 0.8971 | 0.9342 | 0.8526 |
| single18kp-blend50 (plan pairs) | 0.8983 | 0.9000 | 0.8964 | 0.9343 | **0.8605** |
| paired | **-0.017 (5 / 16, p=0.027)** | -0.022 (5 / 16, p=0.015) | -0.001 (4 / 10, p=0.73) | +0.0001 | +0.008 (47 / 46, p=0.20) |

- **Gemini30 4 fixed, as designed:** 0.666 -> **0.943**; q2 0.32 -> 0.92 (pred/tgt 3.04 -> 1.04), q3 0.33 -> 0.90 (2.90 -> 1.05),
  q4 0.72 -> 0.98, q5 0.71 -> 0.98; q0 / q1 0.96 / 0.95 -> 0.94 / 0.93. The diagonal-vs-orthogonal merge is gone.
- **Gemini30 5 does not split:** 0.958 -> 0.951 (q0 0.96 -> 0.96, q1 0.96 -> 0.96, q2 0.95 -> 0.93) -- direction-only pairs do not bring
  back the "another room = another family" error of the 2026-10-07..13 pairs.
- **HF14 12 gives back half of turn-boards' gain:** 0.893 -> **0.805** (q1 0.93 -> 0.82, q3 0.77 -> 0.53; pred/tgt 0.87 / 0.54:
  under-selection is back). The pairs and turn-boards pull against each other on 12's decks. HF14 0 also drops 0.936 -> 0.896
  (q0 0.94 -> 0.90, q1 0.93 -> 0.89), 7 0.946 -> 0.929. Every other HF14 sheet within 0.05; no HF14 sheet gains more than 0.01.
- Other plan sheets moving > 0.05: Gemini30 0 0.682 -> 0.747 (q1 0.62 -> 0.78, the sofa leak shrinks: pred/tgt 1.28 -> 0.95).
  No roof plan moves > 0.03 (Gemini30 17 +0.027). Non-plan: Gemini30 1 0.455 -> 0.650 (q0 0.21 -> 0.39, q2 0.34 -> 0.69), Gemini30
  8 0.838 -> 0.763 (q3 / q4 0.69 / 0.67 -> 0.51 / 0.51, over-selection 1.3 -> 1.6). val: no sheet moves > 0.05.
- **Gate (val per-sheet and Gemini30 >= single18kf-blend50): passed, by 0.0001 on val**, so published:
  `abshetty/floz-refunet-swint-single18kp-blend50` (private, round-trip identical 243 / 243); val / Gemini30 maps re-scored with
  `--save-probs` (IoU identical) on `abshetty/floz-diag` `single18kp-blend50/{val,gemini30}_4096/`. **Not made the best model in
  `startup.md`:** HF14 -0.017 is significant (p=0.027) and sits on the sheet the 14f changes fixed. The gate did not look at HF14.

### 2026-10-15 (GH200) -- HF14 maps of both 18k blends; HF14 in the publish gate

single18kf-blend50 and single18kp-blend50 re-scored on HF14 @4096 (hand boxes) with `--save-probs`: per-question IoU identical to
`eval_baselines/single18k{f,p}-blend50_hf14_4096.json`; maps (51 each) + metrics on `abshetty/floz-diag` `<tag>/hf14_4096/`.
Both blends now have HF14 / val / Gemini30 maps there. **Publish gate from now on** (`startup.md`): Gemini30 up, and neither
val per-sheet nor HF14 down, against the current best.

### 2026-10-15 (CPU) -- HF14 12 under single18kp: the turned decks split again; more turned boards (GEN 2026-10-15b)

HF14 maps `single18k{f,p}-blend50/hf14_4096`, sheet 12. The decks are plank lines: vertical on the top-right strip and the
covered porch, horizontal on the left L-deck and part of the top strip, all one family. single18kp keeps the reference's
direction only: q3 (box on the vertical top-right strip) drops the horizontal L-deck (0.77 -> 0.53), q1 drops the
vertical porch / strip (0.93 -> 0.82). The direction pairs (orthogonal vs 45-degree) were learned as the wider rule
"another direction = another family", outweighing the turned boards.

**Change:** `--turn-boards` default 0.6 -> 0.9, and new `--pair-turns 1` (default): a sheet with a direction pair always
turns its boards, so both rules are on one sheet. 7 of 40 test plan sheets change (most plans have no multi-room
directional family, so turn-boards' reach is limited by the floor mix). Byte-identical: `--turn-boards 0.6 --pair-turns 0`
== the 2026-10-15 code's defaults (40 plan sheets). Not measured yet: needs a training run.

### 2026-10-15 (GH200) -- turn-boards 0.9 + pair sheets always turn (GEN 2026-10-15b): HF14 12 q3 recovered, gate fails on HF14

`TAG=single18kq ./run_single_pass.sh`, same ids / seed (600000+, seed 6, default GEN 2026-10-15b, only `--plan-source`); pool
17,906 ok (same as kp). Against single18kp only the turned boards differ; against single18kf (current best) the plan direction pairs
and the turned boards. Per-question files `eval_baselines/single18kq-blend50_{hf14,hf14fix2,val,gemini30}_4096.json`.

| @4096, hand boxes | HF14 | HF14 fix v2 | val | val per-sheet | Gemini30 |
|---|---:|---:|---:|---:|---:|
| single18kf-blend50 (best) | **0.9157** | **0.9224** | 0.8971 | 0.9342 | 0.8526 |
| single18kp-blend50 (pairs) | 0.8983 | 0.9000 | 0.8964 | 0.9343 | 0.8605 |
| **single18kq-blend50** (pairs + turns) | 0.9088 | 0.9126 | 0.8973 | **0.9351** | **0.8616** |
| kq - kp, paired | +0.011 (10 / 3, p=0.11) | +0.013 (p=0.09) | +0.001 (p=0.21) | +0.0008 | +0.001 (45 / 23, sign p=0.01) |
| kq - kf, paired | -0.007 (7 / 14, p=0.35) | -0.010 (6 / 15, p=0.18) | +0.0002 (p=0.93) | +0.0009 | +0.009 (48 / 48, p=0.13) |

- **HF14 12 (kf / kp / kq):** q0 0.94 / 0.94 / 0.94, q1 0.93 / 0.82 / 0.83, q2 0.94 / 0.94 / 0.94, **q3 0.77 / 0.53 / 0.81** (pred/tgt
  0.83); sheet 0.893 / 0.805 / 0.880. The turns undo the pairs' q3 split; q1 stays down 0.10.
- **Gemini30 4 keeps the pairs' fix:** q0 0.96 / 0.94 / 0.93, q1 0.95 / 0.93 / 0.93, q2 0.32 / 0.92 / 0.92, q3 0.33 / 0.90 / 0.91,
  q4 0.72 / 0.98 / 0.97, q5 0.71 / 0.98 / 0.98; sheet 0.666 -> 0.941.
- **Where HF14 still trails kf (-0.007, not significant):** HF14 2 (rendered elevation, not a plan) 0.910 -> 0.858, 44% of the gap
  (q0 0.89 / 0.85 / 0.81, q1 0.87 / 0.77 / 0.79, both under-selecting, pred/tgt 0.81-0.83 -- it fell in kp too, so it moves with the
  pairs pool rather than the turns), 25 -0.027, 0 -0.021 (q0 / q1 0.94 / 0.93 -> 0.92 / 0.91), 12 -0.014. Other Gemini30 sheets
  vs kf > 0.05: 0 0.68 -> 0.77, 1 0.46 -> 0.63 up, 8 0.84 -> 0.76 down (as in kp). val: no sheet > 0.05.
- **Gate (vs single18kf-blend50: Gemini30 up, val per-sheet and HF14 not down): fails on HF14** (0.9088 < 0.9157; Gemini30 +0.009 and
  val per-sheet +0.0009 pass). Not published; single18kf-blend50 stays the best. Checkpoints `data/runs/ck_single18kq_*` (local).

### 2026-10-15 (GH200) -- single18kq-blend50 made the best (the user's call); publish gate revised

Published `abshetty/floz-refunet-swint-single18kq-blend50` (private, round-trip identical 243 / 243); HF14 / val / Gemini30 re-scored
@4096 with `--save-probs` (per-question IoU identical to `eval_baselines/single18kq-blend50_*`), maps + metrics on `abshetty/floz-diag`
`single18kq-blend50/{hf14,val,gemini30}_4096/` (51 + 72 + 195). It replaces single18kf-blend50 as the best in `startup.md`;
single18kf-blend50 stays listed as the HF14-best reference (0.9157 vs 0.9088). **New gate** (`startup.md`): vs the current best,
Gemini30 up; val per-sheet and HF14 no significant drop (paired p < 0.1); no sheet down > 0.05 from the change itself. kq vs kf
under it: Gemini30 +0.009, val per-sheet +0.0009, HF14 -0.007 (p=0.35); the two sheets down > 0.05 (HF14 2, Gemini30 8) are
elevations, untouched by the plan-only change.

### 2026-10-15 (GH200) -- arms on the Hub; kq blend ratio: DR 0.3 beats 0.5 on HF14 and Gemini30, fails the sheet rule on Gemini30 1

**Arms published** (private, each round-trip identical): `abshetty/floz-refunet-swint-single18k{f,p,q}-{dr,nodr}` -- any blend of
the three runs can now be rebuilt without retraining (float `w * dr + (1 - w) * nodr` of the `model` tensors).

**Blend ratio, eval only** (blendNN = DR share NN%; `data/runs/ck_single18kq_blend{30,70}.pth`, local), @4096 hand boxes, paired vs
single18kq-blend50 (the best). Per-question files `eval_baselines/single18kq-blend{30,70}_*`.

| | HF14 | HF14 fix v2 | val | val per-sheet | Gemini30 |
|---|---:|---:|---:|---:|---:|
| blend50 (best) | 0.9088 | 0.9126 | 0.8973 | 0.9351 | 0.8616 |
| **blend30** (DR 0.3 / no-DR 0.7) | **0.9156** (+0.007, p=0.014) | **0.9221** (+0.009, p=0.028) | 0.8976 (+0.0003) | **0.9357** | **0.8680** (+0.006, p=0.003) |
| blend70 (DR 0.7 / no-DR 0.3) | 0.8963 (-0.013, p=0.003) | -- | 0.8958 (-0.0015, p=0.03) | 0.9342 | 0.8502 (-0.011, p=1e-6) |

- Same direction as the np0 sweep (2026-10-14): more no-DR weight, more Gemini30 -- and this time HF14 rises with it too. blend30
  equals single18kf-blend50 on HF14 (0.9156 vs 0.9157, paired -0.0001) while keeping kq's Gemini30 4 fix (0.94).
- HF14 12: q3 0.81 -> 0.87, q1 0.83 -> 0.84. Gemini30 4 unchanged (0.94-0.98 on all six questions).
- **Gate vs blend50: (1) Gemini30 up, (2) no val / HF14 drop -- pass; (3) fails on Gemini30 1: 0.632 -> 0.557** (q0 0.38 -> 0.34,
  q1 0.82 -> 0.74, q2 0.70 -> 0.60). A blend ratio touches every sheet, so this counts. Gemini30 1 is the sheet that swings most
  between models (0.30 / 0.61 / 0.46 / 0.65 / 0.63 for 18k / np0 / kf / kp / kq). No other sheet falls > 0.05 (blend70: Gemini30 10
  and 11). **Not published** pending the user's call; HF14 / val / Gemini30 maps exist locally (`data/diag/single18kq-blend30/`, IoU
  identical) for a quick upload.

### 2026-10-15 (GH200, last ~40 min of the VM) -- single18kq-blend30 made the best; single18kw nodr arm (wing pairs back on)

- **single18kq-blend30 is the best** (the user's call; it fails the gate's sheet rule only on Gemini30 1, the most volatile sheet):
  `abshetty/floz-refunet-swint-single18kq-blend30` (private, round-trip identical 243 / 243), maps (HF14 / val / Gemini30, 51 + 72 + 195,
  IoU identical to `eval_baselines/single18kq-blend30_*`) on `abshetty/floz-diag` `single18kq-blend30/`. `startup.md` updated.
- **single18kw** (wing colour pairs back on top of GEN 2026-10-15b): pool `single18kw_s600000` = `run_single_pass.sh`'s generator call
  with `GEN="--plan-source data/reference/swiss_dwellings/layouts.pkl.gz --near-pairs 0.6"` (ids 600000+, seed 6): 17,906 ok, 473 s.
  ONLY the nodr arm trained (one-epoch restart of longswiss-swa37-41, run_single_pass.sh's recipe without `--domain-random`), started
  05:28, **finished 05:48:57** and published as `abshetty/floz-refunet-swint-single18kw-nodr` (private, round-trip identical 243 / 243).
  **Not scored. The dr arm was not trained** (VM shutdown). To finish the run: rebuild the pool (deterministic, ~8 min) with
  `TAG=single18kw GEN="--plan-source data/reference/swiss_dwellings/layouts.pkl.gz --near-pairs 0.6" ./run_single_pass.sh` after placing
  the Hub arm at `data/runs/ck_single18kw_nodr/epoch_0.pth` (`scripts/hf_ckpt_to_pth.py`) and `touch data/runs/ck_single18kw_nodr/.done`,
  so the script trains only the dr arm, blends 50/50 and scores. Compare against single18kq (same ids / seed / GEN except the wing pairs);
  the best is single18kq-blend30, so also score the 0.3 / 0.7 blend.

### 2026-10-15 (CPU) -- Gemini30 loss of single18kq-blend30: input scale is the biggest lever found

Loss budget (maps `single18kq-blend30/gemini30_4096`, 0.132): unlabelled drawing selected 0.050 (0.028 far from the target),
other family 0.028 (26 alone 0.016), target missed 0.040, paper 0.011. Eight sheets (26 28 7 19 16 17 23 1) hold 0.085.

**Scale (CPU re-scores of the same model, the same hand boxes, one-sheet pools):**

| single18kq-blend30, Gemini30 | 4096 (protocol) | 2048 |
|---|---:|---:|
| mean IoU | 0.868 | 0.875 |
| sheets > +0.1 | | 26 +0.17, 8 +0.14, 1 +0.13, 0 +0.13, 19 +0.13 |
| sheets < -0.1 | | 6 -0.16, 17 -0.16 |

- Gemini30 1 is a scale + texture case: 0.553 / 0.579 / 0.687 / **0.898** at 4096 / 3072 / 2048 / 1536; a 5x5 median
  filter at 4096 (removes the fill's fine Gemini "grain", keeps the battens) 0.782. The grain is on sheet 1 only (native crops of
  all 30 sheets), so no generator change for it.
- No per-sheet predictor found: the gain from 2048 does not track pattern period, box size or native size (rank corr <= 0.18).

**Probability averaging (offline, maps resized to native with linear interpolation, then > 0.35):**

| | HF14 | val | val per-sheet | Gemini30 |
|---|---:|---:|---:|---:|
| 4096 | 0.9170 | 0.9029 | 0.9376 | 0.8712 |
| 2048 | 0.8592 | 0.8601 | 0.9140 | 0.8784 |
| 4096 + 2048 (1 : 1) | 0.9055 | 0.9015 | 0.9385 | **0.8833** |
| 4096 + 2048 (0.7 : 0.3) | 0.9156 | 0.9027 | 0.9388 | 0.8795 |

- Linear-resize-then-threshold alone is worth +0.003 Gemini30 / +0.0014 HF14 over the evaluator's threshold-then-nearest
  (0.8680 -> 0.8712, 0.9156 -> 0.9170); not adopted (changes the protocol).
- 1 : 1 lifts Gemini30 +0.012 with no Gemini sheet down > 0.02 (64 questions up / 11 down), but costs HF14 0.011, on the
  fine-line sheets (14 -0.048, 27 -0.028, 25 -0.019): they need 4096's detail, Gemini's large-pattern sheets want less.
- So the model is scale-sensitive in both directions. **New training option `--image-max-size-max N`** (`train_refunet.py`,
  `dataset.py`; 0 = off, byte-identical, checked on 6 samples): each training sheet's long side drawn in [`--image-max-size`, N].
- Oracle (each Gemini30 sheet at its better scale of 4096 / 2048): 0.899 vs 0.871 -- the headroom a scale-robust model could
  take. Adding 3072 to the 4096 + 2048 average adds nothing (0.883 either way, 154 questions).
- Checked and ruled out on the way: furniture (Gemini30 23 / 0 draw it white-filled as the generator does, 90% of furnished
  rooms; the orange in my overlays was the missed-target colour, not paint), fill grain (sheet 1 only).
- `run_single_pass.sh` takes `POOL=` (reuse a pool), `TRAINX=` (extra training flags) and `BATCH=`.

### 2026-10-17 (GH200) -- multi-resolution training (single18kms): fails the gate; scale sensitivity not reduced

`TAG=single18kms POOL=data/synthetic/single18kq_s600000 TRAINX="--image-max-size-max 3072" BATCH=4 ./run_single_pass.sh` -- the
kq recipe (same pool, rebuilt at GEN 2026-10-15b: 17,906 ok, same count as kq) with each training sheet's long side drawn in
[2048, 3072]. **Batch 8 OOMs** with the range (dr arm, first step); batch 4 fits (78 GB) but, with no grad accumulation in
`train_refunet.py`, doubles the steps at the same LR (4,431 vs 2,215), so batch size is a second difference from kq. Arms: dr
1,875 s, nodr 2,520 s. Blend30 (0.3 dr / 0.7 nodr) built as for kq. Per-question files `eval_baselines/single18kms-blend{30,50}_*`.
Sanity first: single18kq-blend30 re-scored on HF14 @4096 on this VM = 0.915633, identical to the baseline.

| @4096, hand boxes | HF14 | HF14 fix v2 | val | val per-sheet | Gemini30 | Gemini30 per-sheet |
|---|---:|---:|---:|---:|---:|---:|
| single18kq-blend30 (best) | **0.9156** | 0.9221 | 0.8976 | 0.9357 | **0.8680** | 0.8611 |
| single18kms-blend30 | 0.9089 | **0.9263** | 0.8980 | 0.9359 | 0.8652 | 0.8651 |
| kms30 - kq30, paired | -0.007 (12 / 10, p=0.32) | +0.004 (14 / 9, p=0.58) | +0.0004 (p=0.74) | +0.0002 | -0.003 (58 / 34, p=0.64; sign p=0.016) | +0.004 |
| single18kms-blend50 | 0.9054 | 0.9240 | 0.8989 | 0.9368 | 0.8636 | 0.8659 |

- **Gemini30 per sheet, kq30 -> kms30:** 26 0.470 -> 0.490, 8 0.780 -> 0.828, 1 0.557 -> 0.570, 0 0.771 -> 0.861, 19 0.482 -> 0.491,
  6 0.906 -> 0.900, **17 0.891 -> 0.697**; also 7 +0.05, 22 +0.03, 28 +0.03, 12 -0.02. Most questions improve (58 / 34); the mean
  falls only because of sheet 17 (13 questions, -0.194 each, 459% of the net difference).
- **Gemini30 17** (r4 #016, a hip-roof plan, one shingle family on every plane, native 2400 px so upsampled 1.7x at 4096): kq30
  selects it cleanly (q1-q10 0.93-0.96, pred/tgt ~1.0); kms30 leaves blotchy half-confident coverage over the whole roof (q1 0.96
  -> 0.35, q2 0.95 -> 0.33, q8 0.89 -> 0.47, q10 0.95 -> 0.65; pred/tgt 0.33-0.66). Not a direction split -- a texture-wide
  confidence collapse. It is one of the two sheets that lose 0.16 at 2048 on kq30 (2026-10-15 CPU).
- **HF14 12** (turned deck boards) q1 0.84 -> 0.72, q3 0.87 -> 0.78 (under-selection, pred/tgt 0.77-0.81); sheet -0.048 (0.798 in
  blend50). With fix v2, HF14 18 0.921 -> 0.954 and 2 / 7 +0.03 carry the HF14 fix v2 gain.
- **Gate vs single18kq-blend30: fails (1)** (Gemini30 -0.003) **and (3)** (Gemini30 17 -0.19; HF14 12 -0.048 just under the line).
  (2) passes (no significant HF14 / fix v2 / val drop). Not published. Checkpoints `data/runs/ck_single18kms_{dr,nodr,blend30,blend50}`
  (local).

**Did multi-res make the model scale-robust? No.** Same model scored @2048 (`eval_baselines/single18k{q,ms}-blend30_*_2048`):

| per-sheet 2048 - 4096 | mean abs gap kq30 / kms30 | sd | sheets abs > 0.1 | mean gap |
|---|---:|---:|---:|---:|
| Gemini30 | 0.049 / 0.046 | 0.071 / 0.073 | 7 / 3 | +0.018 / +0.000 |
| HF14 fix v2 | 0.036 / 0.031 | 0.058 / 0.033 | 1 / 1 | -0.028 / -0.029 |
| val | 0.032 / 0.037 | 0.072 / 0.081 | 1 / 2 | -0.024 / -0.028 |

- Gemini30 gaps kq30 -> kms30: 26 +0.17 -> +0.06, 8 +0.14 -> +0.06, 0 +0.13 -> +0.05, 1 +0.13 -> **+0.17**, 19 +0.12 -> **+0.16**,
  6 -0.16 -> **-0.26**, 17 -0.15 -> -0.08 (because 4096 fell), 20 +0.01 -> -0.08. HF14 14 -0.22 -> -0.12; val 13 -0.27 -> -0.29.
- So the 4096 operating point moved (Gemini30's mean gap went to zero: some sheets' 2048 gains now appear at 4096), but the swings
  between scales did not shrink. A 1.5x training range does not cover the 2x test gap, and widening it needs memory (no gradient
  checkpointing in the code) and an LR fix for the smaller batch. Not pursued further today.

### 2026-10-17 (CPU, beside the GH200 runs) -- what the remaining real-sheet loss is, and what mask post-processing buys

`scripts/postproc_probe.py` on single18kq-blend30's @4096 maps (`abshetty/floz-diag single18kq-blend30/{hf14,val}_4096`); it rebuilds
the evaluator's targets with `_prepare_sheet` (baseline IoU reproduces exactly: val 0.8976, HF14 0.9156, fix v2 0.9221). Loss split
per question in IoU units (sum = 1 - IoU; boundary = within 4 native px of the target edge; wrong region = a predicted component
< 10% on the target; missed region = a target component < 25% recalled):

| | HF14 | HF14 fix v2 | val |
|---|---:|---:|---:|
| boundary (fp + fn) | 0.032 | 0.032 | 0.045 |
| under-filled inside found regions | 0.028 | 0.025 | 0.009 |
| missed regions | 0.014 | 0.005 | 0.013 |
| wrong regions (other family + unlabelled) | 0.006 | 0.013 | 0.031 |
| spill near found regions | 0.005 | 0.004 | 0.005 |
| total (1 - IoU) | 0.084 | 0.078 | 0.102 |

**Post-processing, chosen on val, HF14 read once:**

| | val | HF14 | HF14 fix v2 |
|---|---:|---:|---:|
| linear upsample of the probability + fill holes <= 4096 px (inference res) | +0.006 (25 / 1) | +0.002 | +0.002 |
| linear upsample only | +0.005 | +0.0014 | +0.0013 |
| drop components < 0.2% of the mask | +0.001 (2 / 0) | +0.0005 | +0.0007 |
| close 15 / open 15 / hysteresis 0.9 | +0.001 | +0.002 / 0.000 / -0.001 | +0.001 / +0.003 / 0.000 |
| fill every hole | **-0.052** (targets have windows / doors cut out) | | |
| drop components < 3% | -0.024 | | |
| threshold 0.25-0.5 | within +-0.001 | | |

- Morphology cannot close the gap to the user's 0.95 target: it reaches ~+0.002-0.006 of the 0.034-0.05 needed. "Under-filled" is
  not small holes (filling <= 4096 px recovers almost none of it); boundary is partly label precision.
- **Shapes for the frontend (final output = polygons, the user 2026-10-17):** raw predictions have median 12 (HF14) / 17 (val)
  components, p90 39 / 30, vs 2 / 3 in the target; holes median 4 / 8. Speck drop (0.2%) -> components 4 / 5 (p90 6 / 15); + hole
  fill <= 4096 px -> holes 0 / 0 (p90 4 / 8), both IoU-neutral or better. The polygon step (speck drop, small-hole fill, contour
  trace, simplify) should be scored as the model's output: IoU of the rasterised polygons, gated like any other change.

### 2026-10-17 (GH200) -- single18kw finished (wing colour pairs): fails the gate; Gemini30 26 +0.06 but 1 / 17 / 4 / 19 fall

`TAG=single18kw GEN="--plan-source data/reference/swiss_dwellings/layouts.pkl.gz --near-pairs 0.6" ./run_single_pass.sh` with the
Hub nodr arm placed as done: pool rebuilt (17,906 ok, 464 s, same count as kq), dr arm trained (1,190 s, batch 8), published as
`abshetty/floz-refunet-swint-single18kw-dr` (round-trip 243 / 243). Per-question files `eval_baselines/single18kw-blend{30,50}_*`.

| @4096, hand boxes | HF14 | HF14 fix v2 | val | val per-sheet | Gemini30 |
|---|---:|---:|---:|---:|---:|
| single18kq-blend30 (best then) | 0.9156 | 0.9221 | 0.8976 | 0.9357 | **0.8680** |
| single18kw-blend30 | 0.9146 (-0.001, p=0.84) | 0.9224 (+0.0004, p=0.94) | 0.8997 (+0.002, p=0.15) | 0.9356 | 0.8495 (-0.019, p=0.004) |
| single18kw-blend50 | 0.9106 | 0.9187 | 0.9011 | 0.9364 | 0.8454 |

- Gemini30 per sheet kq30 -> kw30: **26 0.470 -> 0.527** (the colour-pair target), 0 +0.04, 7 +0.04, 8 +0.04; **1 0.557 -> 0.328,
  17 0.891 -> 0.677, 4 0.945 -> 0.827, 19 0.482 -> 0.383**. HF14 12 0.895 -> 0.919 (q1 / q3 recover).
- Gate: fails (1) and (3). Gemini30 17 collapses here too (as in kms) -- so that loss is not a multi-res effect; kq is the run
  that happens to get 17 right.

### 2026-10-17 (GH200) -- cross-run weight blends (no training): 0.5 kq30 + 0.5 kms30 passes the gate, the new best

All arms are one-epoch restarts of longswiss-swa37-41, so blends across runs are float averages of the `model` tensors (the
run_single_pass.sh blend with weights). @4096 hand boxes, paired vs single18kq-blend30; per-question files
`eval_baselines/xblend-kq30-{kms30,kw30}_*`.

| | HF14 | HF14 fix v2 | val | val per-sheet | Gemini30 | Gemini30 per-sheet |
|---|---:|---:|---:|---:|---:|---:|
| kq30 (best then) | 0.9156 | 0.9221 | 0.8976 | 0.9357 | 0.8680 | 0.8611 |
| **(a) 0.5 kq30 + 0.5 kms30** | **0.9163** (+0.0007, p=0.89) | **0.9290** (+0.007, p=0.17) | 0.8988 (+0.0012, p=0.08) | 0.9364 | **0.8740** (+0.006, p=0.011; 61 / 16, sign p=2e-7) | 0.8705 |
| (b) 0.5 kq30 + 0.5 kw30 | 0.9156 (0.0000) | 0.9225 (+0.0004) | 0.8994 (+0.0019, p=0.06) | 0.9361 | 0.8680 (+0.0000; 42 / 41) | 0.8613 |

- (a) Gemini30 per sheet (kq30 / a / kms30): 26 0.470 / 0.478 / 0.490, 1 0.557 / **0.654** / 0.570, **17 0.891 / 0.853 / 0.697**,
  0 0.771 / 0.812 / 0.861, 8 0.780 / 0.800 / 0.828, 7 0.805 / 0.836 / 0.855. The blend keeps most of kms's gains and most of kq's
  sheet 17. HF14 12 0.895 / 0.887 / 0.847; fix v2 HF14 18 0.921 -> 0.941. Largest sheet drops: Gemini30 17 -0.038, 12 -0.013, HF14
  18 -0.012 (plain), val 22 -0.005.
- (b) Gemini30 26 0.470 -> 0.526 kept, but 1 0.557 -> 0.482 (-0.075) and 17 -0.047.
- **Gate:** (a) passes all three (Gemini30 up; no significant HF14 / fix v2 / val drop -- val +0.0012 is up; no sheet down > 0.05).
  (b) fails (1) (Gemini30 not up) and (3) (Gemini30 1). (c) (1/3 each of kq30 / kms30 / kw30) not run: it was conditional on both
  (a) and (b) helping.
- **Published (a) as `abshetty/floz-refunet-swint-xblend-kq30-kms30`** (private, round-trip 243 / 243); HF14 / val re-scored with
  `--save-probs` (IoU identical), maps + metrics (51 + 72 + 195) on `abshetty/floz-diag` `xblend-kq30-kms30/`. It is the best in
  `startup.md`. Arms `single18kms-{dr,nodr}` published too (round-trip 243 / 243), so the best decomposes into four Hub arms:
  0.15 kq-dr + 0.35 kq-nodr + 0.15 kms-dr + 0.35 kms-nodr.

### 2026-10-17 (GH200, CPU side) -- post-processing on the new best's Gemini30 maps

`scripts/postproc_probe.py MAPS gemini30` (now reads the Gemini30 local pool with `eval_boxes/gemini30_v1.json`; baseline reproduces
0.8740). The val-chosen "linear upsample of the probability + fill holes <= 4096 px": **Gemini30 0.8740 -> 0.8798 (+0.0058;
questions 81 up / 3 down, worst -0.008; sheets 23 up / 1 down (21, -0.001) / 6 flat**; biggest 1 +0.028, 17 +0.020, 0 / 16 +0.010).
Linear alone +0.0034 (23 sheets up, 0 down). drop0.01 + fill4096 is +0.0079 here but was -0.0022 on val, so not chosen.
Gemini30 loss split (0.126): holes inside found regions 0.030, spill (unlabelled 0.029, other family 0.012), boundary 0.031,
wrong regions 0.024. Raw predictions are fragmented: median 44 components (p90 133) vs 5 in the target -- the polygon stage
matters more on Gemini than on the real sheets (12 / 17). Across the three sets "linear + fill" is +0.006 val / +0.002 HF14 /
+0.006 Gemini30 (worst single question -0.022, on HF14); it changes the evaluation protocol, so it is not adopted
as the metric. It is the natural first step of the polygon output.

### 2026-10-17 (CPU) -- gable tone pairs with one-family twins for Gemini30 26 (GEN_VERSION 2026-10-17)

- **What sheet 26 is** (r4 #078; 0.478 on xblend-kq30-kms30, the largest per-sheet loss: 0.021 of Gemini30's 0.126). One lap
  siding at one spacing (FFT period 12.5 px tan vs 13-14 px olive), two families: tan on the storeys, olive on the gable and
  dormer. Measured fills (CIELAB, eroded pieces): tan (70.6, 7, 31), olive (67, 2, 16) -- the olive is the tan **desaturated**
  (chroma 32 -> 16, hue angle 6 deg apart, L* -3.5), CIEDE2000 7.9-9.3 between pieces, <= 1.7 within each family. The
  2026-10-14 note that it also differs in spacing 1.8x was wrong.
- **Why no synthetic pair covered it:** wing pairs (`--near-pairs`) need a hue angle >= 20 deg and CIEDE2000 >= 10 (7-9 only
  together with spacing 1.4-2x), and sit on a side wing; sheet 26 is a chroma step at 8.5, same spacing, on the gable.
- **Change (`--gable-pairs 0.4`, default):** a colour elevation whose house has a gable (else a second storey) gets with P the
  gable (60% if both) or upper storey as the accent zone in the main wall's exact style, and half of those are a **tone pair**
  (fill chroma x0.35-0.7 or x1.4-2.4, hue +-15 deg, L* +-4, CIEDE2000 6-12, pen recomputed from the fill; two families) and half
  a **twin** (CIEDE2000 0.4-2, the drift Gemini30 1's pieces show; labelled ONE family across the trim band). The twin is the
  new part: the 2026-10-07 / 13 gable / upper-storey pairs had none and split one-family sheets at exactly those boundaries
  (Gemini30 1, 5, 14); wing pairs still broke Gemini30 1. With twins the zone and its trim band carry no information, only the
  colour gap does. Never on a sheet with a wing pair; own RNG streams. Annotation files carry `gable_pair` {zone, kind, de2000}.
- **Verified** (60 elevations, `--mode-weights 1,0,0`, seed 9, ids 820000+): `--gable-pairs 0` is byte-identical to the
  2026-10-15b code (120 / 120 files). Default: 26 colour elevations, 10 planted (6 tone: 5 gable, 1 upper; 4 twin), only those
  10 sheets change. Rendered main-vs-accent fill CIEDE2000 on the tone sheets 8.0 / 8.9 / 10.2 / 10.3 / 10.8 / 16.7 (the last
  one's gables sit in eave shadow); accent area 2-171% of the main wall's. Twins: accent merged into `main` in the labels.
- **Risk to watch:** Gemini30 1 (one family across two trim bands) and 17; HF14 / val one-family elevations (val 9, HF14 14).
  Screen: `TAG=single18kg ./run_single_pass.sh` (GEN 2026-10-17 defaults, ids 600000+, seed 6 -- the kq recipe plus this),
  blend30, then also 0.5 with xblend-kq30-kms30; gate vs xblend-kq30-kms30.

### 2026-10-17 (GH200) -- gable tone pairs (single18kg, GEN 2026-10-17): Gemini30 26 +0.11, but sheet 1 -0.32; both candidates fail the gate

`TAG=single18kg ./run_single_pass.sh` at the new defaults (ids 600000+, seed 6; the kq recipe plus `--gable-pairs 0.4`): pool 17,907 ok
(461 s; one more than kq), 1,621 sheets carry a gable pair (803 tone, 818 twin). Arms dr 1,197 s / nodr 1,252 s, published as
`abshetty/floz-refunet-swint-single18kg-{dr,nodr}` (round-trip 243 / 243). Blend30 as for kq. Per-question files
`eval_baselines/single18kg-blend{30,50}_*`, `eval_baselines/xblend-xa-kg30_*`. (Eight 4096 scoring jobs at once OOM the GH200; four is the limit.)

| @4096, hand boxes, paired vs xblend-kq30-kms30 | HF14 | HF14 fix v2 | val | val per-sheet | Gemini30 |
|---|---:|---:|---:|---:|---:|
| xblend-kq30-kms30 (best) | 0.9163 | 0.9290 | 0.8988 | 0.9364 | **0.8740** |
| kg30 | **0.9223** (+0.006, p=0.35) | 0.9262 (-0.003, p=0.62) | 0.8878 (**-0.011, p=0.048**) | 0.9294 | 0.8613 (**-0.013, p=0.040**; 32 / 66) |
| kg50 | 0.9199 | 0.9232 | 0.8884 | -- | 0.8579 |
| 0.5 xblend + 0.5 kg30 | **0.9231** (+0.007, p=0.09; 10 / 1) | 0.9296 (+0.0005) | 0.8985 (-0.0003) | 0.9363 | 0.8691 (-0.005, p=0.11; 23 / 53) |

- **Gemini30 per sheet (best -> kg30 -> mix):** **26 0.478 -> 0.590 -> 0.498**, **1 0.654 -> 0.335 -> 0.496**, 17 0.853 -> 0.855 -> 0.877,
  19 0.495 -> 0.449 -> 0.465; kg30 also 11 0.883 -> 0.767.
- **What moved (kg30, pred / target area):** sheet 26's merges shrink as intended (q4 / q5 / q7: 3.7-4.2 -> 1.7-2.6, IoU +0.14-0.30; q0 / q3
  +0.23-0.29), but q1 now splits its own family (0.92 -> 0.41, IoU 0.81 -> 0.40). Gemini30 1 (one board-and-batten family across trim
  bands) is split harder (q0 0.42 -> 0.20, q2 0.72 -> 0.20; IoU 0.67 -> 0.18) -- the twins did not protect it. val 9 q2 (one-family
  elevation, flagged as a risk) 0.974 -> 0.626 (pred / tgt 1.01 -> 0.64). Gemini30 11 merges more (1.13 -> 1.66). HF14 / val sheets
  moving > 0.03: only val 9 (-0.061); in the mix none.
- So the tone pairs teach the chroma split sheet 26 needs, and push the model to split within-family drift too (Gemini30 1, val 9, 26 q1).
- **Gate:** kg30 fails (1), (2) (val p=0.048) and (3) (val 9, Gemini30 1, Gemini30 11). The mix fails (1) (Gemini30 -0.005) and (3)
  (Gemini30 1 -0.158); it passes (2) and lifts HF14 +0.007 (p=0.09). Not published; xblend-kq30-kms30 stays the best.

### 2026-10-17 (CPU) -- is fill grain why colour pairs break Gemini30 1? No

Gemini30 1 @4096 (hand boxes, CPU, `--workers 1`), raw sheet vs the same sheet median-filtered 5x5 (removes the fill's fine
grain; the 2026-10-15 probe). Checkpoints rebuilt from the Hub; kg30 = 0.3 single18kg-dr + 0.7 single18kg-nodr.

| | raw (q0 / q1 / q2) | median 5x5 | gain |
|---|---:|---:|---:|
| xblend-kq30-kms30 (best) | 0.655 (0.39 / 0.91 / 0.66) | 0.893 (0.95 / 0.83 / 0.90) | +0.238 |
| single18kg-blend30 | 0.337 (0.19 / 0.64 / 0.18) | 0.518 (0.76 / 0.46 / 0.34) | +0.181 |

- Raw scores reproduce the GPU's (0.654 / 0.335). Grain costs both models ~0.2 on this sheet, but filtering it out does not
  close kg30's gap (0.32 raw, 0.38 filtered): the colour-pair model splits the clean sheet too. So the trade (26 up, 1 down)
  is not grain; the gable-twin colour drift (CIEDE2000 0.4-2) matches sheet 1's piece gaps (0.3-1.6) and still did not
  stop the split. Grain alone is worth ~+0.24 on sheet 1 for the best model (+0.008 Gemini30), sheet 1 only.

### 2026-10-17 (CPU) -- Gemini30's unlabelled false positives are trim bleed: synthetic sheets have no corner boards

Best model (xblend-kq30-kms30) Gemini30 @4096 maps (`abshetty/floz-diag xblend-kq30-kms30/gemini30_4096`). Per question, the
predicted pixels on no label at all, in IoU units, split by distance from the target (scratchpad `fpu/split3.py`). Same split on
four synthetic colour / mono elevations from the training distribution (GEN 2026-10-15b defaults, ids 820002-820012, 52
questions, CPU; larger sheets OOM at 15 GB):

| IoU units per question | synthetic (IoU 0.942) | Gemini30 (IoU 0.874) |
|---|---:|---:|
| inside the target's / any label's openings | 0.0017 | 0.0055 |
| <= 4 px from the target (label precision) | 0.0137 | 0.0183 |
| **4-25 px out (trim boards, casings, rakes)** | **0.0002** | **0.0170** |
| > 25 px away (other drawing, label gaps) | 0.0002 | 0.0191 (sheet 19 alone 0.0085) |
| of <= 25 px: on white (>= 235) pixels | 0.0007 | 0.0128 |
| of <= 25 px: between pieces of the target / along its outer edge | 0.0023 / 0.0116 | 0.0096 / 0.0257 |

- The model stops at trim on our sheets and runs over it on Gemini's. Zooms (sheets 28, 20, 16): Gemini draws trim as white
  boards between two ink lines -- corner boards at every wall end and inside corner, boards up both sides of the chimney, rake
  boards along every gable, window casings -- and next to board-and-batten or lap siding such a strip reads as one more board;
  the model takes the whole strip.
- **Gap found:** the Revit elevation path (`build_elevation_fc`) draws fascia, rakes (slab-edge "trim" faces), belts, gable-trim
  bands, the chimney cap and casings (white on mono sheets, white / dark / pale on colour ones, always 0.3 ft), but **no corner
  boards** -- wall ends and inside corners are a single outline. (v6's own elevation code has `corner_boards`; the Revit path never
  used it.)
- Sheet 19's far-away FP is the unlabelled joist field of an under-floor framing plan (a label gap, 0.0085); the rest of "away"
  (0.011) is spread over sheets.
- Ceiling for a trim fix: the 4-25 px bucket, 0.017 of Gemini30's 0.126; on the real sheets the analogous spill is ~0.005
  (2026-10-17 postproc probe), so the risk to real 28 is small (trim stays cut from the labels).
- **Counterfactual (CPU): corner boards are not the cause.** Corner boards drawn onto the same four synthetic sheets (white
  strip at both ends of every main / accent / accent2 wall piece, ink line on its inner side, cut from the labels; 0.45 ft =
  7-8 px and 0.8 ft = 12-15 px), best model @4096: it selects **0-11% of the board pixels next to the target** (mean ~5%; IoU
  cost 0.0003-0.0072 per question), and sheet IoU moves <= 0.007. So the model already stops at a white board it has never seen
  when the board is drawn in our style; Gemini's 4-25 px bleed comes from how Gemini draws (line weight / softness, board vs
  siding look), not from a missing trim type. No corner-board generator change. Scratchpad `fpu/cb/`.

### 2026-10-17 (GH200) -- single36k: 36k fresh sheets seen once at the kq config -- no clear gain over 18k; the best + s36k30 mix ties

Pool `single36k_s600000` (35,812 sheets) = `single18kq_s600000` (ids 600000-617999, GEN 2026-10-15b) + `single36k_ext_s618000` (ids
618000-635999, 17,906 ok, 465 s): today's code with `--gable-pairs 0`, which is byte-identical to 2026-10-15b (20 kq sheets
regenerated: 40 / 40 files identical; flags equal except `gable_pairs: 0`). Per-id RNG, so it is exactly one 36k build. Hard-linked
union with `merge_manifest.json` (version 2026-10-15b; run with `ALLOW_STALE_POOLS=1` on purpose).
`TAG=single36k N=36000 POOL=data/synthetic/single36k_s600000 ALLOW_STALE_POOLS=1 ./run_single_pass.sh` -- the kq recipe at batch 8,
4,431 steps per arm (fresh one-epoch cosine over 2x the steps): dr 2,349 s, nodr 2,467 s. Arms published
`abshetty/floz-refunet-swint-single36k-{dr,nodr}` (round-trip 243 / 243). Per-question files `eval_baselines/single36k-blend{30,50}_*`,
`eval_baselines/xblend-xa-s36k30_*`.

| @4096, hand boxes | HF14 | HF14 fix v2 | val | val per-sheet | Gemini30 |
|---|---:|---:|---:|---:|---:|
| xblend-kq30-kms30 (best) | **0.9163** | 0.9290 | **0.8988** | **0.9364** | 0.8740 |
| kq30 (same first 18k, same recipe) | 0.9156 | 0.9221 | 0.8976 | 0.9357 | 0.8680 |
| s36k30 | 0.9048 | 0.9211 | 0.8912 | 0.9310 | 0.8704 |
| s36k50 | 0.9043 | 0.9214 | 0.8944 | -- | 0.8682 |
| 0.5 best + 0.5 s36k30 | 0.9126 (-0.004, p=0.37) | **0.9293** (+0.0002) | 0.8976 (-0.001, p=0.38) | 0.9358 | **0.8750** (+0.001, p=0.59; 31 / 32) |

- **Data doubling (s36k30 vs kq30):** HF14 -0.011 (p=0.20; sheet 14 0.880 -> 0.812, 2 +0.046), fix v2 -0.001, val -0.006 (p=0.20; val 9
  0.897 -> 0.846), Gemini30 +0.002 (70 / 44 questions). **Gemini30 26 0.470 -> 0.580** from more data alone (kg's gable pairs gave 0.590,
  with sheet 1 -0.32; here sheet 1 +0.026), 19 +0.047, 17 0.891 -> 0.813 (again: kq is the run that gets 17 right), 6 0.906 -> 0.791.
  Twice the sheets seen once is not a clear win at this recipe.
- **Gate vs xblend-kq30-kms30:** s36k30 fails (1) and (3) (HF14 14, val 9, Gemini30 0 / 1 / 6 / 11 down > 0.05). The mix passes the
  letter of all three -- Gemini30 +0.001, no significant drop (HF14 -0.004, p=0.37), no sheet > 0.05 (Gemini30 per sheet 26 0.478 ->
  0.525, 1 0.654 -> 0.627, 17 0.853 -> 0.853, 19 0.495 -> 0.507) -- but the Gemini30 gain is a coin flip (31 / 32) and it costs plain
  HF14 0.004. **Not published / not made the best: held for the user's call.**

### 2026-10-19 (CPU) -- Gemini line counterfactual: the model is very sensitive to ink line weight; heavier / softer lines bleed less

Best model (xblend-kq30-kms30) @4096 on Gemini30 20 / 10 / 28 / 16, the 3 questions with the most 0-25 px unlabelled FP on each
(12 questions; hand boxes), the whole sheet image transformed, labels unchanged: thin = 3x3 max filter (ink -1 px per side),
thick = 3x3 min filter (+1 px per side), blur = Gaussian sigma 1.2, sharp = unsharp mask (2x - blur 1.5). Raw reproduces the GPU
maps (20: 0.886 vs 0.885). Scratchpad `lc/`.

| 12 q | IoU | 4-25 px FP | of <= 25 px: on white | > 25 px FP | sheets 20 / 10 / 28 / 16 |
|---|---:|---:|---:|---:|---|
| raw | 0.855 | 0.044 | 0.036 | 0.025 | 0.886 / 0.801 / 0.855 / 0.879 |
| thin | **0.532** | 0.063 | 0.055 | 0.216 | 0.496 / 0.242 / 0.734 / 0.656 |
| sharp | 0.737 | 0.050 | 0.036 | 0.032 | 0.856 / 0.514 / 0.784 / 0.794 |
| thick | 0.832 | **0.028** | **0.016** | 0.028 | 0.885 / 0.679 / 0.846 / 0.916 |
| blur | **0.869** | 0.039 | 0.027 | **0.009** | 0.864 / 0.852 / 0.857 / 0.903 |

- **Line weight is a large, unmodelled sensitivity.** Removing ~1 px of ink per side collapses these questions (0.86 -> 0.53;
  sheet 10 0.80 -> 0.24), and crisper lines cost 0.12. Heavier ink cuts the trim bleed by ~40% (white-trim FP -55%) but breaks
  sheet 10; a slight blur helps 3 of 4 sheets (+0.014 overall, far FP 0.025 -> 0.009).
- Fits the 2026-10-15 scale result (Gemini sheets that gain at 2048 -- lines relatively heavier -- and the oracle 0.899): part of
  what scale was buying is line weight relative to the model's training distribution.
- Not yet known: whether Gemini's lines at 4096 are thinner than synthetic lines at the 2048 training size (measure next), and
  whether inference blur holds on all of Gemini30 / HF14 / val. 12 hand-picked questions, one model: a direction, not a size.

### 2026-10-19 (CPU) -- ink line width at model input: Gemini @4096 has ~2.5 px lines, training sheets 1-2 px -- the scale gap again

Ink pixels inside labelled pattern regions (labels eroded 10 px; ink = > 60 grey levels darker than a 9x9 local max; width = min of
the horizontal / vertical ink run through the pixel; integer px), at the resolution the model sees. Synthetic = the 60 GEN
2026-10-15b elevations; real 28 = HF14 + val. Scratchpad `lw/lw.py`.

| at model input | line width median (sheet p25-p75) | ink contrast | ink share of pattern area |
|---|---:|---:|---:|
| synthetic @2048 (training), mono | 1.0 (1-1) | 82 | 0.103 |
| synthetic @2048 (training), colour | 2.0 (2-4) | 98 | 0.026 |
| synthetic @2048 x0.85 (DR arm mean) | 1.0 (1-2.25) | 84 | 0.059 |
| **Gemini30 @4096 (eval)** | **2.5 (2-3)** | 98 | 0.082 |
| Gemini30 @2048 | 1.0 (1-2) | 98 | 0.085 |
| real 28 @4096 (eval) | 2.0 (2-2.25) | 75 | 0.035 |

- Relative to the pattern (ink share) Gemini's lines are not unusual (0.08, between synthetic mono and colour). In pixels, Gemini at
  4096 draws its lines ~2.5x as wide as the training sheets at 2048; at 2048 it matches training. So the "line weight" effect is
  the 2026-10-15 scale mismatch seen from another side, not a separate drawing-style gap.
- **Correction to the line counterfactual above:** a 3x3 max filter removes ~1 px per side, which nearly erases Gemini's 2-3 px
  lines; the "thin" collapse (0.86 -> 0.53) is texture being deleted, not evidence that the model fails on thin lines. "thick"
  and "blur" move the lines further from training, yet blur still helped 3 of 4 sheets -- softening, not width, is what helps
  there. The trim bleed (4-25 px at 4096) is ~2-12 px at 2048 -- a few pixels at the scale the model was trained at.

### 2026-10-19 (CPU) -- `--hires-crop P`: train at the 4096 eval scale without 4096 memory (implemented, untested on GPU)

`train_refunet.py --hires-crop P` (default 0): with probability P a training sample is the sheet at twice `--image-max-size`
(long side 4096 at 2048 -- the scale every eval set is scored at), cropped to at most 2048 x 2048 around the reference box, the
labels and the anchor clipped to the crop; references are still cut from the full-resolution sheet; DR shrink / jitter and the
flips apply after the crop as before. Motivation: the scale gap (2026-10-15, 10-17 kms, 10-19 line width: Gemini @4096 shows
~2.5 px lines where the 2048 training sheets show 1-2 px). The 50/50 mix keeps whole-sheet 2048 samples for context and for the
Gemini sheets that prefer 2048.
- Checked on the 60 GEN 2026-10-15b elevations: P = 0 gives byte-identical samples to the old code (12 / 12, DR on); P = 1:
  long side <= 2048 on all 60, the reference box lies inside the crop and inside the target on every sample, crops look right.
- **Not "the same pixels per sample":** whole elevation sheets at 2048 are wide (median 1.92 M px, max 3.93 M); crops are nearly
  square (median 3.84 M, max 4.19 M). Peak memory should be about the same (+7% at the max, collate pads to the batch max), but
  crop samples cost up to ~2x compute: at P = 0.5 expect ~1.5x time per arm (~30 min instead of ~20).
- Screen: `TAG=single18khc POOL=data/synthetic/single18kq_s600000 TRAINX="--hires-crop 0.5" ./run_single_pass.sh` (kq pool, kq recipe,
  batch 8), blend30; gate vs xblend-kq30-kms30; also 0.5 xblend + 0.5 khc30; also score khc30 @2048 to see whether the per-sheet
  2048-vs-4096 swings shrink (kms did not).

### 2026-10-19 (CPU) -- mask -> polygon post-processing: linear + hole fill + polygon is +0.002..0.008 everywhere; cell snapping +0.017 on Gemini30 only

The frontend gets polygons, so every method is scored as the rasterised polygon output (IoU vs labels), on the best model's
(xblend-kq30-kms30) @4096 maps; chosen on val, read once on HF14 / fix v2 / Gemini30. Scripts `scripts/pp_probe/`
(common / align / stroke / methods; run `python3 scripts/pp_probe/methods.py val`), production function `scripts/polygonize.py`.

**Do labels follow the drawn lines?** Yes. Ink = > 60 grey levels below a 9x9 local max. Label-edge pixels within 2 px of ink:
val 0.93, HF14 0.94, Gemini30 0.99 (prediction edges 0.95 / 0.91 / 0.86). About half the boundary-band errors are the stroke
pixels themselves (val 0.57, HF14 0.50, Gemini30 0.59), and labels split the boundary stroke down the middle: share of stroke
pixels (within 3 px of the label edge) inside the label, median val 0.54, HF14 0.59, Gemini30 0.49 -- predictions 0.60 / 0.58 /
**0.73** (Gemini30's trim bleed again).

| IoU (mean over questions), delta vs the evaluator's nearest-upsampled mask | val (tune) | HF14 | HF14 fix v2 | Gemini30 |
|---|---:|---:|---:|---:|
| base (evaluator) | 0.8988 | 0.9163 | 0.9290 | 0.8740 |
| linear upsample, threshold 0.35 | +0.0049 | +0.0015 | +0.0015 | +0.0034 |
| + fill holes <= 4096 inference px | +0.0058 | +0.0021 | +0.0019 | +0.0059 |
| **+ polygons (drop < 0.2%, contours with holes, Douglas-Peucker 1 px)** | **+0.0061** | **+0.0021** | **+0.0021** | **+0.0082** |
| guided filter on the image, r = 4 native px (r = 8 / 16 lose on val) | +0.0041 | +0.0018 | +0.0020 | -0.0020 |
| cell voting (cells = non-ink components, < 2% of the sheet; mean prob > 0.5; ink follows its nearest cell pixel) | +0.0036 | -0.0031 | -0.0020 | +0.0123 |
| hybrid: cell decision within 4 px of the filled linear mask's edge, filled linear elsewhere | +0.0037 | +0.0010 | +0.0018 | +0.0152 |
| hybrid 8 px + polygons | +0.0024 | +0.0001 | +0.0011 | **+0.0166** |

- **Default polygon output: linear + fill + polygons** (`scripts/polygonize.py`): up on every set, questions up / down val 26 / 1,
  HF14 15 / 3, Gemini30 107 / 2, worst single question -0.023 (HF14). Median 180 (HF14) - 440 (Gemini30) vertices per question.
- **Cell snapping is a Gemini30 tool for now:** hybrid +0.009-0.011 over linear + fill on Gemini30 (148 / 9 questions) but -0.002
  val, -0.001 HF14 vs linear + fill. Its losers are sheets where the ink map is wrong: Gemini30 1 (fill grain), 17 (shingle roof),
  val 17 (markup paint), HF14 18 q8, HF14 14 (faint lines). A noise-robust ink map (median-filtered before thresholding, adaptive
  threshold for faint lines) is the next thing to try before it can be a default.
- The ceiling is lower than the 0.05-0.06 "boundary + under-filled" budget suggested: labels already split strokes at the line
  centre and prediction edges already sit on ink 86-95% of the time; what remains in the band is mostly which side of a stroke
  pixels fall on (worth ~1-2 px of edge).
- **Sturdier ink map (v2) does not rescue cell snapping** (`--ink=v2`: black-hat k = 7, per-sheet threshold = half the 99.5th
  percentile clipped to 15-60, ink pieces < 12 px dropped; designed from the failure cases, not tuned). Hybrid 4 px vs linear +
  fill: val -0.0026 (v1 -0.0021; val 17 now -0.013 per sheet: v2 misses its faint lavender board lines, so cells merge), HF14
  +0.0002 (v1 -0.0011), Gemini30 +0.0095 (v1 +0.0093; Gemini30 1 still -0.04 -- its board lines are found by both; the loss is
  elsewhere). Hybrid 8 px gets worse with v2 on val (-0.013). Line snapping stays a Gemini30-only tool; not pursued further
  without a tuning set that is not val / HF14.

### 2026-10-19 (CPU) -- probability-map rules: tuned on Gemini30 only, none transfers; the polygonize.py default stands

`scripts/pp_probe/pmap.py` (per question, all on the linear-upsampled P, then fill holes <= 4096 inference px, vs P > 0.35):
global thresholds 0.15-0.6, Otsu, box-relative (P > a x median P in the reference box), hysteresis (lo 0.1-0.25, core 0.5),
Gaussian smoothing 1.5-6 px, half-height of the local max (31 px), Laplacian sign (boundary at the curvature change), pruning
pieces whose peak P < m. **Tuned on Gemini30 only** (leave-one-sheet-out CV); HF14 / val read once for the pre-registered picks.
- Gemini30 (raw, vs 0.35): every rule that adds pixels loses (thr 0.15 -0.011, hysteresis lo 0.1 -0.015, smoothing sigma 6
  -0.020, Laplacian -0.005); rules that remove pixels win a little (thr 0.5 +0.0023, box-relative 0.5 +0.0050, prune 0.7 +0.0013).
  The model over-includes on Gemini30; the "under-filled" bucket is not reachable by lowering the bar.
- LOSO CV on Gemini30: global threshold +0.0002 (picks 0.5 on 28 / 30 folds), box-relative +0.0027 (0.5 on 24 / 30), prune +0.0013
  (0.7 on 30 / 30).
- **Read once:** box-relative 0.5: val -0.0004, HF14 -0.0055 (2 up / 13 down), fix v2 -0.0043. Prune 0.7: val +0.0002, HF14
  -0.0003, fix v2 -0.0001. Neither is adopted. The threshold 0.35 and `scripts/polygonize.py` (linear + fill + polygons) stay.

## Prediction-informed classical post-processing (2026-10-19, goal: +0.03 Gemini30, +0.02 val / HF14; tuned on Gemini30 only)
Scripts: scripts/pp_probe/{morph,gcut,inksplit,budget,combo,loso}.py; rows in data/diag/. All deltas are per-question mean IoU.
- **Oracle ceilings over the default (lin_fill+poly)**: label-perfect pixels within 2 px of the predicted edge = Gemini +0.028,
  val +0.035, HF14 +0.025 (4 px: +0.041 / +0.049 / +0.035); perfect removal of false components = +0.019 / +0.028 / +0.004.
  So +0.02 on HF14 needs ~70% of a perfect 2-px boundary fix.
- **Gemini30 (vs default, LOSO)**: orthogonal regularisation +0.0004; snapping axis edges onto drawn lines -0.005 (wrong-line
  snaps, worst -0.25); gated closing 0 / negative; rectangle fit 0; gated opening (k15, p<0.7) +0.0026; marker watershed over
  stroke darkness (6 px band) +0.0055; ink split +0.0009; GrabCut seeded by the prediction b8 -0.003 / b20 -0.040 / b40 -0.078
  (+0.10..+0.30 on the 4 worst sheets, large losses elsewhere; no label-free uncertainty measure predicts which, |rho| <= 0.07);
  GrabCut on a texture image worse. False components (meanP median 0.64 vs 0.98 for true ones) cannot be dropped without
  dropping as much true area.
- **Pre-registered composition (opening k15 t0.7 -> watershed b6), one read** vs the evaluator mask:
  Gemini30 +0.0119 (default +0.0082), val +0.0050 (default +0.0061), HF14 -0.0030 (default +0.0021).
  Does not transfer; not adopted. The default (scripts/polygonize.py) stays.
- **Learned band refinement (scripts/pp_probe/learnref.py, 2026-10-19).** Gradient-boosted trees (sklearn HGB) classify every
  pixel within 4 native px of lin_fill's edge from 25 classical features (probability, its smoothing / gradient / Laplacian,
  signed distance, grey, black-hat 7 / 15, ink v2, ink distance / density, and the local inside-vs-outside Lab colour and
  black-hat texture contrast over 21 / 61 px windows = a per-pixel GrabCut). Trained on Gemini30 only.
  Gemini30 5-fold sheet-grouped CV: **+0.0043** vs lin_fill (82 up / 32 down; 2 sheets down > 0.005); top features P, 61-px colour
  contrast, grey, black-hat 15. One read (vs the evaluator mask): **val -0.0007** (default +0.0061), **HF14 -0.0039** (default
  +0.0021). Same pattern as every Gemini-tuned rule: Gemini gains, real loses. Not adopted.
- **Conclusion:** the limit is the training signal for the post-processor, not the method. Gemini30 boundaries do not behave like
  the real 28's (despite the same half-line label convention), so anything fitted on Gemini transfers negatively. CascadePSP is
  too slow on CPU (fast mode 348 s per question) and moved to the GPU paste with TTA.
- **Cross-fit on the real sheets (user's call, 2026-10-19):** the same learned band refinement trained on one real split, scored
  on the other (sampled 150k band pixels per sheet; threshold 0.5, band 4 px fixed in advance). vs the default (lin_fill+poly),
  Wilcoxon per question:
  | train -> test | per question | per sheet |
  |---|---|---|
  | val -> HF14 | +0.0032 (22 up / 8 down, p=0.018) | +0.0031 (worst -0.005) |
  | val -> HF14 fix v2 | +0.0024 (p=0.056) | +0.0029 |
  | HF14 -> val | **-0.0032** (10 / 31, p=0.003) | +0.0019 (worst -0.013) |
  | val+Gemini -> HF14 | +0.0009 (p=0.17) | +0.0010 |
  | val+Gemini -> HF14 fix v2 | +0.0023 (p=0.07) | +0.0015 |
  | HF14+Gemini -> val | -0.0009 (p=0.56) | +0.0028 |
  Real boundaries are learnable a little (val -> HF14 holds out), but the gain is ~+0.003 and does not hold in the other direction
  per question; adding Gemini dilutes it. Far from the +0.02 goal; not adopted.

### 2026-10-20 (GH200) -- within-run checkpoints and a seed replicate: the swings are seed noise; late-checkpoint averaging does not help; Hub soups fail

`train_refunet.py --save-every N` (new): model weights every N optimizer steps (`step_XXXXX.pth`) and `order_epoch_E.json` (per step,
the image files in its batch). It wraps the same `RandomSampler` on the same generator, so the order and the worker seeds are
unchanged (checked on a toy loader: identical indices and base seeds). kq no-DR arm retrained on the rebuilt `single18kq_s600000`
(17,906 sheets, as before), kq recipe, `--save-every 100` (22 checkpoints + epoch_0), at seed 7 (= the Hub arm's seed) and seed 8.
Every checkpoint scored @4096, hand boxes, on HF14 / fix v2 / val / Gemini30. Scoring beside batch-8 training OOMs (training
takes ~69 GB); 8 scoring jobs at once OOM too -- 4 at a time is safe.

**Same seed reproduces; a different seed does not** (end of epoch, set mean, per-sheet sd of the difference, sheets moving > 0.05):

| | HF14 | fix v2 | val | Gemini30 | sheets > 0.05 |
|---|---:|---:|---:|---:|---|
| Hub kq-nodr (seed 7, 2026-10-15) | 0.9147 | 0.9214 | 0.8954 | 0.8677 | |
| rerun seed 7 | 0.9146 | 0.9209 | 0.8924 | 0.8714 | 0 / 58 (per-sheet sd 0.003-0.012) |
| seed 8 | 0.8986 | 0.9154 | 0.8914 | 0.8556 | HF14 12 -0.09, 27 -0.06; Gemini30 17 -0.15, 11 -0.11, 26 +0.11, +2 (per-sheet sd 0.028 / 0.046) |

- GPU nondeterminism is negligible; **the seed alone moves HF14 and Gemini30 by ~0.016 and 5 / 30 Gemini30 sheets by > 0.05**. The
  gate's "no sheet down > 0.05" rule fires on seed noise for Gemini30 1 / 11 / 17 / 26 and HF14 12 / 27. "Only kq gets Gemini30 17
  right" (2026-10-17) is largely seed 7: kq at seed 8 gives 0.663. Every single-pass run so far used seed 7.
- **Within a run** the sheets oscillate: Gemini30 1 0.14-0.82 (seed 7) / 0.35-0.85 (seed 8) across checkpoints, 17 0.50-0.89; set
  means wobble 0.01-0.02 (seed 7 step 1700: every set -0.05 at once). Step-to-step sheet changes have lag-1 autocorrelation -0.29
  to -0.43 (white noise around a level = -0.5): mostly a wobble that reverses, not lasting jumps.
- **Late-checkpoint averaging does not help** (uniform over steps 1200-end / 1700-end vs the run's epoch_0): seed 7 HF14 -0.008 /
  -0.008, Gemini30 -0.006 / -0.004; seed 8 HF14 -0.001 / -0.002, Gemini30 +0.013 / +0.008; val within +-0.003. It does not shrink
  the seed gap (per-sheet sd 0.023-0.058). The late checkpoints sit around one point; the seed decides the point.
- **Averaging across seeds does:** 0.5 seed 7 + 0.5 seed 8 (no-DR only) beats both seeds on every set -- HF14 0.9151, fix v2 0.9267,
  val 0.8933, Gemini30 0.8719 (vs best single 0.9146 / 0.9209 / 0.8924 / 0.8714); no sheet > 0.01 below both seeds. It is below the
  best (a 4-arm blend; val -0.0055, p=0.0005; Gemini30 1 -0.16).
- **Adding the seed-8 arm to the best** (0.15 kq-dr + 0.175 kq-nodr s7 + 0.175 kq-nodr s8 + 0.15 kms-dr + 0.35 kms-nodr; the
  rebuild without s8 reproduces 0.9163): HF14 -0.0002, fix v2 +0.0014, val -0.0005 (p=0.04), Gemini30 -0.0028 (p=0.008). Fails.
- **Soups of the Hub arms (kq, kms, kw, kg, s36k), eval only:** 0.3 / 0.7 per run: HF14 +0.0040 (p=0.05), fix v2 +0.0019, val
  +0.0006, Gemini30 -0.0017 (sheet 1 -0.10); uniform over all 10 arms: Gemini30 -0.0063 (p=0.014). Both fail the gate.
- **Attribution:** with the wobble reversing step to step, per-window credit would mostly track noise; not pursued. `order_epoch_0.json`
  for both seeds is in `data/runs/ck_kqtrace_s{7,8}_nodr/`.

### 2026-10-20 (CPU) -- working backwards from single errors: error taxonomy, reference-adaptive cell voting
- **Looked at the errors** (HF14 12 q00 0.946, val 13 q05/q08, Gemini30 16 q05/q13, HF14 14 q00, HF14 12 q01, HF14 18 q03): blank
  bands past the last hatch line / trim strips / sills taken in; the last board row before the outline left out; hatch under
  text, symbols, dashed arcs left out (blotchy holes); whole same-texture bands missed (HF14 14 q00, val 13 q05).
- **Taxonomy of error pieces** (`scripts/pp_probe/taxonomy.py`, gain if every piece of a class were fixed): FN under clutter
  HF14 +0.037 / val +0.021 / Gemini +0.014; FP textured (other family) Gemini +0.072 / val +0.039 / HF14 +0.013; thin slivers
  +0.011..0.019; "blank" FP ~0 under the crude density measure.
- **Why cell voting failed before:** ink thresholds (fixed 60, or ink v2's per-sheet clip) miss faint anti-aliased hatch lines
  (black-hat 34-85 on HF14 12), so hatch strips merge into giant cells that fall back to the pixel decision. Fix: threshold from
  the reference box's own stroke contrast (`ref_threshold`: 0.5 x p95 black-hat in the box, clip 10..60).
- **Reference-adaptive cell voting** (`scripts/pp_probe/cellsim.py`, cells <= cap px in iff mean probability > t): picked on
  Gemini30 (vote_t0.4_c0.01: +0.0044 full, LOSO +0.0017), read once: **val -0.0202** (all from val 17, 23 questions at -0.155..-0.160;
  without it -0.0012), **HF14 +0.0020** (p=0.26), fix v2 +0.0020. Not adopted. The edge-cell similarity rule (colour + row
  height vs interior cells) lost on its own motivating cases (Gemini30 16 q05 -0.05..-0.11) and was dropped.
- `--hires-crop` screen (GPU, single18khc): khc30 Gemini30 -0.016 (p<0.01), HF14 -0.007; 0.5 best + 0.5 khc30 Gemini30 -0.004;
  khc30 @2048 far worse. Fails; the best stays.
- **Guarded voting + clutter fill** (`scripts/pp_probe/guard.py`, 2026-10-20). Val 17's collapse was the brick plinth: the light
  mortar network forms "cells" that run out of the region (8.7k px added outside the label), plus thick coloured trims read as ink.
  Guards: additions need the interior colour, only ink within 2 px of a cell follows it, no addition beyond 8 px of the mask.
  Clutter fill: pieces a closing would add, kept if >= rho of their outer ring is in the mask and their colour matches.
  Gemini30: guarded voting **+0.0055** (LOSO +0.0062; 87 up / 28 down), every clutter-fill variant **-0.005..-0.012** (LOSO picks
  none). One read of the pick (vg_c6): **val -0.0062** (p=0.64; val 17 still -0.023; val no longer clean for this rule),
  **HF14 +0.0004** (p=0.19), fix v2 +0.0005. Clutter fill, not picked, for the record: HF14 +0.0013..+0.0026, val +0.0001..-0.0109.
  Neither adopted. Five post-processing families now show the same split: Gemini30 gains, the real sheets do not.

### 2026-10-20 (GH200, CPU side) -- edit-cost proxy for the polygon output (`scripts/editcost.py`)

The user corrects wrong polygons in the frontend (deletes, traces missed sections, moves vertices), so polygons are also scored by
the fixes they need: delete = polygon < 10% on the target; cut = on the target but > 10% of its area and > 0.05% of the sheet
outside the target dilated 4 px (attached spill); add = target piece (>= 0.2% of the target) recalled < 25%; vertices. Best model
(xblend-kq30-kms30) @4096 maps, `polygonize.py` default (IoU reproduces: val 0.9049, HF14 0.9183, Gemini30 0.8821). Rows
`data/diag/rows_editcost_{val,hf14,gemini30}.json`.

| per question | verts median, eps 1 / 2 / 4 | IoU eps 1 / 2 / 4 | delete | cut | add | q needing any fix |
|---|---|---|---:|---:|---:|---:|
| HF14 | 178 / 111 / 66 | 0.9183 / 0.9177 / 0.9157 | 0.33 | 0.00 | 0.12 | 0.31 |
| val | 238 / 133 / 78 | 0.9049 / 0.9056 / 0.9039 | 0.50 | 0.03 | 0.17 | 0.43 |
| Gemini30 | 436 / 251 / 161 | 0.8821 / 0.8816 / 0.8806 | 1.30 | 0.15 | 0.04 | 0.46 |

- Douglas-Peucker eps 2 halves the vertices at the same IoU (-0.0006..+0.0007); eps 4 cuts them ~3x for -0.001..-0.003. eps stays 1
  in `polygonize.py` until the user decides (vertices are what users edit).
- The dominant fix is deleting wrong polygons; attached spill (cuts) is rare on real sheets. Missed sections come second on real.
- **Deleting wrong polygons** (`scripts/pp_probe/delpoly.py`, 2026-10-20; the user's most common edit per `scripts/editcost.py`).
  Per polygon (8-connected piece of the rasterised default output): probability stats, share of the answer, rank, distance to the
  reference box and to the main polygon, Lab / black-hat / stroke-orientation distance to the reference box and the main polygon
  (deployable: no other questions' maps). Wrong = < 10% on the target: Gemini30 294 / 1298 (23%), val 44 / 497, HF14 18 / 186
  (~9%). HGB classifier, delete if P(wrong) > 0.5 (fixed), never the main polygon.
  | train -> test | IoU change | deleted | precision |
  |---|---|---|---|
  | Gemini30 5-fold sheet CV (+ val + HF14 in train) | +0.003..+0.012 every fold (p <= 0.08) | 262 | 72-94% |
  | Gemini30+val -> HF14 | -0.010 (p=0.49) | 38 | 45% (17 / 18 wrong found) |
  | Gemini30+HF14 -> val | -0.005 (p=0.001) | 61 | 33% |
  | val -> HF14 / HF14 -> val (real only) | -0.005 / -0.003 | 10 / 61 | 60% / 31% |
  | Gemini30 + real weighted equally | -0.006 / -0.001 | 35 / 36 | 43% / 58% |
  On the real sheets wrong polygons are rare and look like correct secondary pieces; the deleted correct ones include large,
  confident polygons (HF14 2 q00: 503k px, mean probability 0.81). Automatic deletion is not adopted. Its score could still serve
  the frontend as a per-polygon confidence to flag pieces for one-click review.

### 2026-10-20 (GH200) -- loss variants at two seeds: per-question BCE and family-aware BCE are null; BCE + Dice stays

`train_refunet.py --bce-per-question` (BCE averaged per question, then over the batch, as Dice and the metric weight questions)
and `--bce-other-weight W` (BCE weight W on the pixels of the plan's OTHER labelled families -- the look-alike negatives behind
the wrong polygons users delete; `collate_fn(other_plane=True)`, checked on 40 samples). Defaults reproduce the old loss exactly.
kq no-DR recipe on `single18kq_s600000`, seeds 7 and 8, 0.5 / 0.5 two-seed blend vs the baseline two-seed blend
(`kqtrace_s78_nodr`: HF14 0.9151, fix v2 0.9267, val 0.8933, Gemini30 0.8719); metrics `data/evaluations/loss_{pq,pqo3}_s{7,8,78}`.

| blend vs base blend | HF14 | fix v2 | val | Gemini30 |
|---|---:|---:|---:|---:|
| pq | -0.0062 (p=0.03) | -0.0032 | -0.0001 | +0.0003 |
| pqo3 (pq + other-family weight 3) | -0.0068 (p=0.05) | -0.0033 | -0.0004 | -0.0032 |

- The other-family weight adds -0.001 or less to pq on every set; only per-sheet move > 0.03: Gemini30 17 -0.07 (inside its seed
  swing). As expected with the training loss near zero (~0.03), the loss shape barely reaches the real-sheet errors. Not adopted;
  1 x BCE (pooled) + 2 x per-question Dice stays.
