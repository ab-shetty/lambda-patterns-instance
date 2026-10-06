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

