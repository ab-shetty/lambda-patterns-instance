# Handoff: synth-only transfer search

## Active Goal

Find a synthetic generation/selection schema such that:

- generate exactly **1300 synthetic images**,
- train on those 1300 synth images for **10 epochs**,
- achieve `real_iou > 0.55`,
- and keep `synth_iou - real_iou < 0.05`.

This divergence gate is one-sided. Synthetic validation being **harder** than
real evaluation, i.e. `synth_iou < real_iou`, is acceptable and usually a plus.

This is now a **pure-synth** goal. Mixed real augmentation results are useful
context, but they do not satisfy the goal.

Use the local workspace and `/workspace/probes/*.log` as the source of truth.
Single-run results are noisy, but the current user preference is fast iteration:
seed 0 probes first; only add more seeds for promising candidates.

## Current Best Evidence

### Best real-IoU near miss

Dataset: `/workspace/synth_pool20k_diverse1300`

Built from four 5000-image source pools:

- `/workspace/synth_faintcad5000_rerun`
- `/workspace/synth_cadneg5000_rerun`
- `/workspace/synth_faintcad5000_rerun_b`
- `/workspace/synth_cadneg5000_rerun_b`

Selector:

```bash
python scripts/select_diverse_local.py \
  --sources /workspace/synth_faintcad5000_rerun /workspace/synth_cadneg5000_rerun \
            /workspace/synth_faintcad5000_rerun_b /workspace/synth_cadneg5000_rerun_b \
  --out /workspace/synth_pool20k_diverse1300 \
  --n 1300 \
  --mode-quotas elevation=815,roof_plan=387,freeform=98 \
  --min-score 0.16 --overwrite
```

Run: `/workspace/probes/probe_pool20k_diverse1300_noaug_s0.log`

```text
epoch0 real 0.3529 | synth 0.3092 | div -0.0437
epoch1 real 0.4667 | synth 0.4176 | div -0.0491
epoch2 real 0.4960 | synth 0.5004 | div +0.0044
epoch3 real 0.4737 | synth 0.5274 | div +0.0537
epoch4 real 0.4953 | synth 0.5608 | div +0.0656
epoch5 real 0.5424 | synth 0.6096 | div +0.0673
epoch6 real 0.5232 | synth 0.6441 | div +0.1209
```

Interpretation: closest real-IoU point, but misses both requirements at the
high-real epoch. It needs about `+0.008` real IoU and about `-0.017` divergence.

Seed 1 did not reproduce the high point:

```text
/workspace/probes/probe_pool20k_diverse1300_noaug_s1_s1.log
epoch3 real 0.5226 | synth 0.5707 | div +0.0481
epoch4 real 0.5257 | synth 0.6025 | div +0.0769
epoch5 real 0.5078 | synth 0.5841 | div +0.0763
```

### Best parity near miss

Dataset: `/workspace/synth_pool20k_diverse1300_free140`

This differs from the best dataset by only 45 examples:
42 added freeform, 3 added roof-plan, 45 dropped elevation.

Selector:

```bash
python scripts/select_diverse_local.py \
  --sources /workspace/synth_faintcad5000_rerun /workspace/synth_cadneg5000_rerun \
            /workspace/synth_faintcad5000_rerun_b /workspace/synth_cadneg5000_rerun_b \
  --out /workspace/synth_pool20k_diverse1300_free140 \
  --n 1300 \
  --mode-quotas elevation=770,roof_plan=390,freeform=140 \
  --min-score 0.16 --overwrite
```

Run: `/workspace/probes/probe_pool20k_diverse1300_free140_noaug_s0.log`

```text
epoch0 real 0.3781 | synth 0.3019 | div -0.0761
epoch1 real 0.4718 | synth 0.3877 | div -0.0840
epoch2 real 0.4336 | synth 0.4617 | div +0.0281
epoch3 real 0.5382 | synth 0.5257 | div -0.0124
epoch4 real 0.5267 | synth 0.5970 | div +0.0703
epoch5 real 0.5215 | synth 0.6017 | div +0.0801
```

Interpretation: best synth-harder-than-real point. It misses the real-IoU goal
by `0.0118`, but divergence is excellent at epoch 3.

Seeds 1 and 2 did not reproduce the high real point:

```text
seed1 epoch2 real 0.4808 | synth 0.4857 | div +0.0050
seed1 epoch3 real 0.4572 | synth 0.5207 | div +0.0636
seed2 epoch2 real 0.4999 | synth 0.5080 | div +0.0081
seed2 epoch3 real 0.5043 | synth 0.5457 | div +0.0414
```

Still, `free140` remains the best clue: a small increase in freeform/perimeter
coverage can make synthetic validation harder without fully losing transfer.

## Recent Negative Results

Do not repeat these unless the hypothesis materially changes.

### More of the same 20k-diverse data

Dataset: `/workspace/synth_pool20k_diverse5000`

```text
epoch0 real 0.3549 | synth 0.4392 | div +0.0843
epoch1 real 0.4811 | synth 0.6242 | div +0.1430
epoch2 real 0.5179 | synth 0.6939 | div +0.1760
epoch3 real 0.5163 | synth 0.7216 | div +0.2053
```

Interpretation: 5k images from the same recipe make same-generator synth val
much easier. Real improves some but not enough; divergence explodes.

### Adjacent freeform quotas

```text
free120 epoch3 real 0.4487 | synth 0.5142 | div +0.0656
free135 epoch3 real 0.4929 | synth 0.5912 | div +0.0983
free150 epoch3 real 0.5053 | synth 0.5801 | div +0.0747
free200 epoch3 real 0.5086 | synth 0.5288 | div +0.0202
free200 epoch5 real 0.5038 | synth 0.6087 | div +0.1049
```

Interpretation: `free140` had a useful seed-0 spike, but the nearby quota sweep
did not reveal a robust monotonic improvement. Too much freeform dilutes the
high-transfer elevation/roof mix; too little loses the parity benefit.

### Harder synthetic validation by ordering only

`scripts/build_ordered_hardval_local.py` can place harder examples in the
deterministic `train_split` validation positions.

Results:

```text
hardval epoch1 real 0.3909 | synth 0.2805 | div -0.1104
valmid epoch4 real 0.4966 | synth 0.5415 | div +0.0449
valcal040 epoch5 real 0.5140 | synth 0.6442 | div +0.1302
```

Interpretation: validation ordering can manipulate measured divergence, but it
has not improved the real-IoU ceiling. Do not use it as a substitute for better
training coverage.

### Broad low-annotation selection

Script added: `scripts/select_broad_lowcount_local.py`

Dataset: `/workspace/synth_pool20k_broadlow1300`

```text
epoch0 real 0.3662 | synth 0.4629 | div +0.0967
epoch1 real 0.3845 | synth 0.5168 | div +0.1323
epoch2 real 0.4610 | synth 0.5743 | div +0.1133
```

Interpretation: selecting fewer, larger masks makes synthetic validation too
easy immediately and does not help real enough.

### Mild image degradation

Dataset: `/workspace/synth_pool20k_diverse1300_imgdeg020`

```text
epoch0 real 0.3981 | synth 0.3343 | div -0.0638
epoch1 real 0.4462 | synth 0.4797 | div +0.0335
epoch2 real 0.4682 | synth 0.5335 | div +0.0653
epoch3 real 0.4991 | synth 0.5270 | div +0.0279
epoch4 real 0.5040 | synth 0.6430 | div +0.1390
```

Interpretation: mild degradation delays synth fit but lowers the useful real
curve. It is not enough to fix the gap.

### Train split 0.99

Dataset: original `/workspace/synth_pool20k_diverse1300`, `--train-split=0.99`.

```text
epoch0 real 0.3762 | synth 0.2782 | div -0.0980
epoch1 real 0.3982 | synth 0.4001 | div +0.0018
epoch2 real 0.4446 | synth 0.3977 | div -0.0469
epoch3 real 0.4659 | synth 0.4921 | div +0.0262
```

Interpretation: training on 13 more images did not lift real. The smaller synth
validation split is also noisier, so keep `--train-split=0.98` for comparability.

## Current Training Command

Use this exact probe template unless a run intentionally changes one flag:

```bash
python scripts/probe_multiseed.py --local-data /workspace/<dataset> \
  --seeds 0 --epochs 10 --tag <tag> \
  --image-max-size 1024 --batch-size 1 --num-workers 16 \
  --log-root /workspace/probes \
  --extra "--domain-random --real-eval-indices=12,16,27,7,11,25,23,1,18,2,0,3,14,24 --train-split=0.98 --num-queries=200 --mask-weight=10 --dice-weight=10 --eos-coef=0.03"
```

Important:

- Do **not** add `--realism-aug` by default. It hurt the best path.
- `train_split=0.98` means 1274 train images and 26 synth-val images for a
  1300-image local dataset.
- `probe_multiseed.py` appends `_s<seed>` to the log names. If the tag already
  includes a seed, logs can look like `..._s1_s1.log`.
- Stop runs early when they are clearly off the target path. The user explicitly
  agreed not to run all 10 epochs when it is very unlikely to reach the goal.

## Model And Metric

`RefMask2Former` is a Mask2Former-style, reference-conditioned, class-agnostic
instance segmenter.

Targets are filled material/pattern regions: roof fields, siding, tile/stone
fills, floor/deck/perimeter material regions. They are not 1px CAD linework.

Primary metric in these probes:

```text
mean_gt_iou = per-GT best prediction IoU, averaged over GT masks
divergence = synth_val_iou - real_iou
```

Watch both:

- low divergence with low real IoU is not success,
- high real IoU with large positive divergence means synth is still easier than
  real and not a faithful proxy,
- negative divergence is allowed; it means synth validation is harder than real.

## Useful Diagnostics

Per-real-image diagnostic:

```bash
PYTHONPATH=. python scripts/per_image_real.py \
  --ckpt /workspace/probes/ck_probe_pool20k_diverse1300_noaug_s0/last.pth \
  --image-max-size 1024 --score-thr 0.5
```

For the original best seed-0 checkpoint, weak held-out/full-real examples were:

```text
idx17 iou 0.103, GT 27, pred 107
idx12 iou 0.232, GT 4, pred 71
idx14 iou 0.319, GT 9, pred 71
idx16 iou 0.330, GT 2, pred 82
```

Overlay command:

```bash
PYTHONPATH=. python scripts/overlay_pred.py \
  --checkpoint /workspace/probes/ck_probe_pool20k_diverse1300_noaug_s0/last.pth \
  --image-max-size 1024 --score-thresh 0.5 \
  --indices 12,16,14,24,3,0 \
  --out /workspace/probes/pool20k_diverse1300_last_weak.png
```

Observed failure modes:

- index 12: long thin exterior/perimeter slab/deck strips around a floor plan;
  many false positives inside rooms,
- index 16: broad roof-field segmentation with facet-line context; model
  fragments and leaks,
- index 14: elevation trim/bands plus small rectangles; model overfills large
  wall areas and misses clean small regions,
- index 24 and 3: broad elevation fields where boundary precision and material
  separation are the issue, not total absence of the concept.

## Durable Dead Ends

- Appearance-only realism is a trap. Prior LLM/eyeball style improvements hurt
  or failed to move real.
- Scaling image count alone makes synth validation easy and increases
  divergence.
- Broad low-count/high-area selection makes synth easy.
- Strong/hard validation ordering can make divergence look good while hurting or
  not moving real.
- More freeform is not monotonic: `free140` had one useful spike; nearby values
  did not robustly improve real.

## Files Added During Current Search

- `scripts/select_diverse_local.py`: farthest-point local-data selector over
  `(area score, annotation count, aspect, size)` with mode quotas.
- `scripts/build_ordered_hardval_local.py`: reorder local-data records so the
  deterministic synthetic validation split contains chosen hard examples.
- `scripts/select_broad_lowcount_local.py`: high-area / low-annotation selector;
  currently a negative result, but kept for reproducibility.
- `scripts/degrade_images_local.py`: image-only degradation copy tool.
- `scripts/jitter_annotations_local.py`, `scripts/mosaic_local.py`,
  `scripts/crop_excerpts_local.py`: earlier transforms; none is currently the
  leading path.

## Next Best Moves

The most promising evidence is still the tiny gap between:

- original 20k-diverse epoch 5: high real, too much positive divergence,
- free140 seed-0 epoch 3: excellent divergence, real just below target.

The next useful work should avoid broad sweeps and instead isolate *which exact
examples* in the 45-example original-to-free140 replacement set helped seed 0.
Reasonable next experiments:

1. Build micro-hybrids replacing only 10-25 of the original elevation examples
   with the highest-value freeform additions from free140, not all 42.
2. Inspect the added freeform examples visually and by metadata; prefer those
   matching index-12 perimeter/deck slab failures, not generic room interiors.
3. Add targeted generation for floor-plan perimeter/deck/slab strips while
   preserving the original elevation/roof mix.

Do not mark the goal complete until a current run proves:

```text
real_iou > 0.55
synth_iou - real_iou < 0.05
```

No current run satisfies both.
