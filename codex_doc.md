# Handoff: synth-only transfer search

Authoritative task semantics and metric: [`PROJECT_UNDERSTANDING.md`](PROJECT_UNDERSTANDING.md).

## UPDATE 2026-07-16 — MIXED REAL + BEST SYNTH

The active user goal is no longer synth-only. It is to combine the strongest
synthetic schema with labelled real data and maximize the historical 14-image
HF held-out real mIoU.

The old overall synth-only peak was 0.5593 from the 1600-image
`top80_balanced220` dataset. Its `/workspace` artifact is lost and was never on
Hugging Face; 0.5593 was a single-seed epoch-3 spike (epoch 4 was 0.5049), not a
stable plateau. Do not use the later canonical-500 result (~0.39) as the
project-wide baseline.

The top80 schema was reconstructed locally from four fresh 5000-image
faint-CAD/CAD-negative pools. The final set has the exact old mode counts
(889 elevation / 540 roof / 171 freeform) and a close score mean (0.3868 vs
0.3840). A matched batch-4 pure-synth calibration peaked at 0.5026.

Correct task baseline: **0.3620 reference-conditioned union IoU**. A pattern ID
is local to one image only. Given a user rectangle inside one region, the target
is the union of regions with that same local ID in that image; pattern numbers
must never be compared across images. Both excess selected pixels and missed
pixels lower the score.

Checkpoint:
`data/runs/ck_mix_top80_1600_hf14_rf86_k2_held14_s0_bs4/best_real.pth`.
Correct metrics and visualizations:
`data/visualizations/reference_conditioned_hf14_k2_q200`.

The later class-agnostic Mask R-CNN result (reported as 0.8545) is invalid for
this task because it ignored the reference rectangle and predicted every pattern
region. Its GT-best coverage metric also ignored false positives. Retain its
artifacts only as a negative architecture experiment; do not cite its number as
Floz task performance.

For the old decoder, one variant per scene peaked at
0.5448. Using four variants per real scene
(20% real) peaked lower at 0.5126, so distinct coverage with minimal repetition
remains more important than heavy repetition, while two views gave a small
additional gain. A matched microbatch check also overturned the old hardware-regime
assumption here: batch 4 scored 0.4567 at epoch 1 in 178s, versus batch 1 at
0.3802 in 487s.

Roboflow-only was also measured directly: 86 scenes x four variants, no synth
and no HF training images, peaked at **0.3147** on held-out HF14 at epoch 9.
Afterward local validation rose toward 0.58 while HF transfer stayed near
0.28-0.31. The Roboflow pool is useful as distinct coverage in the top80 mix,
but is not sufficient as a standalone training distribution.

Roboflow `floz-real-pool` version 2 is converted non-destructively at
`data/floz-real-pool-v2-clean`: 86 images, 282 instances, 127 remove polygons
applied as 129 holes. No remote data was deleted or edited.

## Active Goal

Achieve **held-out HF14 reference-conditioned union IoU >= 0.80 after 10
epochs** by combining the best synthetic data with the labelled HF-train and
cleaned Roboflow real pools. The input is a user-selected rectangle and the
output is only matching regions sharing its image-local pattern ID.
Synthetic/real divergence is diagnostic, not an acceptance gate. Mixed real
augmentation results are directly relevant; the old pure-synth-only goal is
superseded.

Use the local workspace and `/workspace/probes/*.log` as the source of truth.
Single-run results are noisy, but the current user preference is fast iteration:
seed 0 probes first; only add more seeds for promising candidates.

## Current Best Evidence

### Active 2000-image probe

Dataset: `/workspace/synth_pool20k_diverse2000_free225`

Built as a fresh 2000-image diverse reselection from the proven four-pool
faintcad/cadneg backbone, not as a duplicate expansion:

```bash
python scripts/select_diverse_local.py \
  --sources /workspace/synth_faintcad5000_rerun /workspace/synth_cadneg5000_rerun \
            /workspace/synth_faintcad5000_rerun_b /workspace/synth_cadneg5000_rerun_b \
  --out /workspace/synth_pool20k_diverse2000_free225 \
  --n 2000 \
  --mode-quotas elevation=1188,roof_plan=587,freeform=225 \
  --min-score 0.16 --overwrite
```

Selection summary:

```text
score_min 0.1600
score_median 0.2647
score_mean 0.3488
score_max 0.8547
ann_count_mean 7.9625
ann_count_median 7
modes: elevation=1188, roof_plan=587, freeform=225
```

Probe started:

```bash
python scripts/probe_multiseed.py \
  --local-data /workspace/synth_pool20k_diverse2000_free225 \
  --seeds 0 --epochs 10 \
  --tag pool20k_diverse2000_free225_noaug_1024 \
  --image-max-size 1024 --batch-size 1 --num-workers 24 \
  --log-root /workspace/probes \
  --extra "--domain-random --real-eval-indices=12,16,27,7,11,25,23,1,18,2,0,3,14,24 --train-split=0.98 --num-queries=200 --mask-weight=10 --dice-weight=10 --eos-coef=0.03"
```

Rationale: 2000 images should be tested first as broader coverage from the
known-transfer distribution. Avoid starting with duplicate expansion; top80
duplicates helped 1600 real IoU but made synthetic validation too easy at the
high-real epoch.

Current result:

```text
epoch0 real 0.4296 | synth 0.3565 | div -0.0731
epoch1 real 0.4573 | synth 0.3910 | div -0.0664
epoch2 real 0.4795 | synth 0.5418 | div +0.0623
```

Interpretation: stopped after epoch 2. The curve stayed too low on real and
crossed the one-sided divergence gate.

Queued/running next candidate:
`/workspace/synth_pool20k_diverse2000_free225_min020`, same quotas but
`--min-score 0.20`. Selection summary: `score_mean 0.3636`,
`score_median 0.2781`, `ann_count_mean 7.903`, modes
`elevation=1188, roof_plan=587, freeform=225`. Hypothesis: more area/coverage
pressure from unique images, without duplicate-driven synthetic validation ease.

Current result:

```text
epoch0 real 0.4394 | synth 0.3818 | div -0.0577
epoch1 real 0.4938 | synth 0.4748 | div -0.0190
epoch2 real 0.4512 | synth 0.5571 | div +0.1059
```

Interpretation: stopped after epoch 2. Raising the min score improved epoch 1
versus the first 2000 run, but real dropped at epoch 2 while synthetic validation
became much easier. This is not the right kind of hardness.

### Active targeted-hard 2000 probe

Dataset: `/workspace/synth_pool20k_areacount2000_quota_hardval_bal`

Built in two steps:

```bash
python scripts/select_area_count_local.py \
  --sources /workspace/synth_faintcad5000_rerun /workspace/synth_cadneg5000_rerun \
            /workspace/synth_faintcad5000_rerun_b /workspace/synth_cadneg5000_rerun_b \
  --out /workspace/synth_pool20k_areacount2000_quota \
  --n 2000 \
  --mode-quotas elevation=1111,roof_plan=675,freeform=214 \
  --target-ann elevation=8,roof_plan=8,freeform=7 \
  --target-cat 2.5 --min-score 0.16 --overwrite

python scripts/build_ordered_hardval_local.py \
  --train-src /workspace/synth_pool20k_areacount2000_quota \
  --hard-src /workspace/synth_pool20k_areacount2000_quota \
  --out /workspace/synth_pool20k_areacount2000_quota_hardval_bal \
  --n 2000 --train-split 0.98 \
  --target-score 0.40 --target-ann-count 8 --target-cat-count 3 \
  --val-mode-quotas elevation=14,roof_plan=14,freeform=12 \
  --min-score 0.18 --max-score 0.86 --overwrite
```

Summary:

```text
train n=1960 score_mean=0.4357 ann_count_mean=6.9541 cat_count_mean=3.2648
val   n=40   score_mean=0.4405 ann_count_mean=6.5000 cat_count_mean=2.6000
```

Rationale: target the real-correlated hard mode: high-coverage, multi-object
examples, while keeping a mixed hard synthetic validation split. This is meant
to be hard in a way that should help the low held-out real indices rather than
just lowering synthetic validation IoU.

Current result:

```text
epoch0 real 0.4162 | synth 0.4453 | div +0.0291
epoch1 real 0.4209 | synth 0.5314 | div +0.1105
```

Interpretation: stopped after epoch 1. Globally high-area/high-count selection
is the wrong hardening direction: it made synth validation easy by epoch 1 and
did not transfer to real. Prefer preserving the top80/top100 transfer backbone
and adding coverage around it, rather than replacing the distribution with pure
top-area examples.

### Top80 backbone plus 400 unique examples

Dataset: `/workspace/synth_pool20k_top80plus400_unique_hardval_bal`

Built by extending the strongest 1600-image top80 dataset with 400 unique
examples from the original four 5k pools:

```bash
python scripts/extend_local_dataset.py \
  --base /workspace/synth_pool20k_diverse1300_upsample1600_top80_balanced220 \
  --sources /workspace/synth_faintcad5000_rerun /workspace/synth_cadneg5000_rerun \
            /workspace/synth_faintcad5000_rerun_b /workspace/synth_cadneg5000_rerun_b \
  --out /workspace/synth_pool20k_top80plus400_unique \
  --n 2000 \
  --add-mode-quotas elevation=222,roof_plan=135,freeform=43 \
  --min-score 0.16 --overwrite
```

Then ordered with a mixed hard validation split:

```text
train n=1960 score_mean=0.4031 ann_count_mean=7.1362 cat_count_mean=2.9837
val   n=40   score_mean=0.3710 ann_count_mean=5.8000 cat_count_mean=2.2500
```

Rationale: keep the transfer-positive top80 backbone and use the extra 400
images for unique coverage instead of replacing the distribution with global
top-area examples.

Current result:

```text
epoch0 real 0.3768 | synth 0.4355 | div +0.0587
epoch1 real 0.4273 | synth 0.4430 | div +0.0157
```

Interpretation: stopped after epoch 1. Divergence was valid at epoch 1, but real
was far below the top80 trajectory. Test the raw top80+400 extension before
discarding the family, because hard-val ordering may have disturbed the useful
training split.

Raw variant running: `/workspace/synth_pool20k_top80plus400_unique`
(`probe_pool20k_top80plus400_unique_raw_noaug_1024_s0.log`).

```text
raw epoch0 real 0.3856 | synth 0.4016 | div +0.0160
raw epoch1 real 0.4069 | synth 0.4237 | div +0.0168
```

Interpretation: stopped after epoch 1. Raw extension also failed, so the extra
400 unique examples diluted the top80 signal rather than improving it.

Next setting probe: run the best 2000 dataset so far
(`/workspace/synth_pool20k_diverse2000_free225_min020`) at `image_max_size=1536`.
At 1024 it reached epoch1 `real 0.4938`, `synth 0.4748`, div `-0.0190` before
collapsing at epoch2; higher resolution is worth one targeted check.

```text
1536 epoch0 real 0.4125 | synth 0.3420 | div -0.0706
1536 epoch1 real 0.4502 | synth 0.4241 | div -0.0261
```

Interpretation: stopped after epoch 1. 1536 keeps synth harder, but real is
worse than the 1024px epoch1 point (`0.4938`) and not on a path to `0.60`.
Higher resolution is not the missing piece for this 2000-image selection.

### Direct mildtransfer-v1 2000 generator pool

Dataset: `/workspace/synth_mildtransfer_v1_2000`

Existing generated pool from `generate_synthetic_v5.py --recipe mildtransfer-v1`.
Composition:

```text
n=2000
modes elevation=1126, roof_plan=541, freeform=333
score_mean=0.2971 score_median=0.1961 score_min=0.0071 score_max=0.9951
ann_count_mean=6.853 ann_count_median=5
```

Rationale: unlike the selection-only variants, this is a true generator-schema
change near the proven faintcad/cadneg family. Probe directly first; if real
transfer appears but coverage is too low, generate/select a higher-coverage
mildtransfer pool next.

Current result:

```text
epoch0 real 0.3858 | synth 0.4043 | div +0.0185
epoch1 real 0.4053 | synth 0.5013 | div +0.0960
```

Interpretation: stopped after epoch 1. Direct mildtransfer-v1 did not transfer;
it started low and synth validation became easy by epoch 1. Do not train this
exact 2000 pool further.

### Highest real-IoU 1600-image point so far

Dataset: `/workspace/synth_pool20k_diverse1300_upsample1600_top80_balanced220`

Built by starting from the 1300-image transfer backbone, duplicating 80
top area/count examples, then filling the remaining 220 duplicate slots with
balanced mode selection.

```text
composition: 1300 base + 300 duplicates
top_score_count=80, remaining duplicates balanced
modes after duplication: elevation=889, roof_plan=540, freeform=171
score_mean 0.3840
ann_count_mean 7.7825

1024 probe:
epoch0 real 0.4325 | synth 0.3577 | div -0.0748
epoch1 real 0.4843 | synth 0.5055 | div +0.0212
epoch2 real 0.4542 | synth 0.4769 | div +0.0227
epoch3 real 0.5593 | synth 0.6197 | div +0.0604
epoch4 real 0.5049 | synth 0.6443 | div +0.1394
```

Interpretation: top80 is the strongest 1600-image real-transfer signal so far,
but the useful epoch misses the one-sided divergence gate by about 0.01 and is
still below the active `real_iou > 0.60` target.

### Best one-sided-valid 1600-image result so far

Dataset: `/workspace/synth_pool20k_diverse1600_free180`

Selector:

```bash
python scripts/select_diverse_local.py \
  --sources /workspace/synth_faintcad5000_rerun /workspace/synth_cadneg5000_rerun \
            /workspace/synth_faintcad5000_rerun_b /workspace/synth_cadneg5000_rerun_b \
  --out /workspace/synth_pool20k_diverse1600_free180 \
  --n 1600 \
  --mode-quotas elevation=950,roof_plan=470,freeform=180 \
  --min-score 0.16 --overwrite
```

Run: `/workspace/probes/probe_pool20k_diverse1600_free180_noaug_s0.log`

```text
epoch0 real 0.4057 | synth 0.3576 | div -0.0481
epoch1 real 0.4827 | synth 0.4749 | div -0.0078
epoch2 real 0.4341 | synth 0.5056 | div +0.0716
epoch3 real 0.4766 | synth 0.5521 | div +0.0754
```

Higher-resolution probe:
`/workspace/probes/probe_pool20k_diverse1600_free180_noaug_1536_s0.log`

```text
epoch0 real 0.4111 | synth 0.3395 | div -0.0716
epoch1 real 0.4874 | synth 0.4833 | div -0.0041
epoch2 real 0.5089 | synth 0.5284 | div +0.0194
epoch3 real 0.4840 | synth 0.5615 | div +0.0775
```

Interpretation: best one-sided-valid 1600 result so far. `image_max_size=1536`
helps and keeps epoch 2 valid, but it is still well below the new `real_iou >
0.60` goal. Positive divergence reopens after epoch 2. Do not rerun this exact
quota unless testing a materially different training/eval setting.

### Best historical 1300-image near misses

These no longer satisfy the active image-count or IoU target, but they are still
the strongest clues about transferable synthetic distributions.

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

### 1600-image quota scaling around the 20k-diverse pool

The active target changed to 1600 images and `real_iou > 0.60`. Simple quota
scaling did not find a path.

```text
/workspace/synth_pool20k_diverse1600_free140
epoch0 real 0.4278 | synth 0.3000 | div -0.1278
epoch1 real 0.4011 | synth 0.3709 | div -0.0301
epoch2 real 0.4707 | synth 0.4348 | div -0.0359
resume epoch3 real 0.4964 | synth 0.4904 | div -0.0060

/workspace/synth_pool20k_diverse1600_free180
epoch0 real 0.4057 | synth 0.3576 | div -0.0481
epoch1 real 0.4827 | synth 0.4749 | div -0.0078
epoch2 real 0.4341 | synth 0.5056 | div +0.0716
epoch3 real 0.4766 | synth 0.5521 | div +0.0754

/workspace/synth_pool20k_diverse1600_free220
epoch0 real 0.3789 | synth 0.2722 | div -0.1067
```

Interpretation: free140/free180 can make synth harder early, but real remains far
below 0.60 at 1024.

Stopping lesson: `free140` should have been allowed exactly one more epoch when
it was at epoch 2 (`real 0.4707`, `synth 0.4348`, div `-0.0359`). The resumed
epoch 3 did improve to `real 0.4964` with valid divergence, but it still did not
match the best 1536px `free180` result. Harder synth is useful under the
corrected one-sided divergence rule, so do not prune a run just because synth
validation starts below real. Let hard-synth curves breathe for one more epoch
when real is still climbing or near the current best; stop soon after if real
stalls/collapses or positive divergence opens.

### Higher-resolution 1600 probes

`image_max_size=1536` is a useful training-setting lever, but it has not solved
the generation/selection problem.

```text
/workspace/synth_pool20k_diverse1600_free180
epoch0 real 0.4111 | synth 0.3395 | div -0.0716
epoch1 real 0.4874 | synth 0.4833 | div -0.0041
epoch2 real 0.5089 | synth 0.5284 | div +0.0194
epoch3 real 0.4840 | synth 0.5615 | div +0.0775

/workspace/synth_pool20k_diverse1600_free200
epoch0 real 0.4069 | synth 0.3279 | div -0.0790
epoch1 real 0.4126 | synth 0.4062 | div -0.0064
epoch2 real 0.4115 | synth 0.4939 | div +0.0824

/workspace/synth_pool20k_diverse1600_free140
epoch0 real 0.4208 | synth 0.3174 | div -0.1034
epoch1 real 0.4521 | synth 0.3953 | div -0.0568
epoch2 real 0.4916 | synth 0.4706 | div -0.0209
epoch3 real 0.4834 | synth 0.5202 | div +0.0369

/workspace/synth_pool20k_diverse1600_roof500_free180
quotas elevation=920,roof_plan=500,freeform=180
epoch0 real 0.3924 | synth 0.2907 | div -0.1017
epoch1 real 0.4744 | synth 0.4268 | div -0.0476
epoch2 real 0.3971 | synth 0.4342 | div +0.0371

/workspace/synth_pool20k_diverse1600_free180 at image_max_size=1792
epoch0 real 0.4512 | synth 0.3574 | div -0.0938
epoch1 real 0.4439 | synth 0.4638 | div +0.0199
epoch2 real 0.4072 | synth 0.4914 | div +0.0842
```

`free200` was stopped during epoch 3 after epoch 2 failed to recover and
divergence turned positive. `free140` at 1536 kept synth harder longer but did
not beat `free180` at 1536; it was stopped during epoch 4. `roof500_free180`
tested a previously fixed mode axis from the same four proven source pools. It
was allowed through epoch 2 because epoch 1 had hard-synth valid divergence, but
epoch 2 collapsed on real while divergence turned positive; do not keep pushing
roof quota upward from this backbone. `free180` at 1792 fits memory and has a
strong hard-synth epoch0, but collapses by epoch2; 1536 remains the useful
higher-resolution setting. A 2048px free180 probe OOMed before epoch 0 with the
current `batch_size=1`, `num_queries=200` setup.

### Geometry-targeted strip/broad-field selection

Scripts added:

- `scripts/select_geometry_targeted_local.py`
- `scripts/build_geometry_hybrid_local.py`

Hypothesis: target known weak real cases by selecting examples with more long
thin strip masks (perimeter/deck/slab failures) and broad material fields
(roof/elevation failures), while keeping the proven `free180` mode quotas.

Full geometry reselection:
`/workspace/synth_geometry_targeted1600_free180`

```text
quotas elevation=950,roof_plan=470,freeform=180
score_mean 0.3836, ann_count_mean 7.2575
strip_area_mean 0.2611 vs 0.0911 for free180
broad_area_mean 0.3147 vs 0.2464 for free180

1536 probe:
epoch0 real 0.4159 | synth 0.4699 | div +0.0540
epoch1 real 0.4158 | synth 0.5569 | div +0.1411
```

Interpretation: this overcorrected. Geometry-heavy examples were easy for
same-generator synth validation and did not move real.

Small hybrid:
`/workspace/synth_geometry_hybrid1600_free180_r150`

Built from `/workspace/synth_pool20k_diverse1600_free180` by replacing only 150
low-geometry examples with high-geometry examples from the full geometry set:
70 elevation, 60 roof-plan, 20 freeform. Quotas stayed unchanged.

```text
selected_strip_mean 0.1251
selected_broad_mean 0.2641
selected_ann_count_mean 7.6963

1536 probe:
epoch0 real 0.4237 | synth 0.3785 | div -0.0452
epoch1 real 0.3978 | synth 0.4519 | div +0.0541
```

Interpretation: even a 150-example geometry replacement hurt the epoch-1 real
trajectory versus `free180`/1536. Do not blindly increase strip/broad-field
geometry coverage from the same source pools; if revisiting this, make it much
smaller or combine with a separate realism/domain intervention.

### Lower-area tail from proven 20k sources

Hypothesis: the original `--min-score 0.16` cutoff might make same-generator
synth validation too easy by excluding low-coverage/harder examples. Tested the
same `free180` quotas with lower `--min-score`, at 1536px.

`/workspace/synth_pool20k_diverse1600_free180_min010`

```text
score_mean 0.3337 vs 0.3513 for free180
score_median 0.2560 vs 0.2680
epoch0 real 0.4087 | synth 0.2938 | div -0.1149
epoch1 real 0.4160 | synth 0.4484 | div +0.0324
```

`/workspace/synth_pool20k_diverse1600_free180_min014`

```text
score_mean 0.3442 vs 0.3513 for free180
score_median 0.2610 vs 0.2680
epoch0 real 0.3859 | synth 0.3147 | div -0.0713
epoch1 real 0.4341 | synth 0.4617 | div +0.0276
```

Interpretation: lowering the area cutoff does make synth harder, but it lowers
the real trajectory too much. The best path remains `min-score 0.16`/free180 at
1536. Do not continue this cutoff sweep unless paired with a new source pool or
different training regime.

### Upsampling the strongest 1300-image set to 1600

Script added: `scripts/upsample_local_dataset.py`

Hypothesis: the old 1300-image diverse set had the highest historical real-IoU
point, and most attempts to add 300 new examples hurt. Test a 1600-image schema
that keeps the 1300 records and fills the remaining 300 by duplicating/upweighting
existing synthetic examples with fresh filenames.

First diagnostic: run the old best 1300 set itself at the useful 1536px training
setting.

```text
/workspace/synth_pool20k_diverse1300 at image_max_size=1536
epoch0 real 0.3802 | synth 0.3106 | div -0.0697
epoch1 real 0.4325 | synth 0.4125 | div -0.0201
```

Interpretation: the old 1300 high-real behavior did not carry over to 1536 in
the early curve. Its 1024 epoch5 spike is not enough evidence that this backbone
is better under the current 1600/1536 search.

Dataset:
`/workspace/synth_pool20k_diverse1300_upsample1600_balanced`

```text
source: /workspace/synth_pool20k_diverse1300
strategy: balanced duplicate selection
composition: 1300 base + 300 duplicates
modes after duplication: elevation=916, roof_plan=488, freeform=196
score_mean 0.3703
ann_count_mean 7.6031

1536 probe:
epoch0 real 0.4203 | synth 0.3347 | div -0.0856
epoch1 real 0.4094 | synth 0.4059 | div -0.0035

1024 probe, matching the old 1300 setting where the parent set spiked late:
epoch0 real 0.3662 | synth 0.2687 | div -0.0975
epoch1 real 0.4739 | synth 0.4774 | div +0.0035
epoch2 real 0.4896 | synth 0.5062 | div +0.0166
epoch3 real 0.4991 | synth 0.5895 | div +0.0904
epoch4 real 0.5107 | synth 0.6325 | div +0.1219
```

Interpretation: upweighting the old 1300 set did not preserve its high-real
behavior under the 1600 requirement. At 1536 it made synth hard but real dropped
by epoch1. At 1024 it recovered to `real 0.5107`, but synth fit ran away and
divergence exceeded the one-sided gate after epoch2. The old 1300 epoch5 spike
does not transfer cleanly once 300 duplicate images are added. Do not repeat
simple duplicate-upsample unless changing the duplication strategy substantially.

Alternative duplicate strategy:
`/workspace/synth_pool20k_diverse1300_upsample1600_topscore`

```text
source: /workspace/synth_pool20k_diverse1300
strategy: duplicate top area/count examples
composition: 1300 base + 300 duplicates
modes after duplication: elevation=815, roof_plan=687, freeform=98
score_mean 0.4127
ann_count_mean 8.3356

1024 probe:
epoch0 real 0.3680 | synth 0.3180 | div -0.0499
epoch1 real 0.5083 | synth 0.5076 | div -0.0007
epoch2 real 0.4175 | synth 0.4190 | div +0.0015
epoch3 real 0.5205 | synth 0.6387 | div +0.1182
epoch4 real 0.5138 | synth 0.6650 | div +0.1512

1536 probe:
epoch0 real 0.4086 | synth 0.3179 | div -0.0907
epoch1 real 0.4170 | synth 0.4408 | div +0.0238
epoch2 real 0.4265 | synth 0.4554 | div +0.0289
```

Interpretation: top-score duplication recovers a useful early real signal
(`real 0.5083` at epoch1, `0.5205` at epoch3), but it makes same-generator synth
validation easy by epoch3. The 1536 version kept synth harder but failed to
recover the real trajectory, so higher resolution is not the fix for this
duplicate strategy. This is better than balanced duplication for real trajectory
at 1024, but still far below `real_iou > 0.60`, and divergence opens badly. The
deterministic val split contains only 5 duplicate examples, so the easy synth-val
readout is not merely duplicate leakage into validation.

Train-preserving low-val variant:
`/workspace/synth_pool20k_diverse1300_upsample1600_topscore_lowval`

Built with `scripts/build_ordered_hardval_local.py`, preserving the top-score
train positions for split seed 0 while replacing the 32 deterministic val
positions with lower-area/count examples:

```text
train n=1568, score_mean 0.4133, ann_count_mean 8.3380
val n=32, score_mean 0.1602, ann_count_mean 4.25

1024 probe:
epoch0 real 0.3321 | synth 0.3165 | div -0.0156
epoch1 real 0.4489 | synth 0.4791 | div +0.0302
epoch2 real 0.3725 | synth 0.4639 | div +0.0914
```

Interpretation: making the synthetic validation split harder did not preserve
the useful top-score real trajectory. It kept divergence valid briefly, but real
was much worse than the un-ordered top-score run. Do not rely on validation
ordering to rescue this path; the training distribution itself still needs to
raise real IoU.

Mild image-degraded variant:
`/workspace/synth_pool20k_diverse1300_upsample1600_topscore_imgdeg025`

Built from the top-score dataset with `scripts/degrade_images_local.py
--strength 0.25` to slow same-generator synth fitting while preserving geometry
and annotations.

```text
1024 probe:
epoch0 real 0.3813 | synth 0.3174 | div -0.0639
epoch1 real 0.4723 | synth 0.4742 | div +0.0020
epoch2 real 0.4684 | synth 0.4899 | div +0.0215
epoch3 real 0.4801 | synth 0.5706 | div +0.0905
```

Interpretation: mild image degradation delayed synth fitting but also removed
the useful top-score real jump. It did not solve the top-score failure mode.
Do not continue image-degradation variants of this path unless paired with a
new data distribution, not just the same duplicated geometry.

Partial top-score duplicate strategy:
`/workspace/synth_pool20k_diverse1300_upsample1600_top100_balanced200`

Built by duplicating 100 top area/count examples first, then filling the
remaining 200 duplicate budget with balanced mode selection:

```text
composition: 1300 base + 300 duplicates
modes after duplication: elevation=882, roof_plan=554, freeform=164
score_mean 0.3875
ann_count_mean 7.8406

1024 probe:
epoch0 real 0.3949 | synth 0.3447 | div -0.0502
epoch1 real 0.4802 | synth 0.4859 | div +0.0057
epoch2 real 0.4601 | synth 0.4573 | div -0.0028
epoch3 real 0.5529 | synth 0.6129 | div +0.0600
epoch4 real 0.5188 | synth 0.6196 | div +0.1008
```

Interpretation: this is the best 1600-image real-IoU point so far (`0.5529`) and
the first 1600 run to clear 0.55, but it misses the one-sided divergence gate by
about 0.01 at the useful epoch and still falls short of the active `real_iou >
0.60` target. The useful direction is a mixed duplicate budget: some top-score
pressure lifts real, but too much makes synth validation run away. Next local
sweep should vary `top_score_count` around 60-100, not return to all-balanced or
all-top-score.

Lighter partial top-score duplicate strategy:
`/workspace/synth_pool20k_diverse1300_upsample1600_top80_balanced220`

```text
composition: 1300 base + 300 duplicates
top_score_count=80, remaining duplicates balanced
modes after duplication: elevation=889, roof_plan=540, freeform=171
score_mean 0.3840
ann_count_mean 7.7825

1024 probe:
epoch0 real 0.4325 | synth 0.3577 | div -0.0748
epoch1 real 0.4843 | synth 0.5055 | div +0.0212
epoch2 real 0.4542 | synth 0.4769 | div +0.0227
epoch3 real 0.5593 | synth 0.6197 | div +0.0604
epoch4 real 0.5049 | synth 0.6443 | div +0.1394
```

Interpretation: top80 is the new best 1600-image real point (`0.5593`) but
misses the divergence gate by essentially the same amount as top100. The
top-score pressure is real-IoU-positive, but the useful epoch still has synth
about 0.06 ahead of real. Continue the local sweep below top80 only if trying to
shave the last ~0.01 divergence; otherwise this family appears capped around
`real 0.55-0.56`.

Too-light partial top-score duplicate strategy:
`/workspace/synth_pool20k_diverse1300_upsample1600_top60_balanced240`

```text
composition: 1300 base + 300 duplicates
top_score_count=60, remaining duplicates balanced
modes after duplication: elevation=895, roof_plan=527, freeform=178
score_mean 0.3807
ann_count_mean 7.7269

1024 probe:
epoch0 real 0.3933 | synth 0.3321 | div -0.0612
epoch1 real 0.4196 | synth 0.4088 | div -0.0108
epoch2 real 0.4045 | synth 0.4600 | div +0.0555
```

Interpretation: top60 loses the real-IoU lift while still crossing the
divergence gate by epoch2. The useful range is not below 60; current best remains
top80/top100, with top80 having the best real point.

### Mildtransfer v1 source-pool expansion

Generator recipe added in `generate_synthetic_v5.py`: `--recipe mildtransfer-v1`.

Intent: conservative source-pool expansion around the proven faintcad/cadneg
direction. Unlike `realhard-v1/v2`, it keeps the broad transfer mix and only
mildly removes shortcuts: fewer perfect outlines, modest construction-sheet/roof
field probability, low realstyle probability, light label jitter, small
crop/scale variation, and a little CAD clutter.

Generated source pool:
`/workspace/synth_mildtransfer_v1_2000`

```text
n=2000
modes elevation=1126, roof_plan=541, freeform=333
score_mean 0.2971, score_median 0.1960
ann_count_mean 6.85, ann_count_median 5
```

Conservative hybrid:
`/workspace/synth_mildtransfer_hybrid1600_free180_r150`

Built from the best current backbone
`/workspace/synth_pool20k_diverse1600_free180` by replacing 150 examples with
`mildtransfer-v1` examples:

```text
replace_counts elevation=80,roof_plan=50,freeform=20
composition 1450 base + 150 mildtransfer
same mode quotas: elevation=950,roof_plan=470,freeform=180
score_mean 0.3629 vs 0.3513 for base free180
ann_count_mean 7.53 vs 7.89
```

1536 probe:

```text
epoch0 real 0.4138 | synth 0.3224 | div -0.0915
epoch1 real 0.4596 | synth 0.4375 | div -0.0221
epoch2 real 0.4792 | synth 0.4695 | div -0.0097
epoch3 real 0.4944 | synth 0.5185 | div +0.0241
epoch4 real 0.4893 | synth 0.5493 | div +0.0600
```

Interpretation: useful hardness and smooth real climb, but still below the
current best `free180`/1536 point (`real 0.5089`, div `+0.0194`). Epoch 4 also
crosses the one-sided divergence gate. The mildtransfer direction is less bad
than geometry/low-score selection, but a 150-example replacement is not enough
to raise the ceiling. Plausible follow-up: test smaller replacement counts
(40-80) or larger source pool + selector that samples mildtransfer only for
examples that preserve high-transfer statistics.

Smaller hybrid:
`/workspace/synth_mildtransfer_hybrid1600_free180_r60`

Built from the same base by replacing only 60 examples:
30 elevation, 20 roof-plan, 10 freeform.

```text
composition 1540 base + 60 mildtransfer
same mode quotas: elevation=950,roof_plan=470,freeform=180
score_mean 0.3571 vs 0.3513 for base free180
ann_count_mean 7.70 vs 7.89

1536 probe:
epoch0 real 0.3474 | synth 0.2986 | div -0.0488
epoch1 real 0.4537 | synth 0.3859 | div -0.0678
```

Interpretation: smaller replacement made synth harder but lowered the real curve
even more than r150. Stop this size path; r150 is the better mildtransfer signal,
though still not a new best.

Stat-matched hybrid:
`/workspace/synth_mildtransfer_statmatch1600_free180_r150`

Built with `scripts/build_statmatched_hybrid_local.py` to test source diversity
without the geometry bias of `build_geometry_hybrid_local.py`. It replaces the
same 150 examples as r150 by mode (80 elevation, 50 roof-plan, 20 freeform), but
matches candidate examples to base examples by area score, annotation count,
sheet aspect, and image size.

```text
composition 1450 base + 150 mildtransfer
same mode quotas: elevation=950,roof_plan=470,freeform=180
selected_score_mean 0.3510 vs 0.3513 for base free180
selected_ann_count_mean 7.8919 vs 7.8869
added_score_mean 0.3626, removed_score_mean 0.3658

1536 probe:
epoch0 real 0.4160 | synth 0.3296 | div -0.0864
epoch1 real 0.4177 | synth 0.4096 | div -0.0081
```

Interpretation: preserving coarse stats did not restore transfer; the
mildtransfer source itself suppresses the early real curve. Do not keep mixing
this source pool unless changing the recipe or using it only for validation
hardness experiments.

### 1600-image top-area hybrid

Dataset: `/workspace/synth_hybrid_base1300_top300_quota_hardval1600`

Built from the 1300 diverse base plus 300 quota-balanced top-area examples
(190 elevation, 90 roof-plan, 20 freeform), then ordered with hard validation:

```text
epoch0 real 0.4388 | synth 0.4559 | div +0.0172
epoch1 real 0.4576 | synth 0.5386 | div +0.0810
epoch2 real 0.4478 | synth 0.5334 | div +0.0856
```

Interpretation: top-area additions raised synth fit, not real transfer. The
ordered hard-val setup was not enough because the training distribution itself
still made synth too easy.

### Direct hard-transfer generation v3

Dataset: `/workspace/synth_hardtransfer_v3_1600`

Generator schema: elevation-heavy (`--mode-weights 70,10,20`), no outlines,
construction-sheet floorplans, roof fields, elevation trims, clean bands,
realstyle, outline floor tiles, clutter boost, moderate label jitter/resolution
scale/cropping. It produced:

```text
modes: elevation=1118, roof_plan=312, freeform=170
ann median 4, score median 0.274
epoch0 real 0.3480 | synth 0.3796 | div +0.0316
epoch1 real 0.3528 | synth 0.4873 | div +0.1346
```

Interpretation: this kind of semantic hardening moved too far away from the
transfer distribution. It made training harder but did not teach the real eval
set; synth still became easy by epoch 1.

### Direct canonical nomarkup generation

Dataset: `/workspace/synth_nomarkup_direct1600`

Generated exactly 1600 images from the canonical/proven generator family, but
removed the synth-only colored lollipop markup artifact:

```bash
python generate_synthetic_v5.py --n 1600 --seed 60626 \
  --tiles /root/.cache/huggingface/hub/datasets--abshetty--floz-assets/snapshots/6dfc52ececbe353f10324a761350b72d535861df/reference_tiles_curated \
  --out /workspace/synth_nomarkup_direct1600 --workers 32 \
  --dense-fill-scope instance --dense-fill-frac 0.45 --dense-fill-opacity 0.18 \
  --mode-weights 65,5,30 --markup-overlay-prob 0.0
```

Stats:

```text
n=1600
modes elevation=1032, roof_plan=485, freeform=83
ann_count_mean 12.8581, ann_count_median 12
score_mean 0.3327, score_median 0.2302

1536 probe:
epoch0 real 0.3611 | synth 0.4286 | div +0.0675
epoch1 real 0.3797 | synth 0.5454 | div +0.1657
```

Interpretation: direct canonical generation without markup is much worse than
the selected 20k-diverse backbone. Removing markup alone does not help under the
pure-synth 1600 goal; synth becomes easy immediately while real stays low. Future
runs should keep using selected/diverse source pools unless the generator schema
changes a structural coverage axis, not just an artifact.

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

### Floorplan-heavy 1600 selection

Dataset: `/workspace/synth_pool20k_diverse1600_floorheavy`

Selector used the same four proven 5000-image pools but changed mode structure
aggressively toward the old floorplan-heavy hypothesis:

```text
quotas elevation=640,freeform=640,roof_plan=320
score_mean 0.3006, score_median 0.2216
ann_count_mean 6.3156, ann_count_median 4

1536 probe:
epoch0 real 0.3802 | synth 0.3904 | div +0.0102
epoch1 real 0.4185 | synth 0.4894 | div +0.0708
```

Interpretation: a large floorplan/freeform shift from the proven pools did not
create useful hard synth and did not improve real transfer. Synth was already
easier than real by epoch1 and real was far below the current best. Do not keep
pushing this mode axis under pure-synth 1600 unless the freeform generator itself
changes; simple reselection is negative.

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
  --image-max-size 1024 --batch-size 1 --num-workers 24 \
  --log-root /workspace/probes \
  --extra "--domain-random --real-eval-indices=12,16,27,7,11,25,23,1,18,2,0,3,14,24 --train-split=0.98 --num-queries=200 --mask-weight=10 --dice-weight=10 --eos-coef=0.03"
```

Important:

- Do **not** add `--realism-aug` by default. It hurt the best path.
- `train_split=0.98` means 1568 train images and 32 synth-val images for a
  1600-image local dataset.
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
- `scripts/select_geometry_targeted_local.py`: object-geometry selector for
  strip/broad-field coverage; current full-reselection result is negative.
- `scripts/build_geometry_hybrid_local.py`: small replacement hybrid builder;
  current 150-example geometry hybrid is negative.
- `scripts/build_statmatched_hybrid_local.py`: source-pool hybrid builder that
  preserves area/count/aspect/size stats; current mildtransfer statmatch result
  is negative.
- `scripts/upsample_local_dataset.py`: duplicate/upweight local synth examples
  to a target image count; current 1300->1600 balanced upsample result is
  negative.
- `generate_synthetic_v5.py`: added `--recipe mildtransfer-v1`; current
  150-example hybrid is a near miss but not a new best.
- `scripts/degrade_images_local.py`: image-only degradation copy tool.
- `scripts/jitter_annotations_local.py`, `scripts/mosaic_local.py`,
  `scripts/crop_excerpts_local.py`: earlier transforms; none is currently the
  leading path.

## Next Best Moves

The 1600/0.60 target is not yet met under simple reselection or duplicate
upsampling. Best observed 1600 real IoU is `0.5593` from top80 upsampling, but
that epoch has div `+0.0604`; best one-sided-valid 1600 real IoU is still
`0.5089` from the 1536px free180 probe. The old best 1300 real IoU was `0.5424`,
so the current problem is both real-transfer ceiling and keeping synth from
becoming easier at the high-real epoch.

Reasonable next experiments:

1. Let hard-synth curves run one more epoch when real is still climbing or near
   best, but avoid spending many epochs after real stalls or synth becomes much
   easier than real.
2. Do not repeat quota scaling (`free140/free180/free220`) or direct realhard-v3
   style semantic hardening. Both are now negative results.
3. Use the 20k-diverse/faintcad/cadneg distribution as the transfer backbone,
   then add targeted *small* interventions. Large schema changes lost transfer.
4. Consider generating larger source pools with mild variants of the proven
   faintcad/cadneg recipe, then selecting 1600 by per-real failure similarity
   rather than global top-area or global diversity.
5. If the user allows training-setting changes, test higher `image_max_size`
   or a regime change separately; no local log found evidence that the current
   1024px pure-synth setup can reach 0.60 real IoU.

Do not mark the goal complete until a current run proves:

```text
real_iou > 0.60
synth_iou - real_iou < 0.05
```

No current run satisfies both.

## Latest 2000 Duplicate-Expansion Candidate

Built from the strongest 1600-image top80 dataset because unique 400-image
extensions diluted transfer.

```text
/workspace/synth_pool20k_top80_upsample2000_balanced
n=2000, base=1600, duplicate=400
modes roof_plan=673, freeform=304, elevation=1023
score_mean=0.3972 ann_count_mean=7.3080

/workspace/synth_pool20k_top80_upsample2000_top120_bal280
n=2000, base=1600, duplicate=400
top_score_count=120, remaining duplicates balanced
modes roof_plan=753, freeform=264, elevation=983
score_mean=0.4150 ann_count_mean=7.4680
```

Probe `top120_bal280` first because prior best real-IoU points needed some
top-score pressure.

Result:

```text
epoch0 real 0.4203 | synth 0.4255 | div +0.0052
epoch1 real 0.4348 | synth 0.5165 | div +0.0817
epoch2 real 0.4862 | synth 0.5606 | div +0.0745
epoch3 real 0.4861 | synth 0.6615 | div +0.1754
```

Interpretation: stopped after epoch 3. It did not reproduce the 1600 top80
epoch3 real spike, and synth became much too easy. Duplicating top80 to 2000
with additional top-score pressure is not enough.
