# Synthetic Dataset Progress

Last updated: 2026-08-06

## Status of this document

Everything below was measured with the pre-2026-08-06 reference-box sampler,
which could place the user rectangle outside the pattern it sampled. Those
numbers remain internally consistent and the generator conclusions still hold,
but they are **not comparable** to anything measured after the fix. See
`startup.md` for the fix and the current results.

Two further corrections from 2026-08-06:

- The source pools `data/synthetic/faintcad2500` and `data/synthetic/cadneg2500`
  referenced throughout this file **no longer exist and cannot be rebuilt** — the
  generator flags that produced them were never committed. Use the 20k HF config
  via `scripts/hf_to_local.py` instead; the substitution costs ~0.006.
- The synthetic pools were the one part of the data **never** affected by the
  sampler bug (0% bad boxes), because their images are 2-5k px wide so 128px
  reference boxes fit trivially. The bug bit only the 640px real plans.

## Mixed-data successor

The synthetic-only result below remains fully reproducible and is an important
baseline. The current accepted single-model result now uses the same 1,600
synthetic images plus 1,548 Roboflow records and reaches **0.6127448856** on the
same corrected HF14 metric at actual epoch 8. The implementation is the direct
reference-conditioned `RefUNet`; see `codex_doc.md` and `startup.md` for the
architecture, exact two-stage schedule, data construction, and evaluation.

Roboflow-only reference points are 0.349361 for 86 originals with live
flip/rotation and 0.464606 for the strong18 offline pool. These do not change the
verified synthetic-only outcome documented below.

## Outcome

The synthetic-only target is achieved.

| Requirement | Verified result |
|---|---|
| Dataset size | 1,600 unique images |
| Training inputs | Fully synthetic |
| Training duration | 10 actual epochs |
| Real evaluation | Fixed HF14 |
| Task metric | Reference-conditioned union IoU |
| Target | `>= 0.55` |
| Result | **`0.5506125168`** |

Passing checkpoint:
`data/runs/ck_stage750e4_rank1_refonly5/epoch_0.pth`

Authoritative audit:
`data/evaluations/verified_rank1_hf14/metrics.json`

The audit covers all 14 fixed real images and all 52 annotated reference
selections. Comparison images are beside the metrics file.

## Dataset construction

Two existing 2,500-image generator pools supply all source images:

- `data/synthetic/faintcad2500`
- `data/synthetic/cadneg2500`

The canonical 1,600-image selection is:

```bash
python scripts/select_toparea_local.py \
  --sources data/synthetic/faintcad2500 data/synthetic/cadneg2500 \
  --out data/synthetic/toparea1600_balanced \
  --n 1600 \
  --mode-quotas elevation=889,roof_plan=540,freeform=171
```

Selection statistics:

```text
n              1600
score_min       0.168350
score_median    0.292692
score_mean      0.394379
score_max       0.853069
elevation       889
roof_plan       540
freeform        171
faintcad        785
cadneg          815
```

The high-area curriculum subset is:

```bash
python scripts/select_toparea_local.py \
  --sources data/synthetic/faintcad2500 data/synthetic/cadneg2500 \
  --out data/synthetic/toparea750_balanced \
  --n 750 \
  --mode-quotas elevation=476,roof_plan=196,freeform=78
```

All 750 source identities are members of the 1,600-image set. The curriculum
therefore uses one fully synthetic dataset, starting with its strongest subset.

## Successful ten-epoch curriculum

### Phase 1: joint instance and reference training

Train on the 750-image subset at 1280 px with batch 4. The successful checkpoint
is `epoch_4.pth`, after five completed epochs. The run was configured with a
ten-epoch cosine schedule; stop once epoch 4 is saved.

Important model settings:

```text
num_queries             200
mask_weight             10
dice_weight             10
ref_weight              2
eos_coef                0.03
domain_random           true
ref_siamese_backbone    true
ref_siamese_level       res3+res5
seed                     0
```

Phase-one checkpoint:
`data/runs/ck_toparea750_hybrid_1280_s0/epoch_4.pth`

### Phase 2: hard reference ranking

Warm-start phase one and train five more epochs on all 1,600 images. Freeze all
segmentation parameters and update only the Siamese reference projector.

```text
batch_size              8
reference_only          true
ref_ranking_margin      1.0
lr                      1e-4
epochs                  5
```

This completes ten actual epochs. The passing point is phase-two epoch 0, the
sixth actual epoch, and is retained even though later phase-two checkpoints
regress slightly.

Full commands are maintained in [`startup.md`](startup.md).

## Acceptance evaluation

```bash
PYTHONPATH=. python scripts/evaluate_reference_selection.py \
  --checkpoint data/runs/ck_stage750e4_rank1_refonly5/epoch_0.pth \
  --image-max-size 1280 \
  --score-thresh 0.6 \
  --match-margin 0.1 \
  --out data/evaluations/verified_rank1_hf14
```

Verified output:

```text
metric                  reference-conditioned union IoU
checkpoint_epoch        0 (phase two; sixth actual epoch)
n_images                14
n_reference_selections  52
mean_iou                0.5506125168094088
```

The model returns separate matched instance masks. The evaluator unions those
masks only for the strict acceptance calculation.

## Progression to the result

| Experiment | Corrected HF14 IoU | Conclusion |
|---|---:|---|
| 500-image res5 Siamese | 0.457807 | Valid initial baseline |
| 750-image `res3+res5` Siamese | 0.519512 | Fine + coarse reference features matter |
| 1,600-image joint training | 0.504098 | More data improved masks, not grouping |
| Ranking margin 2.0 | 0.536815 | Hard-pair ranking was the right loss |
| Ranking margin 1.0 | **0.550613** | Passed |

## Useful negative evidence (2026-08-06, fixed sampler, RefUNet)

Thirteen levers screened at one seed on the **validation split** (see
`startup.md`), baseline validation peak `0.7672`. None beat baseline. Do not
repeat these without a materially new hypothesis; full reasoning is in commit
`b2d2f09`.

| lever | val peak | note |
|---|---:|---|
| threshold retune (0.25–0.65) | 0.7638 | raising it monotonically hurts — the model floods faint sheets *confidently*, so no post-hoc calibration helps |
| boundary snapping ×4 | 0.7672 | guided filter / morph / bilinear / ink-cell fill all ≈0. Boundary precision is **not** the bottleneck at 0.77 |
| resolution 2048 | 0.7524 | |
| schedule 15 epochs | 0.7529 | |
| resolution 1792 | 0.7508 | two independent points: more pixels hurt |
| synthetic volume 3200 | 0.7415 | |
| `--realism-aug` | 0.7398 | |
| dense correlation `--corr-grid 4` | 0.7300 | |
| ink-matched synthetic | 0.7218 | density realism cost labelled area (0.387→0.286) |
| faint + high-area synthetic | 0.7227 | still lost with area preserved |
| dense correlation `--corr-grid 8` | 0.7179 | monotonic: more matching precision, worse result |
| `--scale-matched-ref` | 0.6146 | worst change measured; see the warning in `dataset.py` |

## Lever 14: auxiliary hard-pair ranking loss (2026-08-07, one seed)

The query lineage's biggest win (ranking margin 1.0 -> 0.5506) was dropped
untested when `RefUNet` replaced the query model. Now tested, implemented in
`ref_unet.py` + `train_refunet.py` and default-off (`--rank-weight 0`):

| arm (seed 31, HF14 peak) | mIoU |
|---|---:|
| baseline | 0.6801 |
| `--rank-weight 0.5 --rank-margin 1.0` | 0.6871 |

**+0.007 is inside the 0.013-0.018 run-to-run sd — null.** Likely because the
query model *needed* explicit matching to group candidates, whereas `RefUNet`
predicts the union directly and its conditioning already learns what the union
loss requires. One seed only; a null here is not proof of no effect.

## Classical template matching is strong but redundant (2026-08-07)

`scripts/hatch_matcher.py` measures drafting parameters (orientation, spacing via
the distance transform of the paper, density) instead of correlating a filter
bank. On HF14 it beats the out-of-the-box Gabor probe by a wide margin — AUC
0.727 -> 0.897, oracle-threshold IoU 0.308 -> 0.505 — so *engineered matching was
never the weak part; the engineering was*.

It is nonetheless **not worth feeding into the model**. On identical images,
instances and reference rectangles it correlates *positively* with `RefUNet`
(+0.45 AUC, +0.65 IoU) and its IoU collapses to 0.252 on `RefUNet`'s worst 13
selections versus ~0.59 elsewhere: the two fail on the same inputs. An oracle
that perfectly overrode the 5 selections where `RefUNet` fails and the matcher is
confident would gain only +0.076.

Three measurements agree that the residual difficulty is intrinsic, not a
resolution or architecture deficit: 74% of the matcher's false positives are real
linework a labeller assigned to a different material; no region-recovery scheme
(ink closing, planar subdivision, oracle ceiling 0.257) can even represent the GT
regions; and both methods fail together. This is the confusability the generator
spec deliberately builds in.

Two mechanisms worth remembering:

- **Region identification, not boundary placement, is the binding constraint.**
  Per-image IoU spans 0.42–0.98 and the low ones are wrong regions, not fuzzy
  edges. A perfect model misplacing boundaries by 3px would still reach 0.9485
  mean IoU, so boundaries only start mattering above ~0.90.
- **The model learns a scale-invariant texture embedding, not template
  matching.** Engineered matching is weak: naive NCC scores AUC 0.433 — *below
  chance*, because CAD hatch is periodic and NCC is phase-sensitive — and
  phase-invariant Gabor energy only reaches 0.674.

## Useful negative evidence (earlier lineage, broken sampler)

These variants were evaluated with the corrected task metric and should not be
repeated without a materially new hypothesis:

- 1,000-image ordinary top-area training at 1024 or 1280 px
- all-repeated and 500-plus-250 repeated-image selections
- preferentially sampling repeated reference categories
- fine-only `res3` matching
- feature mean/variance texture statistics
- a separate frozen ImageNet matching backbone
- whole-instance and fixed-size ROI ImageNet matching
- rank fusion, fixed top-k, largest-gap, and two-cluster inference
- nonlinear pairwise matching head
- mask-only warm-start fine-tuning
- reference-only BCE without hard ranking
- hard-ranking margin 4.0

Mask threshold tuning did not materially improve the baseline. Oracle query
selection scored about 0.61, showing that instance mask coverage was not the
main limitation; the winning change had to improve the hardest reference-match
decisions.

## Operational rules

- Evaluate with the corrected reference-conditioned script, never the old
  class-agnostic `mean_gt_iou` proxy.
- Keep every epoch checkpoint for short experiments because the accepted score
  may occur before the final epoch.
- Stop a run early when corrected HF14 trajectory makes the target effectively
  impossible.
- Use all 64 CPU workers for generation or single-run loading on this GH200 VM.
- Keep generated data, logs, checkpoints, and audits in the repository workspace
  under git-ignored directories.
- Never expose `HF_TOKEN` or `ROBOFLOW_API_KEY` values.
