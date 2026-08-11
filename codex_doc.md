# Handoff

Last updated: 2026-08-10

Read `PROJECT_UNDERSTANDING.md` for task semantics and `startup.md` for complete,
copy-paste reproduction commands, the data-rebuild path, and the full comparison
table.

## Pick up here (2026-08-10)

Nothing this session raised real IoU. The headline recipe is unchanged at
**0.6860 ± 0.0175**. Full detail in `synth_progress.md` (2026-08-10 section).

| tried | verdict |
|---|---|
| `--anchor` reference-plane bug, fixed by `--anchor-ref-plane 0.0` | **real bug, fixed**: +0.069 HF14 on mix3652 |
| a *working* anchor vs no anchor | null / slightly negative, 3 seeds, 3 settings |
| disconnected synthetic data | null, 3 seeds, mode- and area-matched |
| data volume, 2,452 → 9,808 records | null |
| mix ratio, 24% → 84% real | null |
| `realaug2x` (2× real-derived records) | **retracted**: +0.015 at one seed, 0.6767 ± 0.0262 at three |

**The anchor was never a shortcut.** It collapsed because the anchor channel was
1.0 across the whole reference crop while the image branch saw a sparse
rectangle, through the same siamese stem. `scripts/anchor_vs_reference_probe.py`
shows the model never obeys the anchor in any pool, including the connected one
where the shortcut is on offer. The documented 0.433 did not reproduce (0.6088);
do not quote it. `--anchor-ref-plane` now defaults to the fixed value, but the
recipe still omits `--anchor` — fixed, it is not an improvement to adopt.

**Screening budget: run-to-run sd on `mix3652` is 0.0262** (seeds 7/31/99). Two
wrong conclusions this session came from reading single-seed gaps of 0.02–0.04
as signal. Run three seeds.

**Where the error is.** The fixed anchor collapses variance (HF14 sd 0.0051 vs
0.0229) without moving the mean, and image 14's failures are wrong-region rather
than fuzzy-boundary: the error is in appearance matching, not localisation or
boundaries. Consistent with more distinct labelled real sources being the
binding constraint — re-augmenting the same 114 sources 18× → 36× → 72× buys
nothing.

**Pick up here: confusable materials.** Every image scoring under 0.65 in either
split is multi-family, single-family images all score 0.92+, and the worst
multi-family image in each split is the one holding a near-duplicate pair
(HF14 img 14, similarity 0.908 → IoU 0.246; validation img 13, 0.886 → 0.385;
corr −0.709 / −0.250, so necessary but not sufficient — img 19 pairs at 0.844 and
still scores 0.915). Synthetic sheets have never contained such a pair, because
`generate_synthetic_v5.py` picks distinct tiles per family by construction, so
generating confusable pairs is the untried lever; measure with the pairwise-family
probe in `synth_progress.md` (2026-08-10) and note the background-similarity
variant of this did **not** replicate.

Unresolved: `--anchor-dropout` was only ever run on top of the bug, so it has
never been tested cleanly.

## Superseded — 2026-08-08 (measurements sound, conclusions retracted)

Share of the target union lying in the connected component holding the user's
rectangle. Reproduces exactly via `scripts/connectivity_stats.py --hf14`, which
needs full resolution (at 1024 nearby components merge and it reads 0.558 / 33%).

| pool | share local | single-component |
|---|---:|---:|
| synthetic 1600 | 0.893 | 84% |
| real 86 | 0.523 | 29% |
| generated 28 | 0.551 | 36% |
| HF14 (eval) | 0.537 | 27% |

The gap is real. Both conclusions drawn from it are not: the anchor evidence was
a bug (above), and selecting for disconnected families is null at 3 seeds — its
mode quotas are also unfillable, only 3 of 2,833 all-disconnected images being
elevations against a quota of 889. Do not re-derive that experiment from here.

**Baseline to compare against.** `data/runs/ck_a10_mix3652_seed31/epoch_4.pth`,
**0.6801** on HF14, one seed. The documented recipe runs unchanged on a 23GB A10
(batch 8 at 1280 peaks at 19.3GiB; use `--num-workers 12`), ~55 min for the full
two-stage run. Single-seed screening only — treat anything under ~0.03 as noise.

**Also worth knowing.** Image 14 is 9 of 52 selections (17% of the metric) and
averages 0.291; fixing it alone would give +0.105. Its failures are wrong-region,
not fuzzy-boundary, and are *not* explained by resolution or aspect ratio.
Two levers were closed this session — see `synth_progress.md`: the auxiliary
ranking loss (null, +0.007) and classical template matching (`scripts/hatch_matcher.py`
beats the Gabor probe by a wide margin but is redundant with `RefUNet`).

## Read this first

`sample_reference_box` — the function that turns a ground-truth instance into the
user's reference rectangle — was returning rectangles partly or entirely outside
the pattern they sampled. It was fixed on 2026-08-06.

**Every number measured before that date used the broken sampler, including the
`>= 0.65` target.** Pre-fix and post-fix numbers are not comparable.
`sample_reference_box_legacy` is retained so the old figures remain reproducible.
Always say which sampler a number came from.

## Current result

Fixed sampler, `RefUNet`, 1,600 synthetic + 1,548 real + 504 generated-realistic:

| Item | Value |
|---|---|
| Recipe mean (seeds 7/31/99) | **0.6860 ± 0.0175** |
| Best single checkpoint | **0.705407920670342** |
| Checkpoint | `data/runs/ck_fix_mix3652_seed31/epoch_6.pth` |
| Metrics | `data/evaluations/refunet_fix_mix3652_s31_e6.json` |
| Visual audit | `data/visualizations/fix_mix3652_s31_e6/` |
| Evaluation | fixed HF14, 52 reference selections, threshold 0.35 |

Quote the 3-seed mean. Run-to-run noise is ~0.013–0.018 sd here, so the best
checkpoint is the top of a spread rather than the expected value.

That 0.6860 picks the epoch by HF14 score, which uses the acceptance set to
choose the checkpoint. Selecting the epoch on the validation split instead gives
**0.6783 ± 0.0081**. Both follow a defensible protocol — the project's metric is
defined as "any checkpoint within ten epochs" — but quote the val-selected number
when it has to hold up. See `startup.md` for the split and its correlation.

Thirteen levers were screened against this baseline on 2026-08-06 (resolution,
synthetic selection, volume, schedule, augmentation, threshold, boundary
snapping, dense reference correlation, scale-matched reference) and **none beat
it**; the table is in `synth_progress.md` and the reasoning in commit `b2d2f09`.
Two of them, `--corr-grid` and `--scale-matched-ref`, are implemented and
default-off. Read the warning above the reference resize in
`refmask2former/dataset.py` before "fixing" the reference scale — it looks like a
bug and correcting it costs 0.153.

The `>= 0.65` target was defined under the broken sampler and has not been
restated. On the fixed metric this recipe averages 0.686; whether that counts as
meeting the goal is a product decision, not a measurement one.

## What changed this session

Starting point was 0.6127 (broken sampler, single run). Two independent gains,
each replicated at three seeds with complete separation between arms:

| Change | Effect | Evidence |
|---|---|---|
| +28 hand-labelled generated plans | +0.0330 | t=2.94, p=0.043 |
| Reference-box sampler fix | +0.0529 | t=4.18, p=0.014 |

The like-for-like progression on the broken sampler is 0.6001 (`mix3148`) →
0.6331 (`mix3652`); the fixed sampler then takes `mix3652` to 0.6860.

The synthetic pools `faintcad2500` / `cadneg2500` proved unrebuildable — their
generator flags were never committed — so the synthetic half now comes from the
20k HF config via the new `scripts/hf_to_local.py`. That substitution cost ~0.006.

## Product and model semantics

A reference rectangle identifies an image-local pattern. The target is the union
of every region with that same image-local grouping ID. IDs such as `pattern1`
have no meaning across plans. Roboflow `remove` polygons are subtracted as holes:
`roboflow_to_local.py` attaches each to every containing pattern and
`render_instance_mask` zeroes rings after the first.

`RefUNet` predicts the selected union directly: a shared ResNet-50 extracts plan
and reference features, four multiscale conditioning blocks combine image
features with the pooled reference, and an FPN decoder emits one mask. Split it
into connected components (preserving holes) if the product needs separate
instances. The earlier query model, which grouped instance candidates by
reference similarity, topped out at 0.5886 on mixed data.

## Relevant implementation

- `refmask2former/ref_unet.py` — shared backbone, conditioning blocks, FPN mask.
- `refmask2former/dataset.py` — `sample_reference_box` (fixed) and
  `sample_reference_box_legacy`; `render_instance_mask` hole convention.
- `scripts/train_refunet.py` — union targets, BCE + Dice, continuation
  checkpoints, per-epoch HF14 diagnostics, explicit schedule length and
  optimizer reset.
- `scripts/evaluate_refunet_selection.py` — authoritative 52-selection eval.
- `scripts/visualize_refunet_selection.py` — the required visual audit.
- `scripts/hf_to_local.py` — HF parquet → local-data format.
- `scripts/roboflow_to_local.py`, `augment_local_dataset.py`,
  `merge_local_datasets.py` — deterministic real-data pipeline.
- `scripts/run_*.sh` — the four experiment drivers from this session.

## Data limitation and where to push next

The mix now contains **114 unique real-ish source plans** (86 scraped real + 28
generated); everything else is synthetic or deterministic offline variants. That
source count remains the binding constraint.

Established by count-matched experiment: one generated plan is worth about as
much as one real plan (0.2608 vs 0.2777 at 28 sources each, a gap inside the
~0.05 run noise), resolution is worth ~0.032, and going 28 → 86 sources buys
+0.073. So **generate more, and generate large** — the 86 scraped plans are
natively 640×640 and can never be improved, while generation resolution is a
free choice.

**Before generating more, read the status section at the top of
`image_generation/README.md`.** The 100-prompt scheme was run once and its
framing was wrong: it produced full sheets with legends and title blocks, then
over-corrected into fragments of buildings. The product gesture is a small
rectangle inside a drawing region — the housed part of a plan, not the legend.
Only 28 of 98 generated images were good enough to label (~29% yield); the
remaining 70 were reviewed and rejected, so there is no labelling backlog to
mine. More data means generating a better round, not labelling what exists.

Untested hypothesis worth pursuing on a validation split: the largest remaining
clean failure is thin wall poche in dense floor plans (image 12), plausibly a
1280px downscaling artifact. It must not be tuned against HF14.

## Evaluation rules

In `startup.md` — single copy, so the two cannot drift.

## Generated artifacts

**A clone has none of this.** Datasets, checkpoints, logs and evaluation JSON
live under git-ignored `data/` and `logs*`, so every `data/...` path quoted in
these docs is a provenance record, not a file you have. Preserve them between VMs
for byte-identical artifacts, or rebuild: `startup.md` covers the headline recipe
and `synth_progress.md` ("Rebuilding what these findings used") covers the
ablation pools, which `startup.md` does not. Source, scripts and documentation
are committed; credentials are never stored in the repo.
