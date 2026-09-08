# Handoff

Last updated: 2026-09-08

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds
every number, command and reproduction path. This file holds only what changed
and where to pick up — if a fact appears in one of those two, it is not repeated
here.

## Pick up here (2026-09-08)

Two things happened this session: the v6d synthetic round was tested on the mix
and **not adopted**, and a **labelling-assist model** was built, which is a new
track. Numbers live in `startup.md` (product), `labeling_assist.md` (labelling)
and `synth_progress.md` (per-experiment history).

**1. v6d added to the documented mix: unsettled, not pursued.** Paired 2x2 on one
GH200, rebuilt from a clean clone (every count matched `startup.md`; `mixbase`
reproduced at 0.7410 val-selected against the recorded 0.7394). `mixv6d` = mix5092
+ 1,600 v6d plans, 2048, seeds 7 and 31: **+0.028 val-selected (positive at both
seeds), null (+0.001) averaged**. At 1.3x the baseline sd that is under the ~2x
threshold requiring a third seed, so it is not a result. Dropped by decision: the
gap to the 0.90 target is ~0.14 and source count, not the generator, is what moves
that. Full entry and the per-epoch traces: `synth_progress.md` (2026-09-08).

**A protocol finding worth keeping from it.** Checkpoint averaging assumes the
run has plateaued. Both `mixv6d` runs peaked at the *final* epoch where both
baselines peaked mid-run, and averaging then reads ~0.04 low -- which is the
entire disagreement between the two selection protocols. **Check whether a run
has turned over before trusting the averaged number.** Not the step-budget
effect: v6d had more optimizer steps and still had not converged.

**2. Labelling assist (`labeling_assist.md`) -- new track.** Roboflow's SAM 3
Label Assist was rejected in practice because its polygons carry ~284 vertices
against a human's 5. That is a post-processing problem: Douglas-Peucker takes
284 -> 7 for 1.7 points of IoU. Fine-tuning SAM 3's mask decoder (encoder frozen,
41 s/epoch) then took held-out polygon IoU **0.798 -> 0.853 mean, and >=0.8 from
71.4% to 84.7%**. Checkpoint `data/runs/sam3_ft/best.pth`, entry point
`scripts/sam3_region_model.py`. One split, one seed -- needs a second before it
is quoted.

**Also settled about SAM 3, so nobody re-derives it:** its vision encoder is
fixed at **1008x1008**, so it is a poor backbone candidate for the product model
(whose largest win is training at 2048). And it **cannot do the product task**:
zero-shot PCS with the reference box as exemplar scores 0.308 on the 52 HF14
selections against RefUNet's 0.769. Details and the measured negatives are in
`labeling_assist.md`.

**Open, in priority order:**

1. **More labelled sources.** The binding constraint, and the only lever sized to
   the remaining ~0.14 (28 -> 86 sources bought +0.073; a generated plan is worth
   1.65x a scraped one). The labelling-assist model exists to make this cheaper;
   a browser UI on a non-GPU box is the next build.
2. **Push resolution past 2048.** 2560 is +0.05 at one seed with averaging and
   the Gemini pool's median long side is 3168px, so 2048 may still be truncating
   the best data. Cheap, and still the largest per-hour model-side lever.
3. **Re-measure the per-pool value at 2048.** The Gemini pool is +0.033 at 1280
   but +0.010 at 2048 with overlapping 2-seed arms. Unresolved as recorded.
4. **Regularization.** The train/HF14 gap is ~0.09 and widens with epochs;
   fitting is not the constraint.

**Do not re-open:** connectivity, tile-similarity confusables, v6e look-alike
pairs, orientation snapping in the polygon regularizer, `--anchor`, and SAM 3 as
a product-task model without fine-tuning it on that task.

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
a bug (`synth_progress.md`), and selecting for disconnected families is null at 3 seeds — its
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

## Facts that live elsewhere

Single copies, so they cannot drift. Do not restate them here.

| what | where |
|---|---|
| task semantics, metric definition, what does *not* count as evidence | `PROJECT_UNDERSTANDING.md` |
| the `sample_reference_box` fix and why pre-2026-08-06 numbers are incomparable | `PROJECT_UNDERSTANDING.md`, mechanism in `startup.md` |
| current result, both selection protocols, the checkpoint and its artifacts | `startup.md` |
| every reproduction command, the data rebuild, the validation split | `startup.md` |
| evaluation rules and the fixed HF14 indices | `startup.md` |
| per-experiment history and the screened levers | `synth_progress.md` |
| the labelling-assist model, its numbers and its negatives | `labeling_assist.md` |

Two standing traps: the reference resize in `refmask2former/dataset.py` looks
like a bug and "fixing" it costs 0.153, and `--corr-grid` / `--scale-matched-ref`
are implemented but screened negative — read the warnings before touching either.

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
- `scripts/family_similarity_probe.py` — per-family appearance similarity:
  `--metrics` (per-image confusability, intra/inter margin, correlations),
  `--tiles` (tile-pool pairs; writes the table `--tile-sim` consumes),
  `--local-data` (pool distribution at training resolution).
- `scripts/generate_images_openai.py` — resumable generation of the unlabelled
  realistic pool on `gpt-image-2`, arbitrary `--size`, receipts carrying the
  billed token usage, and a contact sheet for the visual gates.
- `scripts/upload_unlabelled_roboflow.py` — validates a batch whole, uploads it
  unlabelled to a named project (`--create` to make one), and disposes of a
  review project again with `--delete-project`.
- `scripts/render_image_generation_prompts.py` — `--version v4` is the current
  prompt set, rendered from `specs_v4.jsonl`; v1-v3 are kept so their results
  stay reproducible and v1's recorded SHA still validates.
- `scripts/build_specs_v4.py` — the v4 subject list, built to the evaluation
  set's measured distribution and weighted toward the failure traits.
- `scripts/pool_style_stats.py` — aspect, colour, ink, contrast and fill
  regularity for a pool, so a generated round can be compared with the real 28
  instead of judged by eye.
- `generate_synthetic_v6.py` — the v6 synthetic generator: a house grammar
  (blocks, roofs, dormers, porches, townhouse rows) projected into elevations,
  roof plans and floor plans, with ~20 materials drawn as continuous ruled
  fields in feet. Defaults are the v6d round; the v6e knobs that measured
  negative are kept as named constants with their results in comments.
- `run_synth_v6.sh` — reproduces any synthetic-only arm end to end (pool,
  merge, two-phase train, val-selection, averaging). `./run_synth_v6.sh
  headline 7` is the 0.686 result.
- `scripts/average_checkpoints.py` — averages the last epochs of a run and
  ranks the window on validation; the selection protocol as of 2026-09-07.
- `scripts/select_epoch_on_val.py` — val-selected protocol; one process per run,
  run them in parallel. `--tta {1,2,4,8}` for dihedral test-time augmentation
  (measured +0.004, not worth 8x inference).
- `scripts/train_status.sh` — one-screen progress for every run under
  `data/runs`, including live epoch and running peak.
- `scripts/run_*.sh`, `run_*.sh` — the experiment drivers, one per ablation.

## Data limitation and where to push next

The mix holds **194 unique real-ish source plans** (86 scraped + 28 generated v1
+ 80 Gemini r2/r3); everything else is synthetic or deterministic offline
variants. Source count is still the binding constraint, and generated sources
are now the best-value supply: 1.65x a scraped plan each, and they pick their own
resolution, which the natively-640px scraped pool never can.

**Before generating more, read the status section at the top of
`image_generation/README.md`.** The v1 100-prompt scheme was framed wrongly --
whole sheets with legends, then fragments -- and yielded 28 of 98. v3/v4 fixed it
(72-88% yield) by aiming at the eval set's measured distribution. There is no
labelling backlog worth mining; more data means a better round.

**Resolved 2026-09-06**: the standing hypothesis that image 12's thin wall poche
was a 1280px downscaling artifact is supported -- training at 2048 is worth +0.05
to +0.07 overall. Whether image 12 specifically recovers has not been checked
per-image.

## Evaluation rules

In `startup.md` — single copy, so the two cannot drift.

## Generated artifacts

**A clone has none of this.** Datasets, checkpoints, logs and evaluation JSON
live under git-ignored `data/` and `logs*`, so every `data/...` path quoted in
these docs is a provenance record, not a file you have. Preserve them between VMs
for byte-identical artifacts, or rebuild: `startup.md` covers the headline recipe
and `synth_progress.md` ("Rebuilding what these findings used") covers the
ablation pools, which `startup.md` does not. The v6 synthetic pools and every
synth-only arm rebuild from `./run_synth_v6.sh <arm> <seed>`. Source, scripts and documentation
are committed; credentials are never stored in the repo.
