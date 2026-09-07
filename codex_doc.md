# Handoff

Last updated: 2026-09-06

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds
every number, command and reproduction path. This file holds only what changed
and where to pick up — if a fact appears in one of those two, it is not repeated
here.

## Pick up here (2026-09-06)

The Gemini r2/r3 round is trained, evaluated and recorded. The data pipeline was
rebuilt from a clean clone on a fresh GH200 and every count matched `startup.md`
exactly. Numbers all live in `startup.md`; this is only what changed and what is
open.

**Since then (2026-09-07, later):** checkpoint averaging is the selection
protocol (`scripts/average_checkpoints.py`; +0.02 over single-epoch picks at 2
seeds), the drawn synthetic regime v6 exists and lifts synth-only to 0.686, the
mix with v6b added reads 0.7754 at one seed, 2560 is +0.05 at one seed with
averaging, and dihedral TTA is +0.004. Details: `startup.md`, `synth_progress.md`.

**Two wins, measured separately.** Training at **2048 instead of 1280 is worth
+0.05 to +0.07** on both mixes -- the largest effect in the project, and general
rather than pool-specific. The **80 Gemini plans are worth +0.035** on the mix at
1280. Best checkpoint is now 0.7747, against 0.6860 before.

**The `>= 0.65` target is met** under both selection protocols, at a resolution
the original target never contemplated.

**Open, in priority order:**

1. **Push resolution further.** The Gemini pool's median long side is 3168px, so
   2048 may still be truncating it. 2560/3072 has never been run. Note a longer
   schedule alone is null at 2048 (16 ep vs 9 ep, 2 seeds), so this is about
   pixels, not training time.
2. **Re-measure the per-pool value at 2048.** The Gemini pool is +0.033 at 1280
   but +0.010 at 2048 with overlapping 2-seed arms. Either the effects are
   sub-additive or it is noise; as recorded, it is unresolved.
3. **The synthetic regime is now v6** (`generate_synthetic_v6.py`, drawn
   like a drawing: house grammar, ruled fills in feet, real sheet furniture).
   Synth-only went from 0.573 to **0.686 ± 0.009** (2 seeds, v5+v6d union at
   2048); added to the documented mix at 2048 it reads 0.7754 val-selected at
   one seed (0.7060 for the same seed without it). The full audit trail --
   what each round changed, what was null (v6 volume, 16 epochs, v6e
   look-alike pairs) and the seed spread (~0.05 synth-only) -- is the
   2026-09-07 entry in `synth_progress.md`. Open there: the remaining
   synth-only failure is look-alike families (dark base vs dark roof, two
   light sidings); making the training set harder in that direction was
   negative, so the next idea has to be different. **Do not** re-open
   connectivity or tile-similarity confusables.
4. **Regularization.** Newly justified -- see the generalization finding in
   `startup.md`. The train/HF14 gap is ~0.09 and widens with epochs.

**Three claims corrected this session** (all were wrong in this file):

| claim | correction |
|---|---|
| "label one family per image, not all of them" reads as description | it is a *cost proposal*; r2/r3 are multi-family labelled, 1.75 and 1.52 families per image, 30 of 80 with 2-4 |
| implied augmentation is flip-only | full ×8 dihedral has always been applied online per sample (`dataset.py:387`) to image, masks and reference together; the offline 18x builder does hflip only because the online path covers the rest |
| "a generated plan ≈ a real plan" | retired: 1.65x per source, measured source- and record-matched |

**Tooling added.** `--compile --pad-grid` (1.5x/epoch), `--early-stop-patience`,
`scripts/train_status.sh` for progress at a glance, and val-selection run as one
process per run instead of sequentially (40 min -> 5). Details and the parity
check are in `startup.md`.

**Still unlabelled**: `floz-gen-v4-round1` (50 gpt-image-2), `floz-gen-gemini-r1`
(14), and the two smoke projects. The 28 evaluation images with polygons are in
`perceive-ai/floz-eval28-reference`, tagged `eval-only`/`do-not-train`.

**Generation guidance**, unchanged and now better supported: generate more, and
generate large. `image_generation/README.md` has the v3/v4 prompt history; the
Batch API (`--batch submit|status|fetch`) halves the cost.

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
