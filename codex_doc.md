# Handoff

Last updated: 2026-08-12

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds
every number, command and reproduction path. This file holds only what changed
and where to pick up — if a fact appears in one of those two, it is not repeated
here.

## Pick up here (2026-08-12)

The 2026-08-10 lever — generate synthetic sheets with deliberately confusable
material pairs — was implemented and tested. **It costs −0.104** (0.6823 →
0.5787, 3 seeds, t=3.80, p=0.019, complete separation). The lever is closed;
`--confusable-prob` is committed, default off, kept only so the negative
reproduces. Detail in `synth_progress.md` (2026-08-12).

Three things that reasoning got wrong are worth carrying forward:

| claim | status |
|---|---|
| "synthetic sheets contain no confusable pairs at all" | **false** — the generator picks distinct *paths*, not distinct *appearances*; synthetic already sat at 19.2% vs real 24.4% |
| "every image under 0.65 is multi-family" / "single-family all score 0.92+" | **false** on a rebuilt checkpoint — HF14 imgs 12/7/0 are single-family at 0.393/0.546/0.611 |
| the confusability correlation | **replicates, and more strongly than recorded** (pooled r=−0.754, p=0.0018) — and the intervention still hurt |

**Where the error is now.** ~30% of multi-family images are not decidable from
appearance at all: a family's own instances agree less than that family agrees
with its most confusable neighbour, and those are the worst image in each split
(HF14 14/18, validation 13/17). That matches the 2026-08-07 finding that
`hatch_matcher` and `RefUNet` fail on the same inputs. **The open question is
whether those cases are labelling inconsistencies or genuine semantic
distinctions** — same hatch, different material by drawing context. Look at the
four flagged pairs before designing anything else; if they are label errors that
is a data-quality fix worth ~+0.105 on HF14 from image 14 alone, and if they are
not, ~0.70 is close to the ceiling for a rectangle-only input.

**The lever with replicated positive evidence remains source count**: 28 → 86
real sources bought +0.073, one generated plan ≈ one real plan, and re-augmenting
the same 114 sources 18× → 72× buys nothing. `scripts/generate_images_openai.py`
now drives that round end to end; read the status section of
`image_generation/README.md` first.

Unresolved from 2026-08-10: `--anchor-dropout` has still never been tested
cleanly, having only ever run on top of the reference-plane bug.

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
  realistic pool, with receipts and a contact sheet for the visual gates.
- `scripts/render_image_generation_prompts.py` — `--version v2` is the reframed
  prompt set; v1 is kept byte-identical so its recorded SHA still validates.
- `scripts/run_*.sh`, `run_*.sh` — the experiment drivers, one per ablation.

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
1280px downscaling artifact. It must not be tuned against HF14. This gained
weight on 2026-08-12 — image 12 is **single-family** and still scores 0.393, so
its failure cannot be a confusability effect and no appearance-matching lever
will touch it.

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
