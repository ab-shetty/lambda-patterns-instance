# Handoff

Last updated: 2026-09-05

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds
every number, command and reproduction path. This file holds only what changed
and where to pick up — if a fact appears in one of those two, it is not repeated
here.

## Pick up here (2026-09-05)

**Next step: a GPU box.** Rounds 2 and 3 are labelled and converted, and nothing
further can be learned from them on CPU. 80 of the 100 Gemini images were worth
labelling — 36 of 50 in r2, 44 of 50 in r3, against 28 of 98 in the v1 round —
and they carry 478 pattern instances at a 3168px median long side. Unique
real-ish sources go **114 -> 194**, the first increase since 28 -> 86 bought
+0.073.

```bash
# Roboflow versions are pinned: floz-gen-gemini-r2 v1, floz-gen-gemini-r3 v1
rf.project("floz-gen-gemini-r2").version(1).download("coco-segmentation",
    location="data/roboflow/floz-gen-gemini-r2-raw", overwrite=True)   # 36 imgs
rf.project("floz-gen-gemini-r3").version(1).download("coco-segmentation",
    location="data/roboflow/floz-gen-gemini-r3-raw", overwrite=True)   # 44 imgs
python3 scripts/roboflow_to_local.py --coco .../train/_annotations.coco.json \
  --img-dir .../train --out data/roboflow/floz-gen-gemini-rN-clean
# r2: 36 images, 231 instances, 100 removes
# r3: 44 images, 247 instances, 205 removes, 1 hole dropped (floz_gen_048)
python3 scripts/merge_local_datasets.py --sources <the two clean dirs> \
  --out data/roboflow/floz-gen-gemini-r23-clean                        # 80
python3 scripts/augment_local_dataset.py --src .../r23-clean \
  --out .../floz-gen-gemini-r23-strong18 --aug-per-scene 17 --strong --seed 5858
# then merge as the fourth source: 1600 + 1548 + 504 + 1440 = 5,092
```

Then the `startup.md` two-phase train at **3 seeds** — the baseline is 0.6860 +/-
0.0175, run-to-run sd on mix3652 is 0.0262, so a single seed decides nothing.
Run one baseline seed first to confirm the new box reproduces mix3652.

**Two questions the labelled pool now makes answerable**, both cheap once the GPU
is up: (1) roof plans label one material across several fill orientations while
elevations treat a changed orientation as a new material — train with and without
the roof-plan images and see whether the mixed convention costs anything, noting
they are only 3 of the 80; (2) `remove` holes are 37% of all polygons drawn —
strip the small ones, retrain, and see whether HF14 moves at all.

**Yield by category, r2 + r3.** Labelled: elev_colour 17, elev_faint 16,
plan_mep 16, section_sparse 11, elev_mono 9, plan_finish 5, roof_plan 3,
elev_colour_markup 3. Skipped: section_sparse 4, roof_plan 4, elev_colour 3,
elev_faint 4, elev_mono 3, plan_finish 1, elev_colour_markup 1. Roof plans are
the weakest category by yield and the only one where the labelling convention
was in doubt.

**The gpt-image-2 round (`floz-gen-v4-round1`, 50 images) is still unlabelled**,
as are `floz-gen-gemini-r1` (14) and the two smokes. The 28 evaluation images
with their polygons are in `perceive-ai/floz-eval28-reference` — tagged
`eval-only`/`do-not-train`, never to be merged into a training pool.

**Simpler drawings score far better on fill regularity.** Four single-view,
one-or-two-family specs through `gemini-3-pro-image` read **7,727** against the
real pool's 11,826, the 50-image gpt round's 3,122 and Gemini's own first ten at
2,605. The spec list now matches the eval set on all three simplicity axes
(families 1.9, 45% single-view elevations, sparse clutter on 60%), and the prompt
carries a restraint clause because the model embellishes what it is asked for.
Four images of one category -- promising, not settled. Details in the README.

**Round 1 (50 images) is in `perceive-ai/floz-gen-v4-round1`, unlabelled**, and
the 28 evaluation images with their polygons are in
`perceive-ai/floz-eval28-reference` — tagged `eval-only`/`do-not-train`, never to
be merged into a training pool. Two changes since: family counts now come from
the eval distribution (mean 1.9, 36 single-family, against the 2.7 round 1 used),
and `gemini-3-pro-image` beat `gpt-image-2` on fill regularity in 3 of 4 paired
prompts — including a roof plan at 10,246 against the real pool's 11,826, the
first generated image here to come close. Four pairs is not a decision; the
README has the numbers and the caveats.

**Generate the round through the Batch API** (`--batch submit|status|fetch`):
half price, 24-hour window, and the batch outlives the session, so ~$30 for 300
images rather than ~$60. Fetch maps results by `custom_id` because batch output
order is not input order.

**Tuned after review**: contrast is three levels (47 normal / 24 light / 29
faint) rather than a flag, with a floor in the faint wording, and `plan_mep`
demands a solid filled wall poche plus a second family — as one family of empty
double-outline walls it produced plans with nothing in them to label. Re-smoked:
ink 0.089 against the real pool's 0.111, contrast 0.435 against 0.437. The one
skew left deliberately is 28 of 100 colourised against 16 of 28.

**Labelling cost: label one family per image, not all of them.**
`refmask2former/dataset.py:302-324` picks one target family per image and treats
everything unlabelled as background, so a single fully-labelled family is
a correct record. That leaves 64-71% of the instance polygons. Every occurrence
of the family that IS labelled must be caught, or it teaches false negatives.
Untested and worth an afternoon on a GPU before labelling 300: `remove` holes are
2.2 per image in the generated pool, 37% of all polygons drawn — drop the small
ones from the existing 114, retrain, and see whether HF14 moves at all.

Confusability still has to be measured on the labelled pool, because
`family_similarity_probe.py --local-data` needs annotations. Run it on the first
labelled slice against the real pool's 24.4% before labelling everything.

Two disposable Roboflow review projects hold the smokes:
`perceive-ai/floz-gen-v3-smoke` (4 images) and `perceive-ai/floz-gen-v4-smoke`
(6). Delete with `scripts/upload_unlabelled_roboflow.py --project NAME
--delete-project --execute`.

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
`hatch_matcher` and `RefUNet` fail on the same inputs. Left open, and worth an
hour whenever someone is in the labelling tool anyway: **are those cases
labelling inconsistencies or genuine semantic distinctions** — same hatch,
different material by drawing context? If they are label errors that is a
data-quality fix worth ~+0.105 on HF14 from image 14 alone; if they are not,
~0.70 is close to the ceiling for a rectangle-only input.

**Why generation is the next step.** 28 → 86 real sources bought +0.073, one
generated plan ≈ one real plan, and re-augmenting the same 114 sources 18× → 72×
buys nothing — so distinct sources, not records, are the constraint, and
generation is the only supply that also picks its own resolution (worth ~0.032,
and unretrofittable onto the natively-640px scraped pool). Read the status
sections of `image_generation/README.md` in order: v1 asked for whole sheets, v2
fixed the framing, v3 fixed the drafting-hatch and confuser defects the v2 smoke
test exposed. Generate with v3.

Unresolved from 2026-08-10: `--anchor-dropout` has still never been tested
cleanly, having only ever run on top of the reference-plane bug.

**Architecture is not the lever either (2026-08-12).** Four mechanisms from the
matching / few-shot-segmentation literature — self-support prototypes, a
hypernetwork-generated classifier, a central-surround two-stream reference, and
SimAM shrinkage attention — were implemented and screened on validation. All
null or negative; details and the two false positives they produced are in
`synth_progress.md`. That makes six independent attempts at the conditioning
mechanism, so prefer data supply over model surgery until something changes.

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
- `scripts/run_*.sh`, `run_*.sh` — the experiment drivers, one per ablation.

## Data limitation and where to push next

The mix now contains **194 unique real-ish source plans** (86 scraped real + 28
generated v1 + 80 generated Gemini r2/r3); everything else is synthetic or deterministic offline variants. That
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
