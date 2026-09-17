# Handoff

Last updated: 2026-09-17

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds
every number, command and reproduction path. This file holds only what changed
and where to pick up — if a fact appears in one of those two, it is not repeated
here.

## Pick up here (2026-09-17)

Four things happened this session, on a machine that no longer exists — every
`data/runs/*` checkpoint named below is gone; only what's written here and in
`synth_progress.md` survives. **Nothing below is adopted into the documented
recipe yet.** `synth_progress.md` was also reorganized: the pre-2026-08-06 /
retired-query-model material is now `synth_progress_archive.md`, actually moved
out this time rather than relabeled in place.

**1. A fourth Gemini round (`floz-gen-gemini-r4`, 88 images) is a clean null.**
Merged with r2+r3 into `floz-gen-gemini-r234-clean` (168 images), augmented 18x
(3,024 records), substituted for r23 in the documented mix ->
`mix6676_with_r4` (6,676 records) vs. baseline `mix5092`, 2 seeds each, 2048px:

| protocol | mix5092 | mix6676_with_r4 | Δ |
|---|---:|---:|---:|
| val-selected | 0.7443 ± 0.0263 | 0.7366 ± 0.0000 | −0.0077 |
| averaged (SWA) | 0.7559 ± 0.0147 | 0.7702 ± 0.0172 | +0.0143 |

Sign flips between protocols, both gaps under 1x sd — noise, not signal. r4 is
stylistically indistinguishable from r2/r3 by `pool_style_stats.py` (same
generator, same prompt family) and the model already fits r2/r3/r4 equally well
in-sample (0.850/0.870/0.844 mean IoU) — it isn't under-fit, it's redundant.
Consistent with the existing finding that the Gemini pool's marginal value
already shrank to +0.010 (noise) at 2048px before r4 existed. **Do not re-run
this exact test; a fifth Gemini round would need to look different, not just be
more of the same.**

**2. The documented 9-epoch schedule stops before the real ceiling — found by
warm-restarting, not by a longer single schedule.** The existing "16-epoch
schedule is null" finding (`startup.md`) used one continuously-annealing
cosine. Instead: reset the optimizer and give a **fresh** cosine restart from a
converged checkpoint, repeatedly. On `mix6676_with_r4` seed 31 (one seed):

| epoch | train mIoU (full-mix sample) | val-complement IoU (14 img / 77 sel) |
|---|---:|---:|
| 8 (documented recipe's last epoch) | — | 0.7924 |
| 18 (+1 restart, 10 epochs) | 0.8704 | 0.7993 |
| **20 (+2 restarts, peak)** | 0.8615 | **0.8127** |
| 28 (+2 restarts, final) | 0.8762 | 0.8064 |

Real gain through epoch 20 (+0.033 over the documented recipe's own epoch 8),
then a plateau: epochs 22-28 kept climbing on train (0.85->0.88) without val
following past the epoch-20 peak — the overfitting signature, just not a
blowup. A further low-LR (5e-6, well under the ~1e-5 floor every restart cosine
bottoms out at) sustained continuation from epoch 28 also plateaued (HF14 diag
0.69-0.75, no trend). **One seed. Needs replication before this changes the
documented recipe**, but it means the 9-epoch number in `startup.md` is
probably not this setup's ceiling, and checkpoint-averaging's "check whether the
run has turned over" caution (2026-09-08 entry below) applies to schedule length
too, not just pool composition.

**3. A cross-attention variant of `RefUNet` exists and ties it at both the
documented budget AND under extended training.** `refmask2former/ref_attn_unet.py`:
same ResNet50 backbone, same FPN, same direct-union training objective — only
the two *coarsest* scales' conditioning changes from global-average-pooled-
vector-plus-conv (`ConditionBlock`) to multi-head cross-attention (image tokens
as queries, reference tokens as keys/values, so a location gets a learned blend
of the reference's actual features instead of one broadcast vector). `--model
crossattn` on `train_refunet.py`. On `mix6676_with_r4` seed 31:

| | 9 epochs (documented budget) | +1 warm restart (epochs 9-18) |
|---|---:|---:|
| RefUNet | val-sel 0.7366 | val-complement 0.7993 (ep 18) |
| crossattn | val-sel 0.7314 | val-complement 0.8004 (ep 17) / 0.7992 (ep 18) |

Gaps of −0.005 and +0.001 respectively — a clean tie under two different
training regimes now, not just one. RefUNet went on to a second restart (peak
0.8127 at epoch 20); crossattn's second restart wasn't run — that's the natural
next step, from `ck_crossattn_mixr4_res2048_seed31_cont/epoch_18.pth`.

This is NOT the same territory as `--corr-grid` (screened negative, monotonic
with matching precision): corr-grid only ever produced similarity *scores* as
extra channels, never aggregated the reference's feature *values*. Full
attention did not regress the way corr-grid did — parity, not harm — so the
hypothesis that value-aggregation avoids corr-grid's overfitting-to-spatial-
correspondence failure is not falsified, just not yet confirmed as a win either.

**4. Visual quality audit of the synthetic generator — new, and it changes how
much weight the "generator quality" deprioritization should carry.** Full
write-up and 8 example images: `synth_progress.md` (2026-09-17 entry),
`synth_quality_audit/`. Every prior synth-vs-real comparison in this project
(ink, saturation, contrast, aspect, `pool_style_stats.py`) is an aggregate
statistic; nobody had looked at the images next to real ones until this
session. v6 (`generate_synthetic_v6.py`) is a real composition improvement over
v5 (the pool still in the documented mix) — but has a **confirmed bug** (2-story
level-mark label collision, ~line 1451, one-line fix) and systematically
implausible material colours (blue clay tile, blue cultured stone, pure-green
brick — real instances of these materials don't look like that). v6d's own
+0.028 val-selected (2026-09-08 entry) was measured on a generator carrying
these defects. **The "source count, not the generator" deprioritization in
priority #1 below was reached from statistics that miss this — it should be
read as weaker than it was written.** Cheap next step: fix the label-collision
bug and constrain colourisation to per-material plausible hue ranges, then
re-measure v6d before drawing a stronger conclusion either way.

**Open, in priority order (revised 2026-09-17):**

1. **Replicate the warm-restart extended-training finding** (item 2 above) at a
   second seed, and on `mix5092` too (not just the r4-inclusive mix) — this is
   the single biggest number this session produced and it is one seed.
2. **More labelled sources**, still the structural constraint — but see item 4
   above before treating "generator quality doesn't matter" as settled.
3. **Fix the two confirmed/likely v6 generator defects** (label collision,
   material-colour plausibility) and re-measure v6d on the mix. Cheap, and the
   prior null may not have been a fair test of the generator.
4. **Push resolution past 2048.** Unchanged from 2026-09-08: 2560 is +0.05 at
   one seed with averaging, Gemini's median long side is 3168px.
5. **Run crossattn's second warm restart** (item 3 above) — ties RefUNet
   through the first restart; RefUNet's peak needed a second.
6. Re-measure the Gemini pool's per-pool value at 2048 (unresolved, +0.033 at
   1280 vs. +0.010 at 2048) — today's r4 null is consistent with "already near
   its ceiling at 2048" but doesn't fully resolve it.
7. Regularization — train/HF14 gap widens with epochs, though item 2's finding
   complicates "fitting is not the constraint" somewhat: val DID follow train up
   for 10 extra epochs before plateauing, further than the 9-epoch recipe alone
   showed.

**Do not re-open:** connectivity, tile-similarity confusables, v6e look-alike
pairs, orientation snapping in the polygon regularizer, `--anchor`, SAM 3 as a
product-task model without fine-tuning it on that task, and — new this
session — adding a further same-style Gemini round without a genuinely
different generator or source (see item 1 above). MixUp (pixel-blending two
scenes) was reasoned through and not tried: it directly conflicts with the
project's repeated finding that fine local texture is the signal, so pixel
blending is expected to actively hurt, not just be neutral.

Older handoffs (2026-09-08, 2026-08-08) removed 2026-09-17: their substance was
fully duplicated in `synth_progress.md` (v6d, checkpoint-averaging-plateau
caution, connectivity/disconnection findings, ranking loss, template matching)
and `labeling_assist.md` (SAM 3 fine-tuning, its encoder-resolution and
zero-shot limits), and their priority/do-not-reopen lists were superseded by
the 2026-09-17 ones above. The two facts that weren't preserved elsewhere —
image 14's outsized weight in HF14 (9/52 selections, 0.291 mean) and a 2026-08
A10 GPU memory baseline — are now in `synth_progress.md`; the A10 note was
dropped as stale (measured at 1280px/19.3GiB, the documented recipe is now
2048px/~51GB).

## Facts that live elsewhere

Single copies, so they cannot drift. Do not restate them here.

| what | where |
|---|---|
| task semantics, metric definition, what does *not* count as evidence | `PROJECT_UNDERSTANDING.md` |
| the `sample_reference_box` fix and why pre-2026-08-06 numbers are incomparable | `PROJECT_UNDERSTANDING.md`, mechanism in `startup.md` |
| current result, both selection protocols, the checkpoint and its artifacts | `startup.md` |
| every reproduction command, the data rebuild, the validation split | `startup.md` |
| evaluation rules and the fixed HF14 indices | `startup.md` |
| per-experiment history and the screened levers | `synth_progress.md` (pre-2026-08-06 / retired query-model material: `synth_progress_archive.md`) |
| the labelling-assist model, its numbers and its negatives | `labeling_assist.md` |
| the synthetic generator's visual quality audit and example images | `synth_progress.md` (2026-09-17), `synth_quality_audit/` |

Two standing traps: the reference resize in `refmask2former/dataset.py` looks
like a bug and "fixing" it costs 0.153, and `--corr-grid` / `--scale-matched-ref`
are implemented but screened negative — read the warnings before touching either.

## Relevant implementation

- `refmask2former/ref_unet.py` — shared backbone, conditioning blocks, FPN mask.
- `refmask2former/ref_attn_unet.py` — cross-attention conditioning variant
  (2026-09-17, item 3 above); `--model crossattn` on `train_refunet.py`.
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
