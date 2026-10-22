# Startup archive

Text moved out of `startup.md` and `codex_doc.md` on 2026-10-22, **verbatim**, when the two were de-duplicated. Nothing here is
current: past bests, superseded recipes and finished handoffs. Numbers keep the caveats they were written with (sampler, inference
size, automatic vs hand boxes). Current numbers and commands: `startup.md`; every experiment: `synth_progress.md`.

## From startup.md: the hand-box model table as of 2026-10-22

| model | HF14 hand | val hand | HF14 auto |
|---|---:|---:|---:|
| **`gem2s-x` (best, 2026-10-22: 0.25 gem_s7 + 0.25 gem_s8 + 0.5 xblend; first model with real Gemini labels outside Gemini30; fix v2 0.9320, Gemini30 0.8891)** | **0.9235** | 0.8990 | -- |
| `gke-x` (2026-10-22: 1/6 geme2_s7 + 1/6 gem_s8 + 1/6 gemkms_s7 + 1/2 xblend; fix v2 0.9293, Gemini30 **0.8931**) | 0.9219 | 0.8999 | -- |
| `gk-x` (2026-10-22: 1/6 gem_s7 + 1/6 gem_s8 + 1/6 gemkms_s7 + 1/2 xblend; fix v2 0.9301, Gemini30 0.8913) | 0.9212 | 0.8995 | -- |
| `xblend-kq30-kms30` (previous best, synthetic only, 2026-10-17; fix v2 0.9290, Gemini30 0.8740) | 0.9163 | 0.8988 | -- |
| `single18kq-blend30` (previous best, 2026-10-15; fix v2 0.9221, Gemini30 0.8680) | 0.9156 | 0.8976 | -- |
| `single18kq-blend50` (2026-10-15) | 0.9088 | 0.8973 | -- |
| `single18kf-blend50` (HF14-best reference, 2026-10-15) | **0.9157** | 0.8971 | -- |
| `single18kp-blend50` (GEN 2026-10-15 plan pairs; published, Gemini30 0.8605, 2026-10-15) | 0.8983 | 0.8964 | -- |
| `single18k-blend50` (synthetic only, 2026-10-14) | 0.8877 | 0.8964 | -- |
| `synpre-realmix-lr5e5-e22` | 0.8738 | 0.9087* | 0.8816 |
| `longswiss-swa37-41` (synthetic only, 2026-10-06) | 0.8779 | 0.8766 | -- |
| `mixr4-roiadd-swa15` (shipped) | 0.8466 | 0.8193 | 0.8440 |
| `revit10k-roi-e5` (synthetic only) | 0.8118 | 0.8981 | 0.8126 |

*epoch picked on val (automatic boxes), so optimistic. 4096, threshold 0.35, CPU-scored
2026-10-03; per-question view in the Model Results artifact.

## From startup.md: write-ups of past bests (2026-10-15 .. 10-17)

`xblend-kq30-kms30` (`abshetty/floz-refunet-swint-xblend-kq30-kms30`, round-trip identical 243 / 243; best since 2026-10-17,
passed the gate): float 0.5 / 0.5 weight average of `single18kq-blend30` and `single18kms-blend30` (the multi-resolution arm,
`--image-max-size-max 3072`, batch 4; local only, failed the gate on its own). No training; all arms are one-epoch restarts of
longswiss-swa37-41. HF14 0.9163, fix v2 0.9290, val 0.8988 (per-sheet 0.9364), Gemini30 0.8740. Paired vs single18kq-blend30:
Gemini30 +0.006 (61 / 16 questions, p=0.011, sign p=2e-7), HF14 +0.0007 (p=0.89), fix v2 +0.007 (p=0.17), val +0.0012 (p=0.08),
val per-sheet +0.0007; no sheet down > 0.05 (worst Gemini30 17 0.891 -> 0.853, 12 -0.013, HF14 18 -0.012). Gemini30 1 0.557 ->
0.654, 0 0.771 -> 0.812. Maps (HF14 / val / Gemini30, 51 + 72 + 195) on `abshetty/floz-diag` `xblend-kq30-kms30/`;
`synth_progress.md` (2026-10-17, "cross-run weight blends"). Rebuild: `scripts/hf_ckpt_to_pth.py` on the Hub repo.

`single18kq-blend30` (`abshetty/floz-refunet-swint-single18kq-blend30`, round-trip verified; made the best by the user
2026-10-15): 0.3 x `single18kq-dr` + 0.7 x `single18kq-nodr` (float weight blend of the two arms, both on the Hub; no training).
HF14 0.9156, fix v2 0.9221, val per-sheet 0.9357, Gemini30 0.8680. Against single18kq-blend50: HF14 +0.007 (p=0.014), fix v2
+0.009 (p=0.028), val per-sheet +0.0006, Gemini30 +0.006 (p=0.003); level with single18kf-blend50 on HF14 (paired -0.0001)
while keeping Gemini30 4 at 0.94. **Fails the gate's sheet rule only on Gemini30 1** (0.632 -> 0.557), the most volatile sheet
(0.30 / 0.61 / 0.46 / 0.65 / 0.63 across the last five models); made the best on the user's call. Maps (HF14 / val / Gemini30)
on `abshetty/floz-diag` `single18kq-blend30/`; `synth_progress.md` (2026-10-15, "kq blend ratio").

`single18kq-blend50` (`abshetty/floz-refunet-swint-single18kq-blend50`, round-trip verified; previous best, made the best by the user
2026-10-15): the same recipe at GEN 2026-10-15b -- plan direction pairs (`--near-pairs-plan 0.3`, kinds `direction`),
`--turn-boards 0.9`, pair sheets always turn (`TAG=single18kq ./run_single_pass.sh`, 17,906 sheets); HF14 fix v2 0.9126, val
per-sheet 0.9351, Gemini30 0.8616. Against single18kf-blend50: Gemini30 +0.009 (p=0.13), val per-sheet +0.0009, HF14 -0.007
(p=0.35, not significant), fix v2 -0.010 (p=0.18). Sheets down > 0.05: HF14 2 0.910 -> 0.858 and Gemini30 8 0.838 -> 0.763,
both elevations that the plan-only change cannot touch (both also fell in single18kp). Gemini30 4 0.666 -> 0.941 (the
diagonal / orthogonal merge fixed), HF14 12 0.893 -> 0.880. Maps (HF14 / val / Gemini30) on `abshetty/floz-diag`
`single18kq-blend50/`; paired tests in `synth_progress.md` (2026-10-15, "GEN 2026-10-15b").

`single18kf-blend50` (`abshetty/floz-refunet-swint-single18kf-blend50`, round-trip verified; HF14-best reference): the 50/50 weight
average of two one-epoch restarts of longswiss-swa37-41 on the same 17,907 fresh sheets at GEN 2026-10-14f, with and
without `--domain-random --dr-scale-min 0.7` (`TAG=single18kf ./run_single_pass.sh`); HF14 with fix v2 0.9224, val
per-sheet 0.9342, Gemini30 0.8526. Against `single18knp0-blend50` (same recipe at GEN 2026-10-14b without the wing
pairs): HF14 +0.026 (p=0.006), val and Gemini30 ties. The HF14 gain sits on sheets 12 / 14 / 27,
which the generator changes were designed from, so HF14 is not a clean holdout for it. Maps (HF14 / val / Gemini30) on
`abshetty/floz-diag` `single18kf-blend50/`.
Previous: `single18k-blend50` (`abshetty/floz-refunet-swint-single18k-blend50`): the same recipe at GEN 2026-10-14b
with wing pairs; Gemini30 0.8497 (`synth_progress.md` 2026-10-14, "50/50 blend").

## Current best (2026-10-03): HF14 0.8816 (automatic boxes; 0.8738 with hand boxes)

Synthetic pretraining, then a gentle real-mix fine-tune. swin_t, trained at 2048, inference at
4096, ROI add, one seed: `abshetty/floz-refunet-swint-synpre-realmix-lr5e5-e22` (private Hub;
round-trip verified).

1. Synthetic-only start: `abshetty/floz-refunet-swint-longhardfail-swa16-20` (HF14 0.8668),
   the weight average of restart epochs 16-20 of `run_hard_long.sh` on the mined Revit pool
   `revitfail_hard2000` (see `synth_progress.md` 2026-10-02/03).
2. `run_restart_swa.sh` with `EXTRA="--roi-ref --roi-ref-mode add --lr 5e-5"` on
   `data/mixed/realmix_revithard` (`run_build_realmix.sh`: the mined Revit pool + real 86 x18 +
   generated 28 x18 + Gemini r2-r4 168 x18 = 7,076), restart epoch 22 chosen on validation
   (0.9009), HF14 read once: **0.8816**, about +0.038 over 0.8440.

Fine-tune LR, same start and mix, validation-chosen: 2e-4 SWA 24-26 0.8733, **5e-5 0.8816**,
2e-5 0.8739. A 50/50 weight soup of the 2e-4 and 5e-5 models has the best validation of all
(0.9051) and HF14 0.8753; WiSE interpolations toward the synthetic weights did not help. Picking
across every candidate on validation therefore gives 0.8753 -- the 0.006 gap is noise.

```bash
PYTHONPATH=. python3 scripts/hf_ckpt_to_pth.py --repo abshetty/floz-refunet-swint-synpre-realmix-lr5e5-e22 \
  --out data/runs/ck_best/lr5e5-e22.pth
PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint data/runs/ck_best/lr5e5-e22.pth \
  --indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 --image-max-size 4096 --ref-size 224 --mask-thresh 0.35
# mean_iou 0.8738 (default hand boxes); add --boxes auto for 0.8816
```

Previous best, 0.8440 (2026-09-28): `abshetty/floz-refunet-swint-mixr4-roiadd-swa15`, re-scored
exactly on 2026-10-03.

## Previous best (2026-09-28): HF14 0.8440

swin_t, 282-source mix (`v6dmix_plus_r4`), trained at 2048, **inference at
4096**, with the image-box reference (`--roi-ref --roi-ref-mode add`: the
reference prototype is also pooled from the plan's own features inside the
user's box, so it needs the box at inference -- the evaluator passes it):
`abshetty/floz-refunet-swint-mixr4-roiadd-swa15` (private Hub; round-trip
verified). It is one fresh-cosine restart of `...-swint-mixr4-e8`
(`run_restart_swa.sh` with `EXTRA="--roi-ref --roi-ref-mode add"`), SWA of
restart epochs 12-15 chosen on validation (0.8150), HF14 read once. +0.0269
paired vs 0.8170 (p=0.017). **One seed.**

```bash
PYTHONPATH=. python3 scripts/hf_ckpt_to_pth.py --repo abshetty/floz-refunet-swint-mixr4-roiadd-swa15 \
  --out data/runs/ck_best/swa15.pth
PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint data/runs/ck_best/swa15.pth \
  --indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 --image-max-size 4096 --ref-size 224 --mask-thresh 0.35
# mean_iou 0.8440
```

Previous best, 0.8170 (2026-09-24): `abshetty/floz-refunet-swint-mixr4-restart-e13`,
the same restart without `--roi-ref`; reproduced exactly on 2026-09-28.

Inference size is a model-input choice made on validation, not a sweep on
HF14: `--domain-random` trains on 0.4-1.0x-shrunk sheets, so a 2048 model
sees ~1430 px on average and thin targets gain from a larger input (validation
2048 .785 / 3072 .786 / 4096 .801 / 5120 .799 on the 0.7887 model). **Older
rows below are at 2048 inference; compare like with like.**

## Current reproducible result (RefUNet, the documented recipe)

`RefUNet`, fixed sampler, `mix5092` (1,600 synthetic + 1,548 Roboflow real + 504
generated-realistic + 1,440 Gemini r2/r3 = 5,092 records), **trained at 2048**:

| Item | Value |
|---|---|
| Recipe mean (2 seeds, 9 epochs) | **0.7594 ± 0.0191** |
| Best single checkpoint | **0.7747** (`mix3652` @2048 seed 31) |
| Longer schedule (16 epochs) | 0.7568 ± 0.0090 (2 seeds) -- no gain |
| Evaluation | fixed HF14, 14 plans, all 52 reference selections |
| Mask threshold | 0.35 |

The prior recorded result was 0.6860 ± 0.0175 at 1280 on `mix3652`. Two changes
account for the gain, measured separately (2026-09-06):

1. **Training resolution 1280 -> 2048 is worth +0.05 to +0.07**, on both mixes,
   so it is a general win and not specific to any pool. This is the largest
   single effect found in the project.
2. **The 80 Gemini r2/r3 plans are worth +0.035** (val-selected, 3 seeds,
   t=3.69, p=0.030, complete separation) at 1280.

Note the two are not additive: at 1280 `mix5092` beats `mix3652` by +0.033, at
2048 by only +0.010 -- but with 2 seeds per arm at 2048 those overlap, so the
per-pool value needs re-measuring at 2048 before it is quoted.

**Do not confuse this with source resolution.** The "~0.032" below is the value
of generating *sources* larger. This is the *training* `--image-max-size`, a
different knob, and every number in this repo before 2026-09-06 was measured at
1280 with ~0.06 left on the table.

Rebuilt from a clean clone on 2026-09-06 (GH200): every pipeline count matched
exactly and `mix3652` @1280 reproduced at 0.6947. A rebuild landing in 0.68-0.70
at 1280 is a pass.

Report the **3-seed mean** as the result. Run-to-run noise is ~0.013–0.018 sd on
these mixes, so a single checkpoint is the top of a spread, not the expected
value. See "Seeds" below.

## From startup.md: validation-split subsections (2026-09-07)

### Average the last epochs instead of picking one (2026-09-07)

Single-epoch selection is noisy the other way too: at 2048 the same run swings
0.70-0.76 on HF14 from one epoch to the next, and on 77 validation selections
val can pick the dip. `scripts/average_checkpoints.py` averages the weights of
the last epochs of the cosine schedule (SWA-style, no acceptance data) and
ranks the window on validation:

| `mix5092` run | single epoch, val-picked | averaged, val-picked window |
|---|---:|---:|
| 2048 seed 7 | 0.7060 | 0.7393 (e5-8) |
| 2048 seed 31 | 0.7729 | 0.7779 (e3-8) |
| 2560 seed 7 | 0.7027 | 0.7523 (e3-8) |

Every window on every run scored within 0.01 of the others; the single-epoch
pick is the outlier. Val does not always prefer the average over its own best
single epoch, so this is a variance-reduction protocol adopted a priori, not one
chosen because HF14 liked it -- confirm on each new seed before quoting.
**2048, 2 seeds, averaged: 0.7586** against 0.7394 single-epoch.

### Synthetic-only result (2026-09-07)

Train on synthetic alone, nothing real: **HF14 0.686 ± 0.009 val-selected,
0.692 averaged, 2 seeds**, against 0.573 ± 0.023 for the v5 pool (2 seeds) and
~0.61 for every v5 pool ever recorded. The pool is the union of the v5
`toparea1600_balanced` and 1,600 drawn v6d plans (`generate_synthetic_v6.py`,
defaults = v6d), trained at 2048. What moved it and what did not is in
`synth_progress.md` (2026-09-07 entry).

```bash
./run_synth_v6.sh headline 7      # and seed 31; ~25 min each on a GH200
```

`run_synth_v6.sh` builds the pool, merges it, runs the documented two-phase
schedule at 2048, then val-selection and checkpoint averaging, printing both
numbers. Other arms: `union` (1280), `unionb`, `v5only`, `v6only`, `mixv6b`.

**One reproduction caveat.** The pool that produced 0.686 was generated before
a one-line fix to the excerpt-crop step (`p.buffer(0).intersection(cb)`), so 4
of the first 1,604 draws failed and the 1,600-image subset is ids 0-1603 minus
{234, 523, 988, 1019}. With the fix those four succeed, so a rebuild gives ids
0-1599 -- the same recipe and distribution, four different images. Expect the
result to reproduce within seed noise (~0.05 synth-only), not exactly.

### v6b synthetic added to the mix (2026-09-07, one seed so far)

`data/mixed/v5_1600_v6b_1600_rf1548_gen504_gem1440` (6,692 records: the
documented mix plus 1,600 drawn v6b plans, see `synth_progress.md`) at 2048,
seed 7, two-phase 9 epochs: **HF14 0.7754 val-selected** (val 0.8176, epoch 8,
still rising) against 0.7060 for `mix5092` at the same seed and 0.7729 for its
best seed. Per-epoch peak 0.7775, the highest any 2048 run has reached. One
seed by decision (the session's aim is synth-only); do not quote this as a
recipe change until a second seed is run.

## From startup.md: v5 tiles and tile-sim setup

Anything that runs `generate_synthetic_v5.py` also needs the curated tiles, which
are a separate HF dataset and are **not** fetched by the commands above:

```bash
python3 -c "from huggingface_hub import snapshot_download as d; \
  print(d(repo_id='abshetty/floz-assets', repo_type='dataset'))"
TILES=<printed path>/reference_tiles_curated        # 50 tiles
```

`--tile-sim` (default off, and it should stay off — see the 2026-08-12 entry in
`synth_progress.md`) additionally needs the pairwise table, which is not
committed because it lives under `data/`:

```bash
PYTHONPATH=. python3 scripts/family_similarity_probe.py \
  --tiles $TILES --out data/probes/tile_pairs.json
```

## From startup.md: the 2026-09 data pipeline and training recipe (mix3652 / mix5092)

## Synthetic source

The original pools `data/synthetic/faintcad2500` and `cadneg2500` **cannot be
rebuilt**: they only ever existed on a VM two hops back and the
`generate_synthetic_v5.py` flags that produced them were never committed (the
full history of every branch was searched). The named recipes `realhard-v1`,
`realhard-v2`, and `mildtransfer-v1` are explicitly approximations of that
distribution, not the thing itself.

The reproducible stand-in is the 20k-row default config of
`abshetty/floz-synth-v5`, exported to local-data format:

```bash
python3 scripts/hf_to_local.py --out data/synthetic/hf20k --workers 48
# 20,000 images, 74,465 instances; modes: elevation 8956, freeform 8085, roof_plan 2959
```

`hf_to_local.py` fills in a per-instance `area` (outer ring minus holes) when the
parquet lacks it, because `select_toparea_local.py` ranks on it.

```bash
python3 scripts/select_toparea_local.py \
  --sources data/synthetic/hf20k \
  --out data/synthetic/toparea1600_balanced --n 1600 \
  --mode-quotas elevation=889,roof_plan=540,freeform=171
```

This selects the top 1,600 of 20,000 (top 8%) where the original selected the top
1,600 of 5,000 (top 32%), so the set skews higher-area: score mean `0.518` /
median `0.387` versus the historical `0.394` / `0.293`. Empirically this costs
nothing — see the mix3148 control below.

## Real data from Roboflow

Two projects in workspace `perceive-ai` contribute:

```bash
python3 - <<'PY'
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace("perceive-ai")
rf.project("floz-real-pool").version(2).download(
    "coco-segmentation", location="data/roboflow/floz-real-pool-v2-raw", overwrite=True)
rf.project("floz-generated-realistic-label-pool").version(1).download(
    "coco-segmentation", location="data/roboflow/floz-genreal-v1-raw", overwrite=True)
PY

python3 scripts/roboflow_to_local.py \
  --coco data/roboflow/floz-real-pool-v2-raw/train/_annotations.coco.json \
  --img-dir data/roboflow/floz-real-pool-v2-raw/train \
  --out data/roboflow/floz-real-pool-v2-clean
# 86 images, 282 instances, 127 remove polygons attached 129 times,
# 2 to multiple masks, 1 degenerate pattern dropped, 2 unlabelled skipped

python3 scripts/roboflow_to_local.py \
  --coco data/roboflow/floz-genreal-v1-raw/train/_annotations.coco.json \
  --img-dir data/roboflow/floz-genreal-v1-raw/train \
  --out data/roboflow/floz-genreal-v1-clean
# 28 images, 107 instances, 62 remove polygons attached 62 times
```

A third source, labelled 2026-09-05 and **not yet in any recorded result**:

```bash
rf.project("floz-gen-gemini-r2").version(1).download(
    "coco-segmentation", location="data/roboflow/floz-gen-gemini-r2-raw", overwrite=True)
rf.project("floz-gen-gemini-r3").version(1).download(
    "coco-segmentation", location="data/roboflow/floz-gen-gemini-r3-raw", overwrite=True)
# r2: 36 images, 231 instances, 100 remove polygons attached 100 times
# r3: 44 images, 247 instances, 205 remove polygons, 1 hole dropped (no parent)
```

These are the two Gemini rounds; 80 of the 100 generated were worth labelling.
Merge the two clean dirs into `floz-gen-gemini-r23-clean` before augmenting, so
the 18x replication and the `item_XXXXXX` renaming stay deterministic.

A fourth round, labelled and pulled 2026-09-17, **tested and NOT adopted** (null
result, see "Durable comparison points" below — do not re-add without a reason
beyond "more Gemini data"). It had no version yet:

```bash
rf.project("floz-gen-gemini-r4").generate_version(settings={"augmentation": {}, "preprocessing": {}})
# wait for generation, then:
rf.project("floz-gen-gemini-r4").version(1).download(
    "coco-segmentation", location="data/roboflow/floz-gen-gemini-r4-raw", overwrite=True)
# 88 images, 929 annotations -> roboflow_to_local.py -> 88 images, 612 instances,
# 317 remove polygons attached 317 times
```

If picking this thread back up, merge as `floz-gen-gemini-r234-clean` (168
images) rather than replacing r23, so the null result's baseline (`r23`) stays
rebuildable.

`floz-generated-realistic-label-pool` v1 holds the 28 salvageable, hand-labelled
images from the AI-generation batch. If it has no version yet, generate one with
**no preprocessing and no augmentation** — resizing and offline augmentation are
done locally.

New Roboflow versions will legitimately change these counts; read the conversion
summary rather than forcing the old ones.

### Pool characteristics

| | real 86 | generated 28 | gemini r2+r3 80 |
|---|---|---|---|
| long side | 640 (every image, native) | 1536–2065 | 3168 (median) |
| megapixels (median) | 0.41 | 1.57 | 4.6 |
| instances / image | 2.0 | 3.0 | 6.0 |
| families / image | — | — | 1.6 |
| labelled-area fraction | 0.153 | 0.284 | 0.294 |

The 86 real plans are natively 640×640 web-scraped drawings — Roboflow stores
them at that size, so no re-export recovers detail, and training at 1280 upscales
them 2×.

## Build the training mix

```bash
python3 scripts/augment_local_dataset.py \
  --src data/roboflow/floz-real-pool-v2-clean \
  --out data/roboflow/floz-real-pool-v2-strong18 \
  --aug-per-scene 17 --strong --seed 5858          # 86 -> 1,548

python3 scripts/augment_local_dataset.py \
  --src data/roboflow/floz-genreal-v1-clean \
  --out data/roboflow/floz-genreal-v1-strong18 \
  --aug-per-scene 17 --strong --seed 5858          # 28 -> 504

python3 scripts/merge_local_datasets.py \
  --sources data/synthetic/toparea1600_balanced \
            data/roboflow/floz-real-pool-v2-strong18 \
            data/roboflow/floz-genreal-v1-strong18 \
  --out data/mixed/toparea1600_rf1548_gen504       # 3,652
```

The 80 Gemini images are not in `mix3652`, which is the mix every recorded
result used. Add them as a fourth source rather than editing the three above:

```bash
python3 scripts/merge_local_datasets.py \
  --sources data/roboflow/floz-gen-gemini-r2-clean \
            data/roboflow/floz-gen-gemini-r3-clean \
  --out data/roboflow/floz-gen-gemini-r23-clean            # 80

python3 scripts/augment_local_dataset.py \
  --src data/roboflow/floz-gen-gemini-r23-clean \
  --out data/roboflow/floz-gen-gemini-r23-strong18 \
  --aug-per-scene 17 --strong --seed 5858                  # 80 -> 1,440
# merge as a fourth source: 1,600 + 1,548 + 504 + 1,440 = 5,092
```

Verify both `images/` and `annotations/` hold exactly 3,652 files. Merge order
matters for the deterministic `item_XXXXXX` renaming. Everything under `data/`
is intentionally git-ignored.

## Train

Two optimizer phases: actual epoch 0 on the first epoch of a planned ten-epoch
cosine, then reload, reset the optimizer, and train actual epochs 1–8 on the
first eight of a planned nine-epoch cosine. Do not approximate this with one
uninterrupted command.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=data/runs/ck_mix5092_res2048_seed31
DATA=data/mixed/toparea1600_rf1548_gen504_gem1440
COMMON="--batch-size 8 --num-workers 16 --prefetch-factor 4 \
  --image-max-size 2048 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed 31 --compile --pad-grid 512"

PYTHONPATH=. python3 scripts/train_refunet.py --local-data $DATA \
  --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON

PYTHONPATH=. python3 scripts/train_refunet.py --local-data $DATA \
  --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from $CK/epoch_0.pth
```

**`--image-max-size 2048`, not 1280.** Worth +0.05 to +0.07 (see the top of this
file). At 2048 both mixes were still improving at the final epoch, where at 1280
they peaked mid-run and declined -- higher resolution delays overfitting. A
16-epoch schedule was tried and is **null**: 0.7568 ± 0.0090 against 0.7594 ±
0.0191 at 9 epochs. One seed showed +0.017 and the second erased it; keep 9.

**That null is about ONE long schedule, not about whether more training helps
at all (2026-09-17, one seed, not yet adopted).** Instead of one continuously-
annealing cosine, reset the optimizer and give a fresh cosine restart from a
converged checkpoint, repeatedly (`--reset-optimizer --init-from <checkpoint>`,
same as the documented two-phase recipe already does once). On `mixr4` seed 31:
val-complement climbed from 0.7924 (documented recipe's epoch 8) to a peak of
**0.8127 at epoch 20** (two restarts, ~20 total epochs), then plateaued through
epoch 28 (train mIoU kept rising 0.85→0.88 without val following — the
overfitting signature, not a blowup). One seed; needs replication before it
changes this recipe. See `codex_doc.md` (2026-09-17) and `synth_progress.md`.

**`--compile --pad-grid 512` is optional speed** (1.5x/epoch: channels_last,
`torch.compile`, bucketed padding, fused AdamW). Bucketed padding enters
GroupNorm statistics, so it is not bit-identical; parity checked at 2 seeds on
`mix3652` (compiled 0.7132/0.6806 vs eager 0.6947/0.6698, inside noise). Needs
`--pad-grid >= 256`; `dynamic=True` compilation is *slower* than eager because
`--domain-random` gives every sample a unique shape.

`--mask-thresh 0.35` makes the per-epoch diagnostic directly comparable to the
reported result. `scripts/run_fixedsampler.sh` runs all three seeds.

**Deliberately anchor-free — do not add `--anchor`.** Fixed, it is
neutral-to-slightly-negative (mix3652 −0.014 HF14; synth-only −0.055 / −0.040 on
validation). Its recorded 0.680 → 0.433 "collapse" was a reference-plane bug, not
a shortcut, and does not reproduce; `--anchor-ref-plane` now defaults to the fixed
value, so an anchored run is merely useless rather than harmful. Evidence in
`synth_progress.md`.

**Screening budget.** Run-to-run sd on `mix3652` is **0.0262** (seeds 7/31/99),
so single-seed gaps under ~0.05 mean nothing. A +0.015 single-seed "win" was
retracted at 3 seeds on 2026-08-10. Scale seeds to effect size rather than
always running three: 1 seed settles a gap over ~4x sd, 2 is the screening
default, 3 when the seeds straddle the other arm, the gap is under ~2x sd, or
the number is going to be recorded here.

**Schedule length is a step budget, not an epoch count.** 9 epochs is 5,724
optimizer steps on 5,092 records but only 1,620 on 1,440 -- the 80-Gemini pool
scored 0.593 at 9 epochs and 0.669 at 24 purely for that reason. Set the
schedule per pool, or use `--early-stop-patience` (monitors `val_loss` by
default; `--early-stop-monitor hf14` is selection against the acceptance set and
is screening-only).

Throughput at 1280 is ~82 s/epoch on 3,652 records (~54 s compiled); at 2048
about 3x that. `select_epoch_on_val.py` is one process per run and they are
independent -- run them in parallel, not in one sequential pass.

## From startup.md: seeds policy before 2026-10-20

## Seeds

Run-to-run noise, same data and recipe: sd ~0.013–0.018 on the 3.1k–3.7k mixes,
and up to 0.049 spread on 504-record pools. Effects worth chasing (+0.03) are
only ~2× that, and this repo has already retracted a win that survived three
seeds and died at five (`385f3cb` → `1cbaeea`, `cd71c26`).

Report a multi-seed mean ± sd. Three seeds is the working minimum; use five when
an effect is under ~2× the sd.

**Superseded 2026-09-28 (user's call):** at ~0.85 no more multi-seed runs. One
seed per arm; the evidence is the paired per-question test on HF14 against a
same-recipe control (sign counts and p), as for 0.8440.

## Durable comparison points

All rows are `RefUNet`, HF14, threshold 0.35, fixed sampler, 1280 unless the
resolution column says otherwise. The sampler column is load-bearing.

| Training data | res | seeds | HF14 mIoU |
|---|---|---|---:|
| real 86 only (1,548 records) | 1280 | 3 | 0.3597 ± 0.0448 |
| **gemini 80 only (1,440 records), 9 ep** | 1280 | 3 | 0.5933 ± 0.0170 |
| **gemini 80 only, 24 ep** | 1280 | 2 | 0.6688 ± 0.0350 |
| synth 1600 only (disc / conn pools) | 1280 | 3 | ~0.61 |
| no synthetic (194 sources, 3,492 records) | 1280 | 2 | 0.6683 ± 0.0266 |
| no synthetic (194 sources), 16 ep | 2048 | 1 | 0.7310 |
| `mix3652` (114 sources) | 1280 | 3 | 0.6769 ± 0.0155 |
| `mix3652` | 2048 | 2 | 0.7490 ± 0.0364 |
| **`mix5092` (194 sources)** | 1280 | 3 | 0.7097 ± 0.0157 |
| **`mix5092`** | **2048** | **2** | **0.7594 ± 0.0191** |
| `mix5092`, 16 ep | 2048 | 2 | 0.7568 ± 0.0090 |
| `mix6676_with_r4` (282 sources, +88 4th-round Gemini) | 2048 | 2 | 0.7366 val-sel / 0.7702 avg — **null vs. mix5092**, not adopted (2026-09-17) |
| locally generated synth, `--confusable-prob 1.0` | 1280 | 3 | 0.5787 ± 0.0433 |
| **`mix5092`+3,200 v6d @2560** (2026-09-18) | **2560** | 1 | **0.7834** val-sel — best recorded, one seed |
| `mix5092`+3,200 v6d, 29 ep w/ restarts | 2048 | 1 | 0.7453 val-sel / 0.7676 avg — null vs. recorded 1,600-v6d mix |
| **swin_t**, `v6d1600_rf1548_gen504_gem1440` (194 src) | **2048** | 1 | **0.7403** val-sel (2026-09-21) — below RefUNet's 0.7594 on mix5092 |
| **swin_t**, v6d-only 1,995 | 2048 | 1 | 0.7066 (`codex_doc.md`) |
| **swin_t**, v6r-only 1,995 (region scale matched to real) | 2048 | 1 | 0.7090 — **null**, +0.0024 (2026-09-21) |
| **swin_t**, gemini r23 only (80 src), 20 ep | 1280 | 1 | 0.6025 val-sel; hard train IoU 0.9254 |
| **swin_t**, gemini r234 only (168 src), 20 ep | 1280 | 1 | **0.7148** val-sel; hard train IoU 0.9257 — **+0.112 on identical fit** |
| **swin_t**, `v6dmix_plus_r4` (282 src, 6,671 rec) | **2048** | 1 | **0.7887** val-sel (2026-09-21) — **best recorded**, still rising at the final epoch |
| **swin_t**, `v6dmix_plus_r4` + 1 restart, **inference 4096** | 2048 | 1 | **0.8170** val-sel (2026-09-24) — **best recorded**; same model @2048 inference not read |
| **swin_t, `v6dmix_plus_r4` + 1 restart + `--roi-ref --roi-ref-mode add`, infer 4096** | 2048 | 1 | **0.8440** val-sel (2026-09-28) -- +0.027 paired vs 0.8170 (p=0.017); one seed by policy |
| swin_t, no-gray real aug + `--same-fill-new-colour 0.5` synth, infer 4096 | 2048 | 1 | 0.8153 — but validation prefers the unchanged mix (0.791 vs 0.801); data change not supported |
| swin_t, v7 synth only (1,991) | 2048 | 1 | 0.6834 vs v6d 0.7066 — null (2026-09-23) |
| v6d-only 100,000, single-pass (2 epochs) | 2048 | 1 | 0.7130 — no real data at all |
| **swin_t, Revit-style FreeCAD 1,986 (`generate_synthetic_fc.py --revit --mode-weights 66,16,18`)** | 2048 | 1 | **0.7564** epoch 8 (2026-09-27) — best synthetic-only; val 0.7568; @4096 0.7321 |
| swin_t, r8 flags 1,991 | 2048 | 1 | 0.6304 epoch 8 — r8 hurts |
| swin_t, v6d 8k / 16k (5 / 4 epochs) | 2048 | 1 | 0.7367 / 0.7294 |
| swin_t, v6d 2k, `--refs-per-image 6` | 2048 | 1 | 0.7190 epoch 8 (vs 0.7066 one-question) |
| **swin_t, Revit 2k, `--refs-per-image 6`, 1+4 epochs** | 2048 | 1 | **0.7607** (val 0.7816) — best synthetic-only (2026-09-27) |
| swin_t, Revit 2k new pool (`--revit --revit-plans --shaped`) + ref-jitter fix + `--roi-ref add` | 2048 | 1 | 0.7855 / 0.7970 @2048/4096, epoch 8 (2026-09-30) -- screen control `revit2k-roi-e8` |
| **swin_t, Revit 10k + ROI, 1+5 ep** (`revit10k-roi-e5`) | 2048 | 1 | **0.7868 / 0.8126** @2048/4096; val **0.8861** @4096 -- best synthetic-only, best val of any model (2026-09-30) |
| swin_t, `v6dmix_plus_r4` with v6d_1600 -> Revit 1,600, ref-jitter fix, + ROI restart (`revitmix-roiadd-e10`) | 2048 | 1 | 0.8367 val-sel @4096 (val 0.7843) -- **null/negative vs 0.8440** (-0.007, p=0.58), not adopted (2026-09-30) |
| v6d-only 12,000 | 2048 | 1 | 0.6934 |
| v6d-only 1,600 | 2048 | 1 | 0.6400 |

Older rows, still valid at 1280 with the fixed sampler: `mix3652` 0.6860 ±
0.0175 and its 2026-08-12 rebuild 0.6954 ± 0.0023; locally generated synth
without confusable pairs 0.6823 ± 0.0190. Pre-2026-08-06 (broken sampler) and
retired query-model numbers: `synth_progress_archive.md`, not comparable here.

### Frozen-backbone probe rows (2026-09-20, NOT product numbers)

`swin_b` is a hierarchical vision transformer (windowed self-attention, strides
4/8/16/32). It is the first transformer backbone tested in this project --
`RefCrossAttnUNet`, despite the name, was attention bolted onto a ResNet.

Same decoder, same 200 synthetic images, 1024 px, backbone frozen; HF14 is 52
cached real selections. Useful only for ranking, and far below the real
pipeline's 0.78 because of the freeze and the resolution.

| backbone + decoder | seeds | train | HF14 (probe) |
|---|--:|---:|---:|
| swin_b + selfattn | 4 | 0.7893 ± 0.024 | **0.5591 ± 0.018** |
| swin_b + baseline | 4 | 0.8278 ± 0.028 | 0.5179 ± 0.046 |
| resnet50 + corr4 | 4 | 0.9154 ± 0.006 | 0.4959 ± 0.026 |
| resnet50 + baseline (incumbent) | 4 | 0.9025 ± 0.011 | 0.4661 ± 0.015 |

Swin beats the incumbent by +0.0842 (n=5 each, 5.5 sd at the measured per-run
sd of 0.0243) while fitting worse. Across all 30 probe runs corr(train fit,
HF14) = **-0.18**: fitting the generator better is mildly ANTI-predictive of
real transfer. Checked against an LR sweep (1e-4 to 3e-3): swin_b holds
0.5594-0.5600 at three LRs, resnet50 peaks at 0.4959, so it is not an artefact
of one LR tuned for the incumbent.

### What the comparisons establish

- **Training resolution is the largest single effect: +0.05 to +0.07** (1280 ->
  2048), on both mixes, so it is general rather than pool-specific.
- **A generated plan is worth ~1.65x a scraped plan, per source** (gemini 80
  alone 0.5933 vs real 86 alone 0.3597, matched on sources and records). This
  **retires** the earlier "a generated plan ≈ a real plan" parity, which was
  measured on the mis-specced v1 round.
- **The 80 Gemini plans are worth +0.035** on the mix at 1280 (val-selected, 3
  seeds, p=0.030). At 2048 the same contrast is +0.010 with overlapping arms --
  unresolved, needs re-measuring.
- **Synthetic still earns its place**: removing it costs 0.041 at 1280 and
  0.026 at 2048 (0.7310 vs 0.7568, both 16 ep). Standalone it is ~0.61, second
  only to the Gemini pool -- so keep it, and regenerate it larger.
- **Source resolution is worth ~0.032** (the 28 generated at 640 vs native).
  Distinct from training resolution above.
- **Count dominates**: 28 -> 86 real sources buys +0.073.
- The HF 20k synthetic substitution cost ~0.006; generating the synthetic half
  locally instead costs ~0.013.
- **Deliberately confusable synthetic material pairs cost -0.104** (t=3.80,
  p=0.019). `--confusable-prob` stays off; see `synth_progress.md` (2026-08-12).

So: **train at 2048**, and generate more. Adding *records* without adding
sources does nothing (2026-08-10 volume/ratio nulls in `synth_progress.md`), but
adding generated sources is now the best-value supply there is.

### Cheap architecture screening (2026-09-20)

Architecture and fit questions no longer need a training run. Cache the frozen
backbone once, then train decoder variants against the cached tensors:

```bash
PYTHONPATH=. python3 scripts/cache_backbone_features.py \
  --backbone swin_b --local-data data/synthetic/v6d_train2000 --n 200 \
  --image-max-size 1024 --out data/cache/swin_b_train_1024
PYTHONPATH=. python3 scripts/cache_backbone_features.py \
  --backbone swin_b --source hf14 --image-max-size 1024 \
  --out data/cache/swin_b_hf14_1024            # the 52 fixed questions

PYTHONPATH=. python3 scripts/decoder_search.py \
  --train-cache data/cache/swin_b_train_1024 \
  --fresh-cache data/cache/swin_b_fresh_1024 \
  --hf14-cache  data/cache/swin_b_hf14_1024 --steps 1500
```

40-90 s per variant against ~1 h for a real run. **The speedup does not buy
statistical power on the transfer metric.** Five runs, identical arguments:
train IoU sd **0.0032**, cached-HF14 sd **0.0243**. Fit is settled by one run;
transfer still carries the repo's usual ~0.024 because it is still the 52 hard
questions. **These are ranking numbers, not product numbers** -- frozen
backbone, 200 synthetic images, 1024 px -- so a winner still needs a real run.
It cannot measure backbone finetuning at all.

Before treating any fit target as reachable, run `scripts/label_ceiling.py`:
the stride-4 head costs ~0.000, but the native -> input -> native resize chain
caps HF14 at 0.9739/0.9759/0.9846 (2048/2560/4096) and the v6d pool at
0.9058/0.9210/0.9450. **A "train IoU 0.95" target is above the v6d ceiling
below 4096 px.** Small regions drive it: 0.9023 under 500k native px against
0.9958 above 5M.

`scripts/residual_decomp.py` says which axis a deficit lives on. On every pool
measured, boundary error is at most a fifth of the total and region-level error
(wrong region selected, interior not filled) is 46-56%.

### The limit is generalization, not fit

**Qualified in the large-data regime (2026-09-18).** At 100,000 procedural
plans: fresh-synthetic 0.825 after one pass and 0.798 after two, against HF14
0.711/0.713 — so the ~0.11 synthetic→real transfer gap dominates, and the
second pass *lowers* fresh-synthetic while train rises (overfitting at 2
repeats). The paragraph below still describes the 3-8k mixes it was measured
on. `scripts/fresh_synth_iou.py` measures the train/fresh/real ladder;
`run_capacity_probe.sh` tests whether width also binds (untested).


Train IoU (derived from the Dice term) was still climbing at the final epoch in
every arm -- `mix5092` reached 0.758 while HF14 turned over at epoch 4. Fitting
is not the constraint; `real86only` fits as well as the winning mix (0.749) and
scores less than half as well. Regularization and data quality are the levers
that remain — **with one 2026-09-17 caveat on "longer training push[es] the
wrong lever": see "16-epoch schedule" under Train, one seed, not yet adopted.**

## 2026-09-21 — swin_t meets real data; realism-by-metric fails

**1. swin_t + the right mix is the new best: 0.7887** (282 sources, 2048,
val-selected, one seed), over the 0.7834 record and +0.029 on the 0.7594 ± 0.0191
recipe mean. On the 194-source mix it scored only 0.7403 and looked *worse* than
RefUNet — the mix was the deficit, not the backbone. Val-selection picked the
final epoch and HF14 was still climbing, so **the 9-epoch schedule stops early
for this configuration**; warm restart from `epoch_8.pth` is the next run.
Real data is worth +0.1235 to RefUNet and +0.0337 to swin_t, so the backbones
are not interchangeable and no RefUNet result transfers without re-measuring.
**This was the session's repeated error — check the backbone before citing any
number.**

**2. Transformers degrade on real plans as synthetic training continues.** 100k
v6d @1024, HF14 by epoch: RefUNet 0.5993 → **0.6513 ↑**, swin_t 0.7484 →
**0.6913 ↓**, dinov3_s 0.6895 → **0.6561 ↓**. Synthetic val_loss fell
monotonically for all three. Fit improves while transfer decays — for the
transformers only.

**3. Region-scale realism is a null on swin_t.** v6d regions occupy 0.00896 of
the sheet against real's 0.02411 (2.7x too small) with 11.7 regions/plan against
3.3. `generate_synthetic_v6.py` gained `--view-count-weights` and
`--max-label-fams` (defaults reproduce v6d bit-for-bit); `85,15,0` + cap 3 lands
0.02468 / 5.4, matching real scale to 2.4% while holding labelled area at 0.326
(vs 0.373 — direct union-matching would have cut it to 0.137, the August
ink-matching trap). HF14 moved +0.0024. Five metric-targeted realism attempts,
five nulls. The one synthetic source that ever paid (Gemini, +0.035, p=0.030)
was never metric-matched.

**4. Gemini r4 pays, and the "saturated" verdict was a RefUNet artifact.**
Gemini-only, 80 → 168 sources: **+0.112** HF14 (0.6025 → 0.7148) on **identical**
hard train IoU (0.9254 vs 0.9257), ruling out the 2.1x step-budget confound. In
the mix, 194 → 282 sources is **null on RefUNet** (0.7366 vs 0.7594) and
**+0.048 on swin_t** (0.7403 → 0.7887). Source count on the right backbone is
now the only lever with positive evidence at every scale tested — against
realism 0-for-5 and v6d volume saturated (8x for +0.020). Value tracks the
source *ratio*, not the count, so each further +0.04 costs ~45% more plans:
282 → ~409 next. `startup.md`'s old "do not re-add r4" warning is void for
swin_t; r4 is adopted.

**5. `scripts/fresh_synth_iou.py` scored zero padding as signal.** `collate_fn`
pads to a common size (~40-45% of canvas) and backbones answer differently:
mean P(padding) 0.071 for swin_t, 0.422 for dinov3_s — which alone moved
dinov3_s from 0.86 to 0.33. Now reports a `pixel_mask`-valid number; quote that.
Every hard train IoU recorded before this date (0.8022 included) is the old
full-canvas kind.

**6. The 28-source per-source protocol is underpowered.** Three real 28-draws
gave 0.4301 / 0.4345 / 0.5356 — **draw-noise sd 0.060**, larger than every
between-source difference (gemini +0.041, v6d +0.048, genreal −0.009). The best
single arm was a real draw. Needs ~9 draws/source or ≥56-source arms.

## From codex_doc.md: finished handoffs (2026-10-17 .. 2026-10-21)

## Pick up here (2026-10-17)

**Best model:** `abshetty/floz-refunet-swint-xblend-kq30-kms30` (synthetic only, no training of its own): HF14 0.9163, HF14 fix v2
0.9290, val per-sheet 0.9364, Gemini30 0.8740 @4096, hand boxes. It is the float 0.5 / 0.5 weight average of `single18kq-blend30`
(0.3 x `-single18kq-dr` + 0.7 x `-single18kq-nodr`) and `single18kms-blend30` (same with `-single18kms-{dr,nodr}`, the
multi-resolution arms); all arms are one-epoch restarts of `longswiss-swa37-41` and all are on the Hub. Details and the gate:
`startup.md`. **Always score and gate HF14 with label fix v2 alongside plain HF14** (the user, 2026-10-17).

**The standard experiment** is `TAG=<name> [GEN="..."] [POOL=...] [TRAINX="..."] [BATCH=n] ./run_single_pass.sh` (~8 min pool,
~20 min per arm at batch 8, ~3 min scoring on a GH200; ids 600000+, seed 6, so runs at different generator versions are paired sheet
for sheet). It blends 50/50 and scores without `--save-probs`; also build the 0.3 / 0.7 (DR / no-DR) blend, and -- the cheap win of
2026-10-17 -- the 0.5 / 0.5 blend of that with the current best's components. Compare with `scripts/paired_compare.py` against
`eval_baselines/xblend-kq30-kms30_*`, apply the gate, and publish (Hub + HF14 / val / Gemini30 `--save-probs` maps on
`abshetty/floz-diag` `<tag>/<split>_4096/{metrics.json,probs/}`) only if it passes. Publish the arms too, so blends can be rebuilt.

**Cross-run weight blends are the current lever** (`synth_progress.md` 2026-10-17): kms30 and kw30 each failed the gate alone
(Gemini30 17 collapses in both), but 0.5 kq30 + 0.5 kms30 passes with Gemini30 +0.006. 0.5 kq30 + 0.5 kw30 ties (Gemini30 +0.0000,
sheet 1 -0.075). Untried: other weights (e.g. 0.6 / 0.4), kq30 + kms30 + kf / kp arms, soups of more one-epoch runs.

**Post-processing / polygons:** the frontend receives polygons, not masks (the user, 2026-10-17). Linear upsample + fill holes <= 4096
px is +0.006 val / +0.002 HF14 / +0.006 Gemini30 (23 sheets up, 1 down 0.001), not adopted (protocol change); speck drop + small-hole
fill take predictions from ~12-44 components to ~4. Next: a polygon stage (speck drop, hole fill, contour trace, simplify) scored as
the model output -- IoU of the rasterised polygons. `scripts/postproc_probe.py` (hf14 / val / gemini30).

**User's submission target:** "0.95 on real" (protocol -- HF14 vs val, per question vs per sheet -- not pinned down yet).

**Pick up here (2026-10-20, end of GH200 session; details in `synth_progress.md` 2026-10-20 entries).** Best model unchanged
(xblend-kq30-kms30). Today: seed noise ~0.016 (judge at two seeds, blend vs blend); late-checkpoint averaging, Hub soups,
hires-crop, CascadePSP, per-question / family-aware BCE all null or negative. **Inference / post-processing candidates, held for
the user's decision:** (1) 0.7 x 4096 + 0.3 x 2048 multi-scale (never hurts: val +0.0015, Gemini30 +0.0067, -0.25 fixes / q;
~+25% compute); (2) + TTA x8 (fix v2 0.9429, +0.0117, p=0.07; 8x compute); (3) polygon eps 2 (half the vertices, same IoU);
(4) BPR refiner (`scripts/bpr/bpr.py`, band 4) + sliver revert (refined pieces < 50% on the coarse mask go back to P):
HF14 +0.0046 (p=0.002), fix v2 +0.0033 (p=0.04), val +0.0016, extra deletes +0.06 / +0.02 per q -- small; next for BPR: maps
for the 40% of `bpr_s900000` lost to OOM, real-like patch augmentation, a P-only control, stacking with (1). New tools:
`scripts/editcost.py` (fix counts / vertices), `train_refunet.py --save-every`, `--bce-other-weight`, `--bce-per-question`.
On the Hub (`abshetty/floz-diag`): `bpr/refiner_2026-10-20.pth`, `xblend-kq30-kms30/{split}_2048` (2048 maps),
`xblend-kq30-kms30/bpr_b4/`, `xblend-kq30-kms30/tta/`, `checkpoints/kqtrace_s8_nodr_epoch0.pth` (the seed-8 kq no-DR arm, for
two-seed blends) + `checkpoints/kqtrace_s{7,8}_nodr_order.json`. Parked: TracIn, self-training with automatic boxes (more Gemini
images). **Twin test done (2026-10-20, `synth_progress.md`): Gemini30 1 / 26 recover on their twins (0.65 -> 0.98,
0.48 -> 0.82) = appearance failures; no question fails on both; twins 16 / 20 fail on their own -- inspect those twin images.**
Multi-scale + BPR stack: val +0.0023, fix v2 +0.0031, Gemini30 +0.0089, no added fixes (held for the user too).

**Seed noise (2026-10-20, `synth_progress.md`):** the same seed reproduces (per-sheet sd <= 0.012), but seed 8 vs seed 7 on the kq
no-DR arm moves HF14 / Gemini30 by ~0.016 and 5 / 30 Gemini30 sheets by > 0.05 -- the gate's sheet rule fires on seed noise, and
every single-pass run so far used seed 7. Late-checkpoint averaging within a run does not help; a 0.5 / 0.5 two-seed blend beats
both seeds on every set. Soups of the Hub arms and adding the seed-8 arm to the best fail the gate. `--save-every N` records step
checkpoints and the batch order. **Judge changes at two seeds (blend vs blend)**; revise the sheet rule to "beyond seed noise".
`--hires-crop 0.5` was screened at seed 7 only (fails the gate: Gemini30 -0.016, inside seed noise); no seed-8 replicate.

**Parked until the user has more Gemini images (2026-10-20): self-training with automatic boxes.** Tile each unlabelled sheet
with candidate boxes; ask the best model each one; keep a box only if (a) it lies wholly inside its own answer at P > 0.9, (b) a
second box elsewhere in that answer returns the same answer (reference-swap consistency), (c) the answer is a sensible size with
texture, (d) the box is not mostly text. Survivors + answers are training questions (uncertain pixels ignored), trained at hard
views (other scale / blur / line weight). First validate on val's labelled sheets: share of accepted boxes wholly inside one
family and their answers' accuracy, plus a contact sheet for the user. HF14 / val decide, not Gemini30; Gemini30 never in the pool.

**Synthetic facsimiles of Gemini images (2026-10-20) -- done: `--spec` twins too different (the user), label-shape twins
(`scripts/facsimile_relabel.py`) built and twin-tested on the GPU (see Pick up here). Original plan, kept for reference:** Add `generate_synthetic_fc.py --spec file.json`
that renders a given house + appearance (the dicts `G.make_house` / `G.make_appearance` now sample, `house_for` line 49)
instead of drawing them; describe each Gemini image in that vocabulary (Claude Code views the image; at scale, Gemini Batch
API writes the JSON), clamped to what the generator can draw. Uses: (1) labelled twins of Gemini content with exact masks and
`ref_exclude` (controlled boxes); (2) a gap list of what specs cannot express (e.g. corner boards); (3) the **twin test** on
Gemini30 -- good on the twin but bad on the original = appearance failure (line weight / scale / style), bad on both =
content / layout -- splits Gemini30's loss by cause. Twins of Gemini30 sheets are diagnostic only, never in a training pool.
Copies content, not appearance. Pilot: 10 user Gemini images + 10 Gemini30 twins, then score the twins (GPU, minutes).

**Open problems** (`synth_progress.md` 2026-10-14 .. 17):
- Gemini30 17 (hip-roof shingle plan, native 2400 px) collapses to blotchy half-confidence in both kms and kw (0.89 -> 0.68-0.70);
  only kq gets it right. Fragile, not specific to multi-res.
- Scale sensitivity: 2048 vs 4096 swings sheets +-0.17-0.29; multi-res training at [2048, 3072] did not shrink them (batch 8 OOMs
  with the range; no grad accumulation / checkpointing in `train_refunet.py`).
- HF14 12 q1 / q3 (turned deck boards) and HF14 2 / Gemini30 8 (elevations) move with the plan pairs; Gemini30 1 swings 0.30-0.69.
- Gemini30 0: tile reference takes the sofas / chairs. HF14 18: the two roofs merge under label fix v2.

## Pick up here (2026-10-21): `./run_gemini_holdout.sh` -- does real Gemini appearance move Gemini30?

The twin test says Gemini30's worst sheets fail on how Gemini DRAWS materials, not on layout. The 139 hand-labelled
Gemini r2-r4 sheets that are not in Gemini30 (`scripts/build_gemini_holdout_pool.py`; Gemini30 is a random 30 of r2-r4,
so earlier mixes with all of r2-r4 leaked it) have never been trained on with a clean Gemini30 read. The script trains the
standard one-epoch restart (run_single_pass.sh) on the current best's own kq pool (generator at 2026-10-15b, rebuilt from
commit `6b25130^`) with and without those 139 sheets x18, at seeds 7 and 8, blends the seeds, scores HF14 / fix v2 / val /
Gemini30 @4096 and writes `logs/gemho_compare.txt` (ctl vs gem, paired; plus ctl seed 7 vs single18kq-blend50, which
should match). ~5 h. Read: Gemini30 up well beyond seed noise (~0.016) with HF14 / val not down = labelled Gemini data
is the lever (label more, e.g. round 5); flat = the remaining gap is not appearance data, measure the label ceiling.
The 2026-10-20d / 21a generator defaults (cross gables, wall splits, panels, pattern drift, base bands) are still
unscreened; this run deliberately does not use them.
