# Startup Guide

Read `PROJECT_UNDERSTANDING.md` first — it defines the task, the metric, and what
does not count as evidence. This file owns the numbers and the commands.

## Read this before quoting any number

Every number measured before 2026-08-06 (`0.6127`, `0.5506`, `0.4646`, `0.3494`,
`0.6432`) used the broken `sample_reference_box`, as did the `>= 0.65` target,
and is **not comparable** to anything measured afterwards. Always state which
sampler a number came from. `PROJECT_UNDERSTANDING.md` explains why the fix
matters; "The reference-box fix" below has the mechanism and its effect.

## Current reproducible result

Fixed sampler, `RefUNet`, trained on 1,600 synthetic + 1,548 Roboflow real +
504 generated-realistic records (3,652 total):

| Item | Value |
|---|---|
| Recipe mean (3 seeds) | **0.6860 ± 0.0175** |
| Best single checkpoint | **0.705407920670342** |
| Checkpoint | `data/runs/ck_fix_mix3652_seed31/epoch_6.pth` |
| Metrics | `data/evaluations/refunet_fix_mix3652_s31_e6.json` |
| Visual audit | `data/visualizations/fix_mix3652_s31_e6/` |
| Evaluation | fixed HF14, 14 plans, all 52 reference selections |
| Mask threshold | 0.35 |

Rebuilt from a clean clone on 2026-08-12 (GH200, stack below): every pipeline
count matched exactly and the recipe landed at **0.6954 ± 0.0023** val-selected
(per-seed 0.6953 / 0.6931 / 0.6977), ~0.009 above the recorded figures with a
tighter spread. Both sit inside the documented run-to-run noise. A rebuild
landing in 0.68–0.70 is a pass; anything outside it means check the data counts
against this file before believing a model change.

Report the **3-seed mean** as the result. Run-to-run noise is ~0.013–0.018 sd on
these mixes, so a single checkpoint is the top of a spread, not the expected
value. See "Seeds" below.

## Validation split — tune here, never on HF14

The HF `real-world-test` config has 28 images. HF14 is the acceptance holdout;
its complement is **never trained on and never scored for acceptance**, so it is
the tuning set:

```text
validation indices: 4,5,6,8,9,10,13,15,17,19,20,21,22,26   (14 images, 77 selections)
```

It correlates with HF14 at Pearson r=0.876 / Spearman 0.888 across 27
checkpoints, and reads systematically higher (~0.75 vs ~0.68) because it is an
easier set — use it for *ranking* configurations, not for absolute numbers.
Several of its plans share source PDFs with HF14 (Las Huertas, Ceilhunt, W
Felton, Maple Rd), so it is correlated rather than independent; treat a
validation gain as suggestive, then confirm once on HF14.

### Checkpoint selection is part of the metric

Taking the best epoch *by HF14 score* uses the acceptance set to choose the
checkpoint. Choosing the epoch on validation instead:

| protocol | HF14 |
|---|---:|
| epoch = HF14 peak (project's stated "any checkpoint within ten epochs") | 0.6860 ± 0.0175 |
| epoch chosen on validation | **0.6783 ± 0.0081** |

The +0.0077 gap is selection bias. Both are honest under their own protocol;
quote the val-selected number when you need one that will hold up.

## Environment and credentials

Credentials live in `~/.env` as `HF_TOKEN` and `ROBOFLOW_API_KEY`; load with
`set -a; . ~/.env; set +a` and never print their values.

```bash
pip install --user -r requirements.txt roboflow
pip install --user openai          # only for scripts/generate_images_openai.py
```

Verified working stack: Python 3.10.12, torch 2.7.0, torchvision 0.22.0,
datasets 5.0.1, OpenCV 4.10.0, Pillow 12.3.0, numpy 1.26.4, shapely 2.1.2, on one
NVIDIA GH200 with 64 CPUs. Minor nondeterminism can change the last decimals.

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

Cache the evaluation dataset (never put these HF images into training):

```bash
python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("abshetty/floz-synth-v5", "real-world-test", split="test",
                  cache_dir="./data")
print(len(ds))  # 28
PY
```

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

`floz-generated-realistic-label-pool` v1 holds the 28 salvageable, hand-labelled
images from the AI-generation batch. If it has no version yet, generate one with
**no preprocessing and no augmentation** — resizing and offline augmentation are
done locally.

New Roboflow versions will legitimately change these counts; read the conversion
summary rather than forcing the old ones.

### Pool characteristics

| | real 86 | generated 28 |
|---|---|---|
| long side | 640 (every image, native) | 1536–2065 |
| megapixels (median) | 0.41 | 1.57 |
| instances / image | 2.0 | 3.0 |
| labelled-area fraction | 0.153 | 0.284 |

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
CK=data/runs/ck_fix_mix3652_seed31
COMMON="--batch-size 8 --num-workers 24 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --mask-thresh 0.35 --seed 31"

PYTHONPATH=. python3 scripts/train_refunet.py \
  --local-data data/mixed/toparea1600_rf1548_gen504 \
  --checkpoint-dir $CK --epochs 1 --schedule-epochs 10 $COMMON

PYTHONPATH=. python3 scripts/train_refunet.py \
  --local-data data/mixed/toparea1600_rf1548_gen504 \
  --checkpoint-dir $CK --epochs 8 --schedule-epochs 9 $COMMON \
  --reset-optimizer --init-from $CK/epoch_0.pth
```

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
retracted at 3 seeds on 2026-08-10.

Throughput is ~25 ms per record per epoch: ~85–92 s/epoch on 3,652 records, so a
full 9-epoch run is ~14 minutes and a 3-seed comparison ~45 minutes. Budget
seeds by default.

## Evaluate

```bash
PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py \
  --checkpoint data/runs/ck_fix_mix3652_seed31/epoch_6.pth \
  --indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 \
  --image-max-size 1280 --ref-size 224 --mask-thresh 0.35 \
  --metrics-out data/evaluations/refunet_fix_mix3652_s31_e6.json
# 14 images, 52 reference selections, mean IoU 0.705407920670342
```

Render the required visual audit (input + rectangle, expected union, predicted
union, overlap/excess/missed) with:

```bash
PYTHONPATH=. python3 scripts/visualize_refunet_selection.py \
  --checkpoint data/runs/ck_fix_mix3652_seed31/epoch_6.pth \
  --mask-thresh 0.35 --out data/visualizations/fix_mix3652_s31_e6
```

It reproduces the evaluator's RNG, resize, and threshold exactly, names files by
IoU so failures sort first, and writes `manifest.csv` + `summary.txt`.

## The reference-box fix

`sample_reference_box` treated `min_size` (128px) as a floor without checking a
box that large could fit, so thin regions exhausted all 450 attempts and fell
through to an unchecked `centroid ± 16` box. On a ring-shaped mask — a wall with
a window punched out — that centroid sits **in the hole**, so the crop showed
none of the pattern, then upscaled 7× to 224px.

Incidence before the fix: **57%** of real-plan crops and **36%** of generated-plan
crops lay partly or entirely outside their own mask, but **0%** of synthetic ones
(those images are 2–5k px wide, so 128px boxes fit trivially). That is why it hid
— it only bit the real data.

The fix uses a Chebyshev `cv2.distanceTransform` for the largest square that fits
at every pixel: size is drawn from what actually fits, the centre only from
pixels deep enough to hold it, sides odd so the box is symmetric. `min_size` is
now a preference; containment is asserted across all pools.

Effect on HF14 (`mix3652`, 3 seeds): **0.6331 → 0.6860 (+0.053, t=4.18,
p=0.014)**, with complete separation between arms. Selections scoring below 0.25
dropped from 8 to 1 while the median barely moved — the fix removed impossible
questions rather than making the model broadly better.

## Seeds

Run-to-run noise, same data and recipe: sd ~0.013–0.018 on the 3.1k–3.7k mixes,
and up to 0.049 spread on 504-record pools. Effects worth chasing (+0.03) are
only ~2× that, and this repo has already retracted a win that survived three
seeds and died at five (`385f3cb` → `1cbaeea`, `cd71c26`).

Report a multi-seed mean ± sd. Three seeds is the working minimum; use five when
an effect is under ~2× the sd.

## Durable comparison points

All rows are `RefUNet`, HF14, threshold 0.35. The sampler column is load-bearing.

| Training data | sampler | seeds | HF14 mIoU |
|---|---|---|---:|
| generated 28 only (504 records) | old | 3 | 0.2608 ± 0.020 |
| real 28 only, matched count | old | 3 draws | 0.2777 ± 0.030 |
| real 86 only (1,548 records) | old | 1 | 0.3507 |
| synth 1600 + real 1548 (`mix3148`) | old | 3 | 0.6001 ± 0.0143 |
| synth 1600 + real 1548 + gen 504 (`mix3652`) | old | 3 | 0.6331 ± 0.0132 |
| **`mix3652`** | **fixed** | **3** | **0.6860 ± 0.0175** |
| `mix3652`, rebuilt 2026-08-12 | fixed | 3 | 0.6954 ± 0.0023 |
| locally generated synth, no confusable pairs | fixed | 3 | 0.6823 ± 0.0190 |
| locally generated synth, `--confusable-prob 1.0` | fixed | 3 | 0.5787 ± 0.0433 |

Historical, broken sampler, earlier query-model lineage: synthetic-only 0.5506,
Roboflow-only strong18 0.4646, 86 clean with live flips 0.3494, synth+Roboflow
query model 0.5886, synth+Roboflow `RefUNet` 0.6127.

### What the comparisons establish

- **The 28 generated plans are worth +0.033** on the full mix (t=2.94, p=0.043)
  for a 1.6% increase in records.
- **Per source, a generated plan ≈ a real plan** (0.2608 vs 0.2777 count-matched
  at 28 sources — inside the ~0.05 spread of small pools).
- **Resolution is worth ~0.032** (the same 28 at 640 long side: 0.2330 vs 0.2650).
- **Count dominates**: 28 → 86 real sources buys +0.073.
- The HF 20k synthetic substitution cost ~0.006, so the irrecoverable pools are
  not a loss. Generating the synthetic half locally instead costs ~0.013.
- **Deliberately confusable synthetic material pairs cost −0.104** (t=3.80,
  p=0.019, complete separation). `--confusable-prob` exists but stays off; see
  `synth_progress.md` (2026-08-12) before revisiting confusability at all.

So: generate more, and generate large. Per-image value matches real plans, and
resolution is a free choice at generation time that can never be retrofitted onto
the 640px scraped pool. Adding *records* without adding sources does nothing —
see the 2026-08-10 volume/ratio nulls in `synth_progress.md`.

## Evaluation rules

- HF `real-world-test` images are evaluation-only.
- Fixed indices: `12,16,27,7,11,25,23,1,18,2,0,3,14,24`.
- Evaluate every labelled instance as a deterministic user selection: 52 total.
- Report reference-conditioned union IoU only, with the sampler named.
- Do not report the legacy training `real_iou`, per-GT best coverage, an oracle,
  class-agnostic Mask R-CNN scores, or checkpoint ensembles as product mIoU.
- Threshold 0.35 is fixed. Do not sweep it, the inference resolution, or any
  other knob against HF14 — that is fitting the acceptance set.
- HF14 images 24, 25, and 27 carry human highlighter markup because they came
  from real markup PDFs. That is the real input distribution; keep them in, and
  treat robustness to markup as a model requirement.
