# Startup Guide

Read `PROJECT_UNDERSTANDING.md` first. The task is reference-conditioned region
selection: a user selects a rectangle inside a material pattern and the model
returns every matching region in the same plan. Pattern IDs are image-local.
Roboflow `remove` polygons are holes, never foreground classes.

## Current reproducible result

The best extensible **single checkpoint** is the direct `RefUNet` model:

- training: 1,600 synthetic images + 1,548 Roboflow-derived images;
- unique real sources: 86 (the other 1,462 are deterministic offline variants);
- duration: actual epochs 0 through 8, within the ten-epoch limit;
- evaluation: fixed HF14, 14 plans and all 52 reference selections;
- calibrated mask threshold: `0.35`;
- corrected reference-conditioned union mIoU: **`0.6127448856345988`**;
- checkpoint: `data/runs/ck_refunet_mix3148_w128_s31/epoch_8.pth`;
- metrics: `data/evaluations/refunet_e8_t0.35.json`.

The active target is `>= 0.65`; it has not yet been reached. Do not report the
training-time legacy `real_iou`, an oracle, a class-agnostic score, or an
ensemble as the current result.

`RefUNet` predicts the selected pattern union directly. If the product needs
separate instances, split the binary mask into connected components (preserving
holes) after inference. Training remains one checkpoint that can be continued
when more Roboflow labels arrive.

## Environment and data acquisition

Credentials must be available as `HF_TOKEN` and `ROBOFLOW_API_KEY`; never print
their values.

```bash
pip install -r requirements.txt
pip install roboflow
```

The winning VM used Python 3.10, torch 2.7.0, torchvision 0.22.0, OpenCV 4.10.0,
Pillow 12.3.0, and datasets 5.0.0 on one NVIDIA GH200. Minor nondeterminism can
cause small last-decimal changes.

Cache the evaluation dataset (never put these HF images into training):

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("abshetty/floz-synth-v5", "real-world-test", split="test",
                  cache_dir="./data")
print(len(ds))  # 28
PY
```

Download and convert Roboflow version 2:

```bash
python - <<'PY'
import os
from roboflow import Roboflow
project = (Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
           .workspace("perceive-ai").project("floz-real-pool"))
print(project.versions())
project.version(2).download(
    "coco-segmentation",
    location="data/roboflow/floz-real-pool-v2-raw",
    overwrite=True,
)
PY

python scripts/roboflow_to_local.py \
  --coco data/roboflow/floz-real-pool-v2-raw/train/_annotations.coco.json \
  --img-dir data/roboflow/floz-real-pool-v2-raw/train \
  --out data/roboflow/floz-real-pool-v2-clean
```

Expected version-2 conversion: 86 usable images, 282 pattern instances, 127
`remove` polygons, and 129 hole attachments. One degenerate pattern and two
unlabelled images are skipped. New Roboflow versions will legitimately change
these counts; inspect the conversion summary rather than forcing old counts.

## Rebuild the exact mixed dataset

The canonical synthetic input is `data/synthetic/toparea1600_balanced`. If it
is not preserved from the previous VM, recreate it from the existing generator
pools:

```bash
python scripts/select_toparea_local.py \
  --sources data/synthetic/faintcad2500 data/synthetic/cadneg2500 \
  --out data/synthetic/toparea1600_balanced --n 1600 \
  --mode-quotas elevation=889,roof_plan=540,freeform=171
```

Create 17 deterministic strong offline variants per Roboflow source. The clean
original is retained, producing 18 views per source and 1,548 records total:

```bash
python scripts/augment_local_dataset.py \
  --src data/roboflow/floz-real-pool-v2-clean \
  --out data/roboflow/floz-real-pool-v2-strong18 \
  --aug-per-scene 17 --strong --seed 5858
```

Merge synthetic first and Roboflow second. The merge renames records
deterministically to `item_000000` through `item_003147`:

```bash
python scripts/merge_local_datasets.py \
  --sources data/synthetic/toparea1600_balanced \
            data/roboflow/floz-real-pool-v2-strong18 \
  --out data/mixed/toparea1600_rfstrong1548
```

Verify that both `images/` and `annotations/` contain exactly 3,148 files.
Everything under `data/` is intentionally git-ignored.

## Reproduce the 0.6127448856 checkpoint

The observed run had two deliberate optimizer phases. Phase A trains actual
epoch 0 with a cosine schedule planned for ten epochs:

```bash
PYTHONPATH=. python scripts/train_refunet.py \
  --local-data data/mixed/toparea1600_rfstrong1548 \
  --checkpoint-dir data/runs/ck_refunet_mix3148_w128_s31 \
  --epochs 1 --schedule-epochs 10 \
  --batch-size 8 --num-workers 64 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --seed 31
```

Phase B reloads epoch 0 weights, intentionally resets the optimizer, and trains
actual epochs 1 through 8 using the first eight epochs of a nine-epoch cosine
schedule. This exactly encodes the schedule that produced the retained result:

```bash
PYTHONPATH=. python scripts/train_refunet.py \
  --local-data data/mixed/toparea1600_rfstrong1548 \
  --checkpoint-dir data/runs/ck_refunet_mix3148_w128_s31 \
  --epochs 8 --schedule-epochs 9 \
  --batch-size 8 --num-workers 64 --prefetch-factor 4 \
  --image-max-size 1280 --ref-size 224 --width 128 \
  --lr 2e-4 --backbone-lr-mult 0.1 --train-split 0.99 \
  --domain-random --seed 31 --reset-optimizer \
  --init-from data/runs/ck_refunet_mix3148_w128_s31/epoch_0.pth
```

The training script writes model, optimizer, scheduler, actual epoch, arguments,
and HF14 diagnostics into each checkpoint. It also retains `best_real.pth`, but
the acceptance score must be recomputed with the authoritative evaluator and
the calibrated threshold:

```bash
PYTHONPATH=. python scripts/evaluate_refunet_selection.py \
  --checkpoint data/runs/ck_refunet_mix3148_w128_s31/epoch_8.pth \
  --indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 \
  --image-max-size 1280 --ref-size 224 --mask-thresh 0.35 \
  --metrics-out data/evaluations/refunet_e8_t0.35.json
```

Expected output: 14 images, 52 reference selections, and mean IoU
`0.6127448856345988`.

## Reproduce the synthetic-only 0.5506125168 baseline

Create the 750-image curriculum subset as well as the 1,600-image set described
above:

```bash
python scripts/select_toparea_local.py \
  --sources data/synthetic/faintcad2500 data/synthetic/cadneg2500 \
  --out data/synthetic/toparea750_balanced --n 750 \
  --mode-quotas elevation=476,roof_plan=196,freeform=78
```

Phase 1 was configured for ten epochs but intentionally stopped after
`epoch_4.pth` (five completed epochs):

```bash
python train.py \
  --local-data data/synthetic/toparea750_balanced \
  --image-max-size 1280 --batch-size 4 --grad-accum 1 --epochs 10 \
  --save-every-epoch --num-workers 64 --prefetch-factor 4 \
  --real-eval --real-eval-indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 \
  --domain-random --num-queries 200 \
  --mask-weight 10 --dice-weight 10 --ref-weight 2 \
  --ref-siamese-backbone --ref-siamese-level res3+res5 \
  --eos-coef 0.03 --seed 0 \
  --checkpoint-dir data/runs/ck_toparea750_hybrid_1280_s0 \
  --log-dir data/runs/tb_toparea750_hybrid_1280_s0
```

Phase 2 performs five reference-only hard-ranking epochs. Its `epoch_0.pth` is
the sixth actual epoch and the passing checkpoint:

```bash
python train.py \
  --local-data data/synthetic/toparea1600_balanced \
  --image-max-size 1280 --batch-size 8 --grad-accum 1 --epochs 5 \
  --save-every-epoch --num-workers 64 --prefetch-factor 4 \
  --real-eval --real-eval-indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 \
  --domain-random --num-queries 200 \
  --mask-weight 10 --dice-weight 10 --ref-weight 2 \
  --ref-ranking-margin 1.0 \
  --ref-siamese-backbone --ref-siamese-level res3+res5 \
  --reference-only --eos-coef 0.03 --lr 1e-4 --seed 0 \
  --init-from data/runs/ck_toparea750_hybrid_1280_s0/epoch_4.pth \
  --checkpoint-dir data/runs/ck_stage750e4_rank1_refonly5 \
  --log-dir data/runs/tb_stage750e4_rank1_refonly5

PYTHONPATH=. python scripts/evaluate_reference_selection.py \
  --checkpoint data/runs/ck_stage750e4_rank1_refonly5/epoch_0.pth \
  --image-max-size 1280 --score-thresh 0.6 --mask-thresh 0.5 \
  --match-margin 0.1 \
  --metrics-out data/evaluations/synth_only_0550613.json
```

Expected corrected HF14 mIoU: `0.5506125168094088`.

## Reproduce the strongest Roboflow-only 0.4646064981 baseline

This lineage uses `floz-real-pool-v2-strong18` and no synthetic images. Phase 1
was configured for ten epochs at 2048px and stopped after `epoch_2.pth`:

```bash
python train.py \
  --local-data data/roboflow/floz-real-pool-v2-strong18 \
  --image-max-size 2048 --batch-size 1 --grad-accum 4 --epochs 10 \
  --save-every-epoch --num-workers 24 --prefetch-factor 4 \
  --train-split 0.99 --real-eval \
  --real-eval-indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 \
  --num-queries 200 --mask-weight 10 --dice-weight 10 --ref-weight 2 \
  --ref-siamese-backbone --ref-siamese-level res3+res5 \
  --eos-coef 0.03 --lr 1e-4 --seed 7 \
  --checkpoint-dir data/runs/ck_rf_strong18_bs1_2048_s7 \
  --log-dir data/runs/tb_rf_strong18_bs1_2048_s7
```

Phase 2 warm-starts phase-one epoch 2, emphasizes mask quality, and was stopped
after its first checkpoint:

```bash
python train.py \
  --local-data data/roboflow/floz-real-pool-v2-strong18 \
  --image-max-size 2048 --batch-size 1 --grad-accum 4 --epochs 5 \
  --save-every-epoch --num-workers 24 --prefetch-factor 4 \
  --train-split 0.99 --real-eval \
  --real-eval-indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 \
  --num-queries 200 --mask-weight 20 --dice-weight 20 --ref-weight 0 \
  --ref-siamese-backbone --ref-siamese-level res3+res5 \
  --eos-coef 0.03 --lr 5e-5 --seed 11 \
  --init-from data/runs/ck_rf_strong18_bs1_2048_s7/epoch_2.pth \
  --checkpoint-dir data/runs/ck_rf_2048_maskfocus_from_e2_s11 \
  --log-dir data/runs/tb_rf_2048_maskfocus_from_e2_s11

PYTHONPATH=. python scripts/evaluate_reference_selection.py \
  --checkpoint data/runs/ck_rf_2048_maskfocus_from_e2_s11/epoch_0.pth \
  --image-max-size 2048 --score-thresh 0.3 --mask-thresh 0.05 \
  --match-margin 0.1 \
  --metrics-out data/evaluations/roboflow_only_0464606.json
```

Expected corrected HF14 mIoU: `0.4646064981`.

For the separate 86-original/live-augmentation control, use the clean dataset,
ten full epochs, and no `--domain-random`. `InstanceSegDataset` supplies live
horizontal/vertical flips and quarter-turn rotations:

```bash
python train.py \
  --local-data data/roboflow/floz-real-pool-v2-clean \
  --image-max-size 1280 --batch-size 4 --grad-accum 1 --epochs 10 \
  --save-every-epoch --num-workers 64 --prefetch-factor 4 \
  --train-split 0.99 --real-eval \
  --real-eval-indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 \
  --num-queries 200 --mask-weight 10 --dice-weight 10 --ref-weight 2 \
  --ref-ranking-margin 1.0 \
  --ref-siamese-backbone --ref-siamese-level res3+res5 \
  --eos-coef 0.03 --lr 1e-4 --seed 40 \
  --checkpoint-dir data/runs/ck_rf_clean86_live_fliprot_s40 \
  --log-dir data/runs/tb_rf_clean86_live_fliprot_s40
```

The best checkpoint is epoch 4. At 2048px with score `0.3`, mask `0.05`, and
match margin `0.1`, it scores `0.3493612685`.

## Durable comparison points

| Training data / model | Corrected HF14 mIoU |
|---|---:|
| Synthetic-only query model | 0.550613 |
| 86 clean Roboflow images, live flips/rotations only | 0.349361 |
| 1,548 strong Roboflow views, query model | 0.464606 |
| Synthetic + Roboflow query model | 0.588568 |
| Synthetic + Roboflow direct `RefUNet` | **0.612745** |

The 86-image live-augmentation control completed all ten epochs; its best point
was epoch 4. Offline augmentation adds substantial appearance, crop, scale, and
degradation diversity beyond flips and rotations alone.
