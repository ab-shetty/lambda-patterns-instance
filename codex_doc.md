# Codex Handoff

Last updated: 2026-07-22

Read `PROJECT_UNDERSTANDING.md` for task semantics and `startup.md` for complete,
copy-paste reproduction commands. `startup.md` covers the current mixed result,
the verified synthetic-only baseline, the strongest Roboflow-only baseline, and
the 86-image live-augmentation control.

## Current accepted handoff result

The best extensible single model is `RefUNet`, trained on synthetic + Roboflow:

| Item | Value |
|---|---|
| Synthetic records | 1,600 |
| Roboflow records | 1,548 (86 originals + 1,462 offline variants) |
| Actual epochs | 0 through 8 |
| Evaluation | fixed HF14, all 52 reference selections |
| Metric | reference-conditioned union IoU |
| Mask threshold | 0.35 |
| Corrected mIoU | **0.6127448856345988** |
| Checkpoint | `data/runs/ck_refunet_mix3148_w128_s31/epoch_8.pth` |
| Metrics | `data/evaluations/refunet_e8_t0.35.json` |

The active target is 0.65. It has not been achieved. The user accepted the 0.6
result as the handoff point and explicitly rejected checkpoint ensembling as the
deliverable because future Roboflow labels must extend one training run.

## Product and model semantics

A reference rectangle identifies an image-local pattern. The target is the union
of every region with that same image-local grouping ID. IDs such as `pattern1`
have no meaning across plans. Roboflow `remove` polygons are subtracted as holes.

The earlier query model emitted separate instance candidates and then grouped
them by reference similarity. Its mixed-data ceiling was 0.588568. `RefUNet`
removes that grouping bottleneck: a shared ResNet-50 extracts plan and reference
features, four multiscale conditioning blocks combine image features with the
pooled reference representation, and an FPN decoder predicts one selected union
mask. Connected-component postprocessing can expose separate user-facing
instances without using multiple checkpoints.

Relevant implementation:

- `refmask2former/ref_unet.py`: shared backbone, conditioning blocks, FPN mask.
- `scripts/train_refunet.py`: union targets, BCE + Dice training, continuation
  checkpoints, fixed HF14 diagnostic evaluation, explicit schedule length and
  optimizer reset.
- `scripts/evaluate_refunet_selection.py`: authoritative 52-selection evaluation
  for `RefUNet` checkpoints.
- `scripts/augment_local_dataset.py`: deterministic strong Roboflow variants.
- `scripts/merge_local_datasets.py`: deterministic synthetic/Roboflow merge.

## Exact training nuance

The retained run has two optimizer phases:

1. actual epoch 0, using the first epoch of a planned ten-epoch cosine schedule;
2. reload epoch-0 weights, reset the optimizer, then train actual epochs 1–8
   using the first eight epochs of a planned nine-epoch cosine schedule.

This is now represented explicitly by `--schedule-epochs` and
`--reset-optimizer`; do not approximate it with one uninterrupted command when
trying to reproduce the reported checkpoint. Full commands are in `startup.md`.

## Durable comparison evidence

| Experiment | Corrected HF14 mIoU | Meaning |
|---|---:|---|
| Synthetic-only query curriculum | 0.550613 | Verified prior baseline |
| Clean 86 Roboflow, live flip/rotation | 0.349361 | Full ten-epoch control |
| Strong18 Roboflow-only query lineage | 0.464606 | Offline diversity matters |
| Synthetic + Roboflow query model | 0.588568 | Best old single checkpoint |
| Synthetic + Roboflow direct `RefUNet` | **0.612745** | Current single-model result |

The 86-image control confirms that live orientation augmentation alone is not a
replacement for offline appearance/crop/scale/degradation diversity. The 1,548
Roboflow records still contain only 86 unique real source plans, which is the
main remaining data limitation.

## Evaluation rules

- HF `real-world-test` images are evaluation-only.
- Fixed indices: `12,16,27,7,11,25,23,1,18,2,0,3,14,24`.
- Evaluate every labelled instance as a deterministic user selection: 52 total.
- Report reference-conditioned union IoU only.
- Do not report the legacy training `real_iou`, per-GT best coverage, an oracle,
  class-agnostic Mask R-CNN scores, or checkpoint ensembles as product mIoU.
- Threshold 0.35 is globally calibrated and fixed for the current `RefUNet`.

## Generated artifacts

Datasets, checkpoints, logs, and evaluation JSON files live under git-ignored
`data/` and `logs/`. Preserve those directories between VMs for byte-identical
artifacts, or regenerate them using `startup.md`. Source and documentation are
committed; credentials are never stored.
