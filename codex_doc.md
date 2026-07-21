# Codex Handoff

Last updated: 2026-07-21

Read [`PROJECT_UNDERSTANDING.md`](PROJECT_UNDERSTANDING.md) for product semantics
and [`startup.md`](startup.md) for setup and exact reproduction commands.

## Current result

The active synthetic-data goal is complete.

- Training data: 1,600 unique, fully synthetic images.
- Training duration: exactly ten actual epochs in a two-phase curriculum.
- Evaluation: fixed HF14 holdout, 14 real plans and 52 reference selections.
- Acceptance metric: reference-conditioned union IoU.
- Verified score: **0.5506125168** (target `>= 0.55`).
- Passing checkpoint:
  `data/runs/ck_stage750e4_rank1_refonly5/epoch_0.pth`.
- Metrics and visual audit:
  `data/evaluations/verified_rank1_hf14/`.

The passing checkpoint occurs at the sixth actual training epoch. The complete
lineage continued through ten epochs, satisfying the requirement that the score
may be reached at any point during a ten-epoch run.

## Product semantics

The product behaves like reference-conditioned **instance segmentation**:

1. A user selects a rectangle inside a material or hatch pattern.
2. The model predicts separate candidate instance masks.
3. Each candidate receives an object score and a reference-match score.
4. Matching instances are returned separately, with their own masks and scores.

Pattern names such as `pattern1` are local to one image and have no meaning
across images. The evaluator unions matching instances only to compute a strict
pixel IoU that penalizes both missed instances and false selections. Do not turn
the user-facing result into one semantic mask.

Do not cite class-agnostic GT-best coverage, oracle query selection, Mask R-CNN
coverage, or the training-time `real_iou` proxy as product mIoU. Only
`scripts/evaluate_reference_selection.py` implements the acceptance test.

## Synthetic dataset

The canonical full set is:

`data/synthetic/toparea1600_balanced`

- 1,600 images
- 889 elevation
- 540 roof plan
- 171 freeform
- labelled-area mean: `0.394379`
- labelled-area median: `0.292692`

The first five curriculum epochs use the strict high-area subset:

`data/synthetic/toparea750_balanced`

- 750 images
- 476 elevation
- 196 roof plan
- 78 freeform

An audit confirmed that all 750 source identities are contained in the 1,600
set. Both are selected from `faintcad2500` and `cadneg2500`; no real image is in
either training directory. Their selection manifests are stored inside the
dataset directories (which are git-ignored).

## Model and training

The successful model remains class-agnostic Mask2Former-style instance
segmentation with 200 queries. Reference matching uses a shared Siamese image
backbone and pools both fine `res3` and coarse `res5` features inside each
predicted mask.

Training is a curriculum over the same 1,600-image synthetic universe:

- Phase 1: five joint segmentation/reference epochs on the 750-image high-area
  subset, 1280 px, batch 4.
- Phase 2: five reference-only epochs on all 1,600 images, 1280 px, batch 8.
- Phase 2 freezes segmentation and adds a hardest-positive versus
  hardest-negative soft ranking loss with margin `1.0` logit units.

The ranking loss was the final necessary improvement. It optimizes the failure
that average BCE/AUC concealed: one difficult false-positive or false-negative
instance can substantially reduce union IoU.

Inference for the passing checkpoint uses:

- object score threshold: `0.6`
- relative reference-match margin: `0.1`
- mask threshold: `0.5`

The relative margin is label-free at inference: retain object queries whose
similarity is within `0.1` of the best query for that reference.

## Relevant implementation changes

- `refmask2former/model.py`
  - shared-backbone Siamese instance/reference embeddings
  - selectable `res3+res5` multi-scale pooling
- `refmask2former/criterion.py`
  - optional hard reference-ranking loss
- `refmask2former/dataset.py`
  - deterministic evaluation references
  - optional repeated-reference sampling (not used by the winning recipe)
- `train.py`
  - reference LR grouping, staged freezing, per-epoch checkpoints, ranking flags
- `evaluate.py`
  - reconstructs the saved reference architecture from checkpoint arguments
- `scripts/evaluate_reference_selection.py`
  - authoritative HF14 evaluation and visual audit
  - relative-margin and other diagnostic selection modes
- `scripts/select_toparea_local.py`
  - exact balanced dataset selection and manifests

## Data acquisition

Hugging Face `abshetty/floz-synth-v5`, config `real-world-test`, contains 28
real evaluation images. The fixed HF14 indices are:

`12,16,27,7,11,25,23,1,18,2,0,3,14,24`

Roboflow `perceive-ai/floz-real-pool` version 2 was converted into 86 usable
images, 282 pattern instances, and 127 `remove` polygons attached as 129 holes.
`remove` is never a class: subtract it from surrounding pattern polygons. The
Roboflow data is available for other work but was not used by the successful
fully synthetic training lineage.

## Durable experiment evidence

- The original 500-image Siamese run scored `0.457807` corrected IoU.
- Multi-scale `res3+res5` matching on 750 images raised this to `0.519512`.
- Expanding ordinary joint training to 1,600 images improved mask coverage but
  peaked at only `0.504098` corrected IoU.
- Hard-ranking margin `2.0` reached `0.536815` but did not cross the gate.
- Hard-ranking margin `1.0` reached `0.550613` and passed.
- Reference feature means/variances, a separate frozen ImageNet texture
  backbone, whole-instance ROI matching, fixed-patch ROI matching, pairwise MLP
  matching, all-repeated datasets, and repeated-reference oversampling did not
  beat the multi-scale hard-ranking recipe.
- Oracle query selection reached about `0.61`, proving mask coverage was already
  sufficient and that reference grouping was the decisive bottleneck.

## Repository state

Source and documentation should be committed together. Generated datasets,
checkpoints, logs, and evaluation images remain under `data/` or `logs/` and are
intentionally git-ignored. Preserve those directories when moving to another VM
or regenerate them using [`startup.md`](startup.md).
