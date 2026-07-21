# Startup Guide

Read `PROJECT_UNDERSTANDING.md` first. The product is reference-conditioned
instance selection: a user selects a rectangle inside a material pattern, and
the model must return each matching region as a separate instance mask. Pattern
IDs have no meaning across images. The masks are unioned only for the acceptance
metric; do not collapse the user-facing output into semantic segmentation.

The acceptance metric is reference-conditioned union IoU on the fixed HF14
holdout. The current verified result is `0.550613` (target: `>= 0.55`) during a
completed 10-epoch, synthetic-only curriculum. Do not treat class-agnostic,
per-GT-best, oracle, or Mask R-CNN coverage scores as product performance.

## Fresh VM setup

API credentials must be present as `HF_TOKEN` and `ROBOFLOW_API_KEY`. Never print
their values.

```bash
pip install -r requirements.txt
pip install roboflow
```

Pull the Hugging Face real evaluation set into the local cache:

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset(
    "abshetty/floz-synth-v5",
    "real-world-test",
    split="test",
    cache_dir="./data/huggingface",
)
print(f"downloaded {len(ds)} HF real images")  # expected: 28
PY
```

Download the latest generated Roboflow version as COCO instance segmentation.
Version 2 is the current known version; check `project.versions()` before choosing
a version on a later VM.

```bash
python - <<'PY'
import os
from roboflow import Roboflow

project = (Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
           .workspace("perceive-ai")
           .project("floz-real-pool"))
print(project.versions())
project.version(2).download(
    "coco-segmentation",
    location="data/roboflow/floz-real-pool-v2-raw",
    overwrite=True,
)
PY
```

Convert the raw Roboflow export into the repository's local-data format:

```bash
python scripts/roboflow_to_local.py \
  --coco data/roboflow/floz-real-pool-v2-raw/train/_annotations.coco.json \
  --img-dir data/roboflow/floz-real-pool-v2-raw/train \
  --out data/roboflow/floz-real-pool-v2-clean
```

Roboflow `remove` annotations are not classes. Each is subtracted from every
surrounding pattern as a polygon hole. Keep pattern names (`pattern1`,
`pattern2`, etc.) because they identify matching regions within an image. Never
include `remove` as a training instance.

For version 2, the expected conversion is 86 usable images, 282 instances, 127
remove polygons, and 129 hole attachments. One degenerate pattern and two
unlabelled images are skipped. Investigate unexpected count changes before
training; legitimate new labels or a newer generated version will change them.

The cleaned pool is accepted by `--local-data` and by
`scripts/build_mix.py --real-extra-dir`. Keep the raw export unchanged and write
derived data separately. Everything under `data/` is git-ignored.

Before continuing experiments, read the July 16 updates at the top of
`codex_doc.md` and `synth_progress.md`. Historical `0.5+` results mostly use the
old proxy metric; use `scripts/evaluate_reference_selection.py` and its visual
audit for the actual task.

## Reproduce the verified 1,600-image result

The full dataset is `data/synthetic/toparea1600_balanced`: 1,600 generated
images (889 elevation, 540 roof, 171 freeform) selected from `faintcad2500` and
`cadneg2500`. Its high-area curriculum subset is
`data/synthetic/toparea750_balanced` (476 elevation, 196 roof, 78 freeform).
Every subset image comes from the same 1,600-image source pool, so the training
dataset remains fully synthetic and contains 1,600 unique images. Data is
ignored by git; retain/copy these directories between VMs or regenerate them:

```bash
python scripts/select_toparea_local.py \
  --sources data/synthetic/faintcad2500 data/synthetic/cadneg2500 \
  --out data/synthetic/toparea750_balanced --n 750 \
  --mode-quotas elevation=476,roof_plan=196,freeform=78

python scripts/select_toparea_local.py \
  --sources data/synthetic/faintcad2500 data/synthetic/cadneg2500 \
  --out data/synthetic/toparea1600_balanced --n 1600 \
  --mode-quotas elevation=889,roof_plan=540,freeform=171
```

Training is a ten-epoch curriculum: five joint instance/matcher epochs on the
high-area subset, followed by five reference-only hard-ranking epochs on all
1,600 images. In phase one the LR schedule is intentionally configured for ten
epochs; stop after `epoch_4.pth` is written.

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

Then run the remaining five epochs:

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
```

Evaluate the saved real-validation checkpoint with the calibrated relative
similarity rule:

```bash
PYTHONPATH=. python scripts/evaluate_reference_selection.py \
  --checkpoint data/runs/ck_stage750e4_rank1_refonly5/epoch_0.pth \
  --image-max-size 1280 --score-thresh 0.6 --match-margin 0.1 \
  --out data/evaluations/verified_rank1_hf14
```

The passing checkpoint is phase-two epoch 0 (the sixth actual training epoch)
and scores `0.5506125168` over all 52 reference selections on HF14. The complete
two-phase run finishes ten actual epochs. `--match-margin 0.1` is a label-free,
per-reference relative threshold; it does not use GT at inference.
