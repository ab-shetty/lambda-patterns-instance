# RefMask2Former — Reference-Conditioned Instance Segmentation

Instance-segmentation successor to the reference-based **semantic** pattern
segmentation model. Where the old model merged every matching region into a
single binary mask, this one separates each pattern occurrence into its own
instance — while preserving the product's core behavior: a user selects a
reference patch, and the model returns the instances of *that* pattern.

## Why this design

The patterns are **open-vocabulary**: there is no fixed class list, because the
pattern is whatever the user selects. In the training data (`floz-synth-v5`)
`category_id`s are per-image and arbitrary, so a standard fixed-class detector
is not applicable. Instead the model is **class-agnostic + reference-matching**:

1. A Mask2Former-style query decoder detects **every distinct pattern instance**
   (single foreground class "pattern").
2. A **reference-matching head** produces a normalized embedding per instance.
   The user's reference patch is encoded into the same space; cosine similarity
   selects which detected instances belong to the selected pattern.

This decouples the well-posed problem ("find all instances") from the
open-vocabulary problem ("which match this reference"), and means new/unseen
patterns work at inference without retraining.

## Architecture (`refmask2former/`)

```
image ─▶ ResNet50 backbone ─▶ FPN pixel decoder ─┬─▶ mask_features (stride 4)
                                                 └─▶ multi-scale feats (1/8,1/16,1/32)
                                                          │
              learnable queries ─▶ masked-attention transformer decoder
                                                          │
                            per query: ┌ class logit (pattern / no-object)
                                        ├ mask embedding · mask_features ─▶ mask
                                        └ reference embedding (normalized)
reference patch ─▶ ResNet50 reference encoder ─▶ reference embedding (normalized)
```

| File | Role |
|------|------|
| `backbone.py` | ResNet50, multi-scale features (res2–res5) |
| `pixel_decoder.py` | FPN pixel decoder → mask features + transformer features |
| `transformer_decoder.py` | Mask2Former masked-attention query decoder + 3 heads |
| `reference_encoder.py` | ResNet50 → normalized reference embedding |
| `model.py` | Assembles everything; `forward` + `predict` |
| `matcher.py` | Hungarian matching (class + mask BCE + dice, point-sampled) |
| `criterion.py` | Set loss: class CE + mask BCE/Dice + reference-match BCE, deep supervision |
| `dataset.py` | Parquet ingest, instance masks, reference sampling, padding collate |

Notes:
- **No square resize.** Images keep their aspect ratio (longest side →
  `--image-max-size`); batches are padded to a common size divisible by 32 and a
  pixel mask is carried through attention and the losses so padding is ignored.
- The pixel decoder uses the FPN variant (not multi-scale deformable attention),
  so the whole model uses only standard PyTorch ops — no custom CUDA kernels.

## Dataset

[`abshetty/floz-synth-v5`](https://huggingface.co/datasets/abshetty/floz-synth-v5)
— 20k synthetic architectural images, loaded from the **parquet** files via
`datasets.load_dataset`. Each annotation is one instance (polygon with holes,
`category_name` = `pattern1..N` *local to the image*, `reference_tile`, bbox).
The loader picks a target pattern per image, crops a reference patch from one of
its instances, and labels every instance match/no-match against it.

### Real-world test set

A second config, `real-world-test` (split `test`, 28 samples), holds
**hand-annotated real architectural plans** — not synthetic. It measures
generalization beyond the generator. It is schema-compatible with training and
flows through the same `InstanceSegDataset`, with two differences the loader
handles transparently:

- `annotations` is stored as a parsed list of structs (training stores it as a
  JSON string); `dataset.py` accepts either.
- `mode` is `"unknown"` (real plans aren't classified elevation/floorplan). The
  model doesn't use this field.

```python
load_dataset("abshetty/floz-synth-v5", "real-world-test", split="test")  # 28 rows
```

The default config still returns exactly the 20k train rows and never picks up
the test file.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --image-max-size 2048 --epochs 15 --num-workers 16
```

**Default recipe is `--batch-size 1 --grad-accum 4 --freeze-backbone-bn`** (all on by
default as of 2026-06-22). This is the confirmed-best regime: it lifts held-out
real_iou **+0.03–0.08 vs the old batch-8 setup** (t=4.4, replicated at 1024 and
2048; see `synth_progress.md`). The gain is a regularization effect, not just a
12GB-GPU memory workaround — keep it on a big GPU too. On large GPUs you can raise
throughput with a bigger micro-batch (`--batch-size 4 --grad-accum 1`) at a small
quality cost, or pass `--no-freeze-backbone-bn` to revert backbone BN behaviour.

Quick local run on a slice of the data:

```bash
python train.py --max-records 500 --image-max-size 512 --batch-size 4 --epochs 5
```

Multi-GPU (opt-in) with `--data-parallel`. Final training targets a single GH200.

**Mixed precision.** Training uses **bf16 autocast** by default on CUDA (disable
with `--no-amp`). bf16 — not fp16 — because it keeps fp32's exponent range, so no
`GradScaler` is needed and the loss/matcher stay stable (autocast auto-promotes
BCE/cross-entropy/softmax to fp32). Quality is within run-to-run noise of fp32;
the win is ~halved activation memory and faster tensor-core matmuls.

**Memory.** In fp32, batch 8 at 2048px OOMs on a 94.5GB GH200 (~1.5GB short).
bf16 roughly halves activation memory, so batch 8 fits; the script also exports
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to curb fragmentation. If you
still hit an OOM, drop to `--batch-size 4`.

**Throughput.** ~42 min/epoch (fp32, batch 4) on the full 20k set → a 15-epoch
run is ~10–11 hours; bf16 + the deeper input pipeline (`--num-workers 16
--prefetch-factor 4`, persistent workers) brings it under that. At 2048px the
dataloader can bottleneck the GPU (util swings rather than pinning at 100%), so
watch `nvidia-smi` — if util sags, raise `--num-workers`.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pth
```

Reports, for the reference-conditioned task, mean matched IoU, precision/recall/F1
at IoU≥0.5, reference-match accuracy, and class-agnostic detection recall.

To measure generalization on the held-out real plans (the whole `real-world-test`
config is the eval set — no train/val split):

```bash
python evaluate_real_world.py --checkpoint checkpoints/best.pth
```

Same metrics as above. Reference crops are seeded (`--seed`) so the 28-sample
numbers are reproducible run to run.

## Visualize

```bash
python visualize.py --checkpoint checkpoints/best.pth --num-images 4
```

Saves `predictions.png`: image · reference · GT target instances · predicted matches.

## Key CLI args (`train.py`)

| Arg | Default | Description |
|-----|---------|-------------|
| `--hf-repo` | `abshetty/floz-synth-v5` | HuggingFace dataset repo |
| `--image-max-size` | 1024 | Longest side after aspect-preserving resize (`run_training.sh` uses 2048) |
| `--ref-size` | 224 | Reference patch size |
| `--batch-size` | 1 | Micro-batch size (confirmed-best regime; effective batch = batch-size × grad-accum) |
| `--grad-accum` | 4 | Grad accumulation steps before optimizer step |
| `--freeze-backbone-bn` | on | Freeze backbone BatchNorm; `--no-freeze-backbone-bn` to disable |
| `--epochs` | 50 | Epochs (`run_training.sh` uses 15) |
| `--no-amp` | off | Disable bf16 autocast (train in full fp32) |
| `--num-workers` | 4 | Dataloader workers (`run_training.sh` uses 16) |
| `--prefetch-factor` | 4 | Batches prefetched per worker (num-workers > 0) |
| `--lr` | 1e-4 | Base LR (backbone & reference encoder use ×`--backbone-lr-mult`) |
| `--num-queries` | 100 | Object queries (data has ≤9 instances/image) |
| `--dec-layers` | 9 | Transformer decoder layers |
| `--ref-dim` | 128 | Reference embedding dim |
| `--{class,mask,dice,ref}-weight` | 2/5/5/2 | Loss weights |
| `--max-records` | 0 | Limit records (0 = all) for quick runs |
| `--data-parallel` | off | Wrap in `nn.DataParallel` across visible GPUs |

## Inference sketch

```python
out = model.predict(images, pixel_mask, reference,
                    score_thresh=0.5, match_thresh=0.0)
# out[i] -> {"masks": [n, H, W] bool, "scores": [n], "match_sim": [n]}
```
Instances kept = foreground score > `score_thresh` **and** cosine similarity to
the reference > `match_thresh`. `match_thresh` defaults to **0.0**: the
embeddings are normalized, true-match sims center near +0.32 and non-match near
−0.43 (separability AUC ~0.99), so 0.0 is the natural boundary. A higher cutoff
like 0.5 sits *above* the match cluster and silently drops most true matches.
