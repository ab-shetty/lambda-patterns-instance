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

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
./run_training.sh                      # sensible defaults, single GPU
# or
python train.py --image-max-size 1024 --batch-size 8 --epochs 50 --num-workers 8
```

Quick local run on a slice of the data:

```bash
python train.py --max-records 500 --image-max-size 512 --batch-size 4 --epochs 5
```

Multi-GPU (opt-in) with `--data-parallel`. Final training targets a single GH200.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pth
```

Reports, for the reference-conditioned task, mean matched IoU, precision/recall/F1
at IoU≥0.5, reference-match accuracy, and class-agnostic detection recall.

## Visualize

```bash
python visualize.py --checkpoint checkpoints/best.pth --num-images 4
```

Saves `predictions.png`: image · reference · GT target instances · predicted matches.

## Key CLI args (`train.py`)

| Arg | Default | Description |
|-----|---------|-------------|
| `--hf-repo` | `abshetty/floz-synth-v5` | HuggingFace dataset repo |
| `--image-max-size` | 1024 | Longest side after aspect-preserving resize |
| `--ref-size` | 224 | Reference patch size |
| `--batch-size` | 8 | Batch size |
| `--epochs` | 50 | Epochs |
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
                    score_thresh=0.5, match_thresh=0.5)
# out[i] -> {"masks": [n, H, W] bool, "scores": [n], "match_sim": [n]}
```
Instances kept = foreground score > `score_thresh` **and** cosine similarity to
the reference > `match_thresh`.
