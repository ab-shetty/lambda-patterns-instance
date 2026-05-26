# Synth Realism Handoff

## Real goal
Make the synthetic images look as close as possible to the real-world excerpts, then use `train.py` as the test.

The user's working acceptance criteria are:
- **Primary:** synth and real should approach parity under training, meaning the synth set should behave like the real set instead of becoming much easier over time.
- **Fast read:** **3 epochs is enough** to tell whether a dataset is diverging.
- **Secondary:** `epoch0 real_iou` matters. A run that starts around `0.40` on real is materially better than the older `~0.34-0.38` regime.

Do **not** optimize fill-density as the objective by itself. It is only a guardrail.

## User constraints / preferences
- Use all available CPU for generation. This machine has **64 vCPU**; generation commands should use `--workers 64`.
- For any **500-image generation run**, pipe stdout/stderr to a plain `.log` file so the user can tail it live.
- Put review artifacts the user should inspect **inside the repo**, not under `/tmp`.

## Current high-level read
The big early gap was real: older synth data was too clean, too regular, and too neatly partitioned.

What improved transfer:
- more document-like elevation context
- more rowhouse / townhouse elevation structure
- off-center excerpt crops
- markup overlays
- less perfectly isolated regions

What the user correctly identified later:
1. synth shapes were still too regular
2. adjacent patterns were too cleanly separated instead of touching / interleaving

Those are now partially addressed in `generate_synthetic_v5.py`, but **not enough** to reach synth-real parity.

## Best runs so far

### `v53_row_500` (still the best epoch-0 transfer)
Dataset: `/tmp/synth_v53_row_500`  
Gen log: `/tmp/gen_v53_row_500.log`  
Train log: `/tmp/train_v53_row_3ep.log`

- `ep0`: synth `0.6619`, real `0.4009`, div `+0.2610`
- `ep1`: synth `0.6858`, real `0.3893`, div `+0.2965`
- `ep2`: synth `0.6950`, real `0.3815`, div `+0.3135`

Interpretation:
- best `epoch0 real_iou` seen so far
- still diverges with training

### `v54_context_500`
Dataset: `/tmp/synth_v54_context_500`  
Gen log: `/tmp/gen_v54_context_500.log`  
Train log: `/tmp/train_v54_context_3ep.log`

- `ep0`: synth `0.6348`, real `0.3767`, div `+0.2581`
- `ep1`: synth `0.6577`, real `0.3657`, div `+0.2921`
- `ep2`: synth `0.6668`, real `0.3660`, div `+0.3007`

Interpretation:
- heavier document clutter flattened divergence slightly
- but hurt early transfer

### `v55_contextlite_500`
Dataset: `/tmp/synth_v55_contextlite_500`  
Gen log: `/tmp/gen_v55_contextlite_500.log`  
Train log: `/tmp/train_v55_contextlite_3ep.log`

- `ep0`: synth `0.6695`, real `0.3623`, div `+0.3072`
- `ep1`: synth `0.6929`, real `0.3839`, div `+0.3089`
- `ep2`: synth `0.7048`, real `0.3774`, div `+0.3273`

Interpretation:
- better fill-density match did **not** translate into better parity

### `v56_interleave_500`
Dataset: `/tmp/synth_v56_interleave_500`  
Gen log: `/tmp/gen_v56_interleave_500.log`  
Train log: `/tmp/train_v56_interleave_3ep.log`

- `ep0`: synth `0.6691`, real `0.3748`, div `+0.2943`
- `ep1`: synth `0.6879`, real `0.3633`, div `+0.3246`
- `ep2`: synth `0.7001`, real `0.3900`, div `+0.3100`

Interpretation:
- first run after explicitly attacking regular shapes + clean separations
- geometry got better
- later real score improved a bit, but early transfer regressed vs `v53`

### `v57_complex_500` (latest)
Dataset: `/tmp/synth_v57_complex_500`  
Gen log: `/tmp/gen_v57_complex_500.log`  
Train log: `/tmp/train_v57_complex_3ep.log`

- `ep0`: synth `0.6856`, real `0.3804`, div `+0.3052`
- `ep1`: synth `0.7036`, real `0.3771`, div `+0.3265`
- `ep2`: synth `0.7048`, real `0.3774`, div `+0.3275`

Interpretation:
- better than `v56` at epoch 0
- still worse than `v53`
- roof-plan geometry improved visually, but did not improve training parity

## v58 appearance/cue ablations (2026-05-26) — NEW BEST + 4 negatives

All run as the standard 3-epoch parity check, `--init-from checkpoints_v51/best.pth`,
`--image-max-size 2048 --batch-size 8`. The v58 sets use the canonical recipe
(`--mode-weights 65,5,30 --dense-fill-scope instance --dense-fill-frac 0.45
--dense-fill-opacity 0.18 --markup-overlay-prob 0.35`, seed 5858).

### `v58_base` — NEW BEST epoch-0 transfer (beats v53)
Dataset: `/tmp/synth_v58_base` (canonical recipe, outlines ON, RGB, no degradation).
- `ep0`: synth `0.6819`, real **`0.4253`**, div `+0.2566`
- `ep1`: synth `0.7064`, real `0.3969`, div `+0.3095`
- `ep2`: synth `0.6970`, real `0.3964`, div `+0.3006`
This is the best `ep0 real_iou` recorded (vs v53 `0.4009`) and the new benchmark.
The plain clean recipe — no ablation — is the optimum on these axes.

### Four interventions that ALL HURT real transfer
Each isolates one variable against `v58_base` (or v53 for grayscale):
- **Grayscale** (luminance-only, both synth+real, train+infer): `ep0 real 0.3528` (−0.07).
  Once colour is gone, synth/real low-level texture stats are already close
  (edge-density 0.080 vs 0.063; flat-frac 0.69 vs 0.73), so there was little
  appearance gap left to close. Colour is NOT the divergence driver.
- **`--no-outline`** (remove the per-instance dark loop): `ep0 real 0.3678` (−0.06).
  The outline is a HELPFUL boundary cue, not a harmful synth shortcut. Hypothesis
  rejected.
- **`--realism-aug`** (train-time blur/noise/JPEG/contrast → real PDF look):
  `ep0 real 0.3602` (−0.07). Degrading synth toward real rasterization hurt;
  the model wants clean crisp synth.
- **`--split-scale 0`** (kill all interleave/bay splitting): NO-OP on instance
  count (median stayed ~10). Instance count is baked into the architectural
  decomposition (surfaces/facets), not the splitting machinery.

### Consistent lesson
Every perturbation that REMOVES a cue (colour, outlines) or ADDS noise
(degradation) makes real transfer WORSE; instance count/granularity doesn't move
it. The synth→real gap is a robust feature-space domain shift, NOT an appearance/
cue/count artifact data manipulation can erase. Stop searching the appearance/cue
axis. New code (all default-OFF, so default gen+train = the best config):
`dataset.py` `grayscale` / `realism_aug`; generator `--split-scale`; train/eval
`--grayscale` / `--realism-aug`. Real instance stats for reference: median 3
inst/img (synth ~8–11), bbox min-side med 102px@2048, area-frac p90 0.32 (synth
0.18) — real is COARSER and wider-spread than synth, opposite the earlier
"make it finer" direction.

## What changed in the generator
Main file: `generate_synthetic_v5.py`

Current meaningful changes already in the code:
- rowhouse-heavy elevation generation
- off-center elevation excerpts
- markup overlays
- document effects / partial-crop support
- review-pair generation script in `scripts/build_review_pairs.py`
- **stepped interleaving seams** for large regions
- suppression of heavy region outlines on split/interleaved pieces
- more irregular floor-plan room fills
- more complex roof-plan footprints via appended wings before facet decomposition
- per-item dense-fill bias so split regions do not become too sparse

Important recent additions:
- `_split_poly_interleaved(...)`
- `_surface_items(...)`
- `_augment_roof_wings(...)`

## Current diagnosis
The project is no longer blocked on the obvious failures like:
- perfectly regular rectangles / trapezoids everywhere
- totally separated pattern blocks

Those issues were real and are now reduced.

The remaining gap appears to be more structural:
- roof plans still do not have enough real-style internal complexity
  - too few awkward valley / ridge / pitch-break situations
  - still too wing-level and tidy
- elevations still simplify facade composition too aggressively
  - real excerpts often have messier material transitions and more asymmetric local interruptions
- fill density matters enough to avoid sparse toy regions, but it is **not** the main lever by itself

## Review folders already built
These are in-repo and clickable from JupyterLab:

- [review_pairs/v53_row](/home/ubuntu/lambda-patterns-instance/review_pairs/v53_row)
- [review_pairs/v55_contextlite](/home/ubuntu/lambda-patterns-instance/review_pairs/v55_contextlite)
- [review_pairs/v56_interleave](/home/ubuntu/lambda-patterns-instance/review_pairs/v56_interleave)
- [review_pairs/v57_complex](/home/ubuntu/lambda-patterns-instance/review_pairs/v57_complex)

Each contains 28 pair images plus `manifest.csv`.

Use these before making broad new changes. The user has already been giving useful qualitative feedback from them.

## Fill-density notes
Utility script: `scripts/ink_coverage.py`

How to interpret it:
- use it as a **sanity check**, not the target
- if median collapses too low, the synth set gets too sparse and toy-like
- if it overshoots too high, the set becomes bluntly dense and loses excerpt realism

Relevant recent coverage numbers:
- `v56` first-120 sample: `mean 0.4929`, `median 0.2984`
- `v57` first-120 sample: `mean 0.5350`, `median 0.3097`

That is enough to avoid the worst sparse failure, but still does not explain the remaining training gap.

## How to run the core loop

Reference tiles path:
`/home/ubuntu/.cache/huggingface/hub/datasets--abshetty--floz-assets/snapshots/6dfc52ececbe353f10324a761350b72d535861df/reference_tiles_curated`

Generate 500 images:
```bash
python generate_synthetic_v5.py \
  --n 500 \
  --seed 999 \
  --tiles /home/ubuntu/.cache/huggingface/hub/datasets--abshetty--floz-assets/snapshots/6dfc52ececbe353f10324a761350b72d535861df/reference_tiles_curated \
  --out /tmp/synth_new_500 \
  --workers 64 \
  --dense-fill-scope instance \
  --dense-fill-frac 0.45 \
  --dense-fill-opacity 0.18 \
  --mode-weights 65,5,30 \
  --markup-overlay-prob 0.35 \
  > /tmp/gen_new_500.log 2>&1
```

Train for the 3-epoch parity check:
```bash
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLCONFIGDIR=/tmp/matplotlib \
python train.py \
  --local-data /tmp/synth_new_500 \
  --init-from checkpoints_v51/best.pth \
  --image-max-size 2048 \
  --batch-size 8 \
  --epochs 3 \
  --num-workers 16 \
  --real-eval \
  --checkpoint-dir /tmp/ck_new_3ep \
  --log-dir /tmp/log_new_3ep \
  > /tmp/train_new_3ep.log 2>&1
```

Build side-by-side review pairs in repo:
```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python scripts/build_review_pairs.py \
  --synth-root /tmp/synth_new_500 \
  --out review_pairs/new_run \
  --cache-dir ./data
```

## What the next model should do
Benchmark to beat is now **`v58_base ep0 real_iou = 0.4253`** (clean canonical
recipe), not v53's 0.4009.

Do **not** keep searching the appearance/cue axis. The v58 ablations show
grayscale, outline-removal, and rasterization-degradation all HURT, and instance
count/granularity is a no-op. Density/document-noise sweeps already failed (v54).
That entire family is exhausted — the gap is a feature-space domain shift.

If you still want to attack the gap, the untried axes are model/training-side,
not "make the synth image prettier":
1. **Resolution / small-instance handling** — real instances are tiny
   (min-side ~46–102px @2048). Test higher `--image-max-size` or an
   FPN/query setup that localizes small instances better.
2. **A few-shot real fine-tune / domain-adaptation** head (the 28 real plans are
   the eval set, so you'd need *new* real labels — but even a handful could test
   whether a thin adaptation layer closes most of the gap).
3. **Different/again-pretrained backbone** features (the ImageNet ResNet may be a
   poor prior for line-art plans).

If a new run improves visuals but cannot beat `v58_base` on epoch-0 real transfer,
it is solving the wrong axis. Use the clean `v58_base` recipe as the data default.
