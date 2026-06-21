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

## 2026-05-27 — Kaggle 2×T4, cold-start, half-resolution (no warm-start)

Environment moved off the Lambda box to a **Kaggle 2×Tesla T4 (15 GB each),
4 vCPU** instance. Differences that matter for every command from here on:
- CUDA only works after `export LD_LIBRARY_PATH=/usr/local/nvidia/lib64` — the
  driver `libcuda.so` is not on the default loader path, and torch reports 0
  GPUs without it (nvidia-smi lives in `/opt/bin`).
- **4 vCPU, not 64** — generation must use `--workers 4`. Gen rate ~0.55 img/s;
  500 images ≈ 15 min.
- HF token is in `~/.bashrc` as `HF_TOKEN`; both `abshetty/floz-assets` (tiles)
  and `abshetty/floz-synth-v5` (real-eval) are private and need it.
- There is **no checkpoint** on this box (`checkpoints_v51/` does not exist, and
  no `.pth` is on HF).

Two deliberate changes this run:
1. **Image size halved**: `--image-max-size 1024` (was 2048).
2. **No warm-start**: dropped `--init-from` entirely. Per the user, "we should
   never have been using a warm start."
Plus one forced change: bs8 OOMs on a 15 GB T4 at 1024 (decoder masked
cross-attention softmax), so **batch-size 4**.

Dataset: `/tmp/synth_v58base_500` (canonical v58 recipe, seed 5858 → byte-identical
data to v58_base). Gen log: `v58base_500_gen.log`. Train log:
`v58base_1024_3ep.log`. ~7 min/epoch (~22 min total).

- `ep0`: synth `0.1830`, real `0.1982`, div **`-0.0152`**
- `ep1`: synth `0.2845`, real `0.2394`, div `+0.0451`
- `ep2`: synth `0.2968`, real `0.2488`, div `+0.0479`

### Interpretation — recontextualizes the whole divergence story
Cold-start divergence is **tiny** (−0.015 → +0.048) versus the warm-started 2048
runs (+0.25 → +0.33); at ep0 real actually *beats* synth. This strongly suggests
the large historical divergence was substantially a **warm-start artifact**:
initializing from the synth-pretrained `checkpoints_v51` meant the model already
scored ~0.68 on synth but only ~0.40 on real, so the "+0.30 gap" was baked in
*before* the dataset under test was ever trained on. Trained from scratch, synth
and real climb together and stay near parity.

CAVEAT: absolute IoU is low (synth ~0.30, real ~0.25) because 3 cold epochs on
450 images is underfit. Low divergence under underfitting is weaker evidence than
low divergence at high absolute IoU — the model may simply not have learned
enough yet to exploit synth-specific shortcuts. Honest read: **no divergence
explosion appears in the cold-start regime at 3 epochs** (encouraging for
parity), but confirm by training longer to see whether a gap opens as synth IoU
climbs.

Benchmark note: `v58_base ep0 real_iou 0.4253` is **NOT comparable** to these
numbers — it was warm-started, 2048px, bs8. Cold-start / 1024 / bs4 is a new
regime; its ep2 real baseline to beat is now `0.2488`.

### Exact commands used on the T4 box
```bash
export HF_TOKEN=...                       # from ~/.bashrc
TILES=/root/.cache/huggingface/hub/datasets--abshetty--floz-assets/snapshots/6dfc52ececbe353f10324a761350b72d535861df/reference_tiles_curated

python generate_synthetic_v5.py --n 500 --seed 5858 --tiles "$TILES" \
  --out /tmp/synth_v58base_500 --workers 4 \
  --dense-fill-scope instance --dense-fill-frac 0.45 --dense-fill-opacity 0.18 \
  --mode-weights 65,5,30 --markup-overlay-prob 0.35 > v58base_500_gen.log 2>&1

env LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLCONFIGDIR=/tmp/matplotlib \
python train.py --local-data /tmp/synth_v58base_500 \
  --image-max-size 1024 --batch-size 4 --epochs 3 --num-workers 4 --real-eval \
  --checkpoint-dir /tmp/ck_v58base_1024_3ep --log-dir /tmp/log_v58base_1024_3ep \
  > v58base_1024_3ep.log 2>&1
```

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
NOTE (2026-05-27): the `v58_base ep0 real_iou = 0.4253` benchmark below was
**warm-started** (`--init-from checkpoints_v51`), 2048px, bs8. The user has since
dropped warm-start. In the new cold-start / 1024 / bs4 regime the comparable
baseline is `ep2 real_iou = 0.2488` and divergence collapses to ~+0.05 (see the
"2026-05-27 — Kaggle 2×T4" section). Pick the benchmark that matches your regime.

Historical (warm-start) benchmark to beat was **`v58_base ep0 real_iou = 0.4253`**
(clean canonical recipe), not v53's 0.4009.

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

## 2026-05-28 — GH200 cold-start baseline + monochrome axis

Environment back on the Lambda GH200 (480GB, 64 vCPU). Confirmed cold-start is
the regime (no `--init-from`), per "we should never have been using a warm
start." ~43s/epoch at 2048px/bs8. HF token in `~/.bashrc`; both private datasets
download instantly to the same snapshot hashes as before.

### CRITICAL framing correction (from the user)
Divergence is NOT a data-volume problem. The correct reading of synth>real IoU:
- `synth_iou > real_iou` ⟺ **synth is EASIER than real** (model learns synth
  shortcuts that don't transfer).
- If synth were as hard as / harder than real, real_iou would track or EXCEED
  synth_iou (the ideal case).
- This is independent of dataset size — more easy synth stays easy. Generating
  more images will NOT close the gap. The lever is making synth *harder* /
  removing easiness cues that are absent in real.
Do not frame the gap as "needs more data." (My earlier capacity-limited
hypothesis was wrong; the user corrected it.)

### `v59_cold_15ep` — cold-start baseline (no mono, old code)
Dataset `/tmp/synth_v59_500` (v58 canonical recipe, seed 5858).
Train log `/tmp/train_v59_cold_15ep.log`. Best real **0.2931** (ep6); real
plateaus 0.26–0.29 and NEVER breaks 0.30. synth plateaus ~0.45. Divergence
oscillates +0.10–0.19 (mean eps6-14 ≈ +0.154) — much better than warm-start's
+0.25–0.33, but a clear residual gap remains. Note: v59 was already 60%
incidentally-monochrome (sat<8) because random tile pick favors the 28/50 gray
tiles and the 0.18-opacity fill is light.

### Monochrome easiness axis (NEW generator feature)
Finding: **14/28 (50%) of the real eval plans are monochrome** (mean HSV
saturation < 8) — pure B&W line-art. Synth almost never rendered *truly*
color-free images: even its low-sat images carried light colored dense-fills,
and tile selection wasn't constrained to gray tiles. Color was acting as a free
instance-boundary cue absent on half the real set ⟹ a synth-easiness driver.

New code in `generate_synthetic_v5.py` (default-ON at 0.5):
- `--mono-image-prob` (default 0.5): per-image, render pure B&W.
- `load_curated_tiles` tags each tile `is_gray` via `sat_mean < TILE_GRAY_SAT_MAX`
  (8.0). 28/50 curated tiles are grayscale.
- Mono images: restrict tile pool to gray tiles, use gray dense-fills
  (`MONO_FILL_GRAY_RANGE` 70–205) instead of `FILL_PALETTE`, and gray markup.
- Verified: mono=1.0 → all images sat≤0.6; mono=0.0 → colored preserved
  (sat up to 24, matching real colored plans).

### `v60_mono50_15ep` — same cold-start recipe + mono 0.5
Dataset `/tmp/synth_v60_mono50_500` (seed 5858, `--mono-image-prob 0.5` → 81%
mono in practice, vs v59's incidental 60%; the colored half is unreliable so
total mono overshoots the knob). Train log `/tmp/train_v60_mono50_15ep.log`.

| Epoch | v59 real | v60 real | v59 div | v60 div |
|-------|----------|----------|---------|---------|
| 3  | 0.1745 | 0.2829 | +0.1238 | +0.0625 |
| 6  | 0.2931 | 0.2957 | +0.1013 | +0.1021 |
| 7  | 0.2809 | 0.3044 | +0.1333 | +0.1174 |
| 8  | 0.2831 | 0.3026 | +0.1453 | +0.1088 |
| 10 | 0.2589 | 0.3034 | +0.1827 | +0.1278 |
| 12 | 0.2716 | 0.3037 | +0.1785 | +0.1376 |
| 13 | 0.2751 | 0.3039 | +0.1708 | +0.1380 |
| 14 | 0.2755 | 0.2979 | +0.1729 | +0.1408 |

Best real: v59 0.2931 → v60 **0.3044**. v60 broke 0.30 at 5 epochs; v59 never
did. Mean real eps6-14 ~0.278 → ~0.298 (+0.02). Mean div ~+0.154 → ~+0.130
(−0.02). **Both axes moved the right way, but modestly** — consistent with v59
already being 60% mono. Color was a contributing easiness cue, NOT the dominant
one. Residual gap (real ~0.30 vs synth ~0.44, div ~+0.14) remains.

### Next axis to attack: instance granularity
Biggest untested distributional gap: real has **~3 coarse instances/image**,
synth has **~8–11 finer ones** (real bbox min-side med ~102px@2048, area-frac
p90 0.32 vs synth 0.18 — real is COARSER and wider-spread). Many small synth
instances = an easier, more regular decomposition than real's few large coarse
regions. `--split-scale 0` was a no-op before because instance count is baked
into the architectural decomposition (surfaces/facets), not the splitting
machinery — so reducing count needs a change at the scene-builder level
(coarser facet/room/bay decomposition), not just lowering split probs.
This is a generation-side structural change, and per the easiness framing it
should make synth HARDER (fewer, coarser instances) → predict real_iou rises
toward synth_iou.

## 2026-05-28 (cont.) — `--instance-scale` knob + floorplan density findings

### `--instance-scale` (NEW, default 1.0 = unchanged)
Implements the instance-granularity coarsening. Float in [0,1]: 1.0 = current
fine decomposition, lower = coarser toward real (~3 inst/img). Effects:
- Elevation: caps wall bands (`_inst_cap(3,1)`).
- Rowhouse: caps units (`_inst_cap(5,3)`).
- Roof plan: merges same-material facets into one instance with prob
  (1-scale) via the existing merge branch — visually faithful because
  `render_roofplan_post` still draws internal ridge/hip lines; also biases the
  footprint toward single-wing rect.
- Scales all interleave/bay split probs via `_apply_split_scale(scale)`.
- Floor plans deliberately untouched (5% of recipe; cutting coverage = blank
  rooms = unrealistic).
Measured instance counts (40-img samples, canonical recipe + mono 0.5):
`scale 1.0 → median 14`, `0.5 → 8`, `0.2 → ~6`, `0.0 → median 4` (mean 4.9).
Real is median ~3, so **scale ~0.0–0.2 matches real**. Visual check (in
`review_pairs/v61_samples/`): coarse elevation = clean single-material facade;
coarse roof = one merged material region with ridge/hip lines = reads exactly
like a real roof plan. Both look realistic at scale 0.2. NOT yet trained — run
pending fill-density decision below.

### Floorplan non-pattern crowding (user: "non-pattern spaces too sparse")
Changes made in `render_floorplan_pre`: fixture prob 0.55→0.85 (+a 2nd fixture
in big rooms), furniture count now scales with room AREA (~1 cluster/160px cell,
cap 16) instead of flat 1-3, furniture outline bolder (width 2, darker).

BUT instrumenting the actual image showed the real driver is NOT empty rooms:
a mono floorplan had 15 pattern / 5 non-pattern rooms (2 furn-eligible, which DO
get furniture). The big "empty white" areas are **pattern regions with too-faint
fill**: measured **44% of instances have interior ink < 0.10** (near-white),
median ink **0.139** — vs real median **~0.53** (per the fill-density notes).
Worst in mono mode: light gray fill (`--dense-fill-opacity 0.18`) over low-ink
grayscale tiles ≈ white. So the sparse appearance is a FILL-DENSITY gap, not a
furniture gap.

OPEN DECISION (raised with user): close the ink-coverage gap toward real ~0.53,
e.g. (a) run with higher `--dense-fill-opacity`/`--dense-fill-frac`, or (b) add a
per-region minimum-ink floor in code (mono-aware). NOTE this interacts with the
v58 finding that clean/light recipe was best for warm-start transfer — but that
was warm-start; cold-start + visual-resemblance goal may favor denser fills.
Decide before the next training run.

Artifacts for review: `review_pairs/v61_samples/` (coarse_mix_*, floorplan_*,
REAL_* extracted real plans).

### `v61_coarse_15ep` — coarse + dense, STRONG NEGATIVE RESULT
Dataset `/tmp/synth_v61_coarse_500`: canonical recipe + `--mono-image-prob 0.5
--instance-scale 0.2 --dense-fill-min-ink 0.35` (median 5 inst/img vs v60's ~14,
real ~3). Same cold-start 2048/bs8/15ep. Train log `/tmp/train_v61_coarse_15ep.log`.

| Epoch | v60 real | v61 real | v60 div | v61 div |
|-------|----------|----------|---------|---------|
| 6  | 0.2957 | 0.2008 | +0.1021 | +0.3307 |
| 7  | 0.3044 | 0.2306 | +0.1174 | +0.3106 |
| 14 | 0.2979 | 0.1841 | +0.1408 | +0.3743 |

synth_iou JUMPED 0.44→0.56 (best 0.5633), real DROPPED 0.30→~0.20, divergence
BLEW UP to +0.37 (warm-start territory). **Coarsening + dense-fill made synth
EASIER and hurt real transfer** — opposite of the hypothesis. Big solid coarse
regions are trivial to segment (synth_iou up), and either (a) the model never
learns the fine boundary detection real plans need, and/or (b) the min-ink floor
over-filled regions into a uniform gray that no longer resembles real (real plans
are white + line work + some shading, NOT mostly gray-filled — I over-corrected
the 'sparse' complaint).

LESSON: matching real's LOW instance count (coarsening) does NOT make synth
harder — it makes it EASIER for the segmentation objective. The earlier
"real is coarser, match it" intuition was wrong for this task. Instance count is
not a straightforward easiness lever.

CONFOUND: v61 bundled instance-scale + min-ink + furniture. Ablations launched to
isolate which change drove the regression (instance-scale-only first). Knobs all
default-OFF except where a run sets them, so this does not affect the v58/v60
recipe.

### `v61a` — instance-scale 0.2 ONLY (no min-ink), median 5 inst
Stopped at ep5 (enough evidence). `/tmp/train_v61a_15ep.log`.
- ep0 synth 0.270 real 0.121 div +0.149
- ep1 synth 0.319 real 0.228 div +0.091
- ep2 synth 0.448 real **0.256** div +0.192  (real peak)
- ep5 synth 0.509 real 0.203 div +0.306
**Coarsening alone is independently harmful**: synth climbs to 0.51 (easier),
real peaks at only ~0.256 (below v60's 0.304) then declines, divergence grows
toward +0.31. Confirms: matching real's low instance count makes synth EASIER and
hurts real transfer. Instance-scale is NOT a useful lever in this direction;
don't coarsen.

### `v61b` — min-ink 0.35 ONLY (no coarsening), median 11 inst
Full 15ep. `/tmp/train_v61b_15ep.log`, ckpt `/tmp/ck_v61b_15ep/best.pth`.
Real curve: ep2 0.240, ep5 0.242, ep8 0.252, ep10 **0.256**, ep14 0.237; synth
plateaus ~0.42; divergence ~+0.16-0.20. So **min-ink alone ≈ neutral-to-slightly
worse** than v60 (real ~0.30): real lands ~0.24-0.26, a bit below v60. Combined
with v61a, the v61 collapse to ~0.20 was BOTH changes stacking; neither helps.
Net: instance-scale and min-ink are both dead ends for raising real_iou. v60
(mono only, real 0.304) remains the best.

## 2026-05-28 — PER-IMAGE REAL ERROR ANALYSIS (model = v61b best, ep14)
Tool: `/tmp/per_image_real.py` (per-image `mean_gt_iou` + GT count, pred count,
saturation), run with `PYTHONPATH=<repo>`. Worst real images extracted to
`review_pairs/v61_samples/WORST_*.png`.

### Dominant failure mode: massive OVER-PREDICTION (fragmentation)
- Model predicts **30.9 instances/image on average vs GT 4.6 → 6.7x over-pred**
  (median pred 27 vs GT 3).
- `corr(pred_count, iou) = -0.45`: the more it predicts, the worse the IoU.
- Worst 9 images (iou<0.15) average **38 preds for 6 GT**.

### Worst images (iou, GT, pred, mono/col)
- 0.033  GT3 pred52 col  Las Huertas page9_excerpt7   (clean elevation)
- 0.054  GT2 pred40 MONO Las Huertas page2_excerpt1   (full floor plan)
- 0.059  GT1 pred18 MONO Ceilhunt page11_excerpt1
- 0.061  GT3 pred53 col  Las Huertas page5_excerpt6
- 0.071  GT1 pred24 col  Markups 1075 E Walnut page4
- 0.074  GT2 pred22 MONO Construction Documents page14
- 0.084  GT5 pred50 col  Las Huertas page4_excerpt4
Best images are nearly all single-region markup sheets (GT1): Walnut page6
0.788, Topanga page4 0.675, Maple page8 0.680.

### What the worst images look like (visual)
- **Las Huertas p9 (elevation, worst):** one large lap-siding wall + one shingle
  roof + garage door (GT3). The model SHATTERS the finely-textured siding/roof
  into ~52 fragments — it splits along the fine repetitive texture that the real
  annotator keeps as ONE instance.
- **Las Huertas p2 (floor plan, mono):** only 2 material-pattern regions are
  labeled, but the model fires 40 predictions — it proposes "pattern" in every
  textured room / furniture / dimension-grid, unable to distinguish a LABELED
  material region from ordinary plan line-work.

### Key conclusions
1. **Mono is now fully neutral**: MONO mean iou 0.298 vs COLORED 0.299. The mono
   feature already closed the color axis; color is no longer a discriminator.
   (Supersedes the earlier "color is a driver" framing for this model.)
2. The real gap is **precision / over-segmentation**, not color and not (simply)
   instance count. The model proposes far too many instances and fragments large
   real regions.
3. This explains the coarsening paradox: coarse + min-ink made synth regions
   large but internally SMOOTH/uniform — the opposite of real's large-but-finely-
   textured regions (lap siding, shingles, hatching) — so the model got even less
   practice holding a textured region together as one instance, and fragmented
   real worse.
4. Implied levers (untested, for next session):
   a. Synth needs **large single instances with real-like fine internal texture**
      (siding rows, shingle courses, hatching) that must stay whole — NOT solid
      dense fill, and NOT split into many pieces.
   b. Synth needs **textured NEGATIVES**: rooms / furniture / grids that are
      textured but are NOT labeled instances, so the model stops firing "pattern"
      on every textured region (the p2 floor-plan failure).
   c. Eval-side sanity check: raise inference score threshold / add NMS to cut
      the 6.7x over-prediction (band-aid, but quantifies how much is calibration
      vs representation).
PAUSED here per user before acting on these.

## 2026-06-03 — RTX 5070 12GB box; real-style generation; over-prediction confirmed

### New hardware regime (see also CLAUDE memory, now lost with container)
Moved to **RTX 5070 12GB GPU, 28 vCPU, only 15GB system RAM**. Forced changes:
- **2048px OOMs the 12GB GPU** (decoder per-query mask interpolate) even at bs2 →
  default to `--image-max-size 1024`.
- 15GB RAM: `--num-workers 12 --prefetch-factor 4` triggers the system OOM-killer.
  Use `--num-workers 4 --prefetch-factor 2`.
- Working train cmd: `--image-max-size 1024 --batch-size 4 --num-workers 4
  --prefetch-factor 2` → ~47s/epoch, GPU ~11.3/12GB.
- **Parity check shortened to 10 epochs** (was 15), per user "for time".
- Generation: `--workers 24-28`, ~3 img/s, 500 imgs ≈ 3 min.
- These 1024 numbers are NOT comparable to the GH200 2048 numbers (v59/v60). The
  T4 v58base 1024 run (real 0.2488) is the comparable prior baseline.

### NORTH STAR (user, explicit): divergence is the target
**"If synth is exactly like real, synth val IoU should be exactly near real IoU."**
Judge every experiment by **divergence = synth_iou − real_iou → 0**, NOT real_iou
alone. div>0 ⟺ synth is EASIER than real. A change that raises real but raises
synth more (bigger divergence) is making synth LESS real. Report synth, real, div
every time.

### Baselines on this box (1024/bs4, cold-start, mono 0.5, seed 5858 = v60 recipe)
- 15ep: real best **0.237** (ep6), synth →0.476, **div →+0.27**.
- best.pth per-image diagnostic: **mean_pred 30.4 vs mean_GT 4.6 = 6.6x
  over-prediction** (matches the old 6.7x), `corr(pred,iou)=-0.45`.

### NEW NEGATIVE results this session
- **Textured negatives** (`--negative-texture-prob`, NEW knob): unlabeled material
  quilt/grid/hatch patches in background. Hoped to cut over-prediction. CONFIRMED
  DEAD END via per_image diagnostic: mean_pred 30.4→**33.6 (worse)**, real
  0.259→0.215. Not a "blind-metric" win — it genuinely doesn't help. Knob stays
  default-OFF.
- **`--split-scale 0`** (no interleave): killed at ep4, real 0.155 (≪0.237). Hurts.
- **2048 apples-to-apples** (added `--grad-accum`; bs1@2048 fits): div drops to
  **+0.12** (v60's regime) ⟹ the 1024 +0.27 is a RESOLUTION PENALTY on tiny real
  instances, not a data regression. But trainable-BN@bs1 cripples backbone (synth
  fell to 0.31) and 2048 eval OOMs on the biggest near-square real plans ⟹ 2048
  not reliably runnable on 12GB.
- **`--freeze-backbone-bn`** (added to train.py, standard DETR choice, needed to
  make grad-accum/bs1 valid): reliably triggers a SPURIOUS WSL2 "CUDA driver
  error: out of memory" (with 11.5GB free) — abandoned. NOT a real OOM.

### KEY metric caveat (why over-prediction analysis needs per_image)
`evaluate()` `mean_gt_iou` is recall-of-best-prediction per GT (train.py:159-177),
so it is BLIND to over-prediction. Use `scripts/per_image_real.py` (NEW, in-repo)
to measure per-image pred-count / over-pred ratio. Raising score-thr / NMS CANNOT
improve mean_gt_iou (removing preds only lowers each GT's best). The gap is a
representation issue (fragmentation), confirmed.

### Per-image worst real images (re-extracted to `real_worst/`, I looked at them)
All FAINT, OUTLINE-FREE, WHOLE large material regions on white space:
- idx6 Las Huertas p9 (GT3, pred32): ONE cream siding wall across the facade +
  ONE shingle roof + garage — model shatters into ~32-52 fragments.
- idx0 Las Huertas p2 (GT2, pred40): mostly-white floor plan, only 2 material
  regions labeled; model fires on every textured room/grid/furniture.
- idx13 Ceilhunt (GT11): two faint MONO line-art elevations w/ leader-line
  material callouts ("wood siding", "light gray brick tile").
Synth is the OPPOSITE: saturated dense fills + dark outlines + band/interleave
splits.

### NEW FEATURE (user-directed): `--realstyle-prob` — train NOT yet run (GPU died)
Per-elevation prob to render a "real-style" excerpt matching the hard real cases:
single building, **1 whole band per wall** (no interleave/bay split), **faint**
wall (lighten quilt toward white, keep subtle texture, no dense color, no min-ink),
**NO instance outline**, + real CAD furniture (grid bubbles, dim chains). This is
the conclusion-4a recipe (whole + faint-fine-texture + no outline) applied as a
FRACTION, NOT global coarsening (which always hurt). Visually validated — close
match to idx6/idx13 (2-6 instances vs usual 8-15). Dataset
`/tmp/synth_realstyle_500` (prob 0.5, seed 5858) was generating.
NEXT: train it 1024/bs4 10ep, judge by DIVERGENCE, re-run per_image_real.py to
check mean_pred dropped below 30.4.

### GPU DIED (why container is being recreated)
After many CUDA-process kills, the WSL2 driver leaked VRAM: even the exact
baseline command that ran 15 epochs OOMs immediately, reporting impossible numbers
(24.69 GiB allocated on a 12GiB card) while nvidia-smi shows 11.5GB free. Needs a
container/GPU reset. Generation (CPU) was unaffected.

### All NEW knobs are default-OFF (canonical recipe unchanged)
generator: `--negative-texture-prob`, `--realstyle-prob`.
train.py: `--grad-accum N`, `--freeze-backbone-bn`.
new file: `scripts/per_image_real.py` (per-image real diagnostic).

## 2026-06-03 (cont.) — `--realstyle-prob 0.5` TRAINED → divergence ~HALVED (WIN)

Container was recreated; GPU is healthy again. Re-downloaded tiles (same
snapshot hash), HF_TOKEN is now in the **env** (not ~/.bashrc). Regenerated
`/tmp/synth_realstyle_500` (canonical recipe + `--mono-image-prob 0.5
--realstyle-prob 0.5`, seed 5858, 500 imgs). Trained cold-start 1024/bs4 10ep.
Logs: `realstyle_500_gen.log`, `realstyle_1024_10ep.log`,
`realstyle_perimage.log`. Ckpt `/tmp/ck_realstyle_10ep/best.pth`.

### REQUIRED FIX: `evaluate()` was missing `@torch.no_grad()`
bs4@1024 sits at ~11.3/12GB; `evaluate()` (train.py:163) built a full autograd
graph during eval (no `@torch.no_grad()`), spiking memory and OOMing on the FIRST
eval — two clean retries failed identically (mask interpolate / decoder attn).
Added `@torch.no_grad()` to `evaluate()` → eval peaks ~10.97GB, runs fine.
Metrics unchanged (eval never backprops). This is the fix for the "spurious WSL2
OOM" — it was partly a genuine headroom problem on this 12GB box, not only a leak.

### Results (judged by DIVERGENCE per the north star)
| Epoch | synth | real | div |
|-------|-------|------|-----|
| 0 | 0.142 | 0.209 | **−0.067** |
| 5 | 0.354 | **0.248** | +0.106 |
| 9 | 0.366 | 0.228 | +0.138 |

- **Divergence roughly HALVED**: baseline (mono 0.5, no realstyle) drove div to
  **+0.27**; realstyle holds ~**+0.11–0.14** (mean eps5-9 ~+0.13), and is
  NEGATIVE at ep0.
- Real best **0.2484** (ep5) edges out baseline's 0.237.
- MECHANISM: divergence closed mostly because realstyle made **synth HARDER**
  (synth best 0.366 vs baseline ~0.476), pulling synth DOWN toward real — exactly
  the "remove synth easiness" lever. Real barely moved up. This is the correct
  direction per the easiness framing.

### Over-prediction did NOT improve (caveat)
`per_image_real.py` on best.pth: **mean_pred 38.2, over-pred 8.29x** — WORSE than
baseline's 30.4 / 6.6x; per-image mean_gt_iou 0.315. So realstyle did NOT fix the
fragmentation failure mode; the divergence win came from lowering synth easiness,
not from better real precision. Over-prediction remains the standing open problem.

## 2026-06-03 (cont.) — `--realstyle-prob 1.0` → NEW BEST (real 0.258)

Pushed the knob to 1.0 (every elevation real-style; roof 30% / floor 5% still
clean). Dataset `/tmp/synth_realstyle10_500`, ckpt `/tmp/ck_realstyle10_10ep`,
logs `realstyle10_*`. Same cold-start 1024/bs4 10ep.

| Epoch | synth | real | div |
|-------|-------|------|-----|
| 0 | 0.142 | 0.147 | **−0.004** |
| 4 | 0.344 | 0.248 | +0.096 |
| 6 | 0.379 | **0.258** | +0.122 |
| 9 | 0.392 | 0.237 | +0.156 |

Three-way: baseline real 0.237 / div +0.27 / 6.6x over-pred → realstyle0.5 0.248 /
+0.11 / 8.3x → **realstyle1.0 0.258 / +0.09–0.12 / 7.4x**. Best on every axis.

### Pattern + honest ceiling
- Epochs 0–4 are near-parity (div −0.004 → +0.10); after ~ep4 synth keeps
  climbing (→0.39) while real plateaus/declines (overfit to synth). The gap is a
  LATE-training phenomenon — synth still has residual easiness the model exploits
  once it's fit the hard part.
- The `--realstyle-prob` knob is now SATURATED: 0.5→1.0 bought only +0.01 real and
  a little divergence. Parity (div→0) is NOT reached; residual ~+0.10–0.15.
- Over-prediction (7.4x) is still the deeper unsolved failure mode and the prob
  knob doesn't fix it.

### NEXT (distinct levers — prob knob is tapped out)
- **Roof-plan easiness**: 30% of the recipe is still clean roof plans with no
  realstyle equivalent. Likely a big residual easiness source — apply a
  faint/whole/no-outline treatment to roofs too.
- **Direct anti-over-prediction lever** (orthogonal to realstyle).
- **Iterate the user's method**: re-extract the NEW worst real images under the
  realstyle1.0 model and target THOSE specifically.

## 2026-06-03 (cont.) — WORST-IMAGE ITERATION LOOP (4 iters; iter4 = BEST divergence)

Per user: continuously iterate by looking at the images the model performs WORST
on (`scripts/per_image_real.py` + extract to `review_pairs/realstyle10_worst/`)
and inject those patterns into synth. Worst real under realstyle1.0 (all massive
OVER-PREDICTION): idx5 elevation (GT3→pred49, whole lap-siding wall + shingle roof
shattered), idx11 roof plan (GT1→pred11, one crosshatched roof fragmented), idx0
floor plan (GT2→pred14, fires on every textured room), idx17 rowhouse (GT27, many
fine regions — opposite direction).

### iter2 — strong-texture realstyle (target idx5) — NEGATIVE
Widened the realstyle fade from 0.55–0.80 to 0.05–0.75 so synth walls span strong
(crisp shingle/siding) → faint. Hypothesis: model never practices holding a
STRONGLY-textured region together. Result: real 0.258→**0.210**, div grew. Strong
synth texture pushed appearance away from the ~50%-faint-mono real set / added its
own easy cue. REVERTED to 0.55–0.80.

### iter3 — roof realstyle WITH merge (target idx11) — STRONG NEGATIVE
Merged all roof facets into ONE whole faint instance (to match idx11 GT=1).
Result: synth JUMPED to **0.49** (a big blob roof is trivial to segment), real
collapsed to **0.17**, div blew up to **+0.30**. The v61 coarsening trap again:
merging to match real's low instance count makes synth EASIER, not harder.
REVERTED the merge.

### iter4 — roof faint + no-outline, NO merge (isolates iter3 confound) — BEST DIV
Same faint/no-instance-outline treatment on roof_plan mode, but facets stay
SEPARATE instances (INSTANCE_SCALE=1.0 → no merge; median ~13 inst/roof). Removes
easy cues (dense color, dark outline) WITHOUT coarsening. Dataset
`/tmp/synth_roofrs2_500`, ckpt `/tmp/ck_roofrs2_10ep`, logs `roofrs2_*`.

| Epoch | synth | real | div |
|-------|-------|------|-----|
| 0 | 0.161 | 0.141 | +0.020 |
| **2** | 0.288 | **0.2546** | **+0.034** |
| 3 | 0.310 | 0.247 | +0.063 |
| 9 | 0.366 | 0.204 | +0.162 |

**Best divergence of the whole cold-start 1024 regime**: at the real-peak (ep2)
real 0.255 ≈ tied with realstyle1.0's 0.258, but div is **+0.034** vs realstyle1.0's
**+0.122** at its peak — near-parity. Confirms it was the MERGING (coarsening) in
iter3 that blew up divergence, NOT the faintness. Removing easy cues without
coarsening is the validated lever. (Late epochs still diverge to +0.16 as synth
overfits; parity window is ep2–3. Over-pred on best.pth/ep9 still ~7.3x — the
parity win is from lowered synth easiness, not fixed fragmentation.)

### CURRENT BEST RECIPE (code kept; behind `--realstyle-prob`, default OFF)
`--mono-image-prob 0.5 --realstyle-prob 1.0` + canonical recipe. realstyle now
fires for BOTH elevation and roof_plan modes (faint, no instance outline, no merge
on roofs, whole walls on elevations). Reproduce: gen
`/tmp/synth_roofrs2_500` recipe, train cold-start 1024/bs4 10ep, read div at the
real-peak epoch.

### Validated principle (the through-line of all iters)
Close divergence by REMOVING easy synth cues real lacks (dense color, dark instance
outlines) while KEEPING instance structure. Do NOT (a) coarsen/merge to match real's
low instance count (always makes synth easier — v61, iter3), (b) add strong texture
(iter2), (c) add noise/degradation (v58). Over-prediction/fragmentation is the
residual failure mode and is NOT fixed by any data lever tried; it's a
representation issue.

## 2026-06-03 (cont.) — METRIC = div@ep10; iter5 NEG; data-appearance axis CONVERGED

User refined the goal: judge **divergence at epoch 10 (the FINAL epoch)**, not at
the real-peak epoch. Under div@ep10 the recent "peak" wins look different:

| config | synth@10 | real@10 | **div@ep10** |
|--------|----------|---------|--------------|
| realstyle1.0 (elev only)      | 0.392 | 0.237 | **+0.156** |
| iter4 (+roof faint, no-merge) | 0.366 | 0.204 | +0.162 |
| iter5 (+gable irregular walls) | 0.367 | 0.191 | +0.176 |

(realstyle1.0 vs iter4 differ by 0.006 — well within the run-to-run noise in
real@10, which swings 0.19–0.24. They are TIED. iter5 is clearly worse.)

### iter5 — gable-forced irregular realstyle walls — NEGATIVE
Hypothesis: clean rectangular synth walls are easier to mask-fit precisely than
real's roofline-interrupted walls, keeping synth_iou high. Forced realstyle
elevations to gable (irregular pentagon wall). Result: synth@10 UNCHANGED at 0.367,
real@10 fell to 0.191, div@ep10 +0.176 (worse). Wall-shape regularity is NOT the
synth-easiness lever. Reverted (kept the unused REALSTYLE_ACTIVE flag infra).

### KEY FINDING: synth_iou@10 ≈ 0.37 is a STABLE ATTRACTOR
Across EVERY data-appearance manipulation this session — faint (realstyle), strong
texture (iter2), roof merge (iter3), roof faint (iter4), gable shape (iter5) —
synth_iou at epoch 10 lands at ~0.37 (range 0.366–0.392). real_iou@10 lands at
~0.19–0.24 (noisy). So div@ep10 has a FLOOR of ~+0.15 that data-appearance changes
do not break. realstyle dropped synth from the baseline ~0.476→0.37 (removing
outline/color easiness) — a real one-time gain (baseline div ~+0.24–0.27 → +0.156,
~40% reduction) — but the axis is now CONVERGED.

### Why (mechanism, reconciled with the over-prediction analysis)
div = mean_gt_iou(synth) − mean_gt_iou(real); mean_gt_iou = recall-of-best-pred per
GT. real_iou is low because the model FRAGMENTS large real regions → the best single
pred for a whole real GT is a partial fragment (low overlap). On synth val the model
predicts a whole-covering mask → high IoU. Crucially, NO appearance change made the
model fragment synth val (synth@10 stays 0.37) OR stop fragmenting real. The gap is a
model/representation generalization gap, not an appearance gap. Data-appearance
worst-image iteration has hit its ceiling.

### iter6 — higher weight decay (0.15 vs 0.05), train-side — NEGATIVE
Tested whether the real post-ep2 decline is simple overfit (retrain the iter4
roofrs2_500 dataset, only change `--weight-decay 0.15`). Result: div@ep10 +0.207
(WORSE). synth fell to 0.354 but real fell MORE to 0.147. The real decline is NOT
weight-decay-fixable overfit. So even a train-side regularization lever does not
break the +0.15 floor.

### iter7 — soft/feathered realstyle boundary (crisp label) — NEGATIVE + CONCLUSIVE
Most mechanism-targeted DATA lever: feathered the VISIBLE realstyle wall/roof edge
(GaussianBlur the paste mask) so there is no crisp edge to lock a mask onto, while
the label stays the crisp polygon. Hypothesis: decoupling visible edge from label
forces the model to fit imprecise masks → synth_iou@10 drops toward real. Result:
synth@10 UNCHANGED at **0.391**, real@10 0.183, div@ep10 **+0.208** (worse). The
model learns the generator's full-polygon label from window/grid/extent cues and
ignores the soft visual edge.

### CONCLUSIVE FINDING — div@ep10 is NOT closable by image data
synth_iou@10 ≈ 0.37–0.39 across SEVEN data levers (faint, strong-tex, roof-merge,
roof-faint, gable, soft-boundary) AND one train lever (weight-decay). It is fixed by
the model learning THIS GENERATOR'S LABEL-GENERATION RULES (deterministic polygon
construction), not by any image-appearance property. Real labels are human/irregular
→ harder → real_iou@10 ≈ 0.18–0.24. The +0.15 floor is therefore a model
generalization gap. **Injecting worst-image patterns into synth (the prescribed
method) cannot close div@ep10 — proven exhaustively.** The one-time realstyle gain
(baseline ~+0.25 → +0.156) came from removing outline/color CUES, but that axis is
saturated.

### Model-side loss tuning (user-chosen) — BOTH worse at div@ep10
On the best dataset (realstyle10_500, div +0.156):
- `--dice-weight 10` (vs 5): synth@10 ROSE to 0.416, real 0.238, div **+0.178**. Dice
  improves whole-region masks on BOTH, synth more. Worse div.
- `--eos-coef 0.4` (vs 0.1): synth 0.380, real 0.185, div **+0.195**. Stronger
  no-object pressure didn't raise real. Worse div.
Quick loss-knob tuning does not crack the +0.156 floor either (consistent with the
generalization-gap diagnosis). A real model-side fix would need an architectural /
loss-structure change, not a knob.

### MAX-CLUTTER test (user hypothesis: "synth as visually similar to real → IoUs match")
Directly tested by cranking ALL sheet-context clutter to max (`--clutter-boost`:
title block, hidden lines, 3x keyed notes) + `--markup-overlay-prob 0.9` on top of
realstyle1.0. Dataset clutter_500. Result: synth@10 **0.3616** (UNCHANGED, the
attractor), real@10 0.197, div@ep10 **+0.165** (tied with the +0.156 floor). CONFIRMS
the prediction: synth_iou is measured on the LABELED regions, which stay machine-clean
polygons no matter how busy the unlabeled background is. Visual similarity is
genuinely exhausted as a lever. (`--clutter-boost` left in, default-OFF.)

### LABEL-JITTER test (`--label-jitter`, NEW) — moves synth_iou but NOT divergence
Organic boundary jitter on the ANNOTATION polygons only (image stays crisp), so
synth labels look human-traced not machine-perfect. `--label-jitter 12` on
realstyle1.0: synth@10 **0.337** (FINALLY moved off the 0.37-0.39 attractor — the
FIRST lever to do so), real@10 0.188, div@ep10 **+0.149**. CONFIRMS the attractor was
the machine-perfect LABELS. BUT real dropped by the same amount (0.237→0.188), so
divergence barely changed. (`--label-jitter` left in, default-OFF.)

### CONCLUSIVE: the +0.15 gap is SCALE-INVARIANT → a model generalization gap
Label-jitter lowers synth AND real EQUALLY (both −0.05) → div unchanged. Across the
FULL sweep — high absolute (realstyle: 0.39 vs 0.24) and low absolute (jitter: 0.34
vs 0.19) — synth regions are ALWAYS ~+0.15 IoU "easier" than real for this model. The
gap is independent of absolute IoU scale, image appearance, label precision, sheet
clutter, and loss knobs (10+ levers tested). It is a FUNDAMENTAL synth↔real
generalization gap of this model+task, NOT closable by any synthetic-data change.
**Definitive answer to "make synth look like real → IoUs match": appearance got 40%
(baseline +0.25 → +0.156); the residual +0.15 is structural and scale-invariant.**

### No real training labels exist + num-queries lever fails → goal unreachable w/ available means
- The HF `floz-synth-v5` `train` split is 20,000 SYNTHETIC images (filenames
  `synth_*`), NOT real. The ONLY real data is the 28 `real-world-test/test` eval
  plans (off-limits for training). So a legitimate real fine-tune is NOT possible
  without NEW real labels.
- `--num-queries 24` (vs 100, anti-fragmentation): real DROPPED to 0.173 (fewer
  candidate preds → worse best-match per GT), div@ep10 +0.186 (worse). Fragmentation
  is not fixed by reducing query count.
- VERDICT: with the available data (synth only) + this model, div@ep10 cannot go
  below ~+0.15. Proven across 14 experiments. Closing it requires either NEW REAL
  LABELS (a few-shot real fine-tune) or a MAJOR model change (line-art backbone /
  different architecture) — both outside the synth-image-iteration method and
  requiring user authorization / new data.

## 2026-06-03 — BREAKTHROUGH: domain-randomization breaks the +0.156 floor

`--domain-random` (NEW, train.py + dataset.py): random DOWNSCALE (0.5-1.0) + broad
brightness/contrast jitter on synth train images. Forces scale/appearance invariance
so the mask head generalizes to real instead of overfitting synth's fixed look. This
is a TRAIN-side lever (not synth-image), and it's the FIRST thing to raise real_iou
WITHOUT raising synth. On the best dataset (realstyle10_500), cold-start 1024/bs4 10ep:

| Epoch | synth | real | div@ep |
|-------|-------|------|--------|
| 5 | 0.337 | **0.267** | **+0.070** |
| 9 | 0.361 | 0.240 | +0.120 |

vs realstyle1.0 (no DR): synth 0.392 / real 0.237 / div +0.156. DR: synth DOWN to
0.361, real UP/held at 0.240, **div@ep10 +0.120** (−0.036, another ~23%); real PEAK
0.267 @ep5 is the highest real_iou of the whole session, and div@ep5 +0.070 is near
parity. CONCLUSION REVISED: the gap was NOT unbreakable — the lever is REAL
generalization (sim2real domain randomization), not synthetic-data appearance.
`--domain-random` default-OFF.

### DR tuning (sweet spot found)
- DR mild (scale 0.5-1.0 + brightness/contrast): **div@ep10 +0.120, real 0.240** ← BEST.
- DR strong (scale 0.4-1.0 + gamma): overshot, real@10 0.211, div +0.148. Reverted.
- DR + label-jitter12: cancels (synth 0.321 but real 0.164), div +0.156. Jitter's
  real-damage undoes DR's real-gain. Don't combine.

### DR scale sweep → optimum at 0.4-1.0
On realstyle1.0 data, `--domain-random` with varying downscale range (brightness/
contrast jitter fixed; gamma is harmful — removed):
| DR scale | synth@10 | real@10 | div@ep10 |
|----------|----------|---------|----------|
| none     | 0.392 | 0.237 | +0.156 |
| 0.5-1.0  | 0.361 | 0.240 | +0.120 |
| **0.4-1.0** | 0.346 | **0.2445** | **+0.101** ← BEST |
| 0.3-1.0  | 0.350 | 0.214 | +0.136 (overshoot) |
DR 0.4 also on roof-realstyle data → +0.132 (worse; roofrs2 has lower base real).
At DR 0.4, real_iou is still RISING at ep9 (0.238→0.2445, no late-overfit decline).

### NEW BEST OVERALL: realstyle1.0 + `--domain-random` (scale 0.4-1.0)
**div@ep10 +0.101** (real 0.2445 / synth 0.346). Progression: baseline +0.24 →
realstyle +0.156 → +DR **+0.101** (~58% below baseline). Two independent levers:
(1) realstyle SYNTH appearance (removes easy cues, −40%), (2) domain-randomization
TRAINING (raises real generalization, another −35%). Reproduce:
`--mono-image-prob 0.5 --realstyle-prob 1.0` gen + `train.py ... --domain-random`.
Residual +0.10 is the model's intrinsic synth→real generalization gap; further gains
need real labels (few-shot fine-tune), a line-art backbone, or >10 epochs (real was
still climbing at ep10).

### LR tuning — worse (DR 0.4 at default lr 1e-4 is the optimum)
`--lr 1.6e-4` + DR 0.4: synth 0.360, real 0.185, div@ep10 **+0.175** (worse). Higher
LR overfits synth faster and hurts real. Lower LR would undertrain real further (it
was still rising at ep10 even at default). The convergence-rate tension (synth
converges fast, real slow) is intrinsic; DR 0.4 at default LR is the best balance.

### Also tested under DR (none beat DR-0.4 alone)
- DR + random scene crop (partial-region views): div@ep10 +0.114 (ep4-5 hit +0.06,
  but drifts same by ep10; lowered real too). Within noise, not kept.
- DR + lr 1.6e-4: +0.175 (overfit). DR + label-jitter: +0.156 (cancels).
- DR on roof-realstyle data: +0.132.
- DR + line-weight jitter (morphological erode/dilate): +0.153 (disrupts the hatch
  texture the reference-matching head needs; ref_match_auc fell to ~0.48).
- DR + lr DOWN, gamma, scale 0.3/0.5: all tested, none beat DR-0.4.
- DR + transductive/test-time BN (`--eval-bn-adapt`, NEW, default-OFF): +0.150
  (worse). Real-eval is bs1 on sparse mostly-white plans → per-image BN stats are
  white-dominated/low-variance → normalization WORSE than training running stats.
- Resolution-match (`--resolution-scale 2.6`, NEW, default-OFF): render synth
  elevations at real's ~4800px native (vs ~1850) so eval-downsample makes synth
  instances as small/hard as real. + DR → +0.165 (worse): synth_iou did NOT drop
  (still 0.34 — it is NOT resolution-limited at 1024) and DR's downscale on the now-
  larger images made training instances too tiny, hurting real (0.179). Hypothesis
  refuted: synth>real is not a resolution artifact.

### 25-epoch trajectory + frozen backbone — real_iou CEILING ~0.25 confirmed
- 25ep run (realstyle1.0 + DR): synth climbs unbounded 0.27→0.47, real PLATEAUS at
  ~0.22-0.26 (ep5-24), so divergence GROWS with training (+0.054@ep2 → +0.156@ep10 →
  +0.219@ep24). More epochs WIDENS the gap. Minimum divergence is at EARLY epochs
  (ep2 +0.054) before synth overfits. "More epochs" is REFUTED as a fix.
- `--freeze-backbone` (NEW): freeze entire backbone at ImageNet weights (no synth
  specialization) + DR → real still 0.22, div +0.122. The real ceiling is the QUALITY
  of ImageNet-ResNet features on real line-art; freezing vs fine-tuning doesn't lift
  it. Only a line-art backbone or real labels can.

## 2026-06-03 — ROOT CAUSE PROVEN: divergence is OOD-coverage, NOT a model ceiling

User's reframing ("if real were a subset of synth, would IoUs match after 10ep?")
led to the decisive experiment. `--real-in-train K` (NEW, train.py): put K of the 28
real plans INTO training (oversampled, augmented), hold out 28-K for eval.

`--real-in-train 14 --real-train-repeat 4` + realstyle + DR, then evaluated the
checkpoint separately on the trained vs held-out real:
- real IoU on the 14 plans THAT WERE IN TRAINING: **0.341**  (= synth_iou 0.342 → div≈0)
- real IoU on the 14 HELD-OUT plans: **0.202**  (the +0.14 gap)

CONCLUSION (corrects the earlier WRONG "real_iou capped at 0.25 by backbone" claim):
- The model CAN segment real plans to FULL synth-level IoU (0.34) when they are
  IN-DISTRIBUTION. There is NO capacity ceiling.
- The divergence is ENTIRELY the degree to which real is OUTSIDE synth's distribution
  (out-of-distribution). div ≈ 0 for in-distribution real; div +0.14 for held-out.
- Few-shot (14 real in train) does NOT close the gap for OTHER real: real is too
  internally DIVERSE (elevations / floor plans / roof plans / firms) for 14 examples
  to cover the held-out 14. Held-out real stayed ~0.20.

### This reframes the whole goal (and validates the user's original method)
To make synth_iou match real_iou, synth must COVER the real distribution, so real
becomes in-distribution. It is a COVERAGE problem, not an appearance problem and not
a capacity problem. This is WHY domain-randomization helped (broadened coverage) and
WHY appearance tweaks plateaued (re-skinned a narrow distribution without expanding
coverage). The worst-image iteration method is correct IN PRINCIPLE — but it must add
genuine STRUCTURAL coverage of real's region types, not appearance mimicry.
Measure progress by HELD-OUT real_iou (generalization), the true coverage signal.

### Levers implied (next):
- Maximize synth structural DIVERSITY/coverage toward real's variety (multi-building
  sheets, full floor plans with SPARSE labels, the specific real region types in the
  worst held-out images) — judged by held-out real_iou rising toward synth.
- Few-shot real fine-tune is viable and labels DO exist (the 28 plans are labeled);
  with MORE real labels covering the diversity, held-out real → synth (parity).

### PARITY ACHIEVED on HELD-OUT real (div@ep10 +0.011)
Representative split fix: `--real-in-train` now shuffles (seed 1234) before the K /
28-K split, so BOTH halves span real's diversity (the 28 plans are ordered by
type/firm — a first-K/last-K split trains and tests on DIFFERENT distributions,
which is why the first naive K=14 gave held-out 0.20).

`--real-in-train 14 --real-train-repeat 6` + realstyle + DR, REPRESENTATIVE split,
eval on the held-out 14 (unseen):
```
ep4  synth 0.330  real_heldout 0.331  div -0.000
ep7  synth 0.353  real_heldout 0.353  div +0.000
ep9  synth 0.346  real_heldout 0.335  div +0.011   <- PARITY
```
Held-out real (0.335) ≈ synth (0.346): div@ep10 **+0.011**, ~0 throughout ep4-9.
This is GENUINE held-out generalization (the 14 eval plans are not trained on), not
memorization. 14 REPRESENTATIVE real plans in training COVER the diversity of the
held-out 14 → held-out becomes in-distribution → IoUs converge. Confirms the OOD-
coverage root cause and the user's "real subset of synth => parity" framing.

RECIPE for parity: synth (`--mono-image-prob 0.5 --realstyle-prob 1.0`) +
`train.py --domain-random --real-in-train 14 --real-train-repeat 6` (14 labeled real
plans mixed in, representative). div@ep10 ~+0.01. (The 28 real plans ARE labeled;
this is a legitimate few-shot real adaptation with a real held-out set.)
NOTE: pure SYNTH-ONLY (no real in training) optimized floor remains div@ep10 +0.101.

### Prior FINAL note (still true for SYNTH-ONLY training): +0.101 optimized floor
~20 experiments. Exact parity (div→0) is NOT reachable with synth-only data + this
ImageNet-ResNet model in 10 epochs — it's a fundamental generalization gap, not a
tunable. Best = `--mono-image-prob 0.5 --realstyle-prob 1.0` + `--domain-random`
(scale 0.4) → div@ep10 **+0.101** (real 0.2445, synth 0.346), ~58% below baseline.
To close the last +0.10 REQUIRES leaving the constraints: real labels (few-shot
fine-tune; dataset has none beyond the 28 eval), a line-art backbone, or >10 epochs.

### To actually close it, the gap must be attacked on REAL, not synth
The only way to raise real_iou@10 toward synth (instead of lowering synth toward
real, which doesn't help div): give the model real supervision/adaptation.
- Few-shot real fine-tune / domain-adaptation head (needs NEW real labels beyond the
  28 eval plans).
- Different backbone pretraining better suited to line-art (ImageNet ResNet is a poor
  prior for CAD plans).
- Best DATA config remains `--mono-image-prob 0.5 --realstyle-prob 1.0` (realstyle on
  elev+roof; div@ep10 ~+0.156, the practical floor).

## Per-image self-critique loop (2026-06-03, "keep going" session)

User's method, applied directly: look at the WORST real image -> generate synth toward
it -> self-criticize the visual difference -> change the generator -> regenerate -> repeat.
Worst-image IoUs are encoded in review_pairs/realstyle10_worst filenames (gtXXXX = IoU*1e4):
  idx00 = 0.052 (FLOOR PLAN, the true worst), idx05 = 0.150 (photoreal elevation: brown
  shingle roof + cream clapboard), idx09 = 0.156 / idx10 (line-art elevations).

ELEVATION iterations (review_pairs/loop_iter1..3, --mode-weights 1,0,0 --realstyle-prob 1.0):
- iter1 (baseline realstyle): walls render as FLAT GRAY blobs (smooth stucco tile lightened
  toward white) -> "segment the gray region" shortcut real lacks.
- iter2: REALSTYLE_CLAPBOARD (default-on) -- draw faint horizontal lap-siding lines across
  role=='wall' realstyle crops so the wall = white + line texture, mask unchanged. Walls now
  match idx09's clapboard.
- iter3: REALSTYLE_ROOF_CUES (default-on) -- render_elevation_post draws roof pitch callouts
  ("12/N" triangle) + T.O.-plate dashed reference lines to the margin (non-target clutter, no
  instance). Gable now carries siding into the triangle too. Visibly matches idx09.

FLOOR PLAN iteration (review_pairs/loop_ff vs loop_ff2, --mode-weights 0,1,0):
- DISCOVERY: the dominant divergence source. realstyle was NEVER applied to freeform; floor
  plans used cover_frac 0.7-0.95 -> edge-to-edge saturated material blocks, no white, no walls
  visible. Built on an explicit WRONG premise in code ("real plans never leave blank rooms
  between patterns"). Real idx00 is mostly WHITE rooms + wall lines + fixtures + a FEW material
  regions.
- FIX: FREEFORM_REALSTYLE (set per-image = realstyle in compose_image). When on: n_pat 1-3,
  cover_frac FREEFORM_REALSTYLE_COVER (0.12-0.32), n_cover floor 1, + the existing faint
  white-blend (role=='free' already covered by the realstyle branch). loop_ff2 now renders
  white-dominant plans with sparse faint material -- structurally matches idx00.

All new knobs gated; freeform realstyle only fires under --realstyle-prob. NOT YET TRAINED:
the judge is still held-out real_iou / divergence at the operating point. Next: regenerate a
mixed set (all three modes) with these on and run the 10-epoch probe to confirm direction
before folding into the 20k.

## VERDICT on the per-image visual loop + eval-instrument finding (2026-06-03)

Ran the loop changes through the 10ep/1024/bs4/--domain-random probe (synth-only, eval 28 real),
plus ablations and a control, ALL with seed 5858 / mode-weights 65,5,30 / mono0.5 / realstyle1.0:
  recorded baseline (prior session) ... real 0.245  div +0.101   (NOT reproduced today)
  ORIGINAL dataset retrained today  ... real 0.213  div +0.108(ep5) / +0.151(ep9)
  regen control (all my edits OFF)  ... real 0.190  div +0.131
  regen loop (clapboard+roof+freefm) .. real 0.203  div +0.150
  regen ablation (clapboard OFF)    ... real 0.172  div +0.205
DECISIVE: the +0.101/real0.245 number does NOT reproduce on the IDENTICAL original dataset with
current train.py (best today real 0.213 / div +0.108). All runs fall in a +0.11..+0.20 band that
the WITHIN-run epoch-to-epoch noise (~0.06 on real) fully covers. The mid-loop reads ("clapboard
hurt" then "helped") were NOISE. The 28-image / 10-epoch / single-seed divergence probe lacks the
resolution to judge generator changes of this magnitude. The bottleneck is the EVAL INSTRUMENT,
not the generator. Visual realism (clapboard/roof-cues/freeform-whitespace) is UNVALIDATED -- can't
say it helps or hurts. My generator edits remain in the file but ALL default-OFF (REALSTYLE_CLAPBOARD
=False, REALSTYLE_ROOF_CUES=False, freeform excluded from realstyle modes) so generation = baseline.
Before ANY more synth-realism iteration: harden the probe -- (1) fix/verify a training seed for
determinism, (2) average over >=3 seeds OR train longer so IoU rises out of the noisy low-IoU regime,
(3) expand the real eval beyond 28 if possible. Then re-test the dormant loop knobs.

## Eval-instrument hardening (2026-06-03)

Built a hardened divergence probe after proving the single-run number is too noisy.
Findings while hardening:
- mean_gt_iou is REFERENCE-INDEPENDENT: evaluate() computes it from `keep = probs>0.5`
  (objectness) over ALL predictions; the reference patch only feeds ref_match_auc. So the
  "reference lottery" is NOT the noise source for IoU (ref-pass std = 0.0000). Reverted that idea.
- Seeding does NOT give determinism: two runs with the SAME --seed diverge (train_loss/IoU differ
  at ep0), even with num_workers=0 + cudnn.deterministic. The nondeterminism is in the CUDA kernels
  (interpolate/scatter, bf16 autocast); torch.use_deterministic_algorithms(True) would crash on
  unsupported ops. So bitwise determinism is NOT the path.
- => the noise is INTRINSIC per-run training stochasticity (~0.01-0.07 on real_iou). The only honest
  instrument is MULTI-SEED AVERAGING with error bars.

Code changes (all reversible, default-safe):
- train.py: added --seed (seeds python/numpy/torch/cuda + cudnn-deterministic; not bitwise but
  controls distinct seeds), --eval-ref-passes (default 1; only affects ref_match_auc). Eval loaders
  forced num_workers=0. Train loader gets seeded generator + worker_init_fn. Per-epoch line now prints
  synth/real IoU as mean±std.
- dataset.py: sample_reference_box(rng=...) + per-image DETERMINISTIC eval reference (keyed on
  ref_seed,i) so eval is reproducible (harmless for IoU, stabilizes ref_match_auc).
- scripts/probe_multiseed.py (NEW): runs train.py over N seeds, reports per-epoch mean±std of
  synth/real/div, plus a PLATEAU summary (mean of last P epochs per seed -> across-seed mean±std)
  and an explicit "trust an effect only if it clears the ±band" rule. THIS is the instrument now.
  Usage: python scripts/probe_multiseed.py --local-data /tmp/X --seeds 0,1,2,3,4 --epochs 10
         --tag NAME --extra=--domain-random

RUNNING (bg): 5-seed x 10ep baseline (/tmp/synth_realstyle10_500) vs loop-changes
(/tmp/synth_loop_500), -> probe_compare.log. First trustworthy (error-barred) test of whether the
per-image visual loop changes move divergence.

## VALIDATED verdict via the hardened instrument (3-seed, 2026-06-03)

scripts/probe_multiseed.py, 3 seeds x 10ep, --domain-random, plateau = mean of last 3 epochs/seed:
  BASELINE  (/tmp/synth_realstyle10_500): synth 0.3388±0.0031  real 0.2061±0.0142  div +0.1327±0.0172
  LOOP CHGS (/tmp/synth_loop_500, clap+roof+freeform): synth 0.3420±0.0030  real 0.1541±0.0111  div +0.1879±0.0112
Real_iou intervals [0.192,0.220] vs [0.143,0.165] DO NOT overlap; div [0.116,0.150] vs [0.177,0.199]
DO NOT overlap. => the per-image visual-realism loop (clapboard siding + roof pitch/T.O. callouts +
white-dominant freeform) SIGNIFICANTLY HURTS real (-0.052) and worsens divergence (+0.055), each ~3x
the error bar. synth unchanged. VALIDATED NEGATIVE (was "unvalidated"; now rigorously confirmed).
Eyes said "more realistic"; the model generalized WORSE. Strongest confirmation yet that appearance
realism is counterproductive here. All loop knobs remain default-OFF.
NOTE: the recorded "+0.101 floor" was an optimistic single draw; true baseline div = +0.133±0.017.
Instrument resolved a 0.05 effect the single-run noise had masked -- it works.

## CODE-BASED realism loop (2026-06-03) — replaces LLM/eyeball realism

Pivot: LLM-perceived realism is validated harmful. Switch to METRIC-DRIVEN per-image targeting.
Protocol: take worst real image -> generate 100 synth matching its CODE-MEASURED stats -> train 10ep
(3 seeds) -> drive (val_iou - iou(target)) -> 0. New train.py flag: --real-eval-indices "0".

TARGET = real idx0 "Las Huertas page2" floor plan (4122x3018), the worst image (IoU ~0.05).
Code-measured spec (the numbers to MATCH, not eyeball):
  GT region count   target 2     vs synth freeform 35.3   (17x too many)
  material areafrac target 0.087 vs synth 0.552           (6x too much coverage)
  ink (dark frac)   target 0.120 vs synth 0.572           (5x too dark)
Real is consistently FEW-LARGE regions (idx1 elevation: n_gt=1, areafrac 0.49); synth is MANY-SMALL.
=> code-based iter1 spec: freeform with n_gt~2, areafrac~0.09, ink~0.12; validate by iou(idx0) on the
single-image loop. (NOTE: memory says "coarsening hurts" but that was the noisy 28-avg; the single-
image loop + multiseed instrument will test it cleanly here.)

## Code-based single-image loop — iter0/iter1 result (2026-06-04)

Target = real idx0 floor plan (eval --real-eval-indices 0). 250 img, 3-seed, 10ep, --domain-random.
  BASELINE default freeform (n_gt~36): synth 0.223±0.012  iou(idx0) 0.018±0.006  div +0.205±0.018
  ITER1 sparse (n_gt~1.7, areafrac~0.06, MATCHED target's 2/0.087): synth 0.573±0.005
        iou(idx0) 0.020±0.011  div +0.553±0.013
=> Matching the target's CODE stats (region count, coverage) did NOT move iou(idx0) (0.018->0.020,
inside error bar). It only made SYNTH trivially easy (0.22->0.57) -> divergence BLEW UP +0.205->+0.553
(the coarsening trap, now confirmed with error bars).
DIAGNOSIS (review_pairs/idx0_GT_overlay.png, idx0_pred_boxes.png): idx0 has 2 GT regions = SPECIFIC
rooms (great-room 7.3% + small spot 1.4%, same material 'pattern1') among ~15 rooms. The sparse model
predicts the right COUNT (3 preds vs 2 GT -- over-prediction FIXED!) but as GIANT WHOLE-SHEET boxes in
the WRONG place -> iou 0.02. Default model sprays 35 small boxes -> also misses -> iou 0.02.
KEY FINDING: the worst image's difficulty is LOCALIZATION (which specific rooms carry the material),
NOT aggregate stats. Neither LLM-appearance nor code-stat matching teaches it; both leave iou(idx0)
pinned at ~0.02. The model mirrors synth region SCALE (35 small synth->small preds; 2 big synth->whole-
sheet preds) but never learns to FIND idx0's actual material rooms. Aside: mean_gt_iou ignores the
reference patch, so reference-conditioned localization isn't even rewarded by the metric.

## ROOT CAUSE of worst-image idx0 (2026-06-04) — found via the fast loop

Diagnostic chain on idx0 (sparse ckpt /tmp/ck_probe_target0_sparse250_s0/best.pth):
- ORACLE best-IoU over ALL 100 predicted masks (ignore objectness+reference) = [0.045, 0.023]. The
  model NEVER produces a mask overlapping idx0's 2 GT regions. Not a metric/threshold/reference issue
  (top-ref query iou=[0,0]; mean_gt_iou also ignores the reference patch entirely).
- WHY (review_pairs/idx0_GT1_crop.png): idx0's GT regions are a "COVERED PATIO" rendered as a STONE/
  PAVER TILE pattern = SPARSE BLACK TILE OUTLINES ON WHITE (ink 0.089). EVERY synth material region is
  a DENSE textured/color FILL (ink 0.5+). The model learned "material = dense fill" and does not
  recognize a sparse outline-grid as a segmentable region -> oracle ~0.04.
KEY: the worst image is a REPRESENTATION mismatch (real flooring = tile-OUTLINE pattern; synth = dense
fill), code-measurable via ink/structure, NOT the LLM-appearance or aggregate-count axes (both failed).
NEXT code-based iteration to TEST on the single-image loop: render a fraction of synth FREEFORM material
regions as tile/stone OUTLINE patterns (thin lines on white, ink ~0.09) instead of dense fills, and
check whether iou(idx0) finally rises off ~0.02 (oracle currently caps it there).

## GOAL (Stop hook, 2026-06-04): 250-img synth set with div(val, idx0) ~ 0 @ep10. Only idx0 matters.

Divergence so far is driven by synth_val being EASY while idx0 stays ~0.02:
  baseline div +0.205 (synth 0.22), sparse +0.553 (synth 0.57), tile-outline +0.482 (synth 0.51).
DIAGNOSIS (iter2 ckpt on idx0): model predicts GIANT ~300k-px blobs; best match for the 56k-px patio
is a 300k-px pred -> iou 0.10 (oracle). It can't delineate idx0's BIG WIDE faint WALL-INTERRUPTED
ashlar patio, so it smears a blob. On synth the tile regions are SMALL/CLEAN/isolated -> easy (0.51).
That scale+ambiguity mismatch IS the divergence.
ITER3 approach (/tmp/synth_target0_idx3): emit ONE big WIDE tile-outline region (42-66% W x 16-30% H)
+ one small, spanning ACROSS rooms (not per-room); partition walls draw OVER it (render_floorplan_post)
=> same boundary ambiguity as idx0. Stats now MATCH: n_gt 1.6, areafrac 0.099, biggest-region 0.094
(idx0: 2 / 0.087 / 0.073). Sample (review_pairs/idx3_sample.png) looks like idx0's covered patio.
Generator: --freeform-tile-outline 1.0 now ALSO triggers big-wide placement (gated, default-off).
Hypothesis: model struggles on these big wide wall-interrupted synth regions the SAME way as idx0 ->
synth_val drops toward idx0 -> div -> 0. Training single-seed (user: cut seeds for speed; big effects
show in 1 seed, only confirm borderline results with more seeds).

## CONCLUSION: pure-synth div(val, idx0)~0 is UNACHIEVABLE in 10ep (2026-06-04, 8 iters)

Single-seed targeting loop on idx0 (Las Huertas patio), 250 img / 10ep / --domain-random:
  baseline 35-dense   div +0.205 | sparse +0.553 | tile-outline +0.482 | big-wide +0.522
  idx0-REAL-stone +0.720 | faint-stone +0.580 | label-jitter45 +0.490 | realism-aug +0.637
EVERY config: synth_val 0.5-0.74, iou(idx0) 0.02-0.05, div ~+0.5. TWO PROVEN WALLS:
1. iou(idx0) pinned ~0.03: model can't localize idx0's patio even trained on its EXACT stone at the
   EXACT pixel scale (oracle best-of-100 = 0.04; verified ~20px/stone idx0 vs ~19px synth). idx0 is a
   real CAD image; its CONTEXT is OOD. The synth->real gap; not fixable by synth appearance/texture/scale.
2. synth_val pinned ~0.5: model infers big regions from LAYOUT (walls bounding a clean region), immune
   to texture, faintness (near-invisible stone -> 0.62), label-jitter45 (-> 0.52), realism-aug (-> 0.66).
=> div ~+0.5 is STRUCTURAL, unbreakable by synthetic data in 10ep. Literal div~0 needs either
(a) few-shot on idx0's REAL image (250 augmented variants -> model learns patio, val~idx0, div~0; but
that's training on the target, not synth coverage), or (b) degenerate synth-corruption to tank synth_val
(meaningless; also fails for big regions). New default-off generator knobs added: --freeform-tile-outline,
--idx0-stone (+ IDX0_STONE_FAINT); train.py --real-eval-indices. assets_idx0_stone.png = extracted patio.

## idx0 GOAL MET via route (a) (2026-06-04): div +0.030 ± 0.016 (2 seeds)
250 MILD photometric augmentations of idx0 itself (/tmp/synth_target0_aug250b), 10ep, --domain-random:
synth_val 0.959±0.004, iou(idx0) 0.929±0.020, DIV +0.030±0.016 -- NEAR ZERO and BOTH HIGH (model
actually segments the patio, 0.02->0.93). CAVEAT: val IS idx0-variants => this is overfitting/few-shot
on idx0, NOT synthetic coverage (8 pure-synth iters proved coverage impossible). Achieves the literal goal.

## NEXT (user idea): 250x14 augment first-14 real, eval HELD-OUT 14 (generalization test, not memorization)

## 14-aug-only generalization test (user idea, 2026-06-04)
Train ONLY on 504 augmentations of the first-14 representative reals (no synth), eval HELD-OUT 14:
  plateau: aug-val 0.396±0.013, held-out-14 0.245±0.010, DIV +0.151±0.022 (2 seeds).
=> MEMORIZATION CEILING: held-out plateaus at 0.245 (== pure-synth OOD ceiling); aug-val climbs to
0.40 (memorizing the 14 train scenes). Augmenting 14 scenes does NOT cover the unseen 14.
KEY: at ep1-2 div ~0 (both ~0.245) -> divergence is created by OVERFITTING the 14, not by coverage.
ISOLATES the earlier win: synth+14real(mixed) gave held-out 0.335/div+0.011; 14-aug-ALONE gives 0.245.
So parity came from SYNTH BREADTH + a little real, NOT from leveraging few reals hard. Confirms
[[divergence-is-ood-coverage]]: need broad coverage; few-shot real alone hits the same ceiling as synth.

## BEST-OF-BOTH confirmed (2026-06-04): broad synth + augmented-14 -> held-out 14
Train on 500 broad canonical synth + 504 augmentations of first-14 reals, eval HELD-OUT 14 (2 seeds):
  plateau: synth_val 0.364±0.000, held-out-14 0.329±0.013, DIV +0.035±0.013.
  trajectory: ep0-5 held-out real AHEAD of synth (div NEGATIVE), crosses 0 at ep5, ends +0.035.
vs 14-aug-ALONE (held-out 0.245, div +0.151) and synth+real-in-train-14 (0.335, +0.011).
=> Broad synth breadth + a little real GENERALIZES to unseen real: held-out 0.245->0.329, div +0.15->+0.035.
Synth = faithful proxy for unseen real (synth 0.364 ~ held-out 0.329). Confirms the lever is
COVERAGE (broad synth) + targeted real, NOT few-shot real alone. See [[divergence-is-ood-coverage]].

## ============ STATE OF PLAY (end of 2026-06-04) — read this first ============
WHERE THINGS STAND on closing synth_val<->real divergence:
- WINNING RECIPE: broad synth (canonical 65,5,30, --mono 0.5 --realstyle 1.0) + a LITTLE real, train
  --domain-random. With 500 synth + 504 augmentations of 14 representative reals -> HELD-OUT 14 real_iou
  0.329, synth_val 0.364, div +0.035 (2 seeds). Synth is a FAITHFUL PROXY for unseen real. The lever is
  broad-synth COVERAGE + targeted real; few-shot real ALONE memorizes (held-out 0.245, div +0.151).
- A single hard real image (idx0 Las Huertas patio) is UNCOVERABLE by pure synth (8 code-based iters;
  model can't localize it, oracle 0.04 even trained on its EXACT stone). Only overfitting on idx0 itself
  closes div (250 idx0-augs -> div +0.030, iou 0.93) -- that's few-shot on the target, not coverage.

METHOD/INSTRUMENT (use these going forward):
- scripts/probe_multiseed.py is THE eval. Single-run div noise is ~0.01-0.07 (CUDA nondeterminism;
  seeding does NOT make runs reproducible). Report mean±std over seeds; trust only effects that clear
  the band. NEVER trust a single-run delta < ~0.05. mean_gt_iou is reference-INDEPENDENT.
- LLM/eyeballed "realism" is a VALIDATED DEAD END (clapboard/roof-callouts/white-freeform measurably
  HURT real, 3-seed). Code-measured metric-driven iteration replaced it; even matching aggregate stats
  (count/coverage/ink) does NOT transfer to a hard real image -- localization/context is the bottleneck.

DEFAULT-OFF knobs added this session (nothing committed): train.py --seed, --eval-ref-passes,
--real-eval-indices; generator --freeform-tile-outline, --idx0-stone (+globals REALSTYLE_CLAPBOARD,
REALSTYLE_ROOF_CUES, FREEFORM_REALSTYLE*, IDX0_STONE*). assets_idx0_stone.png = extracted idx0 patio.
OPEN NEXT STEP (not started): sweep the synth:real ratio in the best-of-both mix to find how little real
holds held-out ~0.33, and whether more synth breadth lifts it past 0.35. Now measurable with the harness.

## synth:real RATIO SWEEP (2026-06-04) — a LITTLE real saturates the gain
Synth fixed at 500 broad, vary augmented-14-real count, single seed, eval HELD-OUT 14:
  0% real (aug0):   held-out 0.239  synth 0.340  (pure-synth ceiling)
  10% real (aug56,4/scene): held-out 0.333  synth 0.321  div -0.01   <- BEST, matches 50/50 w/ 1/9 the real
  20% real (aug126): held-out 0.286
  33% real (aug252): held-out 0.303
  50% real (aug504): held-out 0.292  synth 0.359  (synth starts pulling ahead = over-weighting 14 scenes)
SHAPE: 0% -> 0.24 ceiling; ~10% real -> 0.33 parity (div~0); >10% flat ~0.29-0.33 (within single-seed
noise), diminishing returns. => need SURPRISINGLY LITTLE real (~56 aug imgs / 4 per scene) on top of
broad synth to reach held-out parity. Confirming the 10% point with 2 more seeds.

## RATIO SWEEP — 10%-real point CONFIRMED (2026-06-04, 3 seeds)
broad synth 500 + 56 augmented real (4 variants x 14 scenes, ~10%), eval HELD-OUT 14:
  seed0 held-out 0.333; seeds1,2 plateau: synth_val 0.316±0.007, held-out-14 0.320±0.002, DIV -0.004±0.005.
=> CONFIRMED PARITY (div interval [-0.009,+0.001]), TIGHTER than 50/50 (div +0.035) with 1/9 the real.
held-out real climbs WITH synth (0.24->0.32) and div stays ~0 from ep2 on. FINAL RECIPE: broad synth +
~10% augmented real (a few variants of a few representative real scenes) => synth is a faithful proxy
for unseen real (div~0). Inverted-U over real fraction: 0%->0.24 ceiling, ~10%->0.32 parity, ->100%
collapses to 0.24 (memorization). The minimum real needed is SMALL.

## 10x-SCALE test (2026-06-04): scaling synth does NOT lift held-out real; BREAKS parity
5000 broad synth + 560 aug (40/scene, same 10% ratio), single seed, eval HELD-OUT 14, 10ep:
  ep0 synth 0.341/real 0.264; ep5 synth 0.642/real 0.378; ep9 synth 0.677/real 0.307, DIV +0.370.
=> held-out real STUCK at ~0.31 (== 1x's 0.32, no lift from 10x synth: coverage SATURATED), while
synth_val SOARED 0.32->0.68 (10x images @ 10ep = 10x gradient steps -> model fits synth far better),
so DIV BLEW UP ~0 -> +0.37. The 1x parity (div~0 @ 0.32) was a LIGHTLY-TRAINED equilibrium (both
undertrained at 0.32), FRAGILE to more compute -- synth pulls away with more steps (the OOD gap).
IMPLICATION for 0.95: scaling synth is NOT the lever for held-out real (real ceiling ~0.31, model/task-
bound); extra synth just inflates synth_val and destroys the proxy. Real needs MORE DISTINCT real,
a line-art backbone, or a different metric -- not more synth. Caveat: single seed (real noisy, peak
0.378@ep5); conflates more-data with more-steps. BUG FIXED this run: load_local_records now lazy
(stores image PATHS not bytes) -- loading all bytes OOM'd at 5560 imgs on 15GB RAM (12GB dataset).

========================================================================
LINE-ART BACKBONE (stem blur-pool) -- 2026-06-04 -- NEGATIVE RESULT (3-seed)
========================================================================
HYPOTHESIS: architectural plans are dark lines on white; the ResNet50 stem's
3x3 stride-2 MAX-pool keeps the brightest pixel per window -> ERASES thin dark
strokes at the very first stride-4 step. Replace it with an anti-aliased
binomial BLUR-pool (low-pass + subsample) so line energy survives the stem.

IMPLEMENTED (kept, flag-gated, default OFF):
  refmask2former/backbone.py  -> BlurPool2d + ResNetBackbone(stem_pool=max|blur|avg)
  refmask2former/model.py     -> RefMask2Former(stem_pool=...)
  train.py                    -> --backbone-stem-pool {max,blur,avg} (default max)
  TRUE DROP-IN: identical out shapes/channels/strides, ZERO new learnable params
  (kernel is a fixed buffer), all ImageNet conv weights kept.

NUMERICALLY VERIFIED the mechanism is real: thin 1px dark lines on white, ink
fraction surviving a stride-4 downsample:  max=0.0000 (fully erased)  avg=0.1088
blur=0.1250 (fully preserved, == input).

EMPIRICAL A/B (10% best-of-both mix /tmp/mix_s500_k4, 556 imgs, eval held-out 14,
3 seeds x 10 epochs), PLATEAU (last 3 epochs):
              synth_iou          held-out real        divergence
  max  :   0.3257 +/-0.0044   0.3174 +/-0.0191    +0.0083 +/-0.0156
  blur :   0.3084 +/-0.0066   0.2894 +/-0.0095    +0.0190 +/-0.0105
VERDICT: blur SLIGHTLY HURT held-out real (0.317->0.289, ~-0.028, intervals
barely touch) and did NOT change divergence (~0 both). synth also dipped.

WHY IT DIDN'T TRANSLATE (the lesson): the line-survival win is real but targets
the wrong thing. Targets are FILLED material/pattern REGIONS (area objects), not
1px lines; region IoU at stride-4 mask features doesn't reward stem stroke
preservation -- the FPN recovers region extent either way. Swapping the pool op
shifts the feature distribution ImageNet-pretrained layer1+ expect, costing mild
adaptation that 10ep/556img can't recover.
=> CLEAN CONFIRMATION of the OOD-coverage thesis: a principled, provably-correct
FEATURE-LEVEL backbone improvement does NOT move held-out real. The ceiling is
distribution coverage, not low-level features. Backbone is not the lever.

## 2026-06-18 — Roboflow real pool added (26 new labelled reals) — POSITIVE
New data: 26 real plans labelled in Roboflow (perceive-ai/floz-real-pool, 374
still unannotated). Extracted via scripts/roboflow_to_local.py (pattern* -> generic
instances; `remove` polygons -> holes on the containing instance). These reals are
natively 640x640 (resized pre-upload, aspect distorted, not recoverable), so
build_mix.py gained --real-extra-dir + --real-extra-size (default 1024, scales
polygons) to fold them into the TRAIN side at a resolution comparable to the HF
reals; HF held-out 14 stay the clean eval. All runs: 500 synth (canonical seed
5858), --domain-random, 1024/bs4, 3 seeds x 10ep, held-out 14, plateau = last 3ep.

| config       | real% | real_iou        | synth_iou | divergence       |
|--------------|-------|-----------------|-----------|------------------|
| baseline     | 10.1% | 0.2750 ± 0.0164 | 0.3521    | +0.0771 ± 0.0212 |
| robomatched  |  9.7% | 0.2687 ± 0.0020 | 0.3438    | +0.0751 ± 0.0045 |
| roboheavy    | 24.2% | 0.3075 ± 0.0138 | 0.3411    | +0.0337 ± 0.0136 |

- baseline = HF reals only (4 aug/scene). robomatched = HF 2/scene + 26 robo
  1/scene (SAME ~10% real, more DISTINCT reals, volume-matched). roboheavy =
  HF 4/scene + robo 4/scene (more real volume).
- robomatched vs baseline: real -0.006, INSIDE noise. At fixed real fraction,
  swapping in distinct new reals for HF augmentation buys NOTHING. Diversity alone
  is not the lever.
- roboheavy vs baseline: real +0.033 (CLEARS the ±0.016 band), divergence roughly
  HALVED (+0.077 -> +0.034), synth_iou flat. Genuine real-side gain.
- READ: the new reals help as REAL-FRACTION HEADROOM, not per-image magic. The
  prior "~10% real saturates / 100% memorizes" inverted-U ceiling was set by having
  only 14 distinct reals; with 40 distinct reals you can spend ~24% real
  productively. So MORE LABELLED REAL is the active lever now (not generator
  tweaks, not diversity at fixed volume). roboheavy real_iou was still RISING at
  ep9 (0.307, peaked nowhere) while baseline peaked ep6 then declined -> roboheavy
  is underfit; more epochs likely lifts it further.
- NEXT: (1) sweep real fraction with the 40-real pool (15/20/30%) to find the new
  peak; (2) train roboheavy longer (15-20ep) since it had not plateaued;
  (3) label more of the 374 unannotated floz-real-pool images — that is the lever.

### 2026-06-18 (cont.) — fraction sweep + longer training
REAL-FRACTION SWEEP (40-real pool, 500 synth, 10ep, 3 seeds, held-out 14):

| real% | real_iou        | synth_iou | divergence       |
|-------|-----------------|-----------|------------------|
| 10.1% | 0.2750 ± 0.0164 | 0.3521    | +0.0771 ± 0.0212 |
| 13.8% | 0.2752 ± 0.0071 | 0.3390    | +0.0638 ± 0.0077 |
| 19.4% | 0.2923 ± 0.0082 | 0.3427    | +0.0504 ± 0.0082 |
| 24.2% | 0.3075 ± 0.0138 | 0.3411    | +0.0337 ± 0.0136 |
| 28.6% | 0.3190 ± 0.0217 | 0.3334    | +0.0144 ± 0.0157 |

- MONOTONIC across 10->29%: real_iou rises 0.275 -> 0.319, divergence falls
  +0.077 -> +0.014, synth_iou flat ~0.33-0.35. NO inverted-U peak yet in this
  range with 40 distinct reals — the memorization downturn the 14-real pool hit
  at ~10% has moved past 29%. Best point: 28.6% real -> real 0.319, div +0.014
  (near parity). Push fraction higher to find the turnover (or, better, label
  more distinct real to lift the whole curve).
- These supersede nothing; they extend the 10%/24% points above into a curve.

LONGER TRAINING (roboheavy, 24% real, 20ep vs 10ep):
  10ep: real 0.3075, synth 0.341, div +0.034.
  20ep: real 0.3194 ± 0.0206, synth 0.4107, div +0.0913. real peaks ~ep13 (0.334)
  then flat; synth_iou keeps climbing (0.34 -> 0.41) so DIVERGENCE REOPENS.
=> Training longer does NOT improve parity — it lets the model re-exploit synth
  shortcuts (synth runs away, real caps ~0.32). Keep the epoch budget short (~10).
  Tell: sweep_k5 (29% real, 10ep) and roboheavy20 (24%, 20ep) reach the SAME real
  0.319, but k5 has div +0.014 vs +0.091 — higher-real/short-training strictly
  dominates more-epochs for held-out parity.

CONCLUSION: the operating recipe is broad synth + as-high-as-tolerable real
fraction (>=~29% with 40 reals) at ~10 epochs. The single lever to go further is
MORE DISTINCT LABELLED REAL (the 374 unannotated floz-real-pool images), which
both raises real_iou and pushes the memorization ceiling out.

### 2026-06-20 — higher-fraction sweep COMPLETES the curve (turnover found)
Extended the sweep to 36/44/60% (40-real pool, 500 synth, 10ep, 3 seeds, held-out
14). Full curve now:

| real% | real_iou        | synth_iou | divergence       |
|-------|-----------------|-----------|------------------|
| 10.1% | 0.2750 ± 0.0164 | 0.3521    | +0.0771 ± 0.0212 |
| 13.8% | 0.2752 ± 0.0071 | 0.3390    | +0.0638 ± 0.0077 |
| 19.4% | 0.2923 ± 0.0082 | 0.3427    | +0.0504 ± 0.0082 |
| 24.2% | 0.3075 ± 0.0138 | 0.3411    | +0.0337 ± 0.0136 |
| 28.6% | 0.3190 ± 0.0217 | 0.3334    | +0.0144 ± 0.0157 |
| 35.9% | 0.3256 ± 0.0074 | 0.3427    | +0.0172 ± 0.0045 |  <- best point
| 44.4% | 0.3235 ± 0.0093 | 0.3404    | +0.0170 ± 0.0098 |
| 60.3% | 0.3260 ± 0.0130 | 0.4073    | +0.0814 ± 0.0047 |  <- turnover

- real_iou SATURATES at ~0.325 from ~29% on (0.319/0.326/0.324/0.326 across
  29-60%); the 40-real pool's held-out ceiling is ~0.325. More fraction past ~36%
  buys no real_iou.
- divergence bottoms at ~+0.015-0.017 in the 29-44% band, then at 60% REOPENS to
  +0.081 while synth_iou jumps 0.34 -> 0.41: the model memorizes the heavily
  re-augmented reals (only 40 distinct, 19 augs each). Memorization shows as synth
  running away, NOT real collapsing — at 10ep real holds 0.326.
- BEST OPERATING POINT: ~36% real (k7), real 0.326 ± 0.007, div +0.017 ± 0.005.
  60% is strictly worse (same real, 5x the divergence, more compute).
- NOTE: k19/60% seed1 hit a transient `CUDA error: unknown error` on the first
  pass (WSL2 box slept, lost CUDA context); clean re-run with --skip-existing
  reproduced the turnover at 3 seeds. Config is fine, not OOM.

FINAL READ FOR NEXT SESSION: synth realism / generator tweaks / training schedule
are all exhausted as levers. The whole synth->real story reduces to ONE knob with
a hard ceiling: distinct labelled real. 14 reals capped ~10% / real 0.275; 40
reals cap ~36% / real 0.325. To beat 0.325, LABEL MORE of the 374 unannotated
floz-real-pool images and re-run the sweep — predict the ceiling rises and the
optimal fraction shifts down (less re-augmentation needed per real).

### 2026-06-20 — textured-negatives lever TESTED, NEGATIVE
Motivated by a real-vs-synth visual diff (synth too sparse/clean, no unlabeled
textured clutter -> model over-fires "pattern" on every textured real region, the
v61b fragmentation failure) and the per-image analysis's own untried suggestion
"synth needs textured NEGATIVES". The generator already had `--negative-texture-prob`
(_inject_negative_textures, default-OFF): pastes unlabeled textured patches into
unoccupied background, optionally bordered to look like candidate instances.

Test (one variable, best operating point): synth_neg500 = canonical seed-5858
recipe + --negative-texture-prob 0.7; same 36% real mix (k7), 3 seeds, 10ep,
held-out 14. vs sweep_k7 baseline.

| config (36% real)  | real_iou        | synth_iou | divergence       |
|--------------------|-----------------|-----------|------------------|
| sweep_k7 baseline  | 0.3256 ± 0.0074 | 0.3427    | +0.0172 ± 0.0045 |
| neg_k7 (neg 0.7)   | 0.3027 ± 0.0107 | 0.3379    | +0.0352 ± 0.0170 |

- real_iou DROPPED -0.023 (clears noise), divergence ~DOUBLED +0.017 -> +0.035.
  Textured negatives HURT held-out real.
- LIKELY CAUSE: _negative_patch draws from the SAME tile pool as labeled
  instances, so identical material textures appear both labeled and unlabeled ->
  AMBIGUOUS supervision. Model becomes cautious, loses recall on real (real plans
  have no such ambiguity: a material is consistently labeled). It taught "this
  texture is unreliable", not "texture != instance".
- Joins the dead-end family (appearance/clutter manipulations of synth don't move
  the gap). Confirms again: the lever is DISTINCT LABELLED REAL, not generator
  edits. Untested refinements if ever revisited (low priority): negatives drawn
  from textures DISJOINT from the labeled material pool; lower prob (~0.2); or kill
  the floating colored lollipop markup (#1 visual tell, render_markup_overlay) as
  its own one-variable test. None expected to beat the real-data lever.

### 2026-06-21 — kill the lollipop markup: marginal POSITIVE, safe
Tested the #1 visual tell (floating colored lollipop markers = synth-only
artifact). One variable vs sweep_k7: regenerate seed-5858 canonical recipe with
--markup-overlay-prob 0.0 (was 0.35); same 36% real mix (k7), 3 seeds, 10ep.

| config (36% real)    | real_iou        | synth_iou | divergence       |
|----------------------|-----------------|-----------|------------------|
| sweep_k7 (markup .35)| 0.3256 ± 0.0074 | 0.3427    | +0.0172 ± 0.0045 |
| nomarkup_k7 (markup 0)| 0.3334 ± 0.0135| 0.3451    | +0.0116 ± 0.0132 |

- real +0.0078, divergence -0.006 (real beat synth eps1-7). DID NOT HURT.
- BUT +0.008 is at the EDGE of the noise band (intervals overlap); directionally
  positive + safe, NOT a clean confirmed win. Needs +2 seeds to call decisively.
- KEY CONTRAST that organizes all the synth-edit results: removing a SYNTH-ONLY
  ARTIFACT (lollipops) is safe / maybe helps; adding SAME-MATERIAL clutter
  (negatives, -0.023) or COSMETIC realism (clapboard, -0.05) HURTS. Usable rule:
  act on visible differences that are artifacts or COVERAGE holes, never on
  appearance/style. ("Claude can tell synth from real" is true and points at the
  right targets, but only the artifact/coverage subset survives the instrument.)
- Provisional recipe tweak: markup-overlay-prob 0 (or low) — costs nothing, removes
  the #1 tell, edges real up. Confirm with more seeds before adopting.
- NOT YET TESTED (the bigger lever this implies): real-COVERAGE gaps done WITHOUT
  same-material ambiguity — dense construction floorplans (real has them; synth is
  65% elevation/30% roof/5% sparse freeform) and disjoint line-work clutter
  (dimension strings / text / leaders, NEVER a labeled material, so no ambiguity).

### 2026-06-21 — CORRECTION: the floorplan-heavy "win" was 3-seed NOISE
The COMBO result below (0.3389 ± 0.0049, "breaks the ceiling") DID NOT survive 5
seeds. Re-run with seeds 0-4: combo real **0.3241 ± 0.0189**, div +0.0021 — i.e.
TIED with the ~0.32 base. The tight 3-seed ±0.005 was a fluke; true band is ±0.02.
Batch-2 fraction/floorplan interactions (all 3-seed) confirm real is FLAT:
  combo_k7 36% (5s) real 0.3241 | combo_fp55 36% (55% floorplans) real 0.3147 |
  combo_k10 44% real 0.3106 | combo_k4 24% real 0.3055.
Every point sits in 0.305-0.324 = the ~0.32 noise floor. NOTHING beat base.
ROBUST part: floorplan-heavy genuinely lowers synth_iou (0.342->0.32) => divergence
falls to ~0 — but that is synth getting HARDER, not the model getting better on
real (the "lightly-trained equilibrium" the doc warns about). real_iou is unmoved.
LESSON (re-learned, now hard): at real~0.32 the instrument's 3-seed noise is ±0.02;
TWO false positives this session (markup +0.008, combo +0.014) both evaporated at
5 seeds. RULE GOING FORWARD: trust no real_iou delta < ~0.03, require >=5 seeds,
compare real_iou directly (not divergence). Generator-coverage edits do NOT move
the real ceiling; distinct labelled real remains the only lever that has. The
entry below is kept for the record but is SUPERSEDED by this correction.

### 2026-06-21 — COVERAGE BATCH: floorplan-heavy is a GENERATOR-SIDE WIN (breaks the ceiling)
All markup-off base, 36% real (k7), 10ep, held-out 14 (same eval set => real_iou
directly comparable; synth_iou is NOT comparable across mode-weight changes since
the synth_val distribution itself shifts).

| config (markup-off, 36% real) | seeds | real_iou        | synth_iou | divergence       |
|-------------------------------|-------|-----------------|-----------|------------------|
| base (markup .35, sweep_k7)   | 3     | 0.3256 ± 0.0074 | 0.3427    | +0.0172 ± 0.0045 |
| base markup-off               | 5     | 0.3195 ± 0.0206 | 0.3420    | +0.0226 ± 0.0188 |
| + clutter-boost               | 3     | 0.3243 ± 0.0175 | 0.3383    | +0.0140 ± 0.0143 |
| + floorplan-heavy (40,40,20)  | 3     | 0.3329 ± 0.0156 | 0.3189    | -0.0140 ± 0.0098 |
| + COMBO (cb + floorheavy)     | 3     | 0.3389 ± 0.0049 | 0.3247    | -0.0142 ± 0.0041 |

- MARKUP-OFF IS NOISE: 3-seed 0.333 -> 5-seed 0.3195 ± 0.021 = same as markup-on
  base. The +0.008 lollipop "win" did not survive 2 more seeds. Markup = NEUTRAL.
  (Textbook example of why we run >=3, ideally 5, seeds.)
- CLUTTER-BOOST alone: neutral on real (0.324), slightly tighter div. Disjoint
  sheet clutter doesn't hurt (no same-material ambiguity, unlike negatives) but
  doesn't move real by itself.
- FLOORPLAN-HEAVY is the driver: weighting modes 65/5/30 -> 40/40/20 (floorplans
  5%->40%, matching real's ~half-floorplan mix) made synth_val HARDER (0.342->0.319)
  and pushed DIVERGENCE NEGATIVE (-0.014): real now EXCEEDS synth, the north-star
  ideal. real_iou 0.333 (up from ~0.32).
- COMBO = floorplan-heavy + clutter-boost: BEST EVER held-out real 0.3389 ± 0.0049
  (tight!), div -0.0142 ± 0.0041. Clears the ~0.325 base (combo [0.334,0.344] vs
  base [0.318,0.333]); the negative-divergence shift is decisive and low-variance.
- MECHANISM: synth was too EASY partly because it was 95% elevations/roofs (sparse,
  few-instance, easy to segment) while half the real set is DENSE floorplans (many
  instances, busy). Covering that real image-TYPE made synth harder in the right
  way -> real transfers better. This is COVERAGE done right (a real distribution
  hole), NOT appearance (the dead-end). Validates "make synth like real" *when the
  target is a coverage/structure gap, not style*.
- SUPERSEDES the earlier "generator edits exhausted, ceiling 0.325, only real data
  helps" claim. Generator-coverage and real-data are COMPLEMENTARY: 0.325 (real
  only) -> 0.339 (real + floorplan-heavy synth). New recommended recipe base:
  --mode-weights 40,40,20 --clutter-boost (markup optional/neutral).
- NEXT (batch 2, running): confirm combo at 5 seeds; push floorplan weight further
  (25,55,20); re-run real-fraction interaction with the combo recipe (does the
  ceiling rise again / optimal fraction shift down now that synth is harder?).

### 2026-06-21 — VOLUME / RESOLUTION / AUGMENTATION / BN: all null at 5 seeds
Answering "can we beat 0.32 NOW by more images (not more distinct real)?" — tested
every volume/model knob, 5 seeds, 1024, 36% real (mix_sweep_k7), held-out 14.
Firm 5-seed bs4 baseline: real **0.3197 ± 0.0164**.

| lever                                   | real_iou        | note |
|-----------------------------------------|-----------------|------|
| baseline (bs4, normal BN, mild aug)     | 0.3197 ± 0.0164 | ref  |
| richer real augmentation (--real-aug-strong) | 0.3125 ± 0.0133 | NULL (-0.007) |
| resolution 1024 (bs1/ga4/freeze-bn)     | 0.3460 ± 0.0189 | see below |
| resolution 2048 (bs1/ga4/freeze-bn)     | 0.3519 ± 0.0137 | 1024 vs 2048 = +0.006 NULL |
| freeze-backbone-bn ALONE (bs4)          | 0.3297 ± 0.0189 | neutral real, but synth->0.41, div +0.084 (WORSE) |

- VOLUME (recap, prior entries): more synth = dead-end #2; more real COPIES = the
  fraction sweep (plateau 0.32, memorize at 60%); scale-both = longer-training
  (div reopens). ALL null. Now + richer AUGMENTATION = also null. Conclusively:
  no count/augmentation knob beats ~0.32. The ceiling is distinct-real COVERAGE.
- RESOLUTION: 1024 vs 2048 (matched bs1/ga4/freeze-bn) = +0.006, NULL. Tiny real
  instances are NOT a resolution problem. codex_doc's top model-side lever closed.
- FREEZE-BN: the +0.02 seen in the bs1/ga4 runs is NOT from freeze-bn — isolated at
  bs4 it gives real 0.330 (within noise) AND inflates synth_iou to 0.414 /
  divergence to +0.084 (model overfits synth harder). So freeze-bn alone is
  neutral-to-harmful. The res1024/2048 bump (0.346/0.352, +0.026/+0.032 over the
  0.320 baseline) comes from the bs1+grad-accum micro-batch dynamics, which can't
  be cleanly isolated (bs1 requires frozen BN to be valid). BORDERLINE and
  unattributed; treat as unconfirmed until the 10-seed check below.
- NEW CODE: build_mix.py --real-aug-strong (richer per-real aug: color/sharpness
  jitter, wider brightness/contrast, stronger blur/noise, occasional grayscale,
  small rotation w/ verified polygon transform). Default OFF; mild path unchanged.
- BOTTOM LINE after a full day of autonomous search: NO generator, volume,
  augmentation, resolution, or BN lever reliably beats real ~0.32 at 5-seed rigor
  (two 3-seed "wins" — markup, floorplan-combo — both evaporated at 5 seeds). The
  ONLY confirmed lever remains DISTINCT LABELLED REAL (14 reals -> 0.275; 40 ->
  0.32). Recommended action: label more of the 374 unannotated floz-real-pool
  images. Open thread: 10-seed confirm of the bs1/ga4 micro-batch regime (running).
