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
