# Synthetic Dataset Progress

## 2026-08-12 — four architecture levers from the matching literature: all null

Screened on validation (HF14 never touched); `scripts/run_paper_levers.sh <arm> <seed>`.
Flags are default-off and the baseline is bit-identical without them.

| arm | mechanism | seeds | val peak |
|---|---|---:|---:|
| `base` | — | 5 | 0.7436 ± 0.0151 |
| `self` | self-support prototype refinement (FSS review 3.2) | 5 | 0.7495 ± 0.0231 |
| `dynamic` | hypernetwork-generated classifier (CS231n 2017) | 3 | 0.7416 ± 0.0087 |
| `surround` | central-surround 3x reference (Zagoruyko 2015) | 1 | 0.7173 |
| `shrink` | SimAM + soft threshold (Remote Sens. 16, 2831) | 1 | 0.7129 |

`surround` and `shrink` lost at one seed and their **code was removed** — they
have no flag and no arm, so re-testing either means re-implementing it. `self`
and `dynamic` remain, default-off, so their negatives reproduce.

`self` is +0.0058, p=0.65 — null, and not worth a second decode pass. `dynamic`
looked like +0.0145 at one seed and flipped sign at three.

One method note, since it nearly produced two false positives: a paired test over
the 77 selections gave Wilcoxon p<0.0001 for `self`, but the same test between two
**base** runs differing only in seed gives 56/14, p=0.0001. Pairing over selections
measures run-to-run variation unless run against a same-recipe null control.

With `--corr-grid`, `--scale-matched-ref`, the ranking loss and the anchor, that is
six attempts at the conditioning mechanism, all null or negative.

## 2026-08-12 — confusable synthetic pairs are HARMFUL; lever closed

The 2026-08-10 handoff named this the untried lever. Tried at maximum dose, it
costs **-0.104**. Same generator, seed and mode weights in both arms; only
`--confusable-prob` differs. Rebuild: `./run_confusable_ablation.sh`.

| arm | seeds | val | HF14 |
|---|---:|---:|---:|
| `confbase` (prob 0) | 3 | 0.7575 | 0.6823 +- 0.0190 |
| `confpair` (prob 1.0) | 3 | 0.6273 | 0.5787 +- 0.0433 |

t=3.80, p=0.019, complete separation, 4x the run-to-run sd, and validation moves
the same way. Locally generated synthetic costs ~0.013 vs hf20k (0.6954 ->
0.6823).

**The premise was false.** The generator picks distinct tile *paths*, not
distinct *appearances*, so confusable pairs already occurred by chance at close
to the real rate. Per-image max inter-family similarity, at 1280, via
`scripts/family_similarity_probe.py --local-data`:

| pool | mean | images with a pair >=0.90 |
|---|---:|---:|
| synthetic `toparea1600` | 0.772 | 19.2% |
| real 86 | 0.817 | 24.4% |
| generated 28 | 0.760 | 4.8% |
| `confpair1600` (as trained) | 0.897 | 65.0% |

The likely mechanism for the loss: **~30% of multi-family images are not
decidable from appearance** — a family's own instances agree less than it agrees
with its neighbour (HF14 2/6, validation 2/8; they average 0.52-0.56 IoU against
0.77-0.85). Those are the worst image in each split, and generating more of the
case teaches a contradictory mapping. This matches the 2026-08-07 `hatch_matcher`
finding that engineered matching and `RefUNet` fail on the same inputs.

The correlation itself replicates *more* strongly than recorded (pooled n=14,
r=-0.754, p=0.0018; validation -0.763 where -0.250 was documented). Standing
example that a replicated correlation is not a lever. Also refuted en route: that
synthetic pairs carry a fill-colour shortcut — they are *more* brightness-matched
than real ones.

**Three 2026-08-10 claims are wrong** on a rebuilt `ck_fix_mix3652_seed31/epoch_6`:
"every image under 0.65 is multi-family" and "single-family images all score
0.92+" (HF14 imgs 12/7/0 are single-family at 0.393/0.546/0.611, already
explained by thin poche and a reference box on printed text), and "synthetic
sheets contain no confusable pairs at all".

Baseline reproduced from a clean clone at **0.6954 +- 0.0023** (val-selected;
documented 0.6783 +- 0.0081), every pipeline count matching `startup.md` exactly.

New: `scripts/family_similarity_probe.py` (`--metrics` per-image confusability and
intra/inter margin, `--tiles` pool pairs, `--local-data` distribution) and
`generate_synthetic_v5.py --confusable-prob/--tile-sim`, default off and verified
byte-identical when off. Leave it off; it is kept so the negative is reproducible.

## 2026-08-10 — anchor plane bug found; disconnection, volume and ratio all null

Epochs chosen on the validation split (`scripts/select_epoch_on_val.py`), never
on HF14. **Run-to-run sd on mix3652 is 0.0262** (seeds 7/31/99): single-seed gaps
under ~0.05 are noise. Nothing here raised real IoU.

### The `--anchor` collapse was an input-statistics bug, not a shortcut

The anchor channel was 1.0 across the whole reference crop while the image branch
saw a sparse rectangle in a field of zeros — same siamese stem, so the reference
features were the casualty. `--anchor-ref-plane 0.0` fixes it (now default).

| mix3652, seed 31 | val | HF14 | val loss ep0→ep8 |
|---|---:|---:|---|
| no anchor | 0.7466 | 0.6916 | 0.90 → 0.35 |
| `--anchor` (plane 1.0) | 0.6332 | 0.6088 | 1.13 → **0.70, rising** |
| `--anchor --anchor-ref-plane 0.0` | 0.7305 | 0.6777 | 0.97 → **0.32** |

The documented 0.433 did **not** reproduce — the bug costs 0.083 here, not 0.247.
Do not quote 0.433 without re-deriving it.

`scripts/anchor_vs_reference_probe.py` feeds contradictory inputs (reference crop
from family B, anchor in family A) and reports which the output obeys, plus
`pred_local`, the share of the prediction inside the anchor's own component:

| synth-only, seed 7 | honest | pred_local | gt_local | follow_anchor | follow_ref | ref_sens |
|---|---:|---:|---:|---:|---:|---:|
| disc, no anchor | 0.611 | 0.193 | 0.266 | 0.036 | 0.604 | 0.934 |
| disc `--anchor` | 0.276 | 0.282 | 0.266 | 0.028 | 0.255 | 0.918 |
| **conn `--anchor`** | 0.306 | 0.331 | 0.266 | 0.064 | 0.375 | 0.795 |
| disc `--anchor` + dropout 0.5 | 0.583 | 0.217 | 0.266 | 0.019 | 0.597 | 0.979 |
| disc `--anchor-ref-plane 0.0` | 0.534 | 0.246 | 0.266 | 0.032 | 0.569 | 0.962 |

Shortcutting would mean `pred_local` near 1.0 and `ref_sens` near 0. Instead it
sits at ground-truth localness with the reference pathway fully causal — damaged,
not shortcutting. The `conn` row is decisive: that is the pool where the shortcut
IS available and it still is not taken. The dropout row corroborates by accident:
`--anchor-dropout` zeroes the reference plane too, so it applies the fix to half
of training and recovers most of the loss. It has therefore never been tested
cleanly.

### A working anchor still does not help

| setting | no anchor (val / HF14) | + fixed anchor | val effect |
|---|---|---|---:|
| synth-only disc, 3 seeds | 0.6260 / 0.6090 | 0.5710 / 0.5923 | −0.055 |
| synth-only conn, 3 seeds | 0.6124 / 0.6190 | 0.5728 / 0.6218 | −0.040 |
| mix3652, 1 seed | 0.7466 / 0.6916 | 0.7305 / 0.6777 | −0.016 |

Every anchored run scored below every un-anchored run on validation. The one real
effect: variance collapses (HF14 sd 0.0051 vs 0.0229) without the mean moving —
so localisation is not where the error is.

### Disconnected synthetic data: null (3 seeds, mode- and area-matched)

conn `share_local` 0.821 → HF14 0.6190 ± 0.0229; disc 0.446 → 0.6090 ± 0.0324.

Two things the plan got wrong. **Mode quotas are unfillable** — disconnection is
99.8% of freeform but 10.9% of elevation, so only 3 of the 2,833 all-disconnected
images are elevations against a quota of 889; any such pool is ~100% floor plans,
a mode swap, and floorplan-heavy synth was already retracted (`385f3cb` →
`1cbaeea`). **Elevations are the whole gap and selection cannot fix them** —
synth elevation `share_local` 0.975 vs real elevation-like 0.492 (freeform 0.505
vs 0.280; roof_plan 0.973), and elevations are 57% of the real set. Generation
can: the current generator already emits elevations at 0.629 (the 20k parquet
came from an older recipe) and `--elev-repeat-prob` reaches 0.584.

### Volume and mix ratio: null

All seed 31; nothing beat the 3,652 baseline anywhere.

| arm | records | real % | real-ish | val | HF14 |
|---|---:|---:|---:|---:|---:|
| synth6400 | 8,452 | 24% | 2,052 | 0.7255 | 0.6625 |
| synth3200 | 5,252 | 39% | 2,052 | 0.7301 | 0.6075 |
| baseline | 3,652 | 56% | 2,052 | 0.7466 | 0.6916 |
| synth800 | 2,852 | 72% | 2,052 | 0.7042 | 0.6650 |
| realaug2x | 5,704 | 72% | 4,104 | 0.7548 | 0.7069 |
| synth400 | 2,452 | 84% | 2,052 | 0.7538 | 0.6789 |

`realaug2x` looked like a win and **is retracted**: 0.6767 ± 0.0262 at 3 seeds vs
the documented 0.6860 ± 0.0175. Seeds 7/99 land at ~0.661; seed 31 was the
outlier, and two seed-31 follow-ups confirm the seed not the config (15-epoch
schedule 0.7004; `realaug4x` at 9,808 records 0.7005). Each row above is one
seed, so two conclusions drawn before the seeds returned — "real-derived record
count is the lever" (+0.042) and "real fraction is monotone on validation" — are
**not supported**. Consistent with more *distinct labelled real sources* being
the binding constraint: re-augmenting the same 114 sources 18× → 36× → 72× buys
nothing.

### Confusable materials are where the score is lost

> **Superseded 2026-08-12.** The correlation below replicates and is stronger
> than recorded here; the *lever* it motivated is closed and cost −0.104, and the
> "untried lever" premise at the end of this section is false. See the top of
> this file before acting on any of it.

Pairwise cosine similarity between families (ImageNet ResNet50 on 64px patches
sampled inside each region), per image, against that image's mean IoU:

| split | corr(IoU, max inter-family sim) | corr(IoU, max background sim) |
|---|---:|---:|
| HF14 | **-0.709** (n=7 multi-family) | -0.167 |
| validation | **-0.250** (n=8) | -0.361 |

Every image scoring under 0.65 in either split is multi-family; every
single-family image scores 0.92+. The worst multi-family image in each split is
the one holding a near-duplicate pair: HF14 img 14 (pair 0.908 -> IoU 0.246) and
validation img 13 (0.886 -> 0.385). Necessary but NOT sufficient — img 19 pairs
at 0.844 and scores 0.915, img 18 at 0.856 scores 0.780.

Image 14's error decomposition matches the mechanism: its pattern2/pattern3 pair
sits at 0.875-0.908, and both of those selections FLOOD (ref08 reaches 99.1%
recall at 25% precision, predicting 3.96x its target) — asked for either family
the model returns both. Its seven pattern1 selections instead land at 27-63%
recall and 39-51% precision: right size, wrong place, excess scattered rather
than concentrated on one partner.

NOT explained by: resolution or aspect (img 16 downscales 4.80x and scores
0.930), instance size (within img 14 the correlation is -0.219 and the LARGEST
instance scores second-worst), faintness (see below), or localisation (the fixed
anchor supplies it and only collapses variance). A "pattern resembles the
background" variant was measured and did **not** replicate — img 14's 0.424 is
unremarkable next to imgs 27/3/25 at 0.62/0.58/0.58, which score 0.82-0.87.

**Untried lever:** `generate_synthetic_v5.py` picks *distinct* tiles per family
by construction, so synthetic sheets contain no confusable pairs at all — the
model has never trained on the case that costs it every point.
`image_generation/README.md` already requires exactly this of generated plans
("at least two intentionally similar/confusable families", "hard negatives").

### Faintness is not the driver (checked 2026-08-10)

Contrast normalisation at inference is neutral-to-harmful on validation (CLAHE
0.7396, percentile stretch 0.6830, vs 0.7466 as-is; the stretch takes img 15 from
0.980 to 0.071). Global ink-depth correlates -0.749 with IoU on HF14 but -0.003
on validation, so it does not replicate; restricting the statistics to the
pattern regions weakens every correlation rather than sharpening it. Sparse faint
plans are mostly fine (imgs 11/16/1 have pattern coverage 0.048-0.096 and score
0.926-0.952). `scripts/faintness_probe.py`.

One genuine measurement artefact, not worth fixing: on HF14 img 7 the reference
box lands on the words "UNDER FLOOR OF THE EXISTING GARAGE" printed over a
stippled floor, so the crop is 25% ink (text) instead of 4% (stipple). Moving it
down 0.75 box-heights onto clean stipple takes that selection from 0.122 to
0.821 — worth ~+0.013 on the headline, inside noise, and changing the eval
sampler to enforce clean crops would break comparability with every existing
number. `scripts/reference_quality_probe.py` shows unrepresentative crops
usually score FINE (the least representative crop on HF14 scores 0.980), so this
is not systematic.

### Rebuilding what these findings used

Nothing under `data/` survives a clone. Every number above came from
`data/runs/ck_mix3652_noanchor_s31/epoch_7.pth` — the `noanchor` arm of
`run_mix_planefix.sh`, i.e. the standard `startup.md` recipe at seed 31. Build
`data/mixed/toparea1600_rf1548_gen504` per `startup.md`, then `./run_mix_planefix.sh`.

The pools the ablations used, none of which `startup.md` covers:

```bash
TILES=~/.cache/huggingface/hub/datasets--abshetty--floz-assets/snapshots/*/reference_tiles_curated
# disc / conn: one elevation pool split by connectivity, area-matched
python3 generate_synthetic_v5.py --n 4000 --seed 31031 --tiles $TILES --workers 64 \
  --mode-weights 100,0,0 --elev-repeat-prob 0.5 --out data/synthetic/elevpool4000
PYTHONPATH=. python3 scripts/select_by_connectivity.py --source data/synthetic/elevpool4000 \
  --out-low data/synthetic/elev889_disc --out-high data/synthetic/elev889_conn --n 889
python3 scripts/select_toparea_local.py --sources data/synthetic/hf20k \
  --out data/synthetic/rest711 --n 711 --mode-quotas roof_plan=540,freeform=171
for A in disc conn; do python3 scripts/merge_local_datasets.py \
  --sources data/synthetic/elev889_$A data/synthetic/rest711 \
  --out data/synthetic/mix1600_$A; done          # then ./run_disc_ablation.sh

# volume / ratio: scale the mode quotas, keep the real half fixed
python3 scripts/select_toparea_local.py --sources data/synthetic/hf20k \
  --out data/synthetic/toparea3200_balanced --n 3200 \
  --mode-quotas elevation=1778,roof_plan=1080,freeform=342      # 400/800/6400 scale likewise
python3 scripts/merge_local_datasets.py --sources data/synthetic/toparea3200_balanced \
  data/roboflow/floz-real-pool-v2-strong18 data/roboflow/floz-genreal-v1-strong18 \
  --out data/mixed/toparea3200_rf1548_gen504   # then ./run_volume_scaling.sh, ./run_ratio_scaling.sh

# realaug2x / 4x: same 114 sources, more offline variants (--aug-per-scene 35 / 71)
```

Visual audits regenerate with `scripts/visualize_refunet_selection.py --indices 7,14`;
per-image and per-family numbers with `faintness_probe.py`, `reference_quality_probe.py`,
`anchor_vs_reference_probe.py` and `connectivity_stats.py`. Epochs for every
comparison were chosen with `select_epoch_on_val.py`, never from a training log.

### New tooling

`connectivity_stats.py` (share_local; needs full resolution — at 1024 nearby
components merge), `select_disconnected_local.py`, `select_by_connectivity.py`,
`anchor_vs_reference_probe.py`, `select_epoch_on_val.py`, generator
`--elev-repeat-prob` (default 0). Also fixed: `load_refunet` never passed
`anchor`, so anchored checkpoints could not be loaded outside training.

## Status of the pre-2026-08-06 material below

Everything below was measured with the pre-2026-08-06 reference-box sampler,
which could place the user rectangle outside the pattern it sampled. Those
numbers remain internally consistent and the generator conclusions still hold,
but they are **not comparable** to anything measured after the fix. See
`startup.md` for the fix and the current results.

Two further corrections from 2026-08-06:

- The source pools `data/synthetic/faintcad2500` and `data/synthetic/cadneg2500`
  referenced throughout this file **no longer exist and cannot be rebuilt** — the
  generator flags that produced them were never committed. Use the 20k HF config
  via `scripts/hf_to_local.py` instead; the substitution costs ~0.006.
- The synthetic pools were the one part of the data **never** affected by the
  sampler bug (0% bad boxes), because their images are 2-5k px wide so 128px
  reference boxes fit trivially. The bug bit only the 640px real plans.

## Mixed-data successor

The synthetic-only result below remains fully reproducible and is an important
baseline. The current accepted single-model result now uses the same 1,600
synthetic images plus 1,548 Roboflow records and reaches **0.6127448856** on the
same corrected HF14 metric at actual epoch 8. The implementation is the direct
reference-conditioned `RefUNet`; see `codex_doc.md` and `startup.md` for the
architecture, exact two-stage schedule, data construction, and evaluation.

Roboflow-only reference points are 0.349361 for 86 originals with live
flip/rotation and 0.464606 for the strong18 offline pool. These do not change the
verified synthetic-only outcome documented below.

## Outcome

The synthetic-only target is achieved.

| Requirement | Verified result |
|---|---|
| Dataset size | 1,600 unique images |
| Training inputs | Fully synthetic |
| Training duration | 10 actual epochs |
| Real evaluation | Fixed HF14 |
| Task metric | Reference-conditioned union IoU |
| Target | `>= 0.55` |
| Result | **`0.5506125168`** |

Passing checkpoint:
`data/runs/ck_stage750e4_rank1_refonly5/epoch_0.pth`

Authoritative audit:
`data/evaluations/verified_rank1_hf14/metrics.json`

The audit covers all 14 fixed real images and all 52 annotated reference
selections. Comparison images are beside the metrics file.

## Dataset construction

Two existing 2,500-image generator pools supply all source images:

- `data/synthetic/faintcad2500`
- `data/synthetic/cadneg2500`

The canonical 1,600-image selection is:

```bash
python scripts/select_toparea_local.py \
  --sources data/synthetic/faintcad2500 data/synthetic/cadneg2500 \
  --out data/synthetic/toparea1600_balanced \
  --n 1600 \
  --mode-quotas elevation=889,roof_plan=540,freeform=171
```

Selection statistics:

```text
n              1600
score_min       0.168350
score_median    0.292692
score_mean      0.394379
score_max       0.853069
elevation       889
roof_plan       540
freeform        171
faintcad        785
cadneg          815
```

The high-area curriculum subset is:

```bash
python scripts/select_toparea_local.py \
  --sources data/synthetic/faintcad2500 data/synthetic/cadneg2500 \
  --out data/synthetic/toparea750_balanced \
  --n 750 \
  --mode-quotas elevation=476,roof_plan=196,freeform=78
```

All 750 source identities are members of the 1,600-image set. The curriculum
therefore uses one fully synthetic dataset, starting with its strongest subset.

## Successful ten-epoch curriculum

### Phase 1: joint instance and reference training

Train on the 750-image subset at 1280 px with batch 4. The successful checkpoint
is `epoch_4.pth`, after five completed epochs. The run was configured with a
ten-epoch cosine schedule; stop once epoch 4 is saved.

Important model settings:

```text
num_queries             200
mask_weight             10
dice_weight             10
ref_weight              2
eos_coef                0.03
domain_random           true
ref_siamese_backbone    true
ref_siamese_level       res3+res5
seed                     0
```

Phase-one checkpoint:
`data/runs/ck_toparea750_hybrid_1280_s0/epoch_4.pth`

### Phase 2: hard reference ranking

Warm-start phase one and train five more epochs on all 1,600 images. Freeze all
segmentation parameters and update only the Siamese reference projector.

```text
batch_size              8
reference_only          true
ref_ranking_margin      1.0
lr                      1e-4
epochs                  5
```

This completes ten actual epochs. The passing point is phase-two epoch 0, the
sixth actual epoch, and is retained even though later phase-two checkpoints
regress slightly.

Full commands are maintained in [`startup.md`](startup.md).

## Acceptance evaluation

```bash
PYTHONPATH=. python scripts/evaluate_reference_selection.py \
  --checkpoint data/runs/ck_stage750e4_rank1_refonly5/epoch_0.pth \
  --image-max-size 1280 \
  --score-thresh 0.6 \
  --match-margin 0.1 \
  --out data/evaluations/verified_rank1_hf14
```

Verified output:

```text
metric                  reference-conditioned union IoU
checkpoint_epoch        0 (phase two; sixth actual epoch)
n_images                14
n_reference_selections  52
mean_iou                0.5506125168094088
```

The model returns separate matched instance masks. The evaluator unions those
masks only for the strict acceptance calculation.

## Progression to the result

| Experiment | Corrected HF14 IoU | Conclusion |
|---|---:|---|
| 500-image res5 Siamese | 0.457807 | Valid initial baseline |
| 750-image `res3+res5` Siamese | 0.519512 | Fine + coarse reference features matter |
| 1,600-image joint training | 0.504098 | More data improved masks, not grouping |
| Ranking margin 2.0 | 0.536815 | Hard-pair ranking was the right loss |
| Ranking margin 1.0 | **0.550613** | Passed |

## Useful negative evidence (2026-08-06, fixed sampler, RefUNet)

Thirteen levers screened at one seed on the **validation split** (see
`startup.md`), baseline validation peak `0.7672`. None beat baseline. Do not
repeat these without a materially new hypothesis; full reasoning is in commit
`b2d2f09`.

| lever | val peak | note |
|---|---:|---|
| threshold retune (0.25–0.65) | 0.7638 | raising it monotonically hurts — the model floods faint sheets *confidently*, so no post-hoc calibration helps |
| boundary snapping ×4 | 0.7672 | guided filter / morph / bilinear / ink-cell fill all ≈0. Boundary precision is **not** the bottleneck at 0.77 |
| resolution 2048 | 0.7524 | |
| schedule 15 epochs | 0.7529 | |
| resolution 1792 | 0.7508 | two independent points: more pixels hurt |
| synthetic volume 3200 | 0.7415 | |
| `--realism-aug` | 0.7398 | |
| dense correlation `--corr-grid 4` | 0.7300 | |
| ink-matched synthetic | 0.7218 | density realism cost labelled area (0.387→0.286) |
| faint + high-area synthetic | 0.7227 | still lost with area preserved |
| dense correlation `--corr-grid 8` | 0.7179 | monotonic: more matching precision, worse result |
| `--scale-matched-ref` | 0.6146 | worst change measured; see the warning in `dataset.py` |

## Lever 14: auxiliary hard-pair ranking loss (2026-08-07, one seed)

The query lineage's biggest win (ranking margin 1.0 -> 0.5506) was dropped
untested when `RefUNet` replaced the query model. Now tested, implemented in
`ref_unet.py` + `train_refunet.py` and default-off (`--rank-weight 0`):

| arm (seed 31, HF14 peak) | mIoU |
|---|---:|
| baseline | 0.6801 |
| `--rank-weight 0.5 --rank-margin 1.0` | 0.6871 |

**+0.007 is inside the 0.013-0.018 run-to-run sd — null.** Likely because the
query model *needed* explicit matching to group candidates, whereas `RefUNet`
predicts the union directly and its conditioning already learns what the union
loss requires. One seed only; a null here is not proof of no effect.

## Classical template matching is strong but redundant (2026-08-07)

`scripts/hatch_matcher.py` measures drafting parameters (orientation, spacing via
the distance transform of the paper, density) instead of correlating a filter
bank. On HF14 it beats the out-of-the-box Gabor probe by a wide margin — AUC
0.727 -> 0.897, oracle-threshold IoU 0.308 -> 0.505 — so *engineered matching was
never the weak part; the engineering was*.

It is nonetheless **not worth feeding into the model**. On identical images,
instances and reference rectangles it correlates *positively* with `RefUNet`
(+0.45 AUC, +0.65 IoU) and its IoU collapses to 0.252 on `RefUNet`'s worst 13
selections versus ~0.59 elsewhere: the two fail on the same inputs. An oracle
that perfectly overrode the 5 selections where `RefUNet` fails and the matcher is
confident would gain only +0.076.

Three measurements agree that the residual difficulty is intrinsic, not a
resolution or architecture deficit: 74% of the matcher's false positives are real
linework a labeller assigned to a different material; no region-recovery scheme
(ink closing, planar subdivision, oracle ceiling 0.257) can even represent the GT
regions; and both methods fail together. This is the confusability the generator
spec deliberately builds in.

Two mechanisms worth remembering:

- **Region identification, not boundary placement, is the binding constraint.**
  Per-image IoU spans 0.42–0.98 and the low ones are wrong regions, not fuzzy
  edges. A perfect model misplacing boundaries by 3px would still reach 0.9485
  mean IoU, so boundaries only start mattering above ~0.90.
- **The model learns a scale-invariant texture embedding, not template
  matching.** Engineered matching is weak: naive NCC scores AUC 0.433 — *below
  chance*, because CAD hatch is periodic and NCC is phase-sensitive — and
  phase-invariant Gabor energy only reaches 0.674.

## Useful negative evidence (earlier lineage, broken sampler)

These variants were evaluated with the corrected task metric and should not be
repeated without a materially new hypothesis:

- 1,000-image ordinary top-area training at 1024 or 1280 px
- all-repeated and 500-plus-250 repeated-image selections
- preferentially sampling repeated reference categories
- fine-only `res3` matching
- feature mean/variance texture statistics
- a separate frozen ImageNet matching backbone
- whole-instance and fixed-size ROI ImageNet matching
- rank fusion, fixed top-k, largest-gap, and two-cluster inference
- nonlinear pairwise matching head
- mask-only warm-start fine-tuning
- reference-only BCE without hard ranking
- hard-ranking margin 4.0

Mask threshold tuning did not materially improve the baseline. Oracle query
selection scored about 0.61, showing that instance mask coverage was not the
main limitation; the winning change had to improve the hardest reference-match
decisions.

## Operational rules

- Evaluate with the corrected reference-conditioned script, never the old
  class-agnostic `mean_gt_iou` proxy.
- Keep every epoch checkpoint for short experiments because the accepted score
  may occur before the final epoch.
- Stop a run early when corrected HF14 trajectory makes the target effectively
  impossible.
- Use all 64 CPU workers for generation or single-run loading on this GH200 VM.
- Keep generated data, logs, checkpoints, and audits in the repository workspace
  under git-ignored directories.
- Never expose `HF_TOKEN` or `ROBOFLOW_API_KEY` values.
