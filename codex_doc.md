# Handoff

Last updated: 2026-08-08

Read `PROJECT_UNDERSTANDING.md` for task semantics and `startup.md` for complete,
copy-paste reproduction commands, the data-rebuild path, and the full comparison
table.

## Pick up here (2026-08-10)

**Both claims in the 2026-08-08 section below are now retracted.** The
disconnected-repetition gap is real and reproduces exactly, but closing it does
nothing, and the anchor evidence for it was an implementation bug. See
"2026-08-10: the anchor bug" and the retraction notes inline. Read that before
acting on anything under the older heading.

**What moved and what did not, this session:**

| result | verdict | evidence |
|---|---|---|
| `--anchor` reference-plane bug, fixed by `--anchor-ref-plane 0.0` | **real** | +0.069 HF14 on mix3652; loss curves + probe on synth-only |
| disconnected synthetic data | null | 3 seeds, mode- and area-matched |
| a *working* anchor vs no anchor | null / slightly negative | 3 seeds, all three settings |
| data volume (2,452 → 9,808 records) | null | seed 31 curve |
| mix ratio (24% → 84% real) | null | seed 31 curve |
| `realaug2x` (2x real-derived records) | **retracted** | looked like +0.015 at seed 31; 0.6767 ± 0.0262 at 3 seeds |

Nothing this session raised real IoU. The anchor fix removes a large penalty but
only restores parity, so the headline recipe is unchanged at 0.6860 ± 0.0175.

**Measured noise on `mix3652`: sd 0.0262 across seeds 7/31/99.** Two of this
session's wrong conclusions came from reading single-seed gaps of 0.02–0.04 as
signal. If a screen matters, run three seeds.

**Where the remaining error actually is.** The fixed anchor collapses run-to-run
variance (HF14 sd 0.0051 vs 0.0229) without raising the mean — it nails down
*where* to look while leaving the score unchanged. Combined with image 14's
failures being wrong-region rather than fuzzy-boundary, the error is in
appearance matching, not localisation or boundary placement. That is where to
work, and it is consistent with more distinct labelled real sources being the
binding constraint rather than any data-volume knob.

## 2026-08-10: disconnected synthetic data is a NULL result

Built the disconnected pool and trained it synth-only, 3 seeds, epochs chosen on
the validation split (`scripts/select_epoch_on_val.py`):

| arm | share_local | HF14 mean | sd | per-seed |
|---|---:|---:|---:|---|
| conn (control) | 0.821 | 0.6190 | 0.0229 | 0.641 / 0.595 / 0.621 |
| disc | 0.446 | 0.6090 | 0.0324 | 0.583 / 0.598 / 0.646 |

Halving `share_local` with mode mix and labelled-area distribution held
identical moves nothing: disc is +0.014 on validation and −0.010 on HF14, the
sign flips between sets, and both gaps sit inside the per-arm sd. Also
establishes the previously missing **synth-only fixed-sampler RefUNet baseline
at ~0.62** (vs 0.6860 for the full mix).

Two things the 2026-08-08 plan got wrong, both worth knowing before retrying:

- **"Keep the mode quotas" is impossible.** Disconnection in `hf20k` is 99.8% of
  freeform but only 10.9% of elevation and 5.3% of roof_plan. Of the 2,833
  all-disconnected images just 3 are elevations, against a quota of 889. Any
  disconnection-selected pool is ~100% floor plans, i.e. a mode swap, and
  floorplan-heavy synth was already run and retracted (`385f3cb` → `1cbaeea`).
- **Elevations are the whole gap, and selection cannot fix them.** Per mode:
  synth elevation 0.975 / real elevation-like 0.492; synth freeform 0.505 / real
  plan-like 0.280; synth roof_plan 0.973. Elevations are 57% of the real set and
  56% of the pool. Generation fixes this where selection cannot — the *current*
  generator already emits elevations at 0.629 (the 20k parquet came from an
  older recipe), and the new `--elev-repeat-prob` reaches 0.584 by reusing one
  material on non-touching surfaces. Tools: `scripts/connectivity_stats.py`
  (reproduces the table below exactly), `select_disconnected_local.py`,
  `select_by_connectivity.py`.

## 2026-08-10: the anchor bug — `--anchor` was never a shortcut

`--anchor` did not collapse HF14 because the model took a shortcut. It collapsed
because the anchor channel was fed **1.0 across the whole reference crop** while
the image branch saw a sparse rectangle in a field of zeros — through the *same*
siamese stem. Those filters cannot serve both, and the reference features were
the casualty. Setting `--anchor-ref-plane 0.0` fixes it:

| model (synth-only, disc, seed 7) | honest IoU | val loss ep0→ep8 |
|---|---:|---|
| no anchor | 0.611 | 1.60 → 0.35 |
| `--anchor` (reference plane 1.0) | 0.276 | 1.60 → **1.31, rising** |
| `--anchor --anchor-ref-plane 0.0` | 0.534 | 1.53 → **0.34** |

`scripts/anchor_vs_reference_probe.py` is the direct evidence. It feeds
contradictory inputs — reference crop from family B, anchor rectangle in family
A — and asks which one the output obeys, plus `pred_local`, the share of the
prediction lying in the anchor's own connected component:

| model | pred_local | gt_local | follow_anchor | follow_ref | ref_sens |
|---|---:|---:|---:|---:|---:|
| disc no-anchor | 0.193 | 0.266 | 0.036 | 0.604 | 0.934 |
| disc `--anchor` | 0.282 | 0.266 | 0.028 | 0.255 | 0.918 |
| **conn `--anchor`** | 0.331 | 0.266 | 0.064 | 0.375 | 0.795 |
| disc `--anchor` + dropout 0.5 | 0.217 | 0.266 | 0.019 | 0.597 | 0.979 |

A shortcutting model would show `pred_local` near 1.0 and `ref_sens` near 0. The
anchored model instead sits at ground-truth localness with its reference pathway
fully causal — it is damaged, not shortcutting. The `conn` row is decisive: that
is the pool where the shortcut *is* on offer (`gt_local` would reward it) and it
still does not take it. The dropout row corroborates the bug by accident —
`--anchor-dropout` zeroes the reference plane too, so it applies the fix to half
of training and recovers most of the loss (0.276 → 0.583).

Consequence: **the disconnected-data programme was motivated by this artifact.**

### Confirmed on the headline mix recipe (`mix3652`, seed 31, `run_mix_planefix.sh`)

Data rebuilt from scratch through the documented pipeline; the no-anchor arm
reproduces the baseline, so the rebuild is sound.

| arm | val | HF14 | vs documented |
|---|---:|---:|---|
| no anchor | 0.7466 | **0.6916** | reproduces 0.6801 (s31) / 0.6860 (3 seeds) |
| `--anchor` (plane 1.0) | 0.6332 | 0.6088 | documented as 0.433 — **not reproduced** |
| `--anchor --anchor-ref-plane 0.0` | 0.7305 | **0.6777** | +0.069 over the buggy arm |

Val loss tells the same story as synth-only: the buggy arm diverges (train
1.30 → 0.35 while val climbs 0.67 → 0.88 → 0.70), the fixed arm descends
cleanly (0.97 → 0.32).

**Caveat: the documented 0.433 did not reproduce.** The bug costs 0.083 here
(0.6916 → 0.6088), not the 0.247 on record. The bug is real, costly, and fixed,
but whatever produced 0.433 involved something further — different epoch
selection, seed, or an older code state. Do not quote 0.433 as the pre-fix
number without re-deriving it.

## 2026-08-10: data volume and mix ratio are both NULL

Nothing beat the 3,652-record baseline anywhere from 24% to 84% real fraction or
2,452 to 9,808 records. All arms seed 31, epochs chosen on validation.

| arm | records | real % | real-ish | val | HF14 |
|---|---:|---:|---:|---:|---:|
| synth6400 | 8,452 | 24% | 2,052 | 0.7255 | 0.6625 |
| synth3200 | 5,252 | 39% | 2,052 | 0.7301 | 0.6075 |
| baseline | 3,652 | 56% | 2,052 | 0.7466 | 0.6916 |
| synth800 | 2,852 | 72% | 2,052 | 0.7042 | 0.6650 |
| realaug2x | 5,704 | 72% | 4,104 | 0.7548 | 0.7069 |
| synth400 | 2,452 | 84% | 2,052 | 0.7538 | 0.6789 |

`realaug2x` looked like a win at seed 31 (0.7069 vs baseline 0.6916) and it did
not survive: **3 seeds give 0.6767 ± 0.0262**, i.e. −0.009 against the documented
0.6860 ± 0.0175. Seeds 7 and 99 both land at ~0.661; seed 31 was the outlier.
Two follow-ups, both at seed 31, confirm the seed rather than the config: a
15-epoch schedule gives 0.7004 and `realaug4x` (9,808 records, 8,208 real-ish)
gives 0.7005 — every seed-31 run sits at ~0.70 no matter what the data does.

**Read the rest of the table with that in mind.** Each row is one seed, and
run-to-run sd on this recipe is **0.0262**, so gaps below ~0.05 carry no
information. Two conclusions drawn from this table before the seeds came back —
"the number of real-derived records is the lever" (from realaug2x vs synth800,
+0.042) and "real fraction is monotone on validation" — are **not supported**;
both sit inside noise. The one defensible statement is the negative: neither
volume nor ratio moves real IoU.

Consistent with `startup.md`'s standing conclusion that more *distinct labelled
real sources* is the binding constraint. Re-augmenting the same 114 sources
harder (18x → 36x → 72x variants) buys nothing, which is what that predicts.

### Does a working anchor actually help? No.

Synth-only, 3 seeds per arm, epochs chosen on validation:

| pool | no anchor (val / HF14) | + fixed anchor (val / HF14) | effect |
|---|---|---|---|
| disc | 0.6260 / 0.6090 | 0.5710 / 0.5923 | val −0.055, HF14 −0.017 |
| conn | 0.6124 / 0.6190 | 0.5728 / 0.6218 | val −0.040, HF14 +0.003 |

Plus mix3652 above: val −0.016, HF14 −0.014. Every anchored run scores below
every un-anchored run on validation, in all three settings. The fixed anchor is
neutral-to-slightly-negative — recoverable, not useful.

One real secondary effect: the anchor **collapses run-to-run variance** (HF14 sd
0.0051 vs 0.0229 on conn; 0.018 vs 0.032 on disc) without raising the mean. It
supplies reliable localization while the remaining error sits entirely in
appearance matching — which is where the score is lost and where effort belongs.

Unresolved: `--anchor-dropout` has never been tested cleanly, because it was only
ever run on top of the bug. Retest it against `--anchor-ref-plane 0.0` if the
anchor turns out to need regularizing at all.

## Superseded — 2026-08-08 (kept for the measurements, not the conclusions)

**The synthetic half of the mix teaches the wrong task.** Measured, per selection,
as the share of the target union lying in the connected component that contains
the user's rectangle:

| pool | share local | single-component targets |
|---|---:|---:|
| synthetic 1600 | 0.893 | 84% |
| real 86 | 0.523 | 29% |
| generated 28 | 0.551 | 36% |
| HF14 (eval) | 0.537 | 27% |

Real plans repeat a material in disconnected places; `toparea1600_balanced` mostly
does not, so for 44% of records the correct answer is "outline the blob you are
pointing at".

> ~~Independent evidence that the model exploits this: feeding the reference-box
> location as a 4th input plane (`--anchor`, implemented) *collapsed* HF14 from
> 0.680 to 0.433 with val loss rising — the shortcut is available and the model
> takes it.~~
>
> **FALSE, retracted 2026-08-10.** The model does not exploit it and never took
> the shortcut in any pool measured, including the connected one where the
> shortcut is genuinely on offer (`pred_local` 0.331 vs `gt_local` 0.266,
> `follow_anchor` 0.064). The anchor collapse was an input-statistics bug in the
> siamese reference plane — see "the anchor bug" above. The 0.433 figure also
> did not reproduce (measured 0.6088); do not quote it.
>
> The **measurements** in the table above are sound and reproduce exactly: 0.898
> / 84% on a rebuilt `toparea1600_balanced`, 0.537 / 27% on HF14. Use
> `scripts/connectivity_stats.py --hf14`, which needs full resolution; at 1024
> nearby components merge and it reads 0.558 / 33%.

The experiment this section proposed — select for disconnected families, keep the
mode quotas, retrain — **has been run and is null at 3 seeds** (and the mode
quotas turn out to be unfillable: only 3 of the 2,833 all-disconnected images are
elevations against a quota of 889). See "disconnected synthetic data is a NULL
result" above. Do not re-derive it from this section.

**Baseline to compare against.** `data/runs/ck_a10_mix3652_seed31/epoch_4.pth`,
**0.6801** on HF14, one seed. The documented recipe runs unchanged on a 23GB A10
(batch 8 at 1280 peaks at 19.3GiB; use `--num-workers 12`), ~55 min for the full
two-stage run. Single-seed screening only — treat anything under ~0.03 as noise.

**Also worth knowing.** Image 14 is 9 of 52 selections (17% of the metric) and
averages 0.291; fixing it alone would give +0.105. Its failures are wrong-region,
not fuzzy-boundary, and are *not* explained by resolution or aspect ratio.
Two levers were closed this session — see `synth_progress.md`: the auxiliary
ranking loss (null, +0.007) and classical template matching (`scripts/hatch_matcher.py`
beats the Gabor probe by a wide margin but is redundant with `RefUNet`).

## Read this first

`sample_reference_box` — the function that turns a ground-truth instance into the
user's reference rectangle — was returning rectangles partly or entirely outside
the pattern they sampled. It was fixed on 2026-08-06.

**Every number measured before that date used the broken sampler, including the
`>= 0.65` target.** Pre-fix and post-fix numbers are not comparable.
`sample_reference_box_legacy` is retained so the old figures remain reproducible.
Always say which sampler a number came from.

## Current result

Fixed sampler, `RefUNet`, 1,600 synthetic + 1,548 real + 504 generated-realistic:

| Item | Value |
|---|---|
| Recipe mean (seeds 7/31/99) | **0.6860 ± 0.0175** |
| Best single checkpoint | **0.705407920670342** |
| Checkpoint | `data/runs/ck_fix_mix3652_seed31/epoch_6.pth` |
| Metrics | `data/evaluations/refunet_fix_mix3652_s31_e6.json` |
| Visual audit | `data/visualizations/fix_mix3652_s31_e6/` |
| Evaluation | fixed HF14, 52 reference selections, threshold 0.35 |

Quote the 3-seed mean. Run-to-run noise is ~0.013–0.018 sd here, so the best
checkpoint is the top of a spread rather than the expected value.

That 0.6860 picks the epoch by HF14 score, which uses the acceptance set to
choose the checkpoint. Selecting the epoch on the validation split instead gives
**0.6783 ± 0.0081**. Both follow a defensible protocol — the project's metric is
defined as "any checkpoint within ten epochs" — but quote the val-selected number
when it has to hold up. See `startup.md` for the split and its correlation.

Thirteen levers were screened against this baseline on 2026-08-06 (resolution,
synthetic selection, volume, schedule, augmentation, threshold, boundary
snapping, dense reference correlation, scale-matched reference) and **none beat
it**; the table is in `synth_progress.md` and the reasoning in commit `b2d2f09`.
Two of them, `--corr-grid` and `--scale-matched-ref`, are implemented and
default-off. Read the warning above the reference resize in
`refmask2former/dataset.py` before "fixing" the reference scale — it looks like a
bug and correcting it costs 0.153.

The `>= 0.65` target was defined under the broken sampler and has not been
restated. On the fixed metric this recipe averages 0.686; whether that counts as
meeting the goal is a product decision, not a measurement one.

## What changed this session

Starting point was 0.6127 (broken sampler, single run). Two independent gains,
each replicated at three seeds with complete separation between arms:

| Change | Effect | Evidence |
|---|---|---|
| +28 hand-labelled generated plans | +0.0330 | t=2.94, p=0.043 |
| Reference-box sampler fix | +0.0529 | t=4.18, p=0.014 |

The like-for-like progression on the broken sampler is 0.6001 (`mix3148`) →
0.6331 (`mix3652`); the fixed sampler then takes `mix3652` to 0.6860.

The synthetic pools `faintcad2500` / `cadneg2500` proved unrebuildable — their
generator flags were never committed — so the synthetic half now comes from the
20k HF config via the new `scripts/hf_to_local.py`. That substitution cost ~0.006.

## Product and model semantics

A reference rectangle identifies an image-local pattern. The target is the union
of every region with that same image-local grouping ID. IDs such as `pattern1`
have no meaning across plans. Roboflow `remove` polygons are subtracted as holes:
`roboflow_to_local.py` attaches each to every containing pattern and
`render_instance_mask` zeroes rings after the first.

`RefUNet` predicts the selected union directly: a shared ResNet-50 extracts plan
and reference features, four multiscale conditioning blocks combine image
features with the pooled reference, and an FPN decoder emits one mask. Split it
into connected components (preserving holes) if the product needs separate
instances. The earlier query model, which grouped instance candidates by
reference similarity, topped out at 0.5886 on mixed data.

## Relevant implementation

- `refmask2former/ref_unet.py` — shared backbone, conditioning blocks, FPN mask.
- `refmask2former/dataset.py` — `sample_reference_box` (fixed) and
  `sample_reference_box_legacy`; `render_instance_mask` hole convention.
- `scripts/train_refunet.py` — union targets, BCE + Dice, continuation
  checkpoints, per-epoch HF14 diagnostics, explicit schedule length and
  optimizer reset.
- `scripts/evaluate_refunet_selection.py` — authoritative 52-selection eval.
- `scripts/visualize_refunet_selection.py` — the required visual audit.
- `scripts/hf_to_local.py` — HF parquet → local-data format.
- `scripts/roboflow_to_local.py`, `augment_local_dataset.py`,
  `merge_local_datasets.py` — deterministic real-data pipeline.
- `scripts/run_*.sh` — the four experiment drivers from this session.

## Data limitation and where to push next

The mix now contains **114 unique real-ish source plans** (86 scraped real + 28
generated); everything else is synthetic or deterministic offline variants. That
source count remains the binding constraint.

Established by count-matched experiment: one generated plan is worth about as
much as one real plan (0.2608 vs 0.2777 at 28 sources each, a gap inside the
~0.05 run noise), resolution is worth ~0.032, and going 28 → 86 sources buys
+0.073. So **generate more, and generate large** — the 86 scraped plans are
natively 640×640 and can never be improved, while generation resolution is a
free choice.

**Before generating more, read the status section at the top of
`image_generation/README.md`.** The 100-prompt scheme was run once and its
framing was wrong: it produced full sheets with legends and title blocks, then
over-corrected into fragments of buildings. The product gesture is a small
rectangle inside a drawing region — the housed part of a plan, not the legend.
Only 28 of 98 generated images were good enough to label (~29% yield); the
remaining 70 were reviewed and rejected, so there is no labelling backlog to
mine. More data means generating a better round, not labelling what exists.

Untested hypothesis worth pursuing on a validation split: the largest remaining
clean failure is thin wall poche in dense floor plans (image 12), plausibly a
1280px downscaling artifact. It must not be tuned against HF14.

## Evaluation rules

- HF `real-world-test` images are evaluation-only.
- Fixed indices: `12,16,27,7,11,25,23,1,18,2,0,3,14,24`; 52 selections.
- Report reference-conditioned union IoU only, naming the sampler and the seed
  count.
- Do not report the legacy training `real_iou`, per-GT best coverage, an oracle,
  class-agnostic Mask R-CNN scores, or checkpoint ensembles as product mIoU.
- Threshold 0.35 is fixed. Do not sweep it or the inference resolution against
  HF14 — that is fitting the acceptance set.
- HF14 images 24, 25, 27 carry human highlighter markup from real markup PDFs.
  That is the real input distribution; keep them in.

## Generated artifacts

Datasets, checkpoints, logs, and evaluation JSON live under git-ignored `data/`
and `logs*`. Preserve them between VMs for byte-identical artifacts, or
regenerate with `startup.md`. Source and documentation are committed;
credentials are never stored in the repo.
