# Synthetic Dataset Progress

Working summary. Every dated entry (2026-08-06 to 2026-09-26) is kept verbatim
in `synth_progress_archive.md`, Part 1, under its original heading; other docs'
references to "`synth_progress.md` (DATE)" resolve there. Append new results
below "Log".

## Current state (2026-09-28)

- **Best model:** `abshetty/floz-refunet-swint-mixr4-roiadd-swa15`, the
  restart recipe (`run_restart_swa.sh` from `mixr4-e8`) plus `--roi-ref
  --roi-ref-mode add`, inference at 4096: **HF14 0.8440** (+0.027 paired vs
  0.8170, p=0.017; one seed). Previous: `...-mixr4-restart-e13`, 0.8170.
  TTA x4 (+0.0015) and multi-scale inference (0.8123) not adopted.
- **Seeds: one per arm from here** (user's call, 2026-09-28): at ~0.85, judge
  a change by the paired per-question test on HF14 against its same-recipe
  control; no seed replicates.
- **Synthetic source: Revit-style FreeCAD sheets beat v6d** (2026-09-27, one
  seed, synth-only, 2k @2048): `scripts/generate_synthetic_fc.py --revit
  --mode-weights 66,16,18` gives **HF14 0.7564 / val 0.7568** against v6d's
  0.7066 / 0.6702 (`abshetty/floz-refunet-swint-revit2k-e8`, round-trip
  verified). The gain is the Revit RENDERER, not the 3D geometry (FreeCAD
  geometry drawn by v6: 0.7126 / 0.6248). At 4096 inference the lead shrinks
  to +0.016 on both splits. Not yet tested in the real mix. Every v6 generator
  change before it lost or tied: v6e look-alike pairs (negative), v7 (-0.023),
  `--hardscape-plan` / `--same-fill-subtle` (null, 3 seeds), `--mottle`
  (null), v5 confusable pairs (-0.104), **r8 (-0.076 HF14, 2026-09-27)**.
- **Best synthetic-only:** Revit 2k + `--refs-per-image 6`, swin_t @2048,
  only 4 epochs (VM deadline): **val 0.7816, HF14 0.7607**
  (`abshetty/floz-refunet-swint-revit2k-sixq-e4`, one seed) -- vs one-question
  Revit (9 epochs) +0.025 val / +0.004 HF14, both noise, at under half the
  epochs; fresh Revit plans (HF14 protocol) 0.789 vs 0.758. One-question Revit 2k: HF14 0.7564
  (`abshetty/floz-refunet-swint-revit2k-e8`). Previous: v6d 100k single pass on RefUNet 0.7130; swin_t v6d 1,995
  0.7066 (`abshetty/floz-refunet-swint-v6d-e8`).
- **Realism probe** (vs val 14): r8 0.909 < v6d 0.948 < FreeCAD 0.954 <
  Revit 0.960 -- and on HF14 the order is REVERSED (Revit best, r8 worst).
  The probe does not predict training value; don't use it to pick pools.

## How to test a synthetic change

- **Screen at 2048, not 1024** (2026-09-27, `run_res_calibration.sh`): the
  same r8-vs-v6d A/B trained at 1024 and at 2048 (1,600 sheets, seed 7).
  1024 said r8 HELPS on validation (+0.028) where 2048 says it HURTS (-0.046);
  on HF14 both say hurts (-0.055 / -0.037; per-selection delta correlation
  0.61 on HF14, -0.03 on validation). Fine-texture changes alias away at 1024
  (x0.4 `--domain-random` shrink on top). Use `run_revit_2k.sh`'s recipe:
  swin_t, ~2k sheets @2048, 1+8 epochs, one seed, ~20 min per arm alone;
  control = the published v6d 2k (`abshetty/floz-refunet-swint-v6d-e8`).
  Single-epoch HF14 swings 0.02-0.04 between adjacent epochs, so a one-seed
  delta under ~0.04 is noise; a winner still needs the real mix.
- The old 1024 screen (`run_synth_only_screen.sh`): swin_t at 1024, three runs
  concurrently ~20 min, v6d control spans 0.640-0.694 on HF14 across seeds.
- **Don't** use `run_synth_ft_ab.sh`: a gentle fine-tune of a converged model
  leaves 48/52 selections tied and cannot detect a pixel-only change. A full
  retrain (`run_nogray_mix.sh`-style) takes ~80 min.
- Choose epochs and inference size on validation
  (`select_epoch_on_val.py`); read HF14 once. HF14 is 52 fixed questions
  (reference boxes identical across runs), so compare **paired** per
  selection, and only against a same-recipe control: two base runs differing
  only by seed pair at p = 0.0001.
- Checkpoint averaging reads ~0.04 low when a run is still climbing at the last
  epoch; check that it turned over first.
- Transfer: use `scripts/question_difficulty.py`, not `fresh_synth_iou.py`
  ratios (different question mix). `fresh_synth_iou.py` decides "seen" by
  filename, and merged pools are all `item_XXXXXX`. An empty `--trained-pool`
  crashes (FileNotFoundError); for known-disjoint splits (fresh ids 100000+)
  pass the model's own training pool.
- **Realism probe** (`scripts/synth_realism_probe.py`): frozen DINOv2 plus
  logistic regression on crops inside labelled regions, reference val 14 at
  3168 px (indistinguishable from HF14: AUC 0.423), mean of 3 seeds, 320 synth
  sheets (seed 6, ids 200000+). Differences under ~0.04 are noise. It measures
  "looks like these ~12 documents", not "looks like a real drawing" (real
  Boise Revit sheets score 0.998), so it ranks changes and its crop sheets
  show what's off, but it is not a training-value metric.

## Findings that still bind

- **Volume:** v6d-only at 2048: 1,600 -> 0.640, 12,000 -> 0.693, 100,000
  seen once -> 0.713 (RefUNet). Gains shrink ~60% per 8x. With enough unique
  data, one pass beats many epochs; a second pass already lowers
  fresh-synthetic IoU. swin_t (2026-09-27, `run_v6d_scaling.sh`, nested
  pools, fresh v6d scored with the HF14 protocol): 2k 0.660, 8k 0.716, 16k
  0.732 -- a 3-point fit flattens at ~0.77. HF14 0.707 / 0.737 / 0.729.
- **swin_t underfits family grouping, not data** (2026-09-27): the 16k model
  scores 0.736 on its OWN training plans (HF14 protocol) against a
  resolution ceiling of 0.904 (`label_ceiling.py`, same 1,798 questions).
  Single-family sheets 0.957; 4-6 family sheets 0.673 (65% of the error);
  non-look-alike targets hold 60% of it. Asking all families per step
  (`--refs-per-image 6`) matches 16k with 2k plans (fresh 0.727, seen 0.741)
  but hits the same ~0.74 wall; real plans +0.017 val / +0.012 HF14 (noise).
- **Real-plan penalty:** net of question difficulty, every source (Gemini
  included) pays ~-0.14 to -0.17 on real plans. At matched 148 sources Gemini
  and v6d tie on HF14.
- **Where the error is:** target thickness at input resolution (thin < 74 px at
  2048: 0.437, mid 0.716, thick 0.825; the thin third is about half the missing
  IoU). Region identification, not boundary placement: boundary methods and
  threshold sweeps (flat, peak 0.35) give ~0. HF14 image 7's reference box sits
  on printed text (worth ~+0.011; don't chase).
- **Resolution:** `--domain-random` shrinks training sheets 0.4-1.0x, so
  inference above the nominal size helps (+0.009 on v6d-only, +0.028 on the
  real mix at 4096). Training at 2560 beat 2048 once (0.7834,
  `abshetty/floz-refunet-res2560-e4`); a second seed was never run.
- **Distinct labelled real sources** are the binding data constraint:
  re-augmenting the same 114 sources 18x/36x/72x, synthetic volume and mix
  ratio at mix scale are all null.
- **Closed levers** (all null or negative, detail in the archive):
  conditioning mechanisms (`--corr-grid`, `--scale-matched-ref`, ranking loss,
  self/dynamic prototypes, anchor; the anchor-plane bug is fixed, default
  `--anchor-ref-plane 0.0`), decoder width, confusable/look-alike synthetic
  pairs, same-fill recolouring, grayscale-free augmentation (the model does use
  colour), contrast normalisation, two-pass inference, small-reference
  training.
- **Backbone:** swin transfers best and fits worst, stable across LRs;
  material separability of frozen features does not predict transfer.

## Generator flags (`generate_synthetic_v6.py`, all default-off, v6d byte-identical when off)

| flag | status |
|---|---|
| `--hardscape-plan`, `--same-fill-subtle` | null, 3 seeds |
| `--same-fill-new-colour` | negative on the real mix (band worse) |
| `--mottle` | null, one seed |
| `--vocab2` | untested (`run_vocab2_screen.sh` ready) |
| `--gemini-colour`, `--muted-palette` | probe only; r8 uses `--neutral-palette` instead |
| **r8 set:** `--val-details --material-mix --val-fills --mode-weights 86,7,7 --res-degrade 0.15 --neutral-palette --fill-scale --real-labelling` | **negative**: HF14 -0.076, val -0.008 (2k @2048; trained WITHOUT `--mode-weights`, i.e. the `R8` dict in `generate_synthetic_fc.py`) |
| `--real-labelling` alone (on Revit sheets) | **negative on validation**: -0.187 vs Revit, 89% of it val 17 (unlabels the foundation band that val asks about); HF14 -0.008 |
| `--tight-crop` | framing fix (ink box + pad); untested |

FreeCAD (`scripts/generate_synthetic_fc.py`, `scripts/fc_massing.py`,
`scripts/revit_render.py`), all 2k @2048, seed 7, epoch 8, `--mode-weights
66,16,18` (v6's mix):

| pool | val | HF14 |
|---|---:|---:|
| v6d (control) | 0.6702 | 0.7066 |
| `--revit` | **0.7568** | **0.7564** |
| no renderer (v6 drawing on FreeCAD houses) | 0.6248 | 0.7126 |
| `--revit --real-labelling` | 0.5702 | 0.7485 |
| `--r8 --revit` | 0.6891 | 0.6713 |

Revit model on its own fresh plans (HF14 protocol) 0.758 = its HF14 0.756; it
scores 0.507 on fresh v6d, the v6d model 0.724 on fresh Revit.

## Open

- **Revit in the real + Gemini mix** (swap for v6d_1600 in `v6dmix_plus_r4`):
  the only test of whether it helps the shipped model.
- **The ~0.74 grouping wall is not mainly step budget** (2026-09-27): one
  fresh 9-epoch cosine on the six-question model (2x steps, epoch 17,
  `abshetty/floz-refunet-swint-v6d2k-sixq-restart-e17`) moved own-plan fit
  0.741 -> 0.765 and fresh 0.727 -> 0.745, against a 0.904 ceiling; val
  0.688 -> 0.705, HF14 0.719 -> 0.707 (noise). Next suspect: conditioning /
  capacity. Untested: `--swin-decoder selfattn` (code in, CPU-checked; in the
  frozen screen it only TIED baseline at the best LR). Revit + six-question
  ran for 4 epochs only (see "Best synthetic-only"); a full 1+8 run is
  `./run_multiref.sh 7 revit`.
- `--vocab2` screen.

## Log

### 2026-09-26 — realism probe, r8, FreeCAD, Revit (detail: archive Part 1)

- Probe v6d vs Gemini: 0.999. Candidate references are all far apart (eval 28
  vs scraped 86: 0.965 at 640; eval vs Gemini 0.954), so val 14 became the
  reference. It looks like modern CAD/BIM exports with colour markup.
- Flag rounds against val 14 (3-seed means): v6d 0.948, r2 0.919, r4 0.932,
  r6 0.964, r7 0.945, **r8 0.909**. The biggest single gain came from labelling
  like real annotators (`--real-labelling`). Blurring crops to 28 px only
  takes r4 from 0.93 to 0.86, so the gap is coarse content.
- `--tight-crop` came from the user spotting white space: val drawings fill
  92% of the image, v6 62%.
- Boise public-domain ADU sets (`data/reference/boise_adu/`, 6 bid sets): the
  six A201 sheets (elevations plus sections, one firm) score 0.998 against val
  14. Another real document family is trivially separable, so AUC < 0.7 would
  mean overfitting to the eval documents.
- FreeCAD geometry on r8's houses: 0.954. Geometry is correct but the probe
  looks at local rendering. Revit-style renderer (cast shadows from the solids,
  line-weight hierarchy, level datums, view titles): 0.960. Looks like a Revit
  set, but val 14's most typical crops are hand-drafted shingle and colour
  markup, not Revit. The user prefers the Revit look; train on it as a mix arm
  and a Revit-only arm.
- FreeCAD install (container-local, ~5 min): micromamba from
  `conda.anaconda.org/conda-forge`, then
  `micromamba create -p /opt/fc -c conda-forge freecad`.

### 2026-09-27 — Revit beats v6d; 1024 screens mislead; the grouping wall

One seed throughout, synth-only, swin_t, 2048, 1+8 epochs, epoch 8 scored.
Numbers are in "Current state", "How to test" and "Generator flags" above;
drivers: `run_res_calibration.sh` (+ `scripts/res_calibration_report.py`),
`run_revit_2k.sh [seed] [arms]`, `run_v6d_scaling.sh`, `run_multiref.sh [seed]
[v6d|revit]` (env `DEC`, `RESTART_FROM`, `TAG`, `BS`).

- Batch size: swin_t @2048 saturates the GH200 at batch 4 (17.7 / 18.0 /
  18.1 img/s at 4 / 8 / 10; 8 peaks 67 GiB on a full 2048 canvas, 16 OOMs).
  `--compile` never finished compiling in 15 min (`--domain-random` shapes).
- `--refs-per-image K` (dataset + `RefSwinUNet.forward_multi`): backbone once
  per plan, decoder per question; 26 questions/s vs 18; 3 plans x 6 questions
  ~66 GiB worst case. Default 0 is byte-identical (checked against HEAD).
- `--swin-decoder selfattn`: `CrossAttnCondition` (moved to
  `refmask2former/attn_condition.py`) at 1/16 and 1/32; published baseline
  checkpoints still load strictly.


### 2026-09-28 — the grouping wall was the reference and the shrink; image-box reference on real: 0.8440

One seed throughout (seed 7), swin_t, 2048 training. Drivers: `run_overfit64.sh`,
`run_multiref.sh` (env `NODR`, `EXTRA`), `run_restart_swa.sh` (env `EXTRA`,
`NODR`, `CMP`, `SEED`); scorers `scripts/fit_diagnose.py` (per-question error
decomposition), `scripts/clean_loss.py` (un-augmented loss, comparable across
DR / no-DR runs), `scripts/export_reference_boxes.py` (the 28 real sheets with
the evaluator's exact boxes).

**Not data, not steps.** The six-question Revit model fits its own plans no
better than fresh ones (0.785 vs 0.789, ceiling 0.960). Memorising 64 Revit
plans (60 epochs, 2 plans x 6 questions) reaches only 0.799 (ceiling 0.957).
Backbone LR x10: 0.719. `--swin-decoder selfattn`: 0.762 (worse; it had never
run at 2048 because `attn_condition.py` fed non-contiguous tensors to SDPA,
forcing the 27 GiB math kernel -- fixed with `.contiguous()`, 2 GiB).

**Two causes, 64-plan overfit (IoU / thin<40px / wrong regions per q / missed per q):**

| arm | IoU | thin | wrong | missed |
|---|---:|---:|---:|---:|
| base | 0.799 | 0.715 | 0.35 | 0.35 |
| `--roi-ref` (prototype pooled from the IMAGE's features in the box) | 0.822 | 0.764 | 0.12 | 0.54 |
| `--roi-ref-mode add` (crop prototype + zero-init ROI projection) | 0.841 | 0.784 | 0.17 | 0.33 |
| no `--domain-random` | 0.865 | 0.805 | 0.27 | 0.31 |
| roi replace + no DR | **0.901** | 0.859 | 0.09 | 0.24 |
| roi add + no DR | 0.895 | 0.842 | 0.14 | 0.18 |

1. The 224 px crop is magnified ~5x relative to the plan and encoded
   separately, so reference and targets meet at different scales; colour/tone
   is the only scale-free cue and the model matches on it (dark tile ->
   dark walls). Pooling the prototype from the image's own features fixes
   wrong-family picks.
2. `--domain-random`'s 0.4-1.0x shrink makes thin targets thinner in training.

**At 2k (six-question, ROI add, no DR, 1+8 epochs):** own plans 0.786 ->
**0.887**, fresh 0.789 -> 0.872; clean loss 0.268 -> 0.108 (train), 0.255 ->
0.071 (held-out). But val 0.7816 -> 0.718 (paired -0.063, p=0.013; 55% one
sheet) and HF14 0.7607 -> 0.743 (noise). **All of it is small user boxes**
(side at 2048 input): <32 px -0.44, 32-64 -0.088, >=64 +0.01. A sub-cell box
pools background at 1/16-1/32, and without DR synthetic training has no small
boxes. `--roi-min-cells` (coverage-gated ROI term) is written, untested.

**Small boxes are answerable -- do not reject them in the UI.** The shipped
0.8170 model scores 0.827 on the 21 questions under 50 px (val17 q13, a flat
lavender 34 px patch: 0.810); only val17 q00 fails every model. Its remaining
loss is over-selection instead: 8 HF14 questions predicting >1.3x the target
hold 48% of it (val: 9 questions, 51%; val 17 alone is 72% of val's loss).

**On the real mix: HF14 0.8440** (+0.0269 paired vs the shipped 0.8170,
p=0.017, 16 better / 6 worse / 30 tied; val 0.8150 vs 0.808). Recipe = the
shipped restart (`run_restart_swa.sh` from `mixr4-e8`, DR on, one question per
plan) + `--roi-ref --roi-ref-mode add`; validation chose swa_12-15 @4096. No
small-box regression (<50 px 0.900 -> 0.897); gains on 50-100 px (+0.030) and
>=100 px (+0.030) boxes; the shipped model's over-selected questions 0.425 ->
0.504. Val 17 untouched. ONE SEED -- the seed-31 replicate queued in
`logs/queue6.sh` was dropped (no multi-seed runs at ~0.85, user's call); the
no-DR mix queued before it ran (below).

Control reproduced exactly on this machine: `mixr4-restart-e13` @4096 = 0.8170.
Rebuilt `v6dmix_plus_r4` = 6,676 records (recorded 6,671; augmentation is
chunked, not byte-identical).

**Domain-random on real (mix, ROI add, restart from `mixr4-e8`, seed 7):** without
DR the val-selected checkpoint scores **HF14 0.8583** (e10 @4096, val 0.8547;
+0.041 paired vs 0.8170, p=0.060, 19 better / 14 worse) -- but e10 is a val-17
spike (sheet 17: 0.51 -> 0.79 -> 0.49 over e9-e11; 27 of 77 val questions).
Without sheet 17, DR beats no-DR at all 12 candidates (+0.01 to +0.04) and
no-DR degrades over the restart (0.913 e8 -> 0.875 e14) while DR holds ~0.92.
**Keep `--domain-random`.** Record 0.8583 as a val-17 lottery, not a recipe.

**Box-size probe (`scripts/box_size_probe.py`, 0.8440 model, @4096, CPU):**
every question it scores < 0.6 (9 val + 6 HF14: val 17's band x10, HF14 0
paving vs floor tile, HF14 18 roof vs band, HF14 12, 25), re-asked with
larger boxes inside the SAME instance. Mean IoU: evaluator box 0.277, largest
square <= 512 px 0.278, largest uncapped 0.276, square crop + ROI box
elongated along the region (<= 4x, the thin rectangle a user draws on a band)
0.278. Over-selection stays 2-8x the target. **The remaining failures are not
reference-information limited: the model cannot separate these look-alike
materials even from the largest box a user could draw.** No inference-time box
trick or UI guidance reaches 0.90; the fix has to be learned (data/training on
look-alike pairs).

**Six-question training on the real mix: negative.** Restart from the 0.8440
checkpoint (same mix, DR, ROI add, `--refs-per-image 6 --batch-size 3`, 8
epochs, `data/runs/ck_mix_sixq_from844`): val swings 0.753-0.843 per epoch
with no trend; val-selected e18 @4096 -> **HF14 0.8165**, -0.027 paired vs
0.8440 (p=0.050), 7 better / 24 worse (sign p=0.003) -- broad, not one sheet.
The six-question fit gain on synthetic does not carry to real. Best stays
0.8440 (`data/runs/ck_mix_roiadd_dr/swa_12-15.pth`).

### 2026-09-28 — where Revit beats v6d, per question (CPU re-score)

Published `swint-revit2k-e8` vs `swint-v6d-e8`, 2048, re-scored on CPU
(reproduces 0.7566 / 0.7078 HF14, 0.7574 / 0.6719 val).
`scripts/per_question_breakdown.py` with hand-tagged sheet types.

| sheet type (HF14 + val questions) | n | v6d | Revit | delta | better / worse |
|---|---:|---:|---:|---:|---|
| line-only CAD elevations (8-10, 13, 14) | 34 | 0.57 | 0.70 | **+0.13** | 30 / 3 |
| colour-markup elevations (17-27) | 64 | 0.71 | 0.78 | +0.07 | 36 / 7 |
| shaded Revit elevations (2-6) | 17 | 0.80 | 0.85 | +0.05 | 12 / 2 |
| floor plans (0, 7, 11, 12) | 8 | 0.56 | 0.56 | 0.00 | 3 / 3 |
| roof plans (1, 15, 16) | 6 | 0.92 | 0.81 | **-0.11** | 0 / 4 |

- The gain is in questions v6d OVER-selects (>1.3x the target): +0.07 HF14,
  +0.19 val (32 better / 6 worse), about half and 70% of the delta. Revit
  reduces the look-alike over-selection that is the shipped model's main loss.
- Small boxes gain too: 32-64 px +0.10 HF14 / +0.12 val.
- Multi-family sheets gain; single-family sheets are flat (-0.01 / -0.02).
- Val 17 is 53% of the val gain (0.570 -> 0.699, 23 / 0 better/worse).
- Revit loses only on plans: roof plans (4 of 6 worse) and floor plan HF14
  12 (0.44 -> 0.36). Those pool sheets are still v6-drawn (34% of the pool,
  same as v6d), so the loss is elevation-heavy training, not worse plan data.
  Generator next: Revit-style roof and floor plans from the same houses.
