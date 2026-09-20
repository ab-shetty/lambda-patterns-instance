# Handoff

Last updated: 2026-09-20

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds
every number, command and reproduction path. This file holds only what changed
and where to pick up — if a fact appears in one of those two, it is not repeated
here.

## Pick up here (2026-09-20)

A method session. The question was "which architecture reaches ~0.95 train IoU",
asked because every previous answer here cost an overnight run. The answer is
that **fit questions do not need overnight runs**, and once measured cheaply,
**capacity was never the limit** and the 0.95 target was partly unreachable and
wholly beside the point.

**The method: cache the frozen backbone once, and every architecture question
costs seconds.** The backbone is ~25.6M of 28.0M parameters and essentially all
of the per-step cost, and every open architecture question was downstream of
`RefUNet.features()`. `scripts/cache_backbone_features.py` stores `c1..c4` for
image and reference; `scripts/decoder_search.py` trains variants against tensors
already on the GPU. A variant costs **40-90 s** instead of an hour, and the
whole 8-variant sweep took 8 minutes. Seed noise differs sharply by metric -- see the
repeatability numbers below.

It ranks decoders and backbones. It cannot measure backbone finetuning, and its
absolute numbers are not product numbers (frozen backbone, 200 synthetic images,
1024 px). Survivors still need a real run.

**The probe is cheap for FIT and only cheap for fit.** Five runs with identical
arguments: train IoU 0.9079 **sd 0.0032**, HF14 transfer 0.4771 **sd 0.0243**.
Fit is nearly deterministic, so one run settles it. Transfer carries the same
~0.024 this repo has always had, because it is the same 52 hard questions --
the freeze buys speed, not statistical power. Budget transfer seeds exactly as
`startup.md` says, and prefer the paired-per-selection test.

**1. `abshetty/floz-refunet-synth100k-e1` is a byte-identical duplicate of
`abshetty/floz-refunet-res2560-e4`** -- same 372 tensors, same config, same
`source_checkpoint`, same `real_mean_iou` 0.7833. The publish step was pointed
at the wrong `.pth`. **The 100k single-pass checkpoint behind the 2026-09-18
underfitting finding does not exist anywhere.** The entry below it in this file
says it was published; that is wrong. Everything here used `res2560-e4`, whose
training mix contains v6d ids 0-3199, so a regenerated `v6d_train2000`
(`--seed 6 --start 0`, deterministic per `(seed, image_id)`) is genuine
training data for it.

**2. Three cheap measurements killed three hypotheses before any training.**

- **Output stride is not the limit.** `scripts/label_ceiling.py` computes what
  the best possible model with this output geometry would score, including a
  convex hinge optimisation over the stride-4 grid (upsampling is linear, so
  this is at or near the true optimum). On HF14 at 2560: stride-4 optimal
  **0.9798**, stride-1 optimal **0.9798** -- identical. The stride-4 head costs
  ~0.000; the whole ~0.02 shortfall is the evaluator's native -> input -> native
  resize round trip.
- **Threshold is not the limit.** The sweep is flat and 0.35 is already optimal
  (HF14 0.7834 at both 0.35 and 0.40; on train, 0.20 buys +0.0035).
- **The 0.95 train target was above the ceiling.** On the v6d pool the ceiling
  is 0.9058 at 2048, **0.9210 at 2560**, 0.9450 at 4096, because v6d regions are
  small: ceiling 0.9023 for regions under 500k native px, 0.9958 above 5M. So
  "train IoU should reach ~0.95" was measuring against an impossible number on
  that pool. On HF14 the ceiling is 0.9739/0.9759/0.9846 at 2048/2560/4096 and
  never binds -- real targets clear 0.97 in every size band.

  Related: the ceiling rises only +0.015 from 1280 to 2048 while the measured
  score rose +0.050, so **at most a third of the project's largest lever is
  mechanical ceiling-raising** and the rest is real.

**3. The residual is region-level, not boundary-level.**
`scripts/residual_decomp.py` splits every error pixel into five buckets
(it reproduces the published HF14 number to four decimals, 0.7834 vs 0.78331):

| | HF14 real | v6d train | v6d fresh |
|---|---:|---:|---:|
| mean IoU | 0.7834 | **0.7485** | 0.7764 |
| boundary | 19.3% | 16.6% | 21.2% |
| missed whole region | 14.5% | 9.3% | 5.3% |
| missed inside found region | 21.1% | 26.9% | 22.2% |
| **selected wrong region** | **23.9%** | **37.0%** | **42.4%** |
| fringe / spill | 21.2% | 10.2% | 8.8% |

Boundary is at most a fifth of the error anywhere. Note also the model scores
**lower on its own training data (0.7485) than on data it has never seen
(0.7764)** -- a negative train/fresh gap, so nothing image-specific is retained.

**4. Decoder capacity is NOT the limit, and more of it is worse.** Eight
variants on the frozen ResNet50, 200 cached images, 1500 steps:

| variant | decoder params | train IoU |
|---|---:|---:|
| corr4 | 5.57M | 0.9099 |
| selfattn (real self-attention) | 5.48M | 0.9074 |
| baseline (width 128) | 5.12M | 0.9067 |
| crossattn | 5.35M | 0.9037 |
| deep | 6.30M | 0.8816 |
| **w384** | **40.12M** | **0.8675** |
| **w256** | **18.49M** | **0.8662** |

Scaling the decoder 8x makes fit *worse*; conditioning mechanism moves fit by
<0.007. **`run_capacity_probe.sh` would have spent an hour testing the one axis
that does not matter** -- and against a soft-dice train number this repo has
already declared invalid.

**5. Fit is step-limited, and 0.9866 train IoU is reachable today.** Baseline
decoder, frozen backbone, by pool size and step budget:

| pool N | 2 passes | 8 passes | 30 passes | 120 passes |
|---|---:|---:|---:|---:|
| 8 | 0.394 | 0.552 | 0.795 | 0.890 |
| 32 | 0.291 | 0.351 | 0.743 | 0.953 |
| **200** | 0.396 | 0.620 | 0.905 | **0.9866** |

The true optimised ceiling for this setup is **0.9886**, so the existing
width-128 decoder sits essentially on it. The 2026-09-18 "train tops out at
~0.80, a correctly-sized model should reach ~0.95" reading conflated *too few
optimisation steps* with *too little capacity*. (The naive area-pooled ceiling
reported by `decoder_search.py`, 0.9175, is a FLOOR on the true bound, not a
cap -- the model legitimately exceeds it by finding a sharper stride-4 encoding
than area-pooling, exactly as the convex analysis predicts.)

**6. A VISION TRANSFORMER BACKBONE WINS -- the hypothesis this repo has been
deferring since 2026-09-18.** `swin_b` is a Shifted-Window Transformer: real
multi-head self-attention, but hierarchical (strides 4/8/16/32) and windowed
(7x7, shifted between blocks), so it emits the same pyramid a ResNet does and
drops into the existing decoder unchanged. That hierarchy is also why 200
images suffice here: it is ImageNet-pretrained and FROZEN, so only a 4.6M
decoder is fitted on top. Same decoder, same data, same steps; 4 seeds on the
main arms.
HF14 here is 52 cached real selections at 1024 with a frozen backbone, so it is
a RANKING number, far below the product's 0.78:

| backbone | decoder | n | train | fresh synth | **HF14 real** |
|---|---|--:|---:|---:|---:|
| swin_b | selfattn | 4 | 0.7893±0.024 | 0.5534±0.014 | **0.5591±0.018** |
| swin_b | corr4 | 4 | 0.8416±0.010 | 0.5590±0.018 | 0.5262±0.016 |
| swin_b | baseline | 4 | 0.8278±0.028 | 0.5649±0.009 | 0.5179±0.046 |
| resnet50 | corr4 | 4 | 0.9154±0.006 | 0.5813±0.006 | 0.4959±0.026 |
| resnet50 | baseline | 4 | 0.9025±0.011 | 0.5700±0.015 | 0.4661±0.015 |
| convnext_base | baseline | 1 | 0.6839 | 0.4702 | 0.4371 |
| resnet50 | crossattn | 1 | 0.9086 | 0.5725 | 0.3976 |

`swin_b + selfattn` beats `resnet50 + baseline` on real plans by **+0.0842
(n=5 each, 5.5 sd using the measured per-run sd of 0.0243)** while fitting
*worse* (0.789 vs 0.903). Across all 30 probe runs, **corr(train fit, HF14
transfer) = -0.18** and corr(train fit, fresh synthetic) = +0.77. Fitting the
generator better does not transfer; it is mildly anti-predictive.

**It is not an LR artefact.** Sweeping 1e-4 / 3e-4 / 1e-3 / 3e-3 per backbone,
each at its OWN best LR: swin_b reaches 0.5594-0.5600 at three different LRs,
resnet50 tops out at 0.4959 (corr4 @3e-4). The gap narrows but never closes or
flips, and swin_b fits worse at every LR it wins at. Both degrade at 3e-3.

**So the original question answers itself: the architecture that reaches 0.95
on train is the one already in the repo, given more steps -- and reaching it is
not worth doing**, because train fit is anti-correlated with the product metric.

**7. The representation already discriminates materials.**
`scripts/material_separability.py` embeds patches from each labelled family with
a frozen backbone and measures within-image separability. No training at all.
On HF14, frozen ResNet50: same-family cos +0.902, different-family +0.467,
**AUC 0.9936**. So `false_region` -- the single largest error bucket -- is not
caused by features that cannot tell the materials apart. It degrades with patch
size (224/96/48 px -> 0.994/0.966/0.945 on HF14, 0.941/0.921/0.878 on v6d), so
the difficulty is fine-scale and spatial, not semantic. Swin scores *lower*
here (AUC 0.9498, margin +0.0245) while transferring better, so patch-level
separability is not the mechanism behind item 6.

**6b. Does the transformer need MORE data? Frozen, no -- it needs less.**
Baseline decoder, 1024 px, HF14 transfer, 2 runs per cell:

| N images | resnet50 | swin_b | gap | train fit r50/swin |
|---:|---:|---:|---:|---|
| 8 | 0.0920 | 0.2169 | **+0.125** | 0.995 / 0.993 |
| 32 | 0.2294 | 0.3656 | **+0.136** | 0.975 / 0.963 |
| 64 | 0.2571 | 0.2975 | +0.040 | 0.962 / 0.911 |
| 200 | 0.4976 | 0.5313 | +0.034 | 0.910 / 0.746 |

ImageNet paid the transformer's data bill, and the frozen representation is
most valuable exactly where task data is scarcest. **But the gap NARROWS as
data grows** (+0.13 -> +0.03), so do not assume it survives to 5-8k records;
that is the main risk in item 1.

**What the gap DOES grow with is resolution.** At a matched N=64:
1024 px gap **+0.041**, 2048 px gap **+0.190** (13.5 sd, n=6 per arm). An
earlier draft of this entry read the 2048 result as evidence that Swin does
better with LESS data -- that compared 2048/N=64 against 1024/N=200 and
confounded the two. Same resolution, more data narrows it; same data, more
resolution widens it sharply. Production is 2560, which is the favourable
side of both -- but the two effects were never measured together above 2048.

**Size is not the mechanism.** `swin_t` (28.3M, the same class as ResNet50's
25.6M) scores 0.5244 at N=200 against resnet50's 0.4976 and swin_b's 0.5313 --
within noise of the 3x larger model. The win is architectural, and a
size-matched transformer has no more parameters to feed than the incumbent.

**Caveats on item 6, before anyone trains on it.** One learning rate (3e-4) for
every backbone, chosen for the incumbent; ResNet50 carries `IMAGENET1K_V2`
weights against Swin's V1 recipe; 200 synthetic images at 1024 px; frozen
throughout, where production finetunes at `backbone_lr_mult 0.1`. The
consistent 4-seed margin and the fit/transfer inversion are the signal; the
absolute numbers are not.

**Open, in priority order (revised 2026-09-20):**

1. **Run the vision transformer (`swin_b`) in the real pipeline**, unfrozen,
   at 2048-2560 on the documented mix. This is the first lever in months with a
   >5x-sd margin behind it, and `StagedBackbone` already exposes the 4-scale
   pyramid the decoder wants, so the code cost is small. Unfreezing is the
   untested half: everything measured so far holds the backbone fixed.
   Also worth one probe each: a plain ViT / DINOv2 backbone (no hierarchy,
   much stronger pretraining) and `swin_t` (to separate architecture from the
   88M-parameter size of `swin_b`).
2. **Done 2026-09-20: the LR sweep clears item 6** (four LRs per backbone,
   swin_b ahead at each backbone's own best). What it still needs is the real
   pipeline, not more probing.
3. **Attack `false_region` and `missed_inside` directly** -- 46-56% of the
   error and the thing no lever in this repo has ever targeted. The features
   separate materials at AUC 0.99, so this is a propagation/objective problem:
   the model knows what the material looks like and still paints the wrong
   region.
4. **Retire train IoU as a target.** It is anti-correlated with transfer, its
   ceiling is pool-dependent, and the number that motivated it came from a
   checkpoint that no longer exists.
5. More labelled real sources -- unchanged, still the structural constraint.

**Do not re-open:** decoder width/depth as a capacity lever (item 4); output
stride, mask threshold, and boundary methods as HF14 levers (items 2-3);
`run_capacity_probe.sh` as written.

## Pick up here (2026-09-18)

An overnight session on a fresh GH200, rebuilt from a clean clone (every
pipeline count matched `startup.md`). The machine was killed at ~06:50; every
`data/runs/*` path below is gone. Two checkpoints were published:
`abshetty/floz-refunet-res2560-e4` and `abshetty/floz-refunet-synth100k-e1`.
**Correction (2026-09-20): the second upload carries the FIRST one's weights,
byte for byte. The 100k checkpoint was never published and is gone; every
number in this entry that came from it is unreproducible. See the 2026-09-20
entry.**
Full detail and tables: `synth_progress.md` (2026-09-18). **One seed per arm —
nothing below is adopted into the documented recipe.**

**The three things that matter, in order:**

1. **The binding term is synthetic→real transfer, ~0.11.** RefUNet on 100,000
   procedural plans, measured on 300 of the 1,401 held-out draws:

   | | fresh-synthetic | HF14 | train |
   |---|---:|---:|---:|
   | after 1 pass (e0) | **0.8250** | 0.7111 | — |
   | after 2 passes (e1) | 0.7980 | 0.7130 | 0.837 |

   Fresh-synthetic is already 0.83 after ONE pass: the generator is close to
   solved on unseen draws, so more of it cannot help. The ~0.11 gap to real is
   domain transfer and that is the ceiling on the synthetic route.
   **Note the second pass made fresh-synthetic WORSE (0.825 → 0.798) while
   train rose to 0.837** — overfitting onset at 2 repeats, so single-pass is
   the right regime and the train/fresh gap of 0.039 at e1 should NOT be read
   as pure underfitting (an earlier draft of this entry did; corrected).
   Whether capacity also binds is untested: at `--width 128` only ~2.4M of
   28.0M parameters sit outside the ResNet50 backbone. **`run_capacity_probe.sh`
   (width 128/256/384) was queued when the machine died — run it first.**

2. **Training at 2560 gave 0.7834 val-selected**, +0.038 over the identical run
   at 2048 and above the recorded best of 0.7594. One seed, under 2× sd, so not
   a result until replicated — but it is the strongest single lever found.
   It selects an early epoch (4), so schedule it short.

3. **Procedural volume pays past 2.5×, then decelerates.** v6d-only 1,600 →
   12,000 → 100,000 gave 0.6400 → 0.6934 → 0.7130 (+0.053 then +0.020 per ~8×).
   The recorded volume nulls tested 2–2.5× and were under-powered, not wrong.
   With enough unique data you need **steps, not epochs**: 100k seen twice beat
   12k seen eighteen times at fewer steps, and warm restarts are a small-data
   patch. Do not scale past ~100k — item 1 says the ceiling is now fit and
   transfer, not supply.

**Two methodological findings worth more than any single number:**

- **The 52 evaluation reference boxes are byte-identical across every run**
  (verified across epochs, runs and training sets). The eval is 52 fixed
  questions, so arms can be compared **paired per selection** rather than by
  means with sd ~0.02. That is much more power at the same compute and nothing
  in this repo has used it.
- **77% of the HF14 deficit is four images** (14, 12, 18, 7); the other ten sit
  at 0.83–0.99 where only boundary precision remains and four boundary methods
  already measured ~0.000. Median selection IoU is 0.816 against a 0.745 mean —
  a tail problem. Image 7 is the known reference-box-on-text artefact (~+0.011,
  do not chase).

**Open, in priority order (revised 2026-09-18):**

1. **Model capacity has never been tested, and it is the top suspect.**
   Train IoU on 100,000 unique plans with deterministic labels tops out at
   ~0.84. A correctly-sized model should fit far higher — the only irreducible
   floor is the look-alike families v6d injects on purpose. Two facts make
   capacity the obvious candidate:

   - At `--width 128` the model is 28.0M parameters of which ~25.6M is the
     ResNet50 backbone, leaving **~2.4M task-specific** for dense
     reference-conditioned prediction at 2048².
   - **MEASURED 2026-09-18, hard threshold 0.35, 300 training images**
     (`fresh_synth_iou.py` pointed at the training pool). This is the first
     valid train IoU in the project:

     | epoch-1 checkpoint | train | fresh synthetic | HF14 |
     |---|---:|---:|---:|
     | RefUNet (28.0M) | **0.8022** | 0.7980 | 0.7130 |
     | crossattn (27.9M) | **0.7668** | 0.7715 | 0.7042 |

     **Train ~= fresh for both** (+0.004, -0.005): zero generalization gap
     within synthetic, on 100,000 unique plans. Absolute fit is ~0.80 where a
     correctly-sized model on deterministic labels should reach ~0.95.
     **This is unambiguous underfitting and it is not a data problem.**
     Note this supersedes an intermediate "overfitting onset" reading, which
     came from comparing a soft-dice train number against a hard fresh number.

   - **Caveat on every OTHER "train IoU" in this repo (the ~0.84 figure, and
     all earlier entries):**
     it is derived from the SOFT dice term (`mask_loss`, `train_refunet.py`
     ~line 195, which uses `logits.sigmoid()` un-thresholded), so it is NOT
     comparable to the hard threshold-0.35 union IoU that every reported
     result uses. `startup.md`'s "Train IoU (derived from the Dice term)" and
     the 2026-09-17 "train mIoU 0.85->0.88" carry the same defect. **A valid
     hard train IoU has never been measured.** Measure it first — point
     `scripts/fresh_synth_iou.py` at training images — before concluding
     anything from a train/fresh or train/real gap.
   - **`RefCrossAttnUNet` is NOT a transformer and adds NO capacity**: 27.9M
     against RefUNet's 28.0M — *smaller*. It swaps cross-attention in for the
     conditioning block at the two coarsest scales of the same ResNet50+FPN
     CNN. Every "crossattn ties/loses" result in this repo is a conditioning-
     mechanism result at constant, very small capacity. **The transformer
     hypothesis has never actually been tested here.**
     **Partly retired 2026-09-20**: a hierarchical vision-transformer BACKBONE
     (Swin-B, frozen, ImageNet) now beats frozen ResNet50 by +0.0842 on real
     plans (5.5 sd, n=5). Still untested: a plain non-hierarchical ViT
     (DINOv2-class), and ANY transformer backbone finetuned rather than frozen
     -- which is where the "a ViT needs far more than 200 images" objection
     actually bites. See the 2026-09-20 entry.

   `./run_capacity_probe.sh 7` (width 128/256/384 → 28.0/39.6/58.3M, one epoch
   each at 1024 on the 100k pool, ~1h) is written and unrun. It is only a first
   step: it scales the decoder, not the backbone, so even width 384 leaves the
   ResNet50 untouched. A real test of "is the model the limit" wants a larger
   or attention-native backbone (ViT/Swin) at 100k+ single-pass, judged on
   **train IoU reaching ~0.95** before anything else is concluded about data.
2. **Second seed at 2560**, and whether 3072 continues the trend.
3. **Close the 0.085 fresh-synthetic → real transfer gap.** This is now the
   binding term on the synthetic route, and it is a generator-realism problem:
   the 2026-09-17 audit's open items (texture irregularity, implausible
   material colours, the 2-story label-collision bug) are unaddressed.
4. **crossattn on 100k ties overall but WINS where it matters — chase this.**
   Paired per-selection on the 52 fixed questions (same pool, steps, seed):
   mean diff +0.0088 for RefUNet, **t = +0.36 — a tie**. But the per-image
   split is large and structured:

   | crossattn better | | crossattn worse | |
   |---|---:|---|---:|
   | img 7 (1 sel) | **+0.616** | **img 0 (2 sel)** | **−0.677** |
   | **img 14 (9 sel)** | **+0.100** | img 18 (10 sel) | −0.046 |
   | img 16, 2, 12, 24, 1 | +0.001..+0.034 | img 27, 25, 11, 23 | −0.009..−0.042 |

   It gains on **image 14 — 36% of the entire HF14 deficit** and the
   long-range/wrong-material failure — and recovers image 7's text-crop
   reference. The whole tie is paid for by **one catastrophic collapse on
   image 0** (0.861 → 0.185). Diagnose that single failure and the
   architecture is ahead. This is the first evidence that attention conditioning
   helps precisely where the propagation diagnosis says it should.

   Headline numbers, for the record: HF14 val-selected 0.7069 vs 0.7130;
   fresh-synthetic 0.7951 vs 0.8250 (1 pass), 0.7715 vs 0.7980 (2 passes) —
   so it fits the generator worse while matching on real plans.

   Superseded note: crossattn at matched data — after one pass
   each: fresh-synthetic 0.7951 vs 0.8250, HF14 0.7067 vs 0.7111, train_loss
   0.5360 vs 0.5072. It fits *worse*, consistent with a weak-prior model still
   being under-served at 100k rather than with an architecture win. (An earlier
   draft claimed it fit better, from a single noisy tqdm batch — wrong.) Its
   epoch-1 number did not finish. A fair test needs a larger data regime again,
   or more capacity.
5. More labelled real sources — still the structural constraint for 0.9.

**Do not re-open:** everything in the 2026-09-17 list, plus — new — synthetic
volume beyond ~100k plans, and warm restart #2 (both val-selection and
checkpoint-averaging chose windows inside restart 1 on every run that had two).

## Pick up here (2026-09-17)

Four things happened this session, on a machine that no longer exists — every
`data/runs/*` checkpoint named below is gone; only what's written here and in
`synth_progress.md` survives. **Nothing below is adopted into the documented
recipe yet.** `synth_progress.md` was also reorganized: the pre-2026-08-06 /
retired-query-model material is now `synth_progress_archive.md`, actually moved
out this time rather than relabeled in place.

**1. A fourth Gemini round (`floz-gen-gemini-r4`, 88 images) is a clean null.**
Merged with r2+r3 into `floz-gen-gemini-r234-clean` (168 images), augmented 18x
(3,024 records), substituted for r23 in the documented mix ->
`mix6676_with_r4` (6,676 records) vs. baseline `mix5092`, 2 seeds each, 2048px:

| protocol | mix5092 | mix6676_with_r4 | Δ |
|---|---:|---:|---:|
| val-selected | 0.7443 ± 0.0263 | 0.7366 ± 0.0000 | −0.0077 |
| averaged (SWA) | 0.7559 ± 0.0147 | 0.7702 ± 0.0172 | +0.0143 |

Sign flips between protocols, both gaps under 1x sd — noise, not signal. r4 is
stylistically indistinguishable from r2/r3 by `pool_style_stats.py` (same
generator, same prompt family) and the model already fits r2/r3/r4 equally well
in-sample (0.850/0.870/0.844 mean IoU) — it isn't under-fit, it's redundant.
Consistent with the existing finding that the Gemini pool's marginal value
already shrank to +0.010 (noise) at 2048px before r4 existed. **Do not re-run
this exact test; a fifth Gemini round would need to look different, not just be
more of the same.**

**2. The documented 9-epoch schedule stops before the real ceiling — found by
warm-restarting, not by a longer single schedule.** The existing "16-epoch
schedule is null" finding (`startup.md`) used one continuously-annealing
cosine. Instead: reset the optimizer and give a **fresh** cosine restart from a
converged checkpoint, repeatedly. On `mix6676_with_r4` seed 31 (one seed):

| epoch | train mIoU (full-mix sample) | val-complement IoU (14 img / 77 sel) |
|---|---:|---:|
| 8 (documented recipe's last epoch) | — | 0.7924 |
| 18 (+1 restart, 10 epochs) | 0.8704 | 0.7993 |
| **20 (+2 restarts, peak)** | 0.8615 | **0.8127** |
| 28 (+2 restarts, final) | 0.8762 | 0.8064 |

The epoch-20 checkpoint is published: `abshetty/floz-refunet-warmrestart-e20`
(private HF Hub repo, `scripts/publish_refunet.py`) — the machine that trained
it is gone, this is the only surviving copy.

Real gain through epoch 20 (+0.033 over the documented recipe's own epoch 8),
then a plateau: epochs 22-28 kept climbing on train (0.85->0.88) without val
following past the epoch-20 peak — the overfitting signature, just not a
blowup. A further low-LR (5e-6, well under the ~1e-5 floor every restart cosine
bottoms out at) sustained continuation from epoch 28 also plateaued (HF14 diag
0.69-0.75, no trend). **One seed. Needs replication before this changes the
documented recipe**, but it means the 9-epoch number in `startup.md` is
probably not this setup's ceiling, and checkpoint-averaging's "check whether the
run has turned over" caution (2026-09-08 entry below) applies to schedule length
too, not just pool composition.

**3. A cross-attention variant of `RefUNet` exists and ties it at both the
documented budget AND under extended training.** `refmask2former/ref_attn_unet.py`:
same ResNet50 backbone, same FPN, same direct-union training objective — only
the two *coarsest* scales' conditioning changes from global-average-pooled-
vector-plus-conv (`ConditionBlock`) to multi-head cross-attention (image tokens
as queries, reference tokens as keys/values, so a location gets a learned blend
of the reference's actual features instead of one broadcast vector). `--model
crossattn` on `train_refunet.py`. On `mix6676_with_r4` seed 31:

| | 9 epochs (documented budget) | +1 warm restart (epochs 9-18) |
|---|---:|---:|
| RefUNet | val-sel 0.7366 | val-complement 0.7993 (ep 18) |
| crossattn | val-sel 0.7314 | val-complement 0.8004 (ep 17) / 0.7992 (ep 18) |

Gaps of −0.005 and +0.001 respectively — a clean tie under two different
training regimes now, not just one. RefUNet went on to a second restart (peak
0.8127 at epoch 20); crossattn's second restart wasn't run — that's the natural
next step. The epoch-17 checkpoint is published:
`abshetty/floz-refunet-crossattn-e17` (same reason as above); the machine's
`epoch_18.pth` to continue from is not.

This is NOT the same territory as `--corr-grid` (screened negative, monotonic
with matching precision): corr-grid only ever produced similarity *scores* as
extra channels, never aggregated the reference's feature *values*. Full
attention did not regress the way corr-grid did — parity, not harm — so the
hypothesis that value-aggregation avoids corr-grid's overfitting-to-spatial-
correspondence failure is not falsified, just not yet confirmed as a win either.

**4. Visual quality audit of the synthetic generator — new, and it changes how
much weight the "generator quality" deprioritization should carry.** Full
write-up and 8 example images: `synth_progress.md` (2026-09-17 entry),
`synth_quality_audit/`. Every prior synth-vs-real comparison in this project
(ink, saturation, contrast, aspect, `pool_style_stats.py`) is an aggregate
statistic; nobody had looked at the images next to real ones until this
session. v6 (`generate_synthetic_v6.py`) is a real composition improvement over
v5 (the pool still in the documented mix) — but has a **confirmed bug** (2-story
level-mark label collision, ~line 1451, one-line fix) and systematically
implausible material colours (blue clay tile, blue cultured stone, pure-green
brick — real instances of these materials don't look like that). v6d's own
+0.028 val-selected (2026-09-08 entry) was measured on a generator carrying
these defects. **The "source count, not the generator" deprioritization in
priority #1 below was reached from statistics that miss this — it should be
read as weaker than it was written.** Cheap next step: fix the label-collision
bug and constrain colourisation to per-material plausible hue ranges, then
re-measure v6d before drawing a stronger conclusion either way.

**Open, in priority order (revised 2026-09-17):**

1. **Replicate the warm-restart extended-training finding** (item 2 above) at a
   second seed, and on `mix5092` too (not just the r4-inclusive mix) — this is
   the single biggest number this session produced and it is one seed.
2. **More labelled sources**, still the structural constraint — but see item 4
   above before treating "generator quality doesn't matter" as settled.
3. **Fix the two confirmed/likely v6 generator defects** (label collision,
   material-colour plausibility) and re-measure v6d on the mix. Cheap, and the
   prior null may not have been a fair test of the generator.
4. **Push resolution past 2048.** Unchanged from 2026-09-08: 2560 is +0.05 at
   one seed with averaging, Gemini's median long side is 3168px.
5. **Run crossattn's second warm restart** (item 3 above) — ties RefUNet
   through the first restart; RefUNet's peak needed a second.
6. Re-measure the Gemini pool's per-pool value at 2048 (unresolved, +0.033 at
   1280 vs. +0.010 at 2048) — today's r4 null is consistent with "already near
   its ceiling at 2048" but doesn't fully resolve it.
7. Regularization — train/HF14 gap widens with epochs, though item 2's finding
   complicates "fitting is not the constraint" somewhat: val DID follow train up
   for 10 extra epochs before plateauing, further than the 9-epoch recipe alone
   showed.

**Do not re-open:** connectivity, tile-similarity confusables, v6e look-alike
pairs, orientation snapping in the polygon regularizer, `--anchor`, SAM 3 as a
product-task model without fine-tuning it on that task, and — new this
session — adding a further same-style Gemini round without a genuinely
different generator or source (see item 1 above). MixUp (pixel-blending two
scenes) was reasoned through and not tried: it directly conflicts with the
project's repeated finding that fine local texture is the signal, so pixel
blending is expected to actively hurt, not just be neutral.

Older handoffs (2026-09-08, 2026-08-08) removed 2026-09-17: their substance was
fully duplicated in `synth_progress.md` (v6d, checkpoint-averaging-plateau
caution, connectivity/disconnection findings, ranking loss, template matching)
and `labeling_assist.md` (SAM 3 fine-tuning, its encoder-resolution and
zero-shot limits), and their priority/do-not-reopen lists were superseded by
the 2026-09-17 ones above. The two facts that weren't preserved elsewhere —
image 14's outsized weight in HF14 (9/52 selections, 0.291 mean) and a 2026-08
A10 GPU memory baseline — are now in `synth_progress.md`; the A10 note was
dropped as stale (measured at 1280px/19.3GiB, the documented recipe is now
2048px/~51GB).

## Facts that live elsewhere

Single copies, so they cannot drift. Do not restate them here.

| what | where |
|---|---|
| task semantics, metric definition, what does *not* count as evidence | `PROJECT_UNDERSTANDING.md` |
| the `sample_reference_box` fix and why pre-2026-08-06 numbers are incomparable | `PROJECT_UNDERSTANDING.md`, mechanism in `startup.md` |
| current result, both selection protocols, the checkpoint and its artifacts | `startup.md` |
| every reproduction command, the data rebuild, the validation split | `startup.md` |
| evaluation rules and the fixed HF14 indices | `startup.md` |
| per-experiment history and the screened levers | `synth_progress.md` (pre-2026-08-06 / retired query-model material: `synth_progress_archive.md`) |
| the labelling-assist model, its numbers and its negatives | `labeling_assist.md` |
| the synthetic generator's visual quality audit and example images | `synth_progress.md` (2026-09-17), `synth_quality_audit/` |

Two standing traps: the reference resize in `refmask2former/dataset.py` looks
like a bug and "fixing" it costs 0.153, and `--corr-grid` / `--scale-matched-ref`
are implemented but screened negative — read the warnings before touching either.

## Relevant implementation

- `refmask2former/ref_unet.py` — shared backbone, conditioning blocks, FPN mask.
- `refmask2former/ref_attn_unet.py` — cross-attention conditioning variant
  (2026-09-17, item 3 above); `--model crossattn` on `train_refunet.py`.
- `scripts/publish_refunet.py` — publish a `RefUNet`/`RefCrossAttnUNet`
  checkpoint to the HF Hub as safetensors, mirroring `publish_sam3.py`'s
  rationale: `data/runs/` doesn't survive the machine, a Hub repo does.
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
- `scripts/family_similarity_probe.py` — per-family appearance similarity:
  `--metrics` (per-image confusability, intra/inter margin, correlations),
  `--tiles` (tile-pool pairs; writes the table `--tile-sim` consumes),
  `--local-data` (pool distribution at training resolution).
- `scripts/generate_images_openai.py` — resumable generation of the unlabelled
  realistic pool on `gpt-image-2`, arbitrary `--size`, receipts carrying the
  billed token usage, and a contact sheet for the visual gates.
- `scripts/upload_unlabelled_roboflow.py` — validates a batch whole, uploads it
  unlabelled to a named project (`--create` to make one), and disposes of a
  review project again with `--delete-project`.
- `scripts/render_image_generation_prompts.py` — `--version v4` is the current
  prompt set, rendered from `specs_v4.jsonl`; v1-v3 are kept so their results
  stay reproducible and v1's recorded SHA still validates.
- `scripts/build_specs_v4.py` — the v4 subject list, built to the evaluation
  set's measured distribution and weighted toward the failure traits.
- `scripts/pool_style_stats.py` — aspect, colour, ink, contrast and fill
  regularity for a pool, so a generated round can be compared with the real 28
  instead of judged by eye.
- `generate_synthetic_v6.py` — the v6 synthetic generator: a house grammar
  (blocks, roofs, dormers, porches, townhouse rows) projected into elevations,
  roof plans and floor plans, with ~20 materials drawn as continuous ruled
  fields in feet. Defaults are the v6d round; the v6e knobs that measured
  negative are kept as named constants with their results in comments.
- `run_synth_v6.sh` — reproduces any synthetic-only arm end to end (pool,
  merge, two-phase train, val-selection, averaging). `./run_synth_v6.sh
  headline 7` is the 0.686 result.
- `scripts/cache_backbone_features.py` — runs a frozen backbone ONCE over a
  pool and stores `c1..c4` for image and reference, padded to one grid with a
  validity mask. `--backbone {resnet50,convnext_base,convnext_small,swin_b,
  swin_t}` via `StagedBackbone`, which exposes all of them as the same
  stride-4/8/16/32 pyramid; `--source {local,hf14,val14}` so real plans cache
  as the 52 fixed questions.
- `scripts/decoder_search.py` — trains reference-conditioned decoder variants
  against that cache (seconds each, not hours): width, depth, dense
  correlation, cross-attention, real self-attention. Reports hard train IoU,
  fresh-synthetic and cached-HF14 transfer.
- `scripts/label_ceiling.py` — what the best possible model with this output
  geometry and resize chain could score, naive and convex-optimised. Run it
  before treating any fit target as reachable.
- `scripts/residual_decomp.py` — splits missing IoU into boundary / missed
  region / missed interior / wrong region / fringe, plus a threshold sweep.
  Says WHICH axis to spend on.
- `scripts/material_separability.py` — within-image same-vs-different family
  separability of a frozen backbone's features. No training; an upper bound on
  what any decoder reading those features can group.
- `scripts/hf_ckpt_to_pth.py` — rebuilds a loadable `.pth` from a published
  safetensors + config.json pair.
- `scripts/average_checkpoints.py` — averages the last epochs of a run and
  ranks the window on validation; the selection protocol as of 2026-09-07.
- `scripts/select_epoch_on_val.py` — val-selected protocol; one process per run,
  run them in parallel. `--tta {1,2,4,8}` for dihedral test-time augmentation
  (measured +0.004, not worth 8x inference).
- `scripts/train_status.sh` — one-screen progress for every run under
  `data/runs`, including live epoch and running peak.
- `scripts/run_*.sh`, `run_*.sh` — the experiment drivers, one per ablation.

## Data limitation and where to push next

The mix holds **194 unique real-ish source plans** (86 scraped + 28 generated v1
+ 80 Gemini r2/r3); everything else is synthetic or deterministic offline
variants. Source count is still the binding constraint, and generated sources
are now the best-value supply: 1.65x a scraped plan each, and they pick their own
resolution, which the natively-640px scraped pool never can.

**Before generating more, read the status section at the top of
`image_generation/README.md`.** The v1 100-prompt scheme was framed wrongly --
whole sheets with legends, then fragments -- and yielded 28 of 98. v3/v4 fixed it
(72-88% yield) by aiming at the eval set's measured distribution. There is no
labelling backlog worth mining; more data means a better round.

**Resolved 2026-09-06**: the standing hypothesis that image 12's thin wall poche
was a 1280px downscaling artifact is supported -- training at 2048 is worth +0.05
to +0.07 overall. Whether image 12 specifically recovers has not been checked
per-image.

## Evaluation rules

In `startup.md` — single copy, so the two cannot drift.

## Generated artifacts

**A clone has none of this.** Datasets, checkpoints, logs and evaluation JSON
live under git-ignored `data/` and `logs*`, so every `data/...` path quoted in
these docs is a provenance record, not a file you have. Preserve them between VMs
for byte-identical artifacts, or rebuild: `startup.md` covers the headline recipe
and `synth_progress.md` ("Rebuilding what these findings used") covers the
ablation pools, which `startup.md` does not. The v6 synthetic pools and every
synth-only arm rebuild from `./run_synth_v6.sh <arm> <seed>`. Source, scripts and documentation
are committed; credentials are never stored in the repo.
