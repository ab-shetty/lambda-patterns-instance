# Synthetic Dataset Progress

## 2026-09-24 — goal: 0.85 HF14 at training resolution <= 2048

Baseline: `abshetty/floz-refunet-swint-mixr4-e8` (swin_t, 282 sources, 2048),
HF14 0.7887, validation 0.7854. Everything below is chosen on VALIDATION;
`scripts/val_failures.py` reports the thin/mid/thick buckets and the flagged
validation sheets for any metrics file.

**Where the best model still fails on validation.** 63% of its remaining
validation loss is ONE sheet, val 17 (a four-unit townhouse, 27 of 77
selections): its thin base band (pattern1, 9 selections, mean 0.314) and its
siding (pattern2, 18 selections, 0.763). The real-data mix had already fixed
most of what the v6d-only model failed on (thin bucket 0.353 -> 0.645, val
10/9 0.48 -> 0.85, val 13 0.63 -> 0.81).

**Looking at the actual reference crops disproved the "tiny blurry reference"
story.** The band's references are sharp teal stone-brick courses (41-67 px
native on a 2219 px sheet). The roof is drawn with the SAME running-bond course
pattern in grey. The model gets the band and ALSO selects the roof and porch
roofs: it matches texture and ignores colour, the only cue separating them.
v6d almost never contains two families that share a fill and differ only by
colour, so nothing teaches that colour can be the boundary.

**Tried and dropped on the way (validation):**
- `--dr-scale-min 0.8` (new flag; `--domain-random` shrink floor, historical
  0.4), full-LR restart from the published e8: epochs 8-10 at 0.771 / 0.748 /
  0.758, thin bucket not improving; a 2e-4 restart knocks the converged model
  back. Stopped.
- Two-pass inference (`scripts/eval_refine.py`, re-reference from the model's
  confident region): bigbox +0.005 (fires on 15/77, never on val 17), mosaic
  -0.03. Not the lever.
- `--small-ref-prob 0.5` (new flag; deliberately tiny training references),
  gentle fine-tune lr 5e-5: epoch 8 0.776, val 17 and thin WORSE. Its premise
  was disproved by the crops; stopped after one epoch.

**Now testing:** `generate_synthetic_v6.py --same-fill-new-colour 0.5` (new,
default 0 = v6d byte-identical): in colour sheets the accent reuses the main
wall's fill and the foundation band the roof's fill, each in a clearly
different colour, band labelled -- val 17's structure. Pool
`v6dsf05_1600` (seed 6, ids as v6d_1600) swapped for `v6d_1600` in the mix
(`data/mixed/v6dsf05mix_plus_r4`, 6,676 records); gentle fine-tune from the
published e8, lr 5e-5, 4 epochs.

Result: epoch 8 made the band WORSE (0.314 -> 0.099; the small-ref fine-tune
too, 0.103) -- the fine-tuned model now selects the ENTIRE roof confidently.
Any further training on this mix pushes toward texture-only matching.

**Cause: the offline `--strong` augmentation grayscales 15% of every real and
Gemini copy** (`scripts/build_mix.py`, ~760 records in the mix). In those copies
a colour-only family boundary is pixel-identical under different labels, so
the model is trained to ignore colour. `--gray-prob` (new, default 0.15 =
unchanged pools; the draw is always consumed, so 0 changes nothing else)
rebuilds them without it: `*-strong18nogray`, and
`data/mixed/v6dsf05mix_plus_r4_nogray` (6,676 records = v6dsf05_1600 + the
three no-gray pools). `./run_nogray_mix.sh 7` trains the full recipe from
ImageNet weights on it (the published model learned the invariance over all 9
epochs; fine-tuning it did not undo it), then selects epoch and inference size
on validation and reads HF14 once.

**The colour story was then disproved by a direct intervention** (should have
come first; 2 min, no training): on val 17, keep the roof's texture and
recolour it red, then ask with the band references. Roof selected / band IoU:

| model | original colours | roof recoloured red |
|---|---|---|
| published mixr4 e8 | 0.21 / 0.316 | 0.00 / 0.589 |
| no-gray + same-fill, epoch 4 | 0.92 / 0.096 | 0.00 / 0.518 |

Both models USE colour. The band's desaturated teal and the roof's grey are
simply close, so with the same course pattern the model calls them one
material. Grayscale augmentation is not what blinds it, and the no-gray run is
worse on this sheet mid-training. Val 17's band is a subtle-hue question on
one sheet; chasing it further is tuning to validation.

**No-gray + same-fill mix, full recipe (`./run_nogray_mix.sh 7`) -- HF14
0.8153, but the data change is NOT supported.** Val-selected epoch 8 (the
final one); inference size chosen on validation: 2048 -> val 0.768, 4096 ->
val 0.791; HF14 read once at 4096: **0.8153**, paired vs the published 0.7887
+0.027 (t 1.52; sign 28/9, p 0.003). Its own HF14 at 2048 is 0.7809, so the
gain is inference size. And on validation the PUBLISHED model at 4096 scores
0.8014, above this model's 0.791 -- validation prefers the old data. Report
the data change as unsupported; the inference-size lever is real and much
larger on the real-data mix than on v6d-only (+0.009 there).
`run_restart_swa.sh` chooses among the published e8, a completed restart of it
and SWA windows, each at 2048 and 4096, on validation, and reads HF14 once.

**Completed restart -> HF14 0.8170 (new best, validation-supported).**
`./run_restart_swa.sh`: one fresh cosine from the published e8 on its own mix.
Validation @2048/@4096 per restart epoch: e8 .751/.788, e9 .757/.780, e10
.707/.748, e11 .749/.789, e12 .761/.798, **e13 .793/.808**, e14 .778/.786, e15
.791/.795; SWA 12-15 .789/.805, 13-15 .789/.801, 14-15 .780/.794; published e8
.785/.801. Chosen: restart e13 @4096. HF14 read once: **0.8170**, paired vs
0.7887 +0.028 (t 1.48; sign 29/14, p 0.03). Inference size sweep on validation
(published e8): 2048 .785, 3072 .786, 4096 .801, 5120 .799 -- 4096 is the
plateau. TTA x4 at 4096: +0.0015, dropped.

**Back to synthetic data (user direction): two generator changes from the
visual audit**, both default-off in `generate_synthetic_v6.py` (v6d verified
byte-identical after each), each on its own RNG so the rest of a sheet is
unchanged when it fires:

- `--hardscape-plan P` (HF14 0 and 12): a floor plan gets ONE unlabelled
  finish over every room, a forced MEP dashed-arc overlay (70%), and a LABELLED
  exterior hardscape -- rear/L deck band, front walk from the door, optional
  side patio -- as random ashlar, deck planks or pavers. Sheets whose hardscape
  would touch the building or close around it fall back to a normal plan.
  (Debugging note: a translucent orange box over a house in the contact sheet
  was the existing 7% highlighter markup on the L-shape's bounding box --
  intended, like HF14 24/25/27 -- not a fill bug.)
- `--same-fill-subtle P` (val 17): the foundation band reuses the roof's fill
  with a SMALL hue shift (25-60 deg) at similar lightness, labelled -- the
  close-colour case the probe showed the model failing, unlike
  `--same-fill-new-colour`'s large colour differences.

Pool `v6dHS_1600` (seed 6, v6d_1600's ids; both knobs 0.5: 126 hardscape
plans, 323 colour sheets with a labelled band), mix `v6dHSmix_plus_r4`.
`./run_synth_ft_ab.sh`: from restart e13, identical lr 5e-5 x3 fine-tunes,
control (v6d_1600) vs treatment, validation @4096 per epoch, and a
PRE-DECLARED HF14 comparison of the two final checkpoints with images 0/12
as the hardscape change's target (validation has no floor plans).

**Result: null.** HF14 paired treatment - control **-0.0000** (t 0.00; 12
better / 8 worse / 32 tied); the pre-declared targets got slightly WORSE
(image 12 -0.034, image 0 -0.032). Validation @4096 final epoch 0.798 vs
0.791, val 17 0.592 vs 0.577 -- noise. The control fine-tune itself drifts
down (0.808 -> 0.791).

**Why, from looking at the reference crops and the fills side by side:**
- HF14 0's patio is RANDOM (non-coursed) ashlar -- mixed rectangles, no
  continuous course lines -- and the house interior is running-bond planks.
  v6's `stone` fill is COURSED ashlar, i.e. visually running bond: the
  generator taught "stone ~ running bond", exactly the confusion. The
  hardscape pool drew hardscape mostly as `stone`, so it reinforced it (the
  treatment floods image 0's interior MORE than control). No v6 fill looked
  like random ashlar (`rubble` is jittered trapezoids).
- HF14 12's deck boards run horizontally on one side and vertically on
  another, one family; v6 draws each family in one direction.

Fixes: new `ashlar` fill (`_ashlar`: 6x6-unit blocks packed with 2x2/2x1/
1x2/1x1/3x2/2x3 stones, alternate block rows offset so no joint runs
through; checked side by side against HF14 0's crop), hardscape kinds now
ashlar 3/7, plank 2/7 (rear band + patio one direction, side return + walk
perpendicular, labelled as one family via `hardscape@v`), grid, stone. Dose
raised: `--hardscape-plan 1.0` -> 279 hardscape plans in `v6dHS2_1600`
(subtle band 0.5). Treatment-only rerun (`TREAT_DATA=data/mixed/
v6dHS2mix_plus_r4 TREAT_TAG=v6dHS2 ./run_synth_ft_ab.sh`), reusing the
control arm (same base, recipe, seed).

**Result: worse.** Validation @4096 final 0.768 vs control 0.791 (val 13
0.921 -> 0.773); HF14 paired -0.013 (t -1.23; 11/14/27). The targets moved a
little (image 12 +0.046, image 0 +0.007) and the dose cost elsewhere. Two
targeted synthetic rounds on the best model: null, then slightly negative.

**Fill-vocabulary audit** (one reference crop per labelled family across all
28 real sheets, next to a swatch of every v6 fill). Real materials v6 cannot
draw:
1. TEXTURED masonry/roofing: BIM exports render each brick/stone/shingle with
   its own tone plus grain (val 17/18/19's teal stone -- the val 17 failure
   family -- 18's asphalt, 2/3/5's brick). Every v6 fill is flat line-art.
2. LIGHT mortar on coloured brick (2, 3, 4, 5); v6 darkens joints.
3. 3D-shadowed lap siding: bold grey band under each board (25:3, 27:4).
4. Basketweave / parquet (1:1, 8:1).
(0:1 random ashlar is now covered by `ashlar`.)

`--mottle P` (new, default 0 = v6d byte-identical; decided from the style's
own seed): coloured brick/block/stone/rubble/ashlar/shingle/asphalt fills get
a tone per unit (connected components between joints) plus fine grain; half
the masonry ones get light mortar. Tested ALONE: `v6dMO_1600` (seed 6, --mottle
0.7; labels byte-identical to v6d_1600, only pixels differ),
`TREAT_DATA=data/mixed/v6dMOmix_plus_r4 TREAT_TAG=v6dMO ./run_synth_ft_ab.sh`.

**Result: indistinguishable from control -- and that indicts the test, not
only the change.** HF14 paired +0.0003 (t 0.18) with **48 of 52 selections
tied** (|d| < 0.01); validation @4096 final 0.787 vs 0.791. A gentle fine-tune
(lr 5e-5, 3 epochs) of a converged model, with synthetic data a quarter of
the mix, barely changes predictions, so `run_synth_ft_ab.sh` cannot detect a
pixel-only synthetic change; the hardscape rounds moved only because a larger
dose shifted other sheets. **Test synthetic changes with a full retrain**
(`run_nogray_mix.sh`-style, ~80 min on a GH200) or on synth-only at 1024
(`run_transfer_ratio.sh`-style, ~30 min, three arms concurrently), paired
over the 52 selections against a same-recipe control. `--mottle` is untested,
not negative.

**Fair screen, synth-only (`./run_synth_only_screen.sh 7`, swin_t at 1024,
three arms concurrently ~20 min, same 1,600 ids/recipe/seed, epoch 8 at 2048
inference; one seed):**

| arm | validation | HF14 | HF14 paired vs v6d |
|---|---:|---:|---:|
| v6d_1600 | 0.6465 | 0.6401 | - |
| `--mottle 0.7` | 0.6293 | 0.6434 | +0.003 (t 0.21; 17/13/22) |
| `--hardscape-plan 1.0 --same-fill-subtle 0.5` (ashlar) | **0.5545** | **0.6679** | +0.028 (t 1.23; 25/22/5) |

`--mottle`: null in a test that can move predictions. Hardscape pool: the two
splits DISAGREE (HF14 +0.028, validation -0.09). Validation has no floor
plans, and this pool also changes elevations via the subtle band -- confounded.
Not adopted under the protocol. **Next: separate `--hardscape-plan` from
`--same-fill-subtle`** (two more arms of this screen), then a second seed.

**Multi-scale inference** (`scripts/eval_multiscale.py`, restart e13,
probabilities averaged at native size): validation preferred 2048+3072+4096
(0.8165 vs 0.8100 for 4096 alone in the same code); HF14 read once: **0.8123**,
below 0.8170. Not adopted; 0.8170 @4096 stays the best.

## 2026-09-23 — v7: v6 re-tuned by looking at Gemini, not by matching a metric

**Untrained. No HF14 number yet.** `generate_synthetic_v7.py` wraps v6 (v6d
stays bit-for-bit: verified by regenerating 48 v6d draws and diffing). Every
change came from looking at 27 Gemini r2-r4 sheets next to v6d contact sheets,
label overlays included:

| seen in Gemini, missing in v6d | v7 change |
|---|---|
| drawing fills the frame; v6d is a small house on ~70% empty sheet | crop to ink bbox + 1.5-5% pad, resample to 2400-3600 long side |
| cross-gable front projections, set-back upper volumes, dormers, porches | 0-2 front projections, pop-up upper volume, higher dormer/porch/chimney rates |
| material changes by volume (B&B gable / shingle / brick base) | accent almost always, second accent 35% |
| muted olive/tan/beige/grey, brick-red or charcoal roofs, white trim | muted palette; no teal/purple/royal blue or coloured trim |
| readable callouts and level marks | lettering x1.6 |
| crisp dark mono linework | 70% of light/faint ink re-drawn at normal |
| no free-standing poles | v6 bug: corner boards sized from the gable's bbox ran to the apex; `CORNER_BOARD_CLIP` (off in v6) |
| **annotators label wall materials only** -- roof, trim, chimney, foundation unlabelled in elevations | those families are still drawn but labelled at 5-25% |
| windows/doors always cut out | hole prob 0.8 -> 0.97 |

The labelling row was found by looking at label overlays of Gemini, not by
measuring. Realism-by-metric is 5-for-5 null (2026-09-21), so the statistics
below are a **post-hoc sanity check, not the design target** -- nothing was
tuned to move them. They happened to land on Gemini (300 draws each):

| | Gemini r2 / r3 / r4 | v6d | v7 |
|---|---|---|---|
| aspect | 2.38 / 2.44 / 2.36 | 1.75 | 2.04 |
| regions / img | 6.4 / 5.6 / 7.0 | 10.7 | 8.6 |
| families / img | 1.75 / 1.52 / 1.86 | 2.59 | 2.17 |
| labelled fraction | 0.313 / 0.277 / 0.312 | 0.295 | 0.310 |
| median region / sheet | 0.037 / 0.049 / 0.031 | 0.021 | 0.030 |

Still unlike Gemini: roof plans are one concentric-course block where Gemini's
are L-shaped multi-hip plans with valleys and mixed membrane/gravel families;
no photographed-paper sheets; floor plans lack Gemini's furniture density.

**Test:** `./run_v7_ab.sh 7` -- swin_t, v7-only (1,991: 9 of 2,000 draws had
no labels left after the wall-only filter) against the published v6d arm
(`abshetty/floz-refunet-swint-v6d-e8`, 0.7066), identical recipe, paired over
the 52 selections. Remember v6r (region
scale matched) was null on swin_t; v7 differs by also changing what is
labelled and how the sheet looks.

**Result (one seed): null, leaning negative.** v7 **0.6834** val-selected
(epoch 7; epoch-8 0.6886) against v6d **0.7066** (the published checkpoint
re-evaluated here reproduces 0.7066 exactly). Paired over the 52 selections:
**-0.0232, t(51) = -0.96, p = 0.34**, 20 better / 20 worse / 12 tied. It moves
*which* questions are answered, not how many: v7 gains on image 14 (+0.096
over 9 selections, the largest deficit image), 2 (+0.206) and 0 (+0.160), and
loses on 16 (-0.493, a roof plan), 25 (-0.252) and 27 (-0.170, both
highlighter-markup sheets). One candidate story, unconfirmed and subject to
`PROJECT_UNDERSTANDING.md`'s per-image caution: v7 leaves elevation roofs
unlabelled, which may teach "roof courses are background" and hurt the roof
plan. First visually-driven realism round; still no synthetic change on
record that beats v6d on swin_t.

**Train -> real gap: unchanged.** `fresh_synth_iou.py`, valid-pixel, 400
plans each, fresh = ids 100000+ (never generated before):

| | v6d model (e8) | v7 model (e7) |
|---|---:|---:|
| train (seen) | 0.7875 | 0.7989 |
| fresh, own generator | 0.8351 | 0.8069 |
| fresh, other generator | 0.7793 (v7) | 0.7627 (v6d) |
| HF14 | 0.7066 | 0.6834 |
| own-fresh -> HF14 gap | 0.129 | 0.124 |
| HF14 / own-fresh | 0.846 | 0.847 |

v7 is a *harder* pool (its model reaches 0.807 on fresh v7, v6d's reaches
0.835 on fresh v6d) and the transfer ratio is identical to three decimals, so
the lower HF14 is the lower synthetic fit carried through, not worse transfer
-- and the visual realism bought no transfer either. Train < fresh on v6d
(0.788 vs 0.835) is sample mix at n=400, not a finding. Open: whether v7
closes on v6d given the steps to fit it (the pool is harder at a matched
budget), and the roof-label ablation (`./run_v7_ab.sh 7 v7roof`).

**Roof-label ablation: null.** `./run_v7_ab.sh 7 v7roof` -- v7 with elevation
roofs labelled as v6d does, every other label identical (same draw order).
HF14 **0.6894** val-selected (epoch 6). Paired: **+0.006 vs v7** (t = +0.30),
**-0.017 vs v6d** (t = -0.67). It does recover roof-plan image 16 (+0.30 over
v7) but gives back image 14 (-0.054 over 9 selections), so the roof story
explains one image and not the mean. Train-to-real, on its own labels:

| | v6d | v7 | v7roof |
|---|---:|---:|---:|
| fresh, own generator | 0.8351 | 0.8069 | 0.8077 |
| HF14 | 0.7066 | 0.6834 | 0.6894 |
| HF14 / own-fresh | 0.846 | 0.847 | 0.854 |

Three arms inside one seed's noise on transfer. v7 is not adopted; v6d stays
the synthetic source. The generator and both arms stay reproducible for a
second seed or a longer-budget test.

**Does generated data transfer better than procedural at matched size? No.**
`./run_transfer_ratio.sh 7` (~30 min: 1024 px, three arms concurrently, data
prep on 64 chunks). swin_t, 148 sources x18 each, epoch 8, no selection:

| arm | fresh / held-out | HF14 | HF14 / fresh |
|---|---:|---:|---:|
| Gemini, 20 held-out sources (fold A) | 0.7640 | 0.6767 | 0.886 |
| Gemini, fold B | 0.7792 | 0.6397 | 0.821 |
| v6d, 400 fresh plans | 0.7477 | 0.6448 | 0.862 |

Gemini 0.853 mean vs v6d 0.862. The ~0.85 ratio now holds across source
(v6d / v7 / v7roof / Gemini), size (148 to 100k), backbone (RefUNet, swin_t)
and resolution. A gap that ignores the training source is probably not domain
transfer but **question difficulty**: HF14 asks harder questions than a typical
unseen plan of any source (77% of its deficit is four multi-material images).
At matched 148 sources Gemini and v6d also tie on HF14 (0.658 vs 0.645); the
two folds alone differ by 0.037, so the earlier "a generated plan is worth
~1.65x" (measured against the 640px scraped pool) does not extend to "worth
more than a procedural plan". Caveats: one seed, 20 held-out Gemini
selections per fold.

Tooling trap found: `fresh_synth_iou.py` decides "seen" by FILENAME, and
every merged pool is renamed `item_XXXXXX`, so a held-out merged pool against
a merged training pool matches every name and scores n=0 (reported as 0.0000).
Pass an empty `--trained-pool` when the split is known disjoint.

**Question difficulty does NOT explain the gap, and the ~0.85 ratio was
partly a protocol artifact.** `scripts/question_difficulty.py` scores unseen
plans with the exact HF14 protocol (every labelled instance a question) and
gives every question source-agnostic features (families on the sheet, target
regions, target area, and `lookalike` = texture-descriptor similarity of the
target to the nearest other family); `question_difficulty_report.py`
summarises:

| model | synthetic (HF14 protocol) | HF14 | HF14/syn | is_real net of difficulty |
|---|---:|---:|---:|---:|
| swin_t v6d 1,995 @2048 | 0.6489 | 0.7066 | 1.089 | -0.173 (t -4.4) |
| swin_t v6d 148 @1024 | 0.5335 | 0.6448 | 1.209 | -0.172 (t -4.3) |
| swin_t Gemini fold A @1024 | 0.7666 | 0.6767 | 0.883 | -0.138 (t -2.3) |
| swin_t Gemini fold B @1024 | 0.7346 | 0.6397 | 0.871 | -0.172 (t -2.4) |
| swin_t v7 1,991 @2048 | 0.7193 | 0.6834 | 0.950 | -0.142 (t -2.5) |

(OLS, SE clustered by sheet; HF14 has only 14 sheets.)

1. Under the HF14 protocol v6d's questions are HARDER than HF14's (97%
   multi-family, 8 target regions vs 2), so the "universal ~0.85" came from
   `fresh_synth_iou.py`'s one-question-per-plan training sampler, which asks a
   different question mix. Do not compare fresh_synth_iou numbers with HF14
   as a transfer ratio again; use this script.
2. Net of difficulty, every model -- Gemini included -- pays the same
   **~-0.14 to -0.17 on real plans**. A real-plan penalty exists and generated
   data does not escape it, so it is not the procedural fill vocabulary.
3. Where it shows: synthetic single-family questions are the easiest (0.88 to
   0.94); HF14's single-family questions score 0.61 to 0.67, mostly image 12
   (4 selections at ~0.44, thin wall poche) and image 0. HF14's high-lookalike
   questions score the same as synthetic ones (0.548 vs 0.538 on v6d). One
   blind spot: the features count only LABELLED families, so unlabelled ink on
   a real sheet is invisible to them.

**Looking at where the real-plan penalty lives** (`visualize_refunet_selection.py`
for swin_t v6d-148 / Gemini-A-148 @1024 and v6d-1,995 @2048, all 14 HF14
sheets sorted by score, plus native-resolution crops). The two worst sheets
are both **detailed floor plans whose target is exterior hardscape**: image 12
(0.44, a faint deck of fine planks) and image 0 (0.53, random-ashlar patio and
walk). Every model, Gemini-trained included, misses parts of the hardscape and
floods the interior, which on image 0 is an UNLABELLED running-bond plank
floor finish under drop shadows, furniture, room names and dimension strings,
and on image 12 is dashed electrical/ceiling arcs, grey furniture outlines and
notes. No training source contains that sheet type: v6d floor plans are sparse
rooms with a few hatched floors, and Gemini produced ~2 floor plans in the 27
sheets inspected. Other weak sheets have other causes: 14 (0.54, 9 selections)
is two elevations drawn tiny across a wide strip (resolution), 27 has
highlighter markup. Roof plans (16, 1) and most elevations are fine.

Two sheets and six selections, so confirm on the validation split's floor
plans before building anything. If it holds, the next generator should make
floor plans with a full unlabelled interior finish, furniture, shadows, text
and MEP overlays, with the target on exterior paving/decking.

**Validation has no detailed floor plans**, so the floor-plan story cannot be
confirmed there (`scripts/sheet_scores_contact.py` renders any split sorted by
score). Its weak sheets fail differently -- val 13 is HF14 14's type (two tiny
elevations on a 5:1 strip), val 10/9 pick unlabelled lap-siding walls for a
mono shingle roof, val 17 matches a thin dark foundation band to the dark roof
by tone. Aspect ratio does NOT explain scores (Spearman -0.11 over 28 sheets).

**What does: target thickness at input resolution.** Over all 129 real
selections (HF14 + val, swin_t v6d-1,995 @2048), Spearman(IoU, thickness) =
+0.58: thin (< 74 px at the 2048 input) 0.437, mid 0.716, thick 0.825. The
thin third is ~half of all missing IoU -- bands, columns, edge-on roofs, thin
decks. (So v7's Gemini-style unlabelling of trim/chimney/foundation removed
exactly the question type the eval set asks most badly.)

**`--domain-random` shrinks training sheets 0.4-1.0x** (`dataset.py` ~408), so
a "2048" model mostly sees 820-2048 px sheets and is evaluated at the top of
that range. Inference size, chosen on VALIDATION:

| model (trained at) | 1024 | 2048 | 2560 | 3072 | 4096 |
|---|---:|---:|---:|---:|---:|
| swin_t v6d-1,995 (2048), val | - | 0.670 | 0.691 | 0.708 | 0.718 |
| ... thin bucket | - | 0.353 | 0.405 | 0.452 | - |
| swin_t Gemini-A-148 (1024), val | 0.660 | 0.713 | - | 0.707 | - |

Confirmed once on HF14, paired: v6d-1,995 2048 -> 4096 **+0.009** (t 0.88;
sign test 29/14, p 0.03); Gemini-A 1024 -> 2048 **+0.056** (t 3.29, p 0.002).
Real but small at 2048; large when the nominal size is too small. The
training-side version (less downscale, or larger nominal size) is untested.

## 2026-09-20 — Cheap architecture screening, and what it retired

Method session on a fresh GH200, rebuilt from a clean clone. Full narrative and
the priority list are in `codex_doc.md` (2026-09-20); this entry holds the
detail that does not belong there.

**Pools.** `generate_synthetic_v6.py` is deterministic per `(seed, image_id)`
(`compose()` seeds `random.Random(seed * 1_000_003 + image_id)`), and the
documented build seed is 6 (`run_synth_v6.sh`). So a train-side slice and a
genuinely-unseen slice of the same distribution are both reproducible:

```bash
python3 generate_synthetic_v6.py --n 2000 --out data/synthetic/v6d_train2000 \
  --seed 6 --start 0      --workers 48          # 1,995 ok, 16 s
python3 generate_synthetic_v6.py --n 600  --out data/synthetic/v6d_fresh600 \
  --seed 6 --start 100000 --workers 48          # 603 ok
```

ids 0-1999 are inside `res2560-e4`'s v6d 3,200, so `v6d_train2000` is genuine
training data for it; ids 100000+ were never generated before.

**Ceiling by resolution** (`scripts/label_ceiling.py`, naive area-pooled grid):

| res | HF14 | v6d |
|---:|---:|---:|
| 1280 | 0.9587 | 0.8628 |
| 2048 | 0.9739 | 0.9058 |
| 2560 | 0.9759 | 0.9210 |
| 4096 | 0.9846 | 0.9450 |

The convex-optimised bound is the honest one and it is tight against naive on
HF14 (0.9798 vs 0.9759 at 2560) but much looser on small regions -- on the
1024 px probe pool naive reads 0.9108 where the optimum is 0.9886. **Quote the
optimised number, or label the naive one a floor.** Stride does not matter:
optimised stride-4 and stride-1 are both 0.9798 on HF14 at 2560.

The 1280 -> 2048 ceiling gain is +0.0152 against a measured +0.050, so the
resolution lever is mostly real rather than mechanical. Worth remembering the
next time resolution is credited or blamed.

**Threshold sweeps** are flat on every pool: HF14 peaks at 0.35 exactly
(0.7834), v6d train peaks at 0.20 for +0.0035 over 0.35. Calibration is not
hiding anything.

**Residual decomposition** (`scripts/residual_decomp.py`, res2560-e4 at 2560).
The script reproduces the published HF14 mean to four decimals (0.7834 vs
0.78331), which is the check that the whole decomposition is trustworthy.
Buckets as a share of all error pixels:

| | HF14 | v6d train | v6d fresh |
|---|---:|---:|---:|
| mean IoU | 0.7834 | 0.7485 | 0.7764 |
| median | 0.8264 | 0.8510 | 0.8830 |
| boundary (within 4 px of GT edge) | 19.3% | 16.6% | 21.2% |
| missed whole region (recall < 0.25) | 14.5% | 9.3% | 5.3% |
| missed inside a found region | 21.1% | 26.9% | 22.2% |
| selected a wrong region | 23.9% | 37.0% | 42.4% |
| fringe / spill | 21.2% | 10.2% | 8.8% |

Two things to keep: boundary is never more than a fifth, and **train (0.7485)
scores BELOW fresh (0.7764)**. A negative train/fresh gap means nothing
image-specific survives training at all, which is the opposite of the
memorisation the 2026-09-18 entry worried about at 2 repeats.

**Target size does not explain HF14's deficit.** Selections run from
sqrt(area) 189 px to 2972 px, and measured IoU is flat across that range
(0.74-0.82) except the four largest, which are nearly solved (0.9713). The
ceiling is >= 0.97 in every size band, so nothing about small selections is
unreachable on real plans. corr(log size, IoU) = +0.18. The small-region
ceiling problem is a **v6d pool artefact**: synthetic regions are far smaller
relative to the sheet (ceiling 0.9023 under 500k px).

**Decoder sweep** (frozen ResNet50, 200 images, 1500 steps, 1024 px). Full
table in `codex_doc.md`. The shape of it: every conditioning mechanism lands
within 0.007 on fit, and the two WIDE variants are the worst (w256 0.8662,
w384 0.8675 against baseline 0.9067) -- at a fixed step budget, extra width
costs more than it buys. `run_capacity_probe.sh` is superseded: it sweeps
exactly this axis, an hour per point, judged on the invalid soft-dice number.

**Fit ladder.** Baseline decoder, by pool size x step budget, hard IoU at 0.35:

| N | 2 passes | 8 | 30 | 120 |
|---:|---:|---:|---:|---:|
| 8 | 0.394 | 0.552 | 0.795 | 0.890 |
| 32 | 0.291 | 0.351 | 0.743 | 0.953 |
| 200 | 0.396 | 0.620 | 0.905 | 0.9866 |

Against an optimised ceiling of 0.9886. Fit is a step-budget question, not a
capacity question, and at 200 images it is solved.

**Probe repeatability, measured (important).** Five runs of the identical
command, resnet50 + baseline:

| metric | mean | sd |
|---|---:|---:|
| train IoU | 0.9079 | **0.0032** |
| cached-HF14 transfer | 0.4771 | **0.0243** |

Freezing the backbone makes the probe fast, not statistically powerful. Fit is
near-deterministic and one run settles it; transfer carries the same ~0.024 the
full pipeline does, for the same reason (52 hard questions, question difficulty
dominating). An earlier draft of this session's writeup claimed ~0.003 noise
for the probe generally -- that is true only of the fit column, and the
backbone margin was initially quoted with error bars that were too tight
(+0.0930 se 0.0134; corrected to +0.0842 at 5.5 sd with n=5 per arm).

**LR sweep, per backbone** (1e-4 / 3e-4 / 1e-3 / 3e-3, best three shown, HF14):

| backbone | best | second | third |
|---|---|---|---|
| swin_b | baseline@1e-3 0.5600 | selfattn@1e-4 0.5597 | selfattn@3e-4 0.5594 |
| resnet50 | corr4@3e-4 0.4959 | baseline@3e-4 0.4752 | baseline@1e-4 0.4631 |
| convnext_base | corr4@3e-4 0.4528 | baseline@3e-4 0.4371 | crossattn@3e-4 0.4239 |

Swin is ahead at each backbone's own best LR, and is stable across three of
them, so the margin is not an artefact of one LR tuned for the incumbent. Every
backbone degrades at 3e-3. Note swin_b's winning configurations are also its
WORST-fitting ones (train 0.761-0.803 against resnet50's 0.903-0.922), which is
the fit/transfer inversion again.

**Backbone bake-off.** `StagedBackbone` exposes ResNet50, ConvNeXt and Swin as
the same stride-4/8/16/32 pyramid, so the decoder is unchanged across them.
ConvNeXt-Base is worst on both axes (baseline train 0.6839, HF14 0.4371, one
seed). Swin-B transfers best and fits worst; ResNet50 the reverse. 4 seeds on
the main arms, table in `codex_doc.md` and `startup.md`.

**Material separability** (`scripts/material_separability.py`, no training).
Within-image same-vs-different family cosine separability of frozen features,
HF14, level 3 (1/32):

| backbone | same | different | margin | AUC |
|---|---:|---:|---:|---:|
| resnet50 | +0.9020 | +0.4665 | +0.4354 | 0.9936 |
| convnext_base | +0.9178 | +0.6280 | +0.2898 | 0.9644 |
| swin_b | +0.9906 | +0.9661 | +0.0245 | 0.9498 |

By patch size, resnet50: 224/96/48 px -> 0.9936/0.9661/0.9452 on HF14 and
0.9406/0.9207/0.8776 on v6d. So the representation carries the material
identity, degrading at fine scale and on synthetic. Note Swin has the WORST
separability and the BEST transfer, so this metric does not explain the
bake-off -- do not use it to pick a backbone.

Only 7 of the 14 HF14 images have >= 2 labelled families, so the real-plan
separability numbers rest on 7 images. The v6d rows (60 images, many families)
are the better-powered half.


## 2026-09-18 — Procedural volume, single-pass training, and the fit ceiling

Rebuilt from a clean clone on a fresh GH200 (every pipeline count in
`startup.md` matched exactly). One seed per arm unless stated; treat all of it
as screening, not as results.

**1. Synthetic volume works, but only past the scale the old nulls tested.**
v6d-only, 2048, val-selected HF14:

| pool | steps | repeats/image | HF14 |
|---:|---:|---:|---:|
| 1,600 | 5,742 | 29 | 0.6400 |
| 12,000 | 26,730 | 18 | 0.6934 |
| 100,000 (single-pass) | 25,000 | 2 | 0.7130 |

+0.053 then +0.020 per ~8x -- decelerating by ~60% each time, so the next 8x is
worth ~+0.008. The two recorded volume nulls (v5 1,600->3,200; v6b
1,600->4,000) tested 2x and 2.5x jumps at 1280, below where the effect appears.
They are not wrong, they were under-powered. **0.7130 from procedural plans
with zero labelled data exceeds `mix5092` at 1280 (0.7097, 3 seeds).**

**2. With enough unique data you need steps, not epochs.** 100,000 plans seen
TWICE beats 12,000 seen eighteen times, at fewer steps. Warm restarts are
revealed as a small-data patch -- they recycle a pool that has nothing left to
teach. No restarts were used in the 100k run.

**3. The grammar does not saturate in pixels but does in teaching.** At 2,000
drawn plans only ~4% have a near-duplicate (cosine > 0.95) and NN-distance
scaling implies intrinsic dimension ~23, so draws stay genuinely novel. But the
model already handles novel draws: see item 4. Pixel novelty that poses no new
problem is not diversity in the sense that matters. `scripts/fresh_synth_iou.py`.

**4. The binding term is synthetic->real transfer (~0.11), and a second pass
already overfits.** RefUNet, 100k, on held-out draws (300-400 of 1,401):

| checkpoint | train | fresh synthetic | HF14 real |
|---|---:|---:|---:|
| epoch 0 (one pass) | -- | **0.8250** | 0.7111 |
| epoch 1 (two passes) | 0.837 | 0.7980 | 0.7130 |

Fresh-synthetic reaches 0.83 after a SINGLE pass -- the generator is nearly
solved on unseen draws, so additional volume cannot help. **The second pass
lowered fresh-synthetic (0.825 -> 0.798) while train rose**: overfitting onset
at only 2 repeats, which is why single-pass is the right regime.

An earlier draft of this entry read the e1 train/fresh gap of 0.039 as proof of
underfitting and "reversal" of `startup.md`'s "fitting is not the constraint".
That was too strong -- the e0->e1 fresh-synthetic decline is an overfitting
signature. What is solid: fresh-synthetic ~0.80-0.83 against real 0.71, so
**transfer, not supply and not (demonstrably) capacity, is the ceiling**.
The old conclusion was measured on 3-8k records where val turned over; it does
not hold in this regime. The remaining **0.085 fresh-synthetic -> real** gap is
domain transfer, and it -- not data volume -- is what caps the synthetic route.
At `--width 128` the ResNet50 backbone is ~25.6M of 28.0M parameters, leaving
~2.4M for the reference-conditioned decoder, so capacity is the obvious suspect
(width 256 -> 39.6M, 384 -> 58.3M). A capacity sweep was queued and the machine
died first; it is the open question.

**5. Training resolution 2560 beats 2048 (one seed).** Same mix
(`mix5092` + 3,200 v6d = 8,292), same seed, same schedule:

| res | val | HF14 val-selected |
|---|---:|---:|
| 2048 | 0.8119 | 0.7453 (0.7676 averaged) |
| **2560** | **0.8296** | **0.7834** (epoch 4) |

+0.038, above the project's recorded best of 0.7594. One seed and under 2x sd,
so **not a result yet** -- needs a second seed. Checkpoint published:
`abshetty/floz-refunet-res2560-e4`. Note it selected epoch 4: at 2560 the run
peaks early.

**6. Adding 3,200 v6d to the documented mix is null.** 0.7453 val-selected /
0.7676 averaged vs the recorded 1,600-v6d mix at 0.7667 / 0.7454 -- sign flips
between protocols, both gaps under 1x sd. Warm restarts to e28 did not help:
both val-selection and checkpoint-averaging chose windows inside restart 1
(e13-18), and every restart-2 window scored worse.

**7. Where the HF14 deficit actually is.** Per-selection decomposition of the
0.745 arm (`metrics_epoch_*.json` carries every selection):

| image | sel | mean IoU | share of deficit |
|---:|---:|---:|---:|
| 14 | 9 | 0.466 | 36.3% |
| 12 | 4 | 0.441 | 16.9% |
| 18 | 10 | 0.787 | 16.1% |
| 7 | 1 | 0.001 | 7.5% |
| the other 10 images | 18 | 0.83-0.99 | 10.4% |

**Four images carry 77%.** The other ten are boundary-precision-only, and four
boundary methods already screened at ~0.000. Image 7 is the known
reference-box-on-text artefact (worth ~+0.011, do not chase). Median selection
IoU is 0.816 against a 0.745 mean: this is a tail problem.

The visual audit (`scripts/visualize_refunet_selection.py`) shows image 14
selecting the WRONG material (roof band when the reference is wall) and image 12
finding only the poche fragment nearest the reference, not the perimeter run --
i.e. a long-range propagation failure. **Caution:** a "sparse/faint targets"
story fitted all three panels and is contradicted by the record -- faintness
correlates -0.749 on HF14 but -0.003 on validation, and images 11/16/1 are
sparse and score 0.926-0.952. Per-image stories remain cheap and wrong here.

**8. The reference box is identical across every run.** Verified across epochs,
runs and training sets: image 7's is always (1448, 550, 259, 259). So the eval
is 52 FIXED questions, which means arm-to-arm comparison can be **paired** per
selection instead of comparing means with sd ~0.02 -- far more power on the same
compute. Nobody has used this. `scripts/reference_quality_probe.py` says crop
representativeness does NOT predict IoU (corr +0.053 HF14, -0.257 validation),
so hand-picking boxes would recover image 7 and little else.

**Open:** the capacity sweep (item 4), a second seed at 2560 (item 5), and
crossattn on 100k (it fit better mid-run -- dice 0.041 vs RefUNet's 0.089 -- but
its epoch-1 result did not finish before the machine died).

## 2026-09-17 — Visual quality audit: the generator looks synthetic in ways the metrics never caught

Every synth-vs-real comparison in this file and `labeling_assist.md` (ink,
saturation, contrast, aspect, family/instance counts, `pool_style_stats.py`) is
an aggregate statistic. Nobody had actually looked at the images side by side
against real ones until now. Examples: `synth_quality_audit/` (repo root).

**v5 (`toparea1600_balanced`, still the pool in the documented `mix5092` /
`mix6676_with_r4`, i.e. every headline number in `startup.md`).** Roof plans
render as flat hatched-fill polygons with no per-material texture at all
(`01_v5_roofplan_flat_fill.png`). Material-texture assignment can be
structurally wrong: an elevation's roof is filled with a brick/subway-tile grid
pattern instead of shingle coursing (`02_v5_elevation_brick_pattern_on_roof.png`).
Against a real elevation from the same pool (`03_...png`) — individually
rendered shingle courses, lap siding, stone base, proper callouts and title
block — the gap is not subtle.

**v6 (`generate_synthetic_v6.py`, tested standalone in the entries below but
never adopted into the documented mix).** Composition is a real improvement:
townhouse rows, dormers, garages, multi-unit sheets, material callouts with
leader lines (`05_v6_townhouse_saturated_colours.png`,
`06_v6_blue_claytile_stone_plus_label_overlap_bug.png`). Two defects found and
one confirmed in source, none previously documented:

1. **Material colours ignore physical plausibility.** Clay tile and cultured
   stone both rendered in saturated royal blue (`06_...png`); brick rendered
   pure saturated green (`07_v6_green_brick_with_markup_dots.png`). Real
   instances of these materials have a narrow, muted, earthy palette; the
   colourised-mode palette picks arbitrary saturated hues with no per-material
   constraint.
2. **Confirmed bug: 2-story level-mark label collision.**
   `generate_synthetic_v6.py` ~line 1451-1453: `labels = [("FIN. FLOOR", 0.0),
   ("T.O. PLATE", -(fh))]`, then for `stories >= 2` appends `("SECOND FLOOR",
   -fh)` — "T.O. PLATE" and "SECOND FLOOR" land at the identical y-coordinate
   and render on top of each other, illegibly. Visible in `06_...png`. Not
   fixed; a one-line offset fix.
3. **Unexplained:** one roof-plan facet fills with a nested-concentric-rectangle
   pattern matching no real roofing convention (`08_v6_roofplan_unexplained_concentric_pattern.png`).
   No `taper`/`drain`/`concentric` logic found in a quick source search — flagged,
   not confirmed as a bug.
4. All texture line-work (siding, standing-seam, brick coursing) stays
   perfectly regular and evenly spaced in both v5 and v6 — no drafting
   irregularity, no weathering, no line-weight variation. Real and
   Gemini-generated examples (`04_...png`) show genuine irregularity (no two
   drawn stones alike). This is likely the single biggest visual tell and
   nothing in either generator currently addresses it.

**Correction, checked before writing this down:** the scattered dot markup
visible on some v6 samples (`07_...png`) is *not* a bug — it's
`DOT_MARKUP_PROB`, an intentional feature simulating highlighter/review markup,
consistent with `PROJECT_UNDERSTANDING.md`'s note that HF14 images 24/25/27
carry real markup and robustness to it is a model requirement.

**Why this matters for the "source count vs. generator quality" question below
and in `codex_doc.md`:** that conclusion was reached from aggregate statistics
that this audit shows miss the actual, visible defect. v6d's own +0.028
val-selected (positive at both seeds, `synth_progress.md` 2026-09-08) was
measured on a generator that still has at least one confirmed rendering bug and
systematically implausible material colours — the deprioritization of
"generator quality" versus "more sources" should be treated as resting on a
weaker foundation than it reads as. Not re-tested this session; the fixes above
(label-collision offset, constrain colourisation to per-material plausible hue
ranges, investigate the concentric-fill facet) are cheap and worth doing before
the next v6 measurement, independent of the texture-irregularity question,
which is a bigger, unscoped generator change.

## 2026-09-08 — v6d added to the documented mix: unsettled, not pursued

Paired 2x2 on one GH200, rebuilt from a clean clone (every pipeline count matched
`startup.md` exactly; `mixbase` reproduced at 0.7410 val-selected against the
recorded 0.7394). `mixv6d` = the documented mix5092 plus 1,600 v6d plans (6,692
records), both arms at 2048, documented two-phase 9 epochs, seeds 7 and 31.

| arm | s7 val-sel | s31 val-sel | mean | averaged mean |
|---|---:|---:|---:|---:|
| `mixbase` (5,092) | 0.7257 | 0.7563 | 0.7410 +- 0.0216 | 0.7442 |
| `mixv6d` (6,692) | 0.7667 | 0.7712 | 0.7690 +- 0.0032 | 0.7454 |

**+0.028 val-selected (positive at both seeds), null (+0.001) averaged.** At
1.3x the baseline sd this is under the ~2x threshold that requires a third seed,
so it is **not a result**. Not pursued: the remaining gap to the 0.90 target is
~0.14, and source count -- not the generator -- is the lever that moves that
(28 -> 86 sources bought +0.073; removing synthetic entirely costs only 0.026 at
2048). Do not re-open this without a reason beyond "+0.028 looked real".

**The durable finding here is about the protocol, not the pool.** Both `mixv6d`
runs peaked at the *final* epoch; both `mixbase` runs peaked mid-run:

```
mixbase  s7   .622 .673 .684 .705 .757 .738 .709 .693 .726   peaks e4
mixbase  s31  .628 .694 .701 .745 .746 .754 .748 .756 .733   peaks e7
mixv6d   s7   .639 .695 .731 .701 .694 .743 .716 .743 .767   peaks e8 (last)
mixv6d   s31  .668 .718 .712 .732 .729 .764 .696 .764 .771   peaks e8 (last)
```

Checkpoint averaging assumes the tail is a plateau. When a run is still climbing
at the last epoch, averaging blends the best weights with worse mid-run ones and
reads ~0.04 low -- which is the entire disagreement between the two protocols
above. **Check whether a run has turned over before trusting the averaged
number**, on any future pool that trains longer without plateauing. Note this is
not the step-budget effect: 9 epochs on 6,692 records is 7,524 steps against
5,724, so v6d had *more* steps and still had not converged. A longer schedule for
this pool was not run; the recorded 16-epoch null was measured on `mix5092`,
which does turn over by e8, so it does not transfer.

## 2026-09-07 — v6: synthetic drawn like a drawing, not quilted from tiles

**Why.** 80 hand-labelled Gemini plans beat 1,600 v5 synthetic plans, and the
three pools side by side say why. v5 quilts 224px raster tiles into 2-3
full-width colour bands per elevation, floats windows at random positions, and
gives every family its own saturated colour. Measured with
`scripts/pool_style_stats.py` against the real 28: twice the ink (0.224 vs
0.111), 68% colourised vs 57%, square (aspect 1.66 vs 2.59), and *over*-ruled
(48,749 vs 11,826 on fill regularity -- a quilted tile of parallel lines with
nothing interrupting it). The eval set and the Gemini plans are ruled line
fills at physical scale masked by real building geometry, so a boundary is an
eave, a rake, a belt course, a corner board or a change of fill; in the
monochrome half of the set it is never a colour change.

**What v6 is** (`generate_synthetic_v6.py`, 0.01 s/image on 48 cores, output in
the local-data layout so the merge/quota/train tooling is unchanged):

- a house grammar (1-3 blocks; gable/hip/shed/flat roofs with pitch and
  overhang; dormers, chimneys, porches, garages) projected into elevations
  with occlusion, plus roof plans (facets, ridges, hips, skylights) and floor
  plans (rooms, poche, doors, fixtures, finishes, MEP overlay) from the same house;
- ~20 material procedures drawn as continuous ruled fields in feet through one
  px/ft scale (lap, board-and-batten, shingle, brick, stone, stucco stipple,
  standing seam, asphalt, tile, plank, tile grid, poche ...), so a fill
  interrupted by a window resumes on the same grid -- the mechanism the Gemini
  v4 prompt asks for;
- one material schedule per house: the siding recurs on every wall of every
  view, the roof on every plane, an accent on gable ends or a wainscot;
- ~55% colourised BIM-export style, ~45% monochrome at three contrast levels;
  windows with casings and mullions cut as label holes; callouts with leaders,
  level marks, dimension strings, graph-paper grounds, highlighter/dot markup,
  excerpt crops. Sheets carry 1/2/4 views (55/38/7%).

Smoke pool vs real 28: ink 0.091 vs 0.111, contrast 0.411 vs 0.437, coloured
65% vs 57%, 1.9 families and 0.29 labelled area per image (Gemini: 1.6, 0.30).

**Standalone (synth-only, 1,600 records, 1280, two-phase 9 ep).** Val-selected
HF14 with the per-epoch peak in brackets:

| pool | seed 7 | seed 31 |
|---|---:|---:|
| v5 `toparea1600_balanced` | 0.5570 [0.6006] | 0.5890 [0.5890] |
| v6 first cut | 0.5348 [0.5427] | 0.5125 [0.5126] |
| v6b (label policy below) | 0.5044 [0.5325] | -- |

v6 loses standalone -- but not uniformly. Per image at the seed-7 peaks, v6
**wins every monochrome group** (HF14 mono 0.457 vs 0.422; validation mono
0.586 vs 0.543) and the hardest images (0: +0.13, 12: +0.13, 23: +0.29, 13:
+0.12), and loses everything on the colourised multi-family sheets: 17 (-0.38
over 27 selections), 19 (-0.49), 18, 25, 27 (about -0.2 each). Those are the
blue board-and-batten townhouses and olive houses with 3-4 families where trim
bands and window casings are labelled as their own family. The first cut
labelled one or two families and never trim, so a model could select "the big
siding region" without consulting the reference; v5's 2-3 distinct colour
bands force reference matching. **v6b** fixes the label policy: roof and accent
labelled ~always, a second accent on 35% of houses, a trim family (belt
courses, fascia, corner boards, window casings, in a colour) on 35%, and a few
saturated eval-17/19 colours in the palette. Families per sheet 1.9 -> 2.4.
One seed of v6b moved exactly the images the diagnosis named (19: 0.35 -> 0.63,
18: 0.54 -> 0.66, 25: 0.51 -> 0.62) and paid for it on mono roofs and plans
(15, 16, 11, 9, 13), so standalone it still trails v5; image 17 stays at 0.16
for both v6 pools against v5's 0.56. The two pools win on disjoint images, which
is why phase B tests v6b **added to** the documented mix rather than replacing v5.

**Union of the two synth pools (v5 1,600 + v6b 1,600, no real data), 1280, seed 7:
val-selected HF14 0.6179 (val 0.7035)** -- the best synth-only number on this
box, against 0.557/0.589 for v5 alone. The two pools win on disjoint images and
the union keeps most of both: image 17 goes 0.16 (v6) / 0.56 (v5) -> 0.68, and
the mono/hard images keep v6's gains. v6b volume alone (4,000 records) is null:
0.52 val-selected, the flattened held-out loss said as much. Per image the
union's remaining gap to the mix model (0.618 vs 0.713 on HF14) sits in image 0
(+0.41, stone patio), 23 (+0.43), 18 (+0.13 x10), 25, 27 and 14; the visual
audit (`data/evaluations/vis_union`) shows one failure mode behind them:
**over-selection** -- asked for the roof it also takes the brick base (18) or the
wall (23), on the faint sheet (14) it selects blank paper, on the plan (0) it
takes the tinted rooms instead of the patio grid.

Two further generator rounds, both aimed by that audit rather than at images:

- **v6c**: townhouse rows (18% of houses) -- 2-5 identical units with party
  walls, one door/garage/dormer per unit, belt courses, brick base, coloured
  trims; eval images 17-19 are exactly this sheet type.
- **v6d**: (1) a flat fill in mono mode is blank paper with an outline and is no
  longer labelled (eval 21/22 leave those unlabelled; labelling them teaches
  "select blank white"); (2) eave and window-head shadows in colour mode, on
  the wall, label unchanged; (3) unlabelled distractors -- a dashed neighbour
  "beyond", a hatched fence/retaining wall, a tree; (4) light unlabelled room
  tints on 35% of floor plans, as in eval image 0.

- **v6e**: the audit on the v5+v6d model shows the blank-paper over-selection
  gone and one failure left: **look-alike families**. Asked for the dark brick
  base of image 18 it selects the dark roof strips and vice versa; on 25/27 it
  swaps two light sidings. So v6e makes families share a colour and differ only
  in fill more often (accent = main colour on 40% of houses; the base band --
  now brick/block/stone/concrete, 0.6-2.6 ft, labelled 70% -- takes the roof's
  colour 30% of the time). Note the v5 finding that *tile-similarity*
  confusables were harmful; this is colour-matched pairs with distinct ruled
  fills, the case the eval set actually poses. **Result: negative**, 0.583
  against 0.660 for v5+v6d at the same seed -- below the seed spread, and the
  same sign as the v5 finding. The generator's defaults are back to v6d
  (`SAME_COLOUR_PAIR_PROB`, `FOUND_ROOF_COLOUR_PROB`, `FOUND_LABEL_PROB` keep
  the v6e values in a comment). The look-alike failure is real, but making the
  training set harder in that direction does not fix it.

Also measured this session: **dihedral TTA** (`--tta 8` on the evaluator and
`select_epoch_on_val.py`) is +0.0036 on validation (0.8348 -> 0.8384) for 8x
inference. Real, small, not a lever. A 16-epoch v6 synth-only run was started
and stopped: val loss had sat at 0.645-0.70 for five epochs with HF14 flat.

**Synth-only results so far** (1,280 unless noted; val-selected HF14, averaged
window in brackets):

| pool | seed 7 | seed 31 |
|---|---:|---:|
| v5 1,600 | 0.557 | 0.589 |
| v6b 1,600 | 0.504 | -- |
| v6b 4,000 (volume) | 0.520 [0.502] | -- |
| v6c 1,600 (townhouses) | 0.545 [0.542] | -- |
| v6b 1,600 **at 2048** | 0.551 [0.568] | -- |
| v5 + v6b union 3,200 | 0.618 [0.616] | 0.608 [0.607] |
| v5 + v6c union 3,200 | 0.611 [0.623] | -- |
| v5 + v6d union 3,200 | 0.660 [0.663] | 0.559 [0.573] |
| v5 + v6d union, 16 epochs | 0.658 [0.646] | -- |
| **v5 + v6d union at 2048** | **0.680 [0.690]** | **0.692 [0.693]** |
| v5 + v6e union 3,200 | 0.583 [0.583] | -- |

Volume of the new pool alone is null; resolution is worth +0.05 to the new pool
alone; the union with v5 is worth +0.04 over v5 at two seeds; the v6d round
(unlabelled blank flats, eave shadows, distractors, room tints -- the
over-selection fixes from the audit) looked like +0.04 at seed 7 and is **null
at two seeds**: 0.610 mean against 0.613 for v5+v6b. Synth-only seed spread is
~0.05 here, larger than any single generator round, so nothing below that
should be read from one seed again. A 16-epoch schedule is null for the union
too (0.658 vs 0.660), as it was for the mix. Resolution is the one lever that
has moved synth-only every time it was pulled: v6b alone +0.05, the v5+v6d
union +0.02 to +0.03 (0.690 averaged at 2048 is the best synth-only HF14 on
this box). **Synth-only headline: 0.686 ± 0.009 val-selected, 0.692
averaged, 2 seeds** -- v5+v6d union 3,200 at 2048 -- against 0.573 for the v5
pools. Recipe in `startup.md`.

**The mix, for reference** (`startup.md` has the details): v6b added to the
documented mix at 2048, seed 7, is 0.7754 val-selected / 0.7770 averaged, the
best single-seed mix number so far; one seed by decision.

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

## Archived material (pre-2026-08-06 sampler / query-model lineage)

Moved to `synth_progress_archive.md` on 2026-09-17 — not relabeled in place
this time, actually moved out, because last time's "condensation" was just this
same marker with the material left sitting below it, and it grew another 280
lines past that point anyway. That file covers the broken-sampler numbers, the
retired Mask2Former query-model architecture and curriculum, and their
superseded negative-evidence list. Not comparable to anything below or in
`startup.md`; kept for provenance only.

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

Image 14 alone is 9 of the 52 HF14 selections (17% of the metric) and averages
0.291 (2026-08-08); fixing it alone would be worth +0.105. Its failures are
wrong-region, not fuzzy-boundary, and not explained by resolution or aspect —
consistent with the intrinsic-confusability finding above.

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

(Older negative-evidence list and the retired-architecture operational rules:
`synth_progress_archive.md`.)

**Split screen (auto-recorded, `run_split_screen.sh`, seed 7):**
```
v6d: val 0.6465  HF14 0.6401
hsonly: val 0.6253  HF14 0.6600
subtleonly: val 0.6271  HF14 0.6622
   paired mean difference +0.0199   se 0.0229   t(51) = +0.87   p = 0.3885
   hsonly better on 25, worse on 17, tied (|d|<0.01) on 10   sign-test p = 0.28
   paired mean difference +0.0221   se 0.0245   t(51) = +0.90   p = 0.3701
   subtleonly better on 22, worse on 27, tied (|d|<0.01) on 3   sign-test p = 0.5682
```

**Split screen (auto-recorded, `run_split_screen.sh`, seed 31):**
```
v6d: val 0.6167  HF14 0.6940
hsonly: val 0.6610  HF14 0.6786
subtleonly: val 0.7050  HF14 0.6981
   paired mean difference -0.0154   se 0.0246   t(51) = -0.63   p = 0.5336
   hsonly better on 19, worse on 20, tied (|d|<0.01) on 13   sign-test p = 1
   paired mean difference +0.0040   se 0.0153   t(51) = +0.26   p = 0.7929
   subtleonly better on 18, worse on 22, tied (|d|<0.01) on 12   sign-test p = 0.6358
```

**Split screen (auto-recorded, `run_split_screen.sh`, seed 99):**
```
v6d: val 0.6330  HF14 0.6845
hsonly: val 0.5974  HF14 0.6690
subtleonly: val 0.6566  HF14 0.6669
   paired mean difference -0.0155   se 0.0220   t(51) = -0.70   p = 0.485
   hsonly better on 19, worse on 21, tied (|d|<0.01) on 12   sign-test p = 0.8746
   paired mean difference -0.0176   se 0.0158   t(51) = -1.11   p = 0.2725
   subtleonly better on 21, worse on 21, tied (|d|<0.01) on 10   sign-test p = 1
```

**Split screen, three seeds (7 / 31 / 99), synth-only swin_t @1024, epoch 8 @2048
inference, deltas vs the same-seed v6d arm:**

| arm | HF14 delta | mean | validation delta | mean |
|---|---|---:|---|---:|
| `--hardscape-plan 1.0` | +0.020 / -0.015 / -0.015 | **-0.003** | -0.021 / +0.044 / -0.036 | -0.004 |
| `--same-fill-subtle 0.5` | +0.022 / +0.004 / -0.018 | **+0.003** | -0.020 / +0.088 / +0.024 | +0.031 |

Both null on HF14. The v6d control alone spans 0.640-0.694 on HF14 across
seeds, so synth-only single-seed deltas under ~0.05 mean nothing here. Neither
knob is adopted; the subtle band's validation mean (+0.031) is the only
residue, and it rests on one seed's +0.088.

`--vocab2 P` (new, default 0 = v6d byte-identical): 3D shadow bands on lap
siding and basketweave for grid fills, from the fill-vocabulary audit. Pool
recipe: `generate_synthetic_v6.py --n 2000 --seed 6 --start 0 --vocab2 0.7`,
first 1,600. **Untested** -- `run_vocab2_screen.sh` (three seeds against the
existing v6d arms) was stopped at launch when the session ended.

## 2026-09-26 — Synth-vs-Gemini probe; `--gemini-colour`

`scripts/synth_vs_gemini_probe.py`: frozen DINOv2 + logistic regression, crops
inside labelled regions, framing/scale/JPEG q75 equalised, grouped 5-fold.
v6d vs Gemini r2-r4: crop AUC **0.999**. Its most obvious synth crops were flat
saturated colour (purple plans, teal siding). Real eval: 11/28 colour sheets
(3 only markup), median sat 55; Gemini 15%, 54; v6d 53%, 90.

`--gemini-colour` (default off, v6d byte-identical): colour on 40% of
elevations / 25% of roof plans / no floor plans (own RNG), muted palettes,
dark ink lines over colour. Pool: 24% colour sheets. Probe AUC **0.998** -- the
saturated fields are gone from the top crops; what now gives synth away is
flat, near-textureless fills (charcoal roofs with faint lines, plain brown
planes), sparse random stipple on white, and solid black poché. Untested on
HF14.
