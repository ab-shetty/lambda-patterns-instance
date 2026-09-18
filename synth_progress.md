# Synthetic Dataset Progress

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
