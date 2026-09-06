# Realistic Label Pool v1: 100-image generation scheme

## Status as of 2026-08-06 — read before regenerating

This scheme was executed once, partially. **Do not re-run it unchanged**; the
framing below produced images that did not match the product.

What happened and where the output lives:

- The first ~50 prompts produced **full construction sheets with legends and
  title blocks**, and generation was stopped there. That framing is wrong: the
  real user gesture is to see a pattern, select a small rectangle inside it, and
  ask for every matching region **within the drawing** — the house section of a
  floor plan, not the legend, schedule, or sheet border. Prompts that maximise
  whole-sheet realism optimise for the wrong thing.
- The next ~50 were regenerated toward focused areas and over-corrected into
  **fractions of houses and buildings**. Workable, but not the target either.
- 98 images were uploaded to project `perceive-ai/floz-generated-realistic-label-pool`.
  **28 were salvageable and are hand-labelled** (version 1). The other **70 were
  reviewed and rejected as too poor to label** — they are not a labelling backlog
  and should not be treated as cheap extra data. The usable yield of this
  generation round was 28 out of 98, roughly 29%.
- A separate earlier batch, `realistic-label-pool-v1-100` in
  `perceive-ai/floz-real-pool`, holds 25 images that are almost all unlabelled.
  It is superseded; prefer the dedicated project.

**What the 28 bought.** Added to the 3,148-record mix they lifted HF14 from
0.6001 to 0.6331 (+0.033, t=2.94, p=0.043, 3 seeds) for a 1.6% increase in
records. Measured properties of that pool, for calibration:

| | real 86 (scraped) | generated 28 |
|---|---|---|
| long side | 640, native | 1536–2065 |
| instances / image | 2.0 | 3.0 |
| labelled-area fraction | 0.153 | 0.284 |

Count-matched at 28 unique sources each, generated scored 0.2608 and real 0.2777
— indistinguishable inside the ~0.05 run noise. **One generated plan is worth
about as much as one real plan**, so volume is the lever. Resolution is worth
~0.032 (the same 28 downscaled to 640 lose that much), and it is a free choice at
generation time that can never be retrofitted onto the 640px scraped pool.

**Aim the next round at:**

1. the drawing region itself — a complete elevation or the housed portion of a
   plan — with legends, schedules, and title blocks absent or marginal;
2. a whole coherent building or elevation, not a cropped fragment of one;
3. large output: 1,500 px or more on the long side, since resolution measurably
   pays and training runs at 1280;
4. crisp CAD linework. The 28 that worked skew tonal and rendered
   (grayscale-shaded materials) rather than clean vector-like hatching, which is
   probably further from scanned real plans than their content suggests;
5. everything the per-image quality gates below already require — repeated
   families in disconnected regions, confusers, and interruptions suitable for
   `remove` holes. Those gates were not the problem and still hold.

The 100 prompts in `prompts.jsonl` remain a valid distribution of *sheet types*;
what needs rewriting is the framing and crop, not the subject list.

## Update 2026-08-12 — v2 prompts exist, and a third defect was found

`render_image_generation_prompts.py --version v2` renders `prompts_v2.jsonl` from
the same unchanged `specs.jsonl`, with the framing fixed per the five aims above.
v1 is untouched and its recorded SHA still validates.
`scripts/generate_images_openai.py` drives generation end to end:

```bash
set -a; . ~/.env; set +a          # OPENAI_API_KEY
pip install --user openai
python3 scripts/generate_images_openai.py --ids 1,11,21,41 \
  --out data/image_generation/realistic_label_pool_v2 \
  --report data/image_generation/smoke_v2_contact.png
# --dry-run first; --limit N to cap; re-running skips accepted IDs
```

Defaults: `gpt-image-1`, landscape `1536x1024` (the model's largest long side,
which clears the >=1500px aim with nothing to spare), quality `high`, roughly
$0.19 an image. **Scale guidance: ~300 images is the round worth doing** — at the
historical 29% yield that is ~86 usable sources, the same jump that bought
+0.073, for ~$50-60 and several hours of hand-labelling. The labelling, not the
API, is the binding cost.

Outputs and `generation_receipt.jsonl` land under git-ignored `data/`, so a
clone has neither: resumability is per-VM, and the four smoke-test images from
2026-08-12 (ids 1, 11, 21, 41) exist only on the VM that made them. Nothing has
been uploaded to Roboflow.

A 4-image smoke test (ids 1, 11, 21, 41) confirmed the framing fix — no title
blocks, no legends, no sheet borders, no cropped fragments. It also exposed a
defect neither the framing nor the subject list accounts for: **the image model
substitutes realistic materials for drafting hatch, and ignores the confuser
requirement.** Spec 1 asks for "close 45-degree hatch, wider 45-degree hatch,
fine masonry stipple" with "similar diagonal spacings"; what came back was brick
coursing, scalloped shingles and board siding, with no 45° hatch and no
confusable pair.

That is measurable, and it already happened once: the delivered 28 are the
**least** confusable pool in the project (4.8% of multi-family images hold a pair
at ≥0.90, against 24.4% for the scraped real plans —
`scripts/family_similarity_probe.py --local-data`). Gate 5 was specified, not
met, and nothing could tell until there was a probe.

Before scaling, fix the prompt and re-smoke:

1. name the hatch as *drafting notation* — parallel ruled lines at a stated angle
   and spacing, explicitly not a depiction of the material's real appearance, and
   explicitly no brick coursing, shingle scallops or photoreal texture;
2. state the confuser tolerance — two families differing only slightly, e.g. 45°
   hatch at 3mm against 4mm spacing;
3. then re-measure the labelled pool with the probe rather than trusting the
   prompt.

Note the tension to respect: deliberately confusable *synthetic* data cost −0.104
(`synth_progress.md`, 2026-08-12). The aim here is to match the real
distribution, which the current generated pool undershoots — not to maximise
confusability.

## Update 2026-08-19 — v3 prompts, and gpt-image-2

Two changes, both to things the 2026-08-12 smoke test found wanting.

**`--version v3` renders `prompts_v3.jsonl`** from the same unchanged
`specs.jsonl`. It is v2 plus the two blocks that smoke test asked for, and
nothing else: a *Hatching* block that defines every fill as drafting notation —
ruled parallel lines, crosshatch or dots at one constant angle and spacing,
identical wherever the family recurs, explicitly not brick coursing, shingle
scallops, board siding, grain, or tonal texture — and a *Confusable families*
block that states the tolerance as a number (45 degrees at 3 mm against 4 mm, or
45 against 40 degrees at the same spacing) because "similar" did not survive the
model's reading of it. v1 and v2 are untouched and v1's recorded SHA still
validates.

Do not raise the confuser dose further. Deliberately confusable *synthetic* data
cost -0.104 (`synth_progress.md`, 2026-08-12). The aim is the real distribution
— 24.4% of multi-family images holding a pair at >=0.90 — which the delivered 28
undershoot at 4.8%, not the maximum.

**`generate_images_openai.py` now defaults to `gpt-image-2`** (2026-04-21),
which takes an arbitrary `WIDTHxHEIGHT` instead of gpt-image-1's three fixed
sizes: both axes divisible by 16, aspect ratio within 1:3..3:1, up to 3840x2160.
That turns resolution from a constraint into a choice. The default is now
`2496x1664` (3:2) against gpt-image-1's 1536x1024, which had cleared the
>=1500px aim with nothing to spare; `--size` overrides it and is validated
locally before a request is spent, and `--orientation legacy-landscape`
reproduces the old geometry. Billing is per token — $30/1M image output tokens
against gpt-image-1's $40 — so the receipt now records the `usage` block per
image and the run prints the total.

**Review a generated batch in Roboflow.** On a headless box the contact sheet is
not enough to apply the visual gates. `upload_unlabelled_roboflow.py` now takes
`--project` with `--create`, uploads whatever images are in the directory
(`--expect 100` restores the strict v1-round manifest check), and can remove a
disposable review project again with `--delete-project --execute` — which moves
it to Roboflow's trash, recoverable there for 30 days.

```bash
python3 scripts/generate_images_openai.py --ids 1,11,21,41 \
  --out data/image_generation/realistic_label_pool_v3
python3 scripts/upload_unlabelled_roboflow.py \
  --images data/image_generation/realistic_label_pool_v3 \
  --project floz-gen-v3-smoke --batch v3-smoke --create \
  --tags generated-realistic-v3,needs-labels --execute
```

### The v3 smoke test (same four IDs, so the prompt is the only variable)

4 of 4 accepted at 2496x1664, about three minutes wall-clock at `--workers 4`
and **$0.287 an image** (9,452 image output tokens each) — so the round guidance
below is now ~$86 for 300, not $50-60. That buys 2.4x the pixels of the v2
geometry; drop to `--size 1536x1024` if the ratio matters more than the pixels.

**The drafting-hatch defect is fixed.** Inspected at 100%, id 1 draws its walls
as ruled 45-degree hatch and fine stipple, id 11 as diagonal hatch against two
tile grids, id 21 as diagonal ruled fills against a dot membrane, id 41 as
crosshatch, concrete dots and batt insulation. No brick coursing, no shingle
scallops, no board siding, no tonal shading anywhere in the four. The framing
gains from v2 held: no title block, legend or sheet border, complete views, and
tags and dimension strings interrupting fills for `remove` holes.

**Confusability is still unverified, and cannot be verified here.**
`family_similarity_probe.py --local-data` reads per-family masks, so it needs
annotations; the prompt now states the tolerance but only the probe run on the
labelled pool proves it landed. That is the same mistake as last round if it is
skipped — gate 5 was specified and unmet for a year before anyone measured it.
Run the probe on the first labelled slice, not on the whole round.

Review copies of these four are in `perceive-ai/floz-gen-v3-smoke` (created
2026-08-19, unlabelled). It is a disposable review project; delete it with
`--delete-project --execute`.

## Update 2026-08-19 (later) — v4, aimed at the evaluation set

Everything above aims at an idea of what a construction drawing looks like.
`scripts/pool_style_stats.py` and a look at all 28 evaluation images say the
idea was wrong, and had been since v1. Measured, every image capped to 1280
first (`data/probes/pool_style_2026-08-19.json`):

| pool | aspect | over 3:1 | colourised | ink | fill regularity |
|---|---:|---:|---:|---:|---:|
| real 28 (the eval set) | 2.59 | 10/28 | 16/28 | 0.111 | **11,826** |
| scraped 86 (training) | 1.00 | 0 | 9/88 | 0.139 | 8,552 |
| generated 28 (v1 round) | 1.50 | 6/28 | 0/28 | 0.095 | 2,751 |
| v3 smoke | 1.50 | 0/4 | 0/4 | 0.063 | 1,348 |
| **v4 smoke** | **2.20** | **2/6** | **2/6** | 0.025 | **2,796** |

Three things follow, and two of them contradict the sections above.

**The eval set is mostly colourised elevations, not CAD hatch sheets.** Roughly
two thirds of the 28 are elevations; 16 of 28 carry colour; their materials are
lap siding, board-and-batten, shingle courses and flat painted fields as often as
drafting hatch; several are markup-tool screenshots with highlighter fills,
magenta dot markers and tool UI chrome inside the frame. Wall details, site
plans, RCPs and structural sheets — 40 of the 100 v1 specs — do not appear at
all. So v3's "flat line art only: not tonal, not grayscale-shaded, not rendered"
aimed away from 16 of the 28, and the 2026-08-06 note calling the delivered
pool's tonal skew "probably further from scanned real plans" has it backwards.

**The model's worst images share a measurable trait.** The four worst HF14
images are all monochrome (saturation ~0) and sparse: ink 0.026-0.12 against the
pool median 0.23. Image 14 (0.291) is two nearly blank faint elevations at 4.2:1
whose materials are named in callout text and barely differ; image 12 (0.393) is
thin gray wall poche under a dense electrical overlay; image 7 (0.546) is a
mostly empty under-floor plan. `specs_v4.jsonl` is weighted toward those traits —
61 of 100 faint or sparse, 14 plans under an MEP overlay, 24 wider than 3:1 —
on top of the eval set's overall shape. That weighting rests on four data points,
so it is an aiming choice, not a theory; `PROJECT_UNDERSTANDING.md` records two
per-image stories that were confidently wrong.

**Pattern consistency is a generation defect that no prompt has moved.** Real
fills read 11,826 on the dominant-FFT-peak measure; the delivered generated 28
read 2,751 and the v3 smoke 1,348, and v3 was the version that asked in capitals
for one constant angle and spacing. v4 reaches 2,796 — the level of the pool
that already bought +0.033, still ~4x short of real — largely because siding
courses and coursing lines are coarser and more periodic than fine 45-degree
hatch. Treat this as a selection problem, not a wording problem:
`--min-regularity N` re-rolls an image whose fills score below N, and since
labelling is the binding cost, the cheaper move is to generate the whole round
without a gate and then label in descending regularity order.

### Running v4

```bash
python3 scripts/build_specs_v4.py                       # -> specs_v4.jsonl
python3 scripts/render_image_generation_prompts.py --version v4
python3 scripts/generate_images_openai.py --ids 3,8,12,2,1,4 \
  --out data/image_generation/realistic_label_pool_v4
python3 scripts/upload_unlabelled_roboflow.py \
  --images data/image_generation/realistic_label_pool_v4 \
  --project floz-gen-v4-smoke --batch v4-smoke --create \
  --tags generated-realistic-v4,needs-labels --execute
```

Each v4 manifest row carries its own aspect, and the generator sizes the request
from it under a fixed pixel budget, so an image costs about $0.20 rather than
the $0.287 of a fixed 2496x1664. Anything past 3:1 — the API's limit — is asked
for as a wide band with white margin above and below, generated at 3:1, and
trimmed to width by dropping the emptiest rows. A 300-image round is ~$60.

The six-image smoke covered `elev_colour_markup`, `elev_colour`, `elev_faint`,
`plan_mep`, `roof_plan` and `section_sparse`. It reproduces the eval set's look
far more closely than v3 did: id 8 is a colourised elevation with magenta dot
markup and material callout leaders, like eval images 21/22; id 3 is four faint
elevations whose materials are legible mainly from their callouts, like eval
image 14; id 2 is a sparse under-floor framing plan with note blocks at 4:1,
like eval image 7. Copies are in `perceive-ai/floz-gen-v4-smoke` (disposable —
`--delete-project --execute`).

### Tuned after review of the smoke

Two defects in the first v4 draft, both found by looking at the images:

**Faintness was a flag, and it went too far.** 61 of 100 specs were flat-out
faint, which put the pool's median ink at 0.025 against the real pool's 0.111,
and the roof of the smoke's faint elevation came back pale enough to read as
empty page. Contrast is now three levels — 47 normal, 24 light, 29 faint — and
the faint wording carries a floor: "every material fill must still be plainly
visible on the paper, subtle, not vanishing, and never fading out to blank white
in the middle of a region". A fill nobody can see is a region nobody can label.

**`plan_mep` produced images with nothing in them.** The category asked for one
family, the wall poche, and the model drew walls as empty double outlines — so
the smoke's floor plan had no patterned region at all and could not have been
annotated. All 14 now demand a SOLID FILLED dark gray poche ("filled in, not a
pair of empty outlines") plus a second family, a floor finish in the wet rooms
and garage, with the reason stated in the prompt: a plan whose walls are empty
outlines has nothing in it to select.

Both fixes were re-smoked (batch `v4-tuned` in `floz-gen-v4-smoke`). The
tuned pool's ink is 0.089 against the first draft's 0.025 and the real pool's
0.111, and contrast lands at 0.435 against the real 0.437. The regenerated
`plan_mep` plan now carries solid filled wall poche broken into disconnected
runs by every opening, plus a tile grid in the wet rooms, under dashed duct runs
and grille tags — the same shape as eval image 12. The faint elevation keeps
visible brick coursing across all four views instead of a roof that reads as
blank page.

**The consistency requirement is now its own block.** In v3 and the first v4 it
was a clause inside a longer paragraph, describing the ideal. It now leads the
prompt after the material list, names the failures rather than the ideal —
wavering or hand-drawn lines, lines curving to follow a wall or bend around an
opening, spacing opening up across a region, a fill changing weight or scale
between two regions — and asks for the mechanism: draw each family as one
continuous ruled field across the whole drawing and let the geometry mask it, so
a fill interrupted by a window resumes on the same grid.

Paired against the previous wording on the same three specs, regenerated:
2,489 -> 3,772, 1,553 -> 2,035 and 5,363 -> 5,522 on fill regularity. All three
moved the same way, but three pairs with no repeats is weak evidence and this
repo has retracted better-supported wins; treat it as free and directionally
right, not as established. A proper paired test is ~$2 in batch mode.

Still a deliberate skew, left as is: 28 of 100 specs are colourised against 16
of 28 in the eval set, because the images the model fails on are the monochrome
ones.

### Batch mode for the full round

The Batch API halves the token price -- $15 per 1M image output tokens against
$30 -- so a 300-image round is ~$30 rather than ~$60, in exchange for a 24-hour
completion window. `/v1/images/generations` is supported; the limits that matter
here are 50,000 requests and a 200MB input file, neither close.

```bash
python3 scripts/generate_images_openai.py --out data/image_generation/round1 \
  --batch submit                      # writes batch_state.json with the id
python3 scripts/generate_images_openai.py --out data/image_generation/round1 \
  --batch status
python3 scripts/generate_images_openai.py --out data/image_generation/round1 \
  --batch fetch                       # writes images, crops, and receipts
```

The batch outlives the shell that submitted it, which is what makes this usable
from a session that will not still be open in 24 hours. Two cautions: the output
file carries every image as base64, so budget ~5.5MB per 4-megapixel PNG (a few
GB for a full round), and output line order does not match input order — the
fetch step maps by `custom_id`, never by position.

## Round 1, 2026-08-19 — 50 images through Batch

Stratified across all eight v4 categories (11 `elev_faint`, 8
`elev_colour_markup`, 7 `plan_mep`, 7 `section_sparse`, 6 `elev_colour`, 4
`roof_plan`, 4 `elev_mono`, 3 `plan_finish`). **50 images, $4.83, $0.097 each,
zero failures**, and the batch came back in about half an hour rather than
anywhere near its 24-hour window. Eleven were generated at the API's 3:1 limit
and trimmed to their wider target aspect.

| | real 28 | round 1 |
|---|---:|---:|
| aspect | 2.59 | 2.50 |
| over 3:1 | 10/28 | 11/50 |
| colourised | 16/28 | 13/50 |
| ink | 0.111 | 0.056 |
| contrast | 0.437 | 0.360 |
| fill regularity | 11,826 | 3,122 |

Aspect now matches. The pool is still lighter and sparser than the real set, and
fill regularity is where it has always been -- above the delivered generated 28
(2,751), a quarter of the real plans. Images live in
`perceive-ai/floz-gen-v4-round1`, unlabelled.

**One operational trap, fixed in `upload_unlabelled_roboflow.py`.** The upload
was interrupted and resumed; on the resume `workspace.project(name)` raised
`does not exist or cannot be loaded` for a project that plainly existed -- raw
HTTP returns it fine -- so the `--create` path made it a second time, and
Roboflow answered the colliding name by silently minting
`floz-gen-v4-round1-pcfow`. The round ended up split 29/22 with one image in
both. The project handle now comes from the raw project record instead of that
SDK call. Also note the upload receipt records the project **name that was
asked for**, not the one Roboflow used, so it could not detect this: the
reconciliation had to read both projects back.

## 2026-08-19, later still — family counts, and a Gemini comparison

**The round was more complicated than the thing it imitates.** The eval set
holds **1.8 material families per image, and 13 of 28 have exactly one**; the
first v4 list asked for three on 72% of specs. The real difficulty there is
faintness, clutter and materials that differ barely — not stacked families.
`build_specs_v4.py` now draws the count from that distribution (45/35/20 for
one/two/three), landing at mean 1.9 with 36 single-family specs. It also cuts
labelling by roughly a third, which is the binding cost.

**The 28 evaluation images are now in Roboflow** as
`perceive-ai/floz-eval28-reference`: 28 images, 207 polygons, families as
`pattern1..N` with holes as separate `remove` polygons, built by
`scripts/upload_eval28_roboflow.py`. Every image is tagged `eval-only` and
`do-not-train`, and the 14 acceptance images additionally `hf14-acceptance`.
**It must never be merged into a training pool** — `startup.md`'s rule that
these are evaluation-only has not changed; this exists so a generated round can
be compared against the target by eye, in the same tool.

**Gemini holds a ruled fill better on 3 of 4 paired prompts.**
`scripts/generate_images_gemini.py` runs the identical manifest through
`gemini-3-pro-image` at 2K. Fill regularity, same prompt, same measurement:

| spec | gpt-image-2 | gemini-3-pro-image |
|---|---:|---:|
| 006 `plan_mep` | 2,081 | **6,168** |
| 011 `roof_plan` | 5,033 | **10,246** |
| 019 `elev_colour_markup` | 2,487 | **3,122** |
| 042 `elev_faint` | **3,550** | 1,992 |

For scale, the real plans read 11,826 and round 1 as a whole 3,122. The roof
plan is the striking one: shingle courses, a dot field and standing-seam lines,
all machine-regular, at 10,246 — the first generated image in this project to
come near the real pool. The elevation it lost on adds a faint paper texture
that the measure reads as noise.

### Ten images through Gemini, on the new family distribution

`perceive-ai/floz-gen-gemini-r1`, `gemini-3-pro-image` at 2K, one or two per
category, unlabelled. Against the eval set and the gpt-image-2 round:

| | real 28 | gpt round 1 (50) | gemini (10) |
|---|---:|---:|---:|
| aspect | 2.59 | 2.50 | 2.50 |
| ink | 0.111 | 0.056 | **0.107** |
| contrast | 0.437 | 0.360 | 0.385 |
| fill regularity | 11,826 | 3,122 | 2,605 |

**Ink density lands on the real set**, where the gpt pool drew at half its
weight — that gap had survived every prompt change. Regularity reads below the
gpt-50, but these are different specs under the new family distribution, so it
is not the paired comparison; the four-pair test above is, and it went the other
way three times out of four. Aspect handling works: 21:9 generations trimmed to
4.6:1, 3.8:1 and 3.4:1 as the manifest asked.

Caveats before treating this as settled: four pairs, one draw each, and this
repo has retracted better-supported results. Gemini also constrains aspect to a
fixed list whose widest is 21:9 (2.33:1, narrower than the eval median of 2.59),
so wide images still need trimming, and there is no batch discount — pro at 2K
is ~$0.13 an image against the $0.097 measured on gpt-image-2 batch.

## Simplicity, taken from the eval set (2026-08-19)

Family count was only one axis. The real 28 are simpler than the round in three
ways, and the spec list now matches all three:

| | real 28 | v4 before | v4 now |
|---|---|---|---|
| families / image | 1.8, 13 of 28 single | 2.7, none single | 1.9, 33 single |
| views / sheet | mostly ONE (25, 26, 27 among them) | every elevation >=2, 13 of 100 at four | 45% single, 48% two, 7% four |
| drafting clutter | a few leaders and level marks | full apparatus on every spec | sparse on 60% of elevations |

Two mechanics were needed to make it stick. The prompt now carries a
**restraint clause** — "draw ONLY what is listed above: exactly the views named
and no others, exactly the material families named and no others... Empty white
paper around and between the views is correct and expected" — because the model
embellishes: asked for two elevations it draws four. And `VIEWS_ONE` is kept
deliberately wide, since the spec builder rejects duplicate
subject+layout+patterns triples and a short single-view list loses draws to
collisions, quietly pulling the round back toward multi-view sheets.

**Simplicity is also what finally moved fill regularity.** Four single-view
specs through `gemini-3-pro-image`:

| pool | fill regularity |
|---|---:|
| real 28 | 11,826 |
| gpt-image-2 round 1 (50) | 3,122 |
| gemini, first 10 | 2,605 |
| **gemini, four simplified** | **7,727** |

That is three times the previous best generated figure and two thirds of the
real pool, from prompts that differ only in asking for less.

Two defects in that first simplified batch, both in this repo rather than in the
model, and both fixed:

* **The crop was cutting the drawings.** `trim_to_aspect` took the densest band
  of the requested height, which is fine when the generation is already close to
  the target -- but Gemini caps at 21:9, so reaching a 4.6:1 spec meant slicing
  the bottom off the building. The ink now decides: rows carrying drawing are
  never removed, and if the target would cut them the crop stops at the widest
  aspect the margins allow and the receipt records what was achieved (those four
  landed at 3.0:1 and 2.5:1 rather than their nominal targets). The "wide band
  with white margin above and below" instruction also now applies from 2.3:1,
  since that is where Gemini's cap bites, not 3:1.
* **The green was modelled wrong, twice.** In the real set the colour is not a
  highlighter swipe over part of a drawing: it is a SOLID FILL OF WHOLE PATTERN
  REGIONS, with the material's own lines drawn on top at full strength and the
  windows and doors inside left unfilled. Eval 27 is 21.9% green with ink
  density 0.99 inside it; eval 24 is 31.2%. Asking for a partial highlighter
  produced smudges, and asking for "hard edges" produced an outline glow around
  the regions. Describing it as a paint-bucket fill of one family — "every
  region of that one family is SOLID GREEN THROUGHOUT its interior... not an
  outline, not a glow or halo around the edges" — lands at 20.6% coverage
  against the real 21.9%. Note also that a flat coloured plane in the image is
  not a defect: it is simply a surface nobody annotates. Only the families named
  in the spec have to carry a pattern.
* **Instruction language leaked onto the sheet.** Telling the model that
  unnamed surfaces "will not be annotated" produced a drawing labelled "METAL
  ROOF (NO PATTERN)". The prompt now forbids writing the brief into the drawing:
  callouts name materials, never `pattern`, `family` or `markup`.
* **The highlighter markup came back as green smudges.** The markup wording now
  demands a flat, even, hard-edged translucent fill that stops at the region
  boundary, "like a digital highlighter rectangle: no soft brush strokes, no
  airbrush, no smudges, no smears, no feathered or faded edges, no gradients".
  Regenerated, it reads like the markup in eval images 24-27: a clean bounded
  rectangle with the linework fully visible through it. The mechanism is
plausible -- one view and one or two families means large uninterrupted regions,
and less for the model to keep consistent -- but it is four images of one
category, so treat it as the most promising lead here rather than a settled
result. They are in `perceive-ai/floz-gen-gemini-r1`, batch `gemini-simple`.

## Round 2 — 50 images through Gemini Batch (2026-08-19)

`perceive-ai/floz-gen-gemini-r2`, unlabelled. `gemini-3-pro-image` at 2K,
submitted with `--batch submit` and fetched with `--batch fetch`: Gemini's Batch
API is the same bargain as OpenAI's — half the token price, a 24-hour target
window, and a job that outlives the shell. Requests go up as a JSONL file, not
inline, because inline batches cap at 20MB and fifty 2K images come back far
past that. 50 of 50 returned, none failed, ~$0.067 an image.

Stratified across all eight categories, on the simplified spec list: families
mean 1.9, 45% of elevations single-view, markup cut to 6 of 100 specs.

| | real 28 | gpt-image-2 round 1 | **gemini round 2** |
|---|---:|---:|---:|
| ink | 0.111 | 0.056 | **0.104** |
| contrast | 0.437 | 0.360 | **0.411** |
| fill regularity | 11,826 | 3,122 | **7,270** |
| aspect | 2.59 | 2.50 | 2.36 |
| over 3:1 | 10/28 | 11/50 | 5/50 |

**The simplicity result held at scale.** 7,270 across 50 images and every
category, against 7,727 on the four-image probe and 3,122 for the gpt round —
so the jump was the simpler drawings, not a lucky draw. Ink and contrast now sit
on the real pool as well; the pool that had been half the weight of real plans
for three rounds is no longer distinguishable on either.

The one measure that moved the wrong way is aspect: 5 of 50 past 3:1 against the
real 10 of 28. That is the ink-aware crop refusing to cut drawings — Gemini caps
at 21:9 and the specs asking for 4:1 and wider simply cannot be reached without
slicing the building, so they stop where the margins run out. Reaching the real
set's wide tail needs the model to draw a wide band inside a 21:9 frame, not a
harder crop.

## The prompt was asking for photographs of paper (2026-08-19)

Round 2 returned drawings that read as a phone photo of a printed sheet — page
shadow in a corner, a visible sheet edge, uneven lighting. Two lines caused it:

* the artifact vocabulary asked **69 of 100 specs** for degradation ("light scan
  skew", "pale photocopy-like export", "JPEG ringing"), with only 10 asking for
  a clean vector export;
* the medium line asked for "the texture of a scanned or exported construction
  document" — and "texture of a scanned document" is an invitation to draw the
  paper rather than the drawing.

The eval set does not support that at any share. **None of the 28 is a scanned
or photographed sheet** — they are digital PDF excerpts. A page-lighting
measurement (real 28 median 0.016 against round 2's 0.035) appeared to flag five
real images as scan-like, but those five were 3, 6, 17, 18 and 19: the
colourised elevations. The metric was reading colour fills, not paper. Do not
repeat that inference — low-frequency brightness variation tracks tinted regions
as readily as page lighting.

Now: the artifact list is clean digital exports only, with no scan or photocopy
option at all, and the medium line says what the page is not — "This is the
digital page itself... It is NOT a photograph of a printed sheet: no page edges
or corners, no drop shadows, no curl or creases, no uneven or angled lighting,
no desk, table or background visible, and no paper grain." Regenerating the
worst offender took its lighting swing from 0.047 to 0.024.

## Round 3 — 50 clean images (2026-08-19)

`perceive-ai/floz-gen-gemini-r3`, unlabelled. Same recipe as round 2 with the
scan wording removed and the simplified spec list. Round 2 is kept as it is.

| | real 28 | round 2 | round 3 |
|---|---:|---:|---:|
| fill regularity | 11,826 | 7,270 | **8,302** |
| ink | 0.111 | 0.104 | 0.104 |
| contrast | 0.437 | 0.411 | 0.390 |
| aspect | 2.59 | 2.36 | 2.43 |
| page-lighting swing | 0.016 | 0.035 | **0.024** |

One bug fixed on the way: `choose_ratio` took the widest ratio *not exceeding*
the target, which sent a 1.3 target to 1:1 because 4:3 sits just above it — four
plans came back square and the upload gate caught them. It now picks the nearest
ratio in log space and trims only when the pick is narrower. The batch receipt
records the ratio as well, so this is visible next time.

---

This directory is the portable source of truth for the 100 unlabelled images
that will be generated by an image model and manually labelled in Roboflow.
The images themselves must be produced by image generation, not procedural
drawing code. `render_image_generation_prompts.py` only materializes exact text
prompts from the checked-in specifications; it does not render pixels.

Generation is stochastic. The prompt set, filenames, distribution, and quality
gates are exact and reproducible, but a second run will not produce byte-identical
pixels.

## Labelling outcome for rounds 2 and 3 (2026-09-05)

**80 of the 100 images were worth labelling** — 36 of 50 in r2, 44 of 50 in r3 —
against 28 of 98 in the v1 round. That is the round's real verdict; fill
regularity (8,302 against the real 11,826) predicted it correctly.

| | r2 | r3 |
|---|---|---|
| labelled | 36 | 44 |
| instances | 231 | 247 |
| `remove` holes | 100 | 205 |
| families / image | 1.75 | 1.52 |
| labelled-area fraction | 0.314 | 0.278 |
| long side, median | 3168 | 3168 |

Skipped by category across both rounds: section_sparse 4, elev_faint 4,
roof_plan 4, elev_colour 3, elev_mono 3, plan_finish 1, elev_colour_markup 1.
Roof plans are the weakest category by yield — 3 labelled, 4 skipped — and the
only one where the labelling convention itself was in doubt (see the annotation
contract below: group by material, not by fill direction).

Roboflow versions are pinned at v1 for both projects; `startup.md` has the
download, conversion and mix commands.

## Files

- `specs.jsonl`: 100 ordered, unique sheet specifications.
- `prompts.jsonl`: generated canonical full prompts (checked in).
- `scripts/render_image_generation_prompts.py`: validates IDs and renders prompts.
- `scripts/upload_unlabelled_roboflow.py`: validates all 100 files and performs a
  resumable unlabelled upload.

Regenerate and verify the prompt manifest:

```bash
python scripts/render_image_generation_prompts.py
python - <<'PY'
import json
rows = [json.loads(x) for x in open("image_generation/prompts.jsonl") if x.strip()]
assert len(rows) == 100
assert [r["id"] for r in rows] == list(range(1, 101))
assert len({r["filename"] for r in rows}) == 100
assert len({r["prompt"] for r in rows}) == 100
print("validated 100 unique prompts")
PY
```

Canonical SHA-256 fingerprints for this version:

```text
specs.jsonl   603b8ddb1692960278c1f7c69f3d4d1c133167938d3c386eb7a51ca1873b7b9b
prompts.jsonl c89f4a66f8f64d0d3328c5db338df635e55aca60ecaea362f1c67da0ccd8c34b
```

## Dataset distribution

There are ten images in each family:

1. elevations;
2. floor plans;
3. roof plans;
4. building sections;
5. wall/details;
6. site plans;
7. reflected-ceiling plans;
8. interior-finish plans;
9. structural plans;
10. mixed-view construction sheets.

Every prompt requires 3-4 image-local pattern families, intentional visual
confusers, 2-6 disconnected occurrences per family, realistic drafting clutter,
hard negatives, and interruptions suitable for `remove` holes. HF evaluation
images are never used as references or inputs.

## Generation procedure for a future Codex session

1. Read this file and `prompts.jsonl` completely.
2. Create `data/image_generation/realistic_label_pool_v1/`.
3. For each manifest row in ID order, issue one image-generation model call using
   the exact `prompt` value. Do not replace it with SVG, CAD, Pillow, OpenCV, or
   other code-rendered imagery.
4. Copy the selected generated output to the exact `filename` in the dataset
   directory. Never overwrite an accepted image silently.
5. Inspect every image visually at full resolution. If it fails a gate below,
   regenerate the same prompt until it passes; do not repair the drawing with
   procedural graphics.
6. Maintain `generation_receipt.jsonl` beside the images with ID, filename,
   generation tool/model if exposed, generation timestamp, attempt count, and
   final status. Resume by skipping only IDs with both an accepted file and an
   accepted receipt row.
7. Do not upload anything until all 100 local files pass validation.

The default Codex image-generation skill uses one built-in call per asset. It
may take several hours and can be left running. CLI `generate-batch` is allowed
only if the user explicitly chooses that image-model API path and has
`OPENAI_API_KEY`; it still must consume these 100 exact prompts and output these
100 filenames.

## Per-image quality gates

- full construction sheet visible, landscape orientation;
- minimum 1,024 pixels on both axes;
- predominantly white paper with black/gray professional CAD linework;
- no photographic building or decorative blueprint-art styling;
- at least three visibly different hatch/material families;
- at least two intentionally similar/confusable families;
- each intended family visibly repeats in disconnected regions;
- plausible geometry bounds every patterned region;
- text, openings, fixtures, or symbols interrupt at least some regions;
- sufficient clean interiors for small and medium reference selections;
- no colored mask, annotation overlay, watermark, logo, or obvious duplicate;
- not merely a rotated, flipped, recolored, or relabelled prior image.

Dataset-level review must confirm exactly ten accepted images per family and
meaningful diversity in layout, line weight, raster quality, scale, clutter,
pattern spacing, and region shape.

## Roboflow upload

Set `ROBOFLOW_API_KEY` without printing it. First run the dry validation:

```bash
python scripts/upload_unlabelled_roboflow.py \
  --images data/image_generation/realistic_label_pool_v1 \
  --batch realistic-label-pool-v1-100
```

Only after it reports 100 present and zero missing, upload:

```bash
python scripts/upload_unlabelled_roboflow.py \
  --images data/image_generation/realistic_label_pool_v1 \
  --batch realistic-label-pool-v1-100 --execute
```

The upload is unlabelled to workspace `perceive-ai`, project
`floz-real-pool`, split `train`, with tags `generated-realistic-v1` and
`needs-labels`. The local receipt makes retries resumable. Afterward, verify in
the Roboflow UI that the named batch contains exactly 100 images before starting
annotation.

## Annotation contract

- Use image-local `pattern1`, `pattern2`, etc.; names do not carry across sheets.
- Label every disconnected occurrence belonging to the same visible pattern.
- Use separate instance polygons even when occurrences share a pattern name.
- Label exclusions inside a surrounding pattern as `remove`; do not use `remove`
  as a material class.
- Do not label text, dimensions, furniture, fixtures, openings, or blank regions
  as patterns unless they are genuinely part of the material fill.
- **Group by material, not by fill direction.** A roof plan's courses turn with
  each plane's slope, so one material appears at several orientations and stays
  ONE pattern — that is how eval 15 and 16 are labelled, one family spanning
  every orientation with the flat roof excluded as a different material. In an
  elevation a change of direction almost always means a different material
  (board-and-batten against lap siding), which is why "different orientation,
  different pattern" holds there. The material decides, not the angle.
