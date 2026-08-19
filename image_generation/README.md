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

Two things to tune before the full round. The smoke's median ink is 0.025
against the real pool's 0.111: at 61% faint the whole pool sits at the sparse
extreme rather than spanning it, so consider dropping to ~45%. And 28 of 100
specs are colourised against 16 of 28 in the eval set — a deliberate skew,
since the worst images are monochrome, but it is a skew.

---

This directory is the portable source of truth for the 100 unlabelled images
that will be generated by an image model and manually labelled in Roboflow.
The images themselves must be produced by image generation, not procedural
drawing code. `render_image_generation_prompts.py` only materializes exact text
prompts from the checked-in specifications; it does not render pixels.

Generation is stochastic. The prompt set, filenames, distribution, and quality
gates are exact and reproducible, but a second run will not produce byte-identical
pixels.

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
