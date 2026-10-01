# Floor plans v3: blind real-vs-synthetic judging (2026-10-01)

`--plan-v2 2` (opt-in; default stays 1). v3 runs on its own RNG stream (`p3`), so levels 0 and 1
are unchanged byte for byte.

## Method

Each round:

1. Generate one fresh floor plan at a **random** id with the current generator
   (`--revit --revit-plans --shaped --mode-weights 0,0,100 --plan-v2 2`). The id is from `SystemRandom`, never hand-picked, and the sheet type is whatever comes out.
2. Show it beside HF14 sheet 0 to one judge and beside HF14 sheet 12 to another. Both images are resized to 2000 px and saved as JPEG q90, in random A/B order.
3. The judges are two fresh subagents per round. They see only the two images and must name the synthetic one, give a confidence and list the visual tells.
4. Each round's fixes go into the generator, not into that one plan. The next round draws a new random plan.

From round 11 on, the judges were told to disregard text, leaders, dimensions, grids, title blocks, electrical/MEP symbols and layout logic. The user said these don't matter for pattern matching. Before that, many tells were about text and electrical symbols.

Script: the scratchpad `turing_round.py` (not kept). A round is about 20 lines: generate, resize and shuffle, then print the key.

## Rounds

| round | synthetic sheet type | vs sheet 0 | vs sheet 12 | main graphics tells -> change |
|---|---|---|---|---|
| 1 | rendered | 95% | 95% | rooms empty (furniture skipped in non-rectangular rooms and rooms with stairs); plain-rectangle car; MEP ellipses on an ordinary plan; framing cropped the dimensions away |
| 2 | under-floor | 88% | 88% | crawlspace and slab drawn with the same dot lattice; pier grid perfectly regular; no wall poché; no slab edge |
| 3 | finish | 95% | 95% | thin double walls the same weight as the furniture; herringbone about 2 ft per plank in heavy pen; bare box and ellipse fixtures |
| 4 | under-floor (partly finished) | 88% | 92% | regular dot stipple; wall fragments; identical piers |
| 5 | rendered | **85%** | 95% | floating walk and deck rectangles; hard uniform shadows inside rooms; furniture overlaps |
| 6 | finish | **85%** | 92% | "2-CAR GARAGE" zoned as a bedroom, so it took the bedroom finish and family (a bug that predates v3); rugs don't mask the floor; identical beds |
| 7 | finish | 88% | 90% | hatch under furniture; basketweave scale |
| 8 | finish | 92% | 95% | high-contrast floor hatch everywhere (real floors are soft: sheets 0, 7, 12); no thresholds |
| 9 | ceiling | 92% | 93% | mostly electrical and leader tells; soffit hatch uniform |
| 10 | ceiling | 92% | 93% | mostly electrical tells; soffit as concentric offsets with no change of direction per face |
| 11* | finish | 92% | 92% | bare evenly spaced plank lines with no stagger or tone; walls one weight |
| 12* | rendered | 90% | 92% | one tile grid lined up across rooms; inset deck border too perfect; chair backs missing; inconsistent shadows |
| 13* | under-floor (partly finished) | 90% | 92% | pier sizes mixed; concrete without aggregate triangles; gaps between the fill and the partitions; partitions with no doors |

| 14* | ceiling | 92% | 92% | soffit drawn as 3–4 sparse lines; every wall uses one rendering |
| 15* | rendered | **88%** | 92% | square-grid paving; wall style switching part way along a wall (from the random room subset) |
| 16* | finish | 90% | 93% | floors stop at doorways and leave white gaps; exterior and interior walls look alike; shadows (judge B) |
| 17* | finish (colour) | **88%** | 93% | windows project past the wall face (the "sill" style); uniform shadows; plank texture |
| 18* | rendered | 90% | 95% | deck lines uniform with no butt joints; inset rim border; shadows (judge B) |

\* graphics-only prompt.

Rounds 13–18 sit at 88–93% vs sheet 0 and 92–95% vs sheet 12. Most of the sheet-12 judge's verdicts rest on drop shadows, which sheet 0 has and sheet 12 doesn't. Every rendered sample therefore fails against sheet 12 and every unshaded one against sheet 0, so a mixed pool can't satisfy both at once. The remaining recurring tells are drawing-craft items: line-weight hierarchy, the detail of windows and doors, wall joins, and "library-quality" furniture. They matter less for pattern matching than the fills, which are now mostly not the top tell.

Best so far: 85% against sheet 0 (rounds 5 and 6, rendered and finish sheets). No judge has called the wrong image yet.

## Calibration: real vs real

The same prompt was run on two pairs of **real** sheets, with the judges still told one was synthetic: sheet 7 vs 0 got 65% and sheet 7 vs 12 got 60%. Both judges called sheet 7 "synthetic" and wrote the same kinds of tells, e.g. "uniform hatch clipped to a polygon" and "flat line-weight hierarchy". So "can't tell" for these judges is about 60–65% confidence, not 50%. The 88–93% plateau is a real gap.

## What v3 changed (all in `scripts/revit_plans.py`)

- **Framing.** 80% of sheets are cropped to the drawing plus its dimension and grid ring, sometimes cutting through the edges. Real floor plans fill about 75% of the frame; ours filled 47%.
- **Furniture in every room.** Beds come with pillow, fold, quilt, throw and nightstand variants. Also sofas with arms and cushions, armchairs, dining sets with chair backs, kitchen L counters, islands, sinks and ranges, vanities with basins, toilets with tank and rim, tubs, showers, desks, laundry, closets, cars and mudroom benches. Layout:
  - Furniture is laid along whichever axis fits, and the best of 5 random layouts is kept.
  - An item is dropped whole, with its detail lines, if it doesn't fit the room minus the stairs or if it overlaps another item.
  - Furniture is solid white and masks the floor. It gets soft shadows only when the walls are shadowed. Rugs mask the floor.
  - Ceiling plans show it as a light-grey ghost.
  - Patio furniture is drawn on the paving and included in its label.
- **Floors.**
  - Herringbone and wood-grain tiles at plank scale (×0.3–0.5); basketweave smaller.
  - Pattern lines mixed 30–65% toward the base colour, so the pen is soft.
  - Bare plank lines become staggered planks.
  - A light tone under all floors on 40% of sheets.
  - A pattern origin per room, with threshold lines across openings.
  - Jittered dot stipple, and AR-CONC triangles in concrete.
- **Walls.** Thin double walls become heavy outline, poché or diagonal hatch. Grey walls get a heavier outline. Shadows are soft and cast mostly outside the building, and also appear on 30% of finish sheets.
- **Hardscape.** Front walks run on to a notched wall instead of floating. Deck boards get staggered butt joints (the rim board was tried and dropped). Square-grid paving becomes irregular stone 65% of the time.
- **Walls and openings, rounds 14–17.** 40% of sheets draw a second wall rendering for a whole wall class (the partitions or the shell); 60% of sheets thicken the exterior shell. Floor finishes run on through doorways. Windows always sit inside the wall thickness (the projecting-sill style is dropped).
- **Ceiling sheets.** Soffit boards change direction per face, mitred at the corners, along or across the band. They get a trim line and are spaced differently from the deck boards.
- **Under-floor sheets.**
  - Pier spans and spacing are uneven; each pier is a footing with a post, round or square, one size per sheet.
  - Girders are drawn as double lines in 45% of sheets, and there are joist span arrows.
  - The slab gets an edge line and never shares the crawlspace's pattern kind.
  - Foundation walls get poché.
  - 45% of sheets have part of the level as finished rooms (as on sheet 7), with walls and doors and the fill running to the wall centre lines.
- **Other.**
  - "2-CAR GARAGE" gets the garage zone (v3 only).
  - Repeated room names are renamed.
  - No MEP scatter on ordinary plans.

## Not changed (judges' tells outside the brief)

Text, leaders, dimensions, grids, electrical logic, door placement and room layout logic. Layout tells came up often: wall stubs, door swings that collide and odd room shapes. They come from `plan_layout`, which v3 does not touch.

## Next

Nothing has been trained on v3 yet. On the next GPU session, generate a pool with `--plan-v2 2` under a new name and check HF14 0 and 12 against the plan-v2 and earlier pools.
