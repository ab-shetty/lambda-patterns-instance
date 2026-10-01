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

\* graphics-only prompt.

Best so far: 85% against sheet 0 (rounds 5 and 6, rendered and finish sheets). No judge has called the wrong image yet.

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
- **Hardscape.** Front walks run on to a notched wall instead of floating. Decks get a rim board (30%).
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
