# Visual quality audit — synthetic generators vs. real data

Full write-up: `synth_progress.md`, 2026-09-17 entry ("Visual quality audit").
This folder exists because that finding came from *looking at images*, not
metrics, and the images are the evidence.

| file | what it shows |
|---|---|
| `01_v5_roofplan_flat_fill.png` | v5 (`toparea1600_balanced`, still in the documented mix): a roof plan rendered as two flat hatched-fill triangles. No per-material texture, no drafting detail. |
| `02_v5_elevation_brick_pattern_on_roof.png` | v5 elevation: the roof is filled with a brick/subway-tile grid pattern — the wrong texture family for a roof (should read as shingle coursing). |
| `03_real_elevation_for_comparison.png` | Real scraped plan (`floz-real-pool-v2`): individually rendered shingle courses, lap siding, stone base, proper callouts, scale bar, title block. |
| `04_real_gemini_foundation_plan_for_comparison.png` | Real photographed plan (`floz-gen-gemini-r4`): irregular hand-drawn stone veneer (no two stones alike), cross-hatched slab fill, archival lettering. |
| `05_v6_townhouse_saturated_colours.png` | v6 generator: much better composition (townhouse row, dormers, garages, material callouts) but cartoonishly saturated teal/red/grey palette. |
| `06_v6_blue_claytile_stone_plus_label_overlap_bug.png` | v6: a clay-tile roof and cultured-stone wall both rendered in saturated royal blue — a material-color combination that does not exist in reality. Also shows the confirmed 2-story level-mark label collision (`T.O. PLATE` / `SECOND FLOOR` at the same y-coordinate — `generate_synthetic_v6.py` ~line 1451). |
| `07_v6_green_brick_with_markup_dots.png` | v6: correct per-material texture assignment (brick on walls, stippled shingle on roof) but implausibly saturated pure-green brick. The magenta dots are **not** a bug — `DOT_MARKUP_PROB`, intentional simulated review markup. |
| `08_v6_roofplan_unexplained_concentric_pattern.png` | v6 roof plan: one facet fills with a nested-concentric-rectangle pattern that doesn't match any real roofing convention. Mechanism not identified — flagged, not confirmed as a bug. |

Regenerate more v6 samples any time with:

```bash
python3 generate_synthetic_v6.py --n 20 --out /tmp/v6_sample --seed <n>
```
