#!/usr/bin/env python3
"""Render the canonical 100 image-model prompts from checked-in specifications.

Two framings, both rendered from the same unchanged `specs.jsonl` subject list:

v1  the framing that was actually executed on 2026-08-06. It asks for a full
    construction SHEET, which is what produced legends and title blocks and a 29%
    usable yield. Kept byte-identical so the SHA-256 fingerprints recorded in
    `image_generation/README.md` still validate; do not edit it.

v2  the rewrite the README's "Aim the next round at" section asks for: the
    drawing region itself, one complete coherent building, no title block or
    legend or sheet border, crisp vector-like CAD hatching rather than tonal
    shading. The subject list was never the problem, so it is reused verbatim.

v3  v2 plus the two defects the 2026-08-12 smoke test exposed, which v2's
    framing does not address: the model draws each material's REAL APPEARANCE
    (brick coursing, shingle scallops, board siding) instead of drafting hatch,
    and it ignores the confuser requirement entirely -- the delivered pool is
    the least confusable in the project, 4.8% of multi-family images against
    24.4% for the scraped real plans. So v3 names the fills as notation and
    states the confuser tolerance numerically. Current version; use this one.
"""

import argparse
import json
from pathlib import Path


TEMPLATE = """Use case: scientific-educational
Asset type: unlabelled training image for reference-conditioned architectural-pattern segmentation
Primary request: Create one highly realistic full architectural construction drawing sheet, suitable for later manual polygon annotation. This is dataset item {id:03d} of 100 and must be visually unique, not a variation of another item.
Sheet family: {family}
Layout: {layout}.
Patterns: {patterns}. Every pattern family must recur in 2-6 spatially disconnected regions. Preserve enough clean interior in each family for both tiny and medium user reference rectangles.
Difficulty: {difficulty}.
Drafting content: add plausible professional dimensions, leaders, level marks, detail bubbles, symbols, fixtures, openings, tags, notes, and drawing borders. These elements should interrupt some patterned regions so an annotator can label subtraction holes. Include visually confusing unpatterned hard negatives.
Style/medium: authentic black and gray CAD linework rasterized from a professional construction-document PDF; mixed line weights, realistic drafting density, {artifacts}; not an illustration and not blueprint artwork.
Composition/framing: full sheet visible on white paper, wide landscape approximately 3:2, no cropped drawing border.
Constraints: no colored rendering, no photographic building, no watermark, no logo, no large readable headline, no segmentation overlay, no bounding boxes, no colored masks, and no pre-existing annotation marks."""


TEMPLATE_V2 = """Use case: scientific-educational
Asset type: unlabelled training image for reference-conditioned architectural-pattern segmentation
Primary request: Create one highly realistic excerpt of an architectural construction drawing — the DRAWING REGION ITSELF as it sits inside a sheet, never the whole sheet. This is dataset item {id:03d} of 100 and must be visually unique, not a variation of another item.
Subject: {family}
Layout: {layout}. Every view present must be drawn COMPLETE — no facade cut off mid-wall, no plan truncated mid-footprint, no zoomed-in fragment of a larger building — and all views must belong to the same project.
Patterns: {patterns}. Every pattern family must recur in 2-6 spatially disconnected regions. Preserve enough clean interior in each family for both tiny and medium user reference rectangles.
Difficulty: {difficulty}.
Drafting content: plausible professional dimensions, leaders, level marks, detail bubbles, symbols, fixtures, openings, tags, and notes, at the density of a working drawing. Some of these must interrupt patterned regions so an annotator can label subtraction holes. Include visually confusing unpatterned hard negatives.
Excluded content: no title block, no legend, no material schedule, no revision table, no sheet border or frame, and no large readable headline. The drawing itself fills essentially the whole image.
Style/medium: crisp black and gray CAD linework rasterized from a professional construction-document PDF — vector-like hatching at consistent spacing with mixed line weights, {artifacts}. Flat line art only: not tonal, not grayscale-shaded, not rendered, not an illustration, not blueprint artwork, not a photograph.
Composition/framing: the drawing fills the frame on white paper, landscape orientation, with only a thin margin and nothing important cropped at the edges.
Constraints: no colored rendering, no photographic building, no watermark, no logo, no segmentation overlay, no bounding boxes, no colored masks, and no pre-existing annotation marks."""


# v3 = v2 + the "Hatching" and "Confusable families" blocks. Those two are the
# whole delta: the smoke test showed that framing alone gets a clean drawing of
# the wrong thing -- real materials rendered realistically, every family easy to
# tell apart. The tolerance is stated as a number because "similar" did not
# survive the model's interpretation. Do not raise it further: deliberately
# confusable SYNTHETIC data cost -0.104 (synth_progress.md, 2026-08-12). The aim
# is to match the real distribution, which this pool undershoots.
TEMPLATE_V3 = """Use case: scientific-educational
Asset type: unlabelled training image for reference-conditioned architectural-pattern segmentation
Primary request: Create one highly realistic excerpt of an architectural construction drawing — the DRAWING REGION ITSELF as it sits inside a sheet, never the whole sheet. This is dataset item {id:03d} of 100 and must be visually unique, not a variation of another item.
Subject: {family}
Layout: {layout}. Every view present must be drawn COMPLETE — no facade cut off mid-wall, no plan truncated mid-footprint, no zoomed-in fragment of a larger building — and all views must belong to the same project.
Patterns: {patterns}. Every pattern family must recur in 2-6 spatially disconnected regions. Preserve enough clean interior in each family for both tiny and medium user reference rectangles.
Hatching — the most important requirement: every material fill is DRAFTING NOTATION, not a picture of the material. Draw each family as a regular field of ruled straight parallel lines, crosshatch, or evenly spaced dots, at ONE constant angle and ONE constant spacing, identical everywhere that family appears. Do NOT draw what the material actually looks like: no brick coursing or individual bricks, no shingle scallops or roof tiles, no board or lap siding, no stone or block outlines, no wood grain, no photoreal or tonal texture, no shading or gradients inside a region. A wall of brick is a field of 45-degree ruled lines, not a picture of bricks.
Confusable families: at least two families must be deliberately hard to tell apart, differing only in one small parameter — for example 45-degree ruled hatch at 3 mm spacing against 45-degree at 4 mm, or 45-degree against 40-degree at the same spacing. They must stay genuinely distinguishable when looked at closely, and hard to separate at a glance. Do not make every family obviously different.
Difficulty: {difficulty}.
Drafting content: plausible professional dimensions, leaders, level marks, detail bubbles, symbols, fixtures, openings, tags, and notes, at the density of a working drawing. Some of these must interrupt patterned regions so an annotator can label subtraction holes. Include visually confusing unpatterned hard negatives.
Excluded content: no title block, no legend, no material schedule, no revision table, no sheet border or frame, and no large readable headline. The drawing itself fills essentially the whole image.
Style/medium: crisp black and gray CAD linework rasterized from a professional construction-document PDF — vector-like hatching at consistent spacing with mixed line weights, {artifacts}. Flat line art only: not tonal, not grayscale-shaded, not rendered, not an illustration, not blueprint artwork, not a photograph.
Composition/framing: the drawing fills the frame on white paper, landscape orientation, with only a thin margin and nothing important cropped at the edges.
Constraints: no colored rendering, no photographic building, no watermark, no logo, no segmentation overlay, no bounding boxes, no colored masks, and no pre-existing annotation marks."""


# v4 assembles rather than formats: a spec carries optional parts (colour,
# markup, tool chrome) and an empty one must leave no dangling sentence.
def render_v4(spec):
    parts = []
    parts.append("Use case: scientific-educational")
    parts.append("Asset type: unlabelled training image for reference-conditioned "
                 "architectural-pattern segmentation")
    parts.append(
        "Primary request: Create one excerpt from a real residential construction "
        "plan set — a single drawing as it looks when one page of the PDF is "
        f"cropped down to it. This is dataset item {spec['id']:03d} of 100 and "
        "must be visually unique, not a variation of another item.")
    parts.append(f"Subject: {spec['subject']}")
    parts.append(
        f"Layout: {spec['layout']}. Every view present must be drawn COMPLETE — "
        "nothing cut off mid-wall or mid-footprint — and all views must belong to "
        "the same project.")
    parts.append(
        f"Materials: {spec['patterns']}. Every material family must appear in "
        "several spatially separated places in the drawing.")
    # The consistency requirement gets its own block, ahead of the drawing's
    # content, and names the failures rather than describing the ideal: v3 asked
    # for "ONE constant angle and ONE constant spacing" inside the hatching
    # paragraph and scored the worst fill regularity of any pool measured.
    parts.append(
        "PATTERN CONSISTENCY — THE MOST IMPORTANT REQUIREMENT, ABOVE REALISM: "
        "each material family must be drawn EXACTLY THE SAME in every region it "
        "appears in. One spacing, one angle, one line weight, chosen once and "
        "held everywhere — in small regions as well as large, at the edges of a "
        "region as well as in the middle. The lines must be machine-ruled and "
        "perfectly straight, as if plotted from CAD. SPECIFICALLY FORBIDDEN: "
        "lines that waver, wobble or look hand-drawn; lines that curve or bend "
        "to follow a wall, a roof slope or an opening; spacing that opens up or "
        "tightens across a region; a fill that fades out, changes weight, or "
        "changes scale between two regions of the same material; two regions of "
        "the same material that do not look like the same material. Where an "
        "opening, a symbol or a note interrupts a region, the fill continues on "
        "the far side ON THE SAME GRID — draw each family as one continuous "
        "ruled field across the whole drawing, then let the geometry mask it.")
    parts.append(f"Deliberate difficulty: {spec['confuser']}.")

    if spec["presentation"] == "colourised":
        parts.append(
            f"Presentation: a colourised permit-set drawing — the surfaces are "
            f"filled with flat {spec['colour']} and other muted architectural "
            "tints, with each material's line pattern drawn ON TOP of its tint. "
            "This is a coloured drawing, not black-and-white linework, and not a "
            "photorealistic rendering. Each material named above is drawn with "
            "its own line pattern over its tint — a roof of shingles is course "
            "lines over a tint, not a plain coloured plane. Other surfaces may "
            "be plain colour.")
    else:
        parts.append("Presentation: black and gray linework on white paper, as "
                     "exported from CAD to PDF. No colour fills.")
    # Three levels, not a flag. The first v4 draft treated faint as on/off and
    # came back with roofs so pale they read as empty page -- a fill nobody can
    # see is a region nobody can label, so faintness now has a floor.
    if spec["faint"] == "faint":
        parts.append(
            "Contrast: faint. Pale gray linework and low contrast, as if lightly "
            "printed, with much of the page empty white. Every material fill "
            "must still be plainly visible on the paper — subtle, not vanishing, "
            "and never fading out to blank white in the middle of a region.")
    elif spec["faint"] == "light":
        parts.append(
            "Contrast: light. Thin gray linework, lighter than a bold CAD plot "
            "but fully legible, with the material fills easy to see at a glance.")
    if spec["markup"]:
        parts.append(
            f"Markup: one of the material families has been colour-filled for "
            f"review — {spec['markup']}. Fill it the way a paint-bucket fills a "
            "shape: every region of that one family is SOLID GREEN THROUGHOUT "
            "its interior, from one edge of the region to the other, with the "
            "green stopping exactly at the region outline and leaving the "
            "windows, doors and openings inside it unfilled. The family's own "
            "lines stay drawn on top of the green at full strength. Not an "
            "outline, not a glow or halo around the edges, not a brush stroke, "
            "not a soft or faded wash — the interior is green.")
    if spec["ui_chrome"]:
        parts.append(f"Screen capture: this is a screenshot of that review tool, "
                     f"so {spec['ui_chrome']}.")
    parts.append(
        f"Drafting content: {spec['clutter']}. Some of it must cross patterned "
        "regions so an annotator can label subtraction holes, and openings, "
        "fixtures and symbols must interrupt the fills.")
    # The model embellishes: asked for two elevations it draws four, asked for
    # three materials it invents a fourth. The eval set is plainer than that --
    # 1.8 families per image, most excerpts a single view -- so the restraint has
    # to be stated as a requirement rather than implied by the spec.
    parts.append(
        "Restraint — draw ONLY what is listed above: exactly the views named and "
        "no others, exactly the material families named and no others, and no "
        "extra annotation beyond what the drafting content asks for. This is a "
        "plain working drawing, not a showcase sheet. Empty white paper around "
        "and between the views is correct and expected.")
    parts.append(
        "Excluded content: no title block, no legend, no material schedule, no "
        "revision table, no sheet border or frame, and no large headline. This is "
        "one drawing cropped out of a sheet, not the sheet.")
    parts.append(f"Medium: {spec['artifacts']}; the texture of a scanned or "
                 "exported construction document, not an illustration.")
    if spec["aspect"] > 2.3:                    # both APIs cap below the eval tail
        # The API stops at 3:1, so the extra width is won back by trimming the
        # margin afterwards -- which only works if the margin is there to trim.
        parts.append(
            f"Composition: landscape. The drawing itself must be a wide band of "
            f"roughly {spec['aspect']:.1f}:1 sitting across the middle of the "
            "frame, with empty white paper above and below it and nothing "
            "important near the top or bottom edge.")
    else:
        parts.append(
            f"Composition: landscape, roughly {spec['aspect']:.1f}:1, the drawing "
            "filling the frame with a thin margin and nothing important cropped "
            "at the edges.")
    parts.append(
        "Never write any of this brief onto the drawing: labels must be ordinary "
        "architectural callouts naming materials, never words like 'pattern', "
        "'family', 'no pattern', 'markup' or 'typical of this dataset'.\n"
        "Constraints: no photographic building, no watermark, no logo, no "
        "segmentation overlay, no bounding boxes, no coloured region masks of the "
        "kind an annotation tool draws, and no pre-existing polygon annotations.")
    return "\n".join(parts)


TEMPLATES = {"v1": TEMPLATE, "v2": TEMPLATE_V2, "v3": TEMPLATE_V3,
             "v4": render_v4}
DEFAULT_OUT = {"v1": "image_generation/prompts.jsonl",
               "v2": "image_generation/prompts_v2.jsonl",
               "v3": "image_generation/prompts_v3.jsonl",
               "v4": "image_generation/prompts_v4.jsonl"}
# v1-v3 share one subject list; v4 is built from the evaluation set's shape.
DEFAULT_SPECS = {"v4": "image_generation/specs_v4.jsonl"}
DEFAULT_SPEC = "image_generation/specs.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", default=None,
                        help="defaults to the spec list for the version")
    parser.add_argument("--version", choices=sorted(TEMPLATES), default="v1")
    parser.add_argument("--out", default=None,
                        help="defaults to prompts[_v2,_v3].jsonl for the version")
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_OUT[args.version]
    if args.specs is None:
        args.specs = DEFAULT_SPECS.get(args.version, DEFAULT_SPEC)
    specs = [json.loads(line) for line in Path(args.specs).read_text().splitlines()
             if line.strip()]
    if [spec["id"] for spec in specs] != list(range(1, 101)):
        raise ValueError("spec IDs must be exactly 1 through 100 in order")
    template = TEMPLATES[args.version]
    render = template if callable(template) else (lambda spec: template.format(**spec))
    rows = [{"id": spec["id"],
             "filename": f"floz_gen_{spec['id']:03d}.png",
             "prompt": render(spec),
             "aspect": spec.get("aspect"),
             "category": spec.get("category")} for spec in specs]
    Path(args.out).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
