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

TEMPLATES = {"v1": TEMPLATE, "v2": TEMPLATE_V2}
DEFAULT_OUT = {"v1": "image_generation/prompts.jsonl",
               "v2": "image_generation/prompts_v2.jsonl"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", default="image_generation/specs.jsonl")
    parser.add_argument("--version", choices=sorted(TEMPLATES), default="v1")
    parser.add_argument("--out", default=None,
                        help="defaults to prompts.jsonl / prompts_v2.jsonl")
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_OUT[args.version]
    specs = [json.loads(line) for line in Path(args.specs).read_text().splitlines()
             if line.strip()]
    if [spec["id"] for spec in specs] != list(range(1, 101)):
        raise ValueError("spec IDs must be exactly 1 through 100 in order")
    template = TEMPLATES[args.version]
    rows = [{"id": spec["id"],
             "filename": f"floz_gen_{spec['id']:03d}.png",
             "prompt": template.format(**spec)} for spec in specs]
    Path(args.out).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
