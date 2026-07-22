#!/usr/bin/env python3
"""Render the canonical 100 image-model prompts from checked-in specifications."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", default="image_generation/specs.jsonl")
    parser.add_argument("--out", default="image_generation/prompts.jsonl")
    args = parser.parse_args()
    specs = [json.loads(line) for line in Path(args.specs).read_text().splitlines()
             if line.strip()]
    if [spec["id"] for spec in specs] != list(range(1, 101)):
        raise ValueError("spec IDs must be exactly 1 through 100 in order")
    rows = [{"id": spec["id"],
             "filename": f"floz_gen_{spec['id']:03d}.png",
             "prompt": TEMPLATE.format(**spec)} for spec in specs]
    Path(args.out).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
