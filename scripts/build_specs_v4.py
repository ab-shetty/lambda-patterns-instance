#!/usr/bin/env python3
"""Build the v4 spec list from the measured shape of the real evaluation set.

The v1 spec list was ten families of ten -- elevations, floor plans, roof plans,
sections, wall details, site plans, RCPs, interior-finish plans, structural
plans, mixed-view sheets. `scripts/pool_style_stats.py` and a look at all 28
evaluation images say that list is not what the metric is scored on:

  * roughly two thirds of the real 28 are elevations, and 16 of 28 are
    colourised or rendered rather than black-on-white CAD;
  * their "patterns" are lap siding, board-and-batten, shingle courses and flat
    painted fields as often as they are drafting hatch;
  * several are markup-tool screenshots, with highlighter fills, magenta dot
    markers, dashed work-area boxes, and tool UI chrome inside the frame;
  * median aspect is 2.59 and ten of the 28 are wider than 3:1;
  * wall details, site plans, RCPs and structural sheets do not appear at all.

And the four worst-scoring HF14 images are all monochrome (saturation ~0) and
sparse -- ink 0.026-0.12 against the pool median 0.23. Image 14 (0.291) is two
nearly blank faint elevations at 4.2:1 whose materials are named in callout text
and barely differ visually; image 12 (0.393) is thin wall poche under a dense
electrical overlay. So this list is weighted toward faint, sparse, cluttered and
wide, on top of the eval set's overall shape.

That weighting is a bet on four data points. `PROJECT_UNDERSTANDING.md` records
two per-image failure stories that were confidently wrong, so treat it as an
aiming choice, not a theory: the round is still mostly the eval distribution.

    python3 scripts/build_specs_v4.py --out image_generation/specs_v4.jsonl

Deterministic given `--seed`; it prints the realised distribution and refuses to
write a list with duplicate subjects.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# --- shared vocabulary -------------------------------------------------------

HOUSE = ["single-storey ranch house", "two-storey house with a gabled roof",
         "split-level house", "two-storey house with a hipped roof",
         "single-storey house with an attached garage", "duplex",
         "three-unit townhouse row", "two-storey house with a shed dormer",
         "bungalow with a covered porch", "small apartment block, three storeys",
         "detached accessory dwelling unit", "house with a walk-out basement",
         "cottage with a steep gable", "carriage house over a garage",
         "L-shaped ranch house", "two-storey house with a wraparound deck"]

# The real excerpts are mostly ONE view. Counted over the 28: about half show a
# single elevation, most of the rest two, and only one shows four. The first v4
# list had no single-view option at all and put four views on 13 of 100.
# Kept wide on purpose: the uniqueness check rejects repeated
# subject+layout+patterns triples, so a short single-view list loses draws to
# collisions and the round drifts back toward multi-view sheets.
VIEWS_ONE = ["a single front elevation, on its own",
             "a single rear elevation, on its own",
             "one side elevation, on its own",
             "a single proposed front elevation, on its own",
             "one street-facing elevation, on its own",
             "a single garage-side elevation, on its own",
             "one courtyard-facing elevation, on its own",
             "a single elevation of the addition only, on its own",
             "one rear elevation showing the deck, on its own",
             "a single north elevation, on its own",
             "one south elevation, on its own",
             "a single elevation with the neighbouring property line shown"]
VIEWS_TWO = ["front and rear elevations side by side",
             "left and right side elevations side by side",
             "front elevation with the adjacent side elevation",
             "two proposed elevations above one another",
             "an existing and a proposed elevation side by side"]
VIEWS_FOUR = ["all four elevations in a row"]


def choose_views(rng):
    roll = rng.random()
    if roll < 0.50:
        return rng.choice(VIEWS_ONE)
    if roll < 0.95:                                  # only 1 of the real 28 shows four
        return rng.choice(VIEWS_TWO)
    return rng.choice(VIEWS_FOUR)


# Clutter, likewise, is drawn rather than maximised: a permit-set elevation
# carries a handful of leaders and level marks, not the full apparatus. Only the
# MEP plans are meant to be dense -- that density is the point of that category.
ELEV_CLUTTER_SPARSE = [
    "a few material callout leaders naming the finishes, level marks at floor "
    "and plate, and one dimension string",
    "two or three callout leaders and level marks, and nothing else",
    "level marks, a single overall dimension string, and one or two labels"]

# Materials as the real plans draw them: line families and flat fields, not
# always drafting hatch. Each entry is (name, how it is drawn).
SIDING = [("horizontal lap siding", "evenly spaced horizontal course lines"),
          ("board-and-batten siding", "evenly spaced vertical batten lines"),
          ("shingle siding", "staggered horizontal shingle course lines"),
          ("stucco", "a flat field with a very fine even stipple"),
          ("brick veneer", "regular horizontal coursing lines with head joints"),
          ("stone veneer", "an irregular blocky outline field"),
          ("standing-seam metal", "evenly spaced vertical seam lines"),
          ("fibre-cement panel", "a plain field divided by thin reveal joints"),
          ("wood plank soffit", "closely spaced parallel plank lines"),
          ("smooth painted trim band", "a plain unfilled field")]

COLOURS = ["sage green", "pale lavender", "warm tan", "cream", "slate gray",
           "soft blue-gray", "muted olive", "light taupe", "dusty rose",
           "pale sand"]

MARKUP = ["a translucent green highlighter fill covering one material's regions",
          "translucent magenta dot markers placed along one material",
          "a blue dashed work-area boundary box over part of the drawing",
          "a red revision line and a translucent yellow highlighter band",
          "a translucent green highlighter fill plus a blue dashed scope box"]

UI_CHROME = ["a markup-tool panel with a small tool icon and a label intruding "
             "at one corner, partly over the drawing",
             "a thin markup-application toolbar along the top edge with a "
             "stamp label visible"]

MEP_CLUTTER = ["dashed circuit-home-run arcs sweeping across the rooms, "
               "receptacle and switch symbols on every wall",
               "duct runs, ceiling supply and return grille tags, and "
               "thermostat symbols over the whole plan",
               "smoke and CO detector symbols, fixture tags and long callout "
               "leaders with two and three lines of note text each",
               "lighting symbols with switch legs curving between rooms and "
               "dimension strings on all four sides"]

PLAN_FINISH = [("ceramic tile", "a small square tile grid"),
               ("large-format tile", "a large square tile grid"),
               ("carpet", "an even fine stipple"),
               ("wood flooring", "closely spaced parallel plank lines"),
               ("polished concrete", "a sparse dot field")]

ROOF_FILL = [("asphalt shingles", "staggered horizontal course lines"),
             ("standing-seam metal", "evenly spaced parallel seam lines"),
             ("single-ply membrane", "a plain field with a sparse dot pattern"),
             ("built-up gravel roof", "a dense fine stipple"),
             ("tile roof", "closely spaced parallel course lines")]

SPARSE_SUBJECT = [
    "a partial building section through an addition, mostly empty page around it",
    "an under-floor framing plan with a single hatched crawlspace region",
    "a foundation plan with one filled slab region and long note blocks",
    "a wall section with three stacked levels and generous white space",
    "a stair section with one hatched wall running through it",
    "a garage under-floor plan with one filled region and a large note block"]

ARTIFACTS = ["faint PDF export with pale gray linework",
             "light scan skew and slightly soft lines",
             "JPEG ringing around the darker lines",
             "a clean vector PDF export at moderate line weight",
             "a pale photocopy-like export with thin lines"]

CONFUSERS = [
    "two of the material families differ only in course spacing -- one about "
    "25% tighter than the other -- and are otherwise identical",
    "two families share the same line direction and weight and differ only "
    "slightly in spacing",
    "one material is a plain painted field and another is the same field with "
    "a barely visible stipple, so they separate only on close inspection",
    "two families differ only in line angle by a few degrees at the same spacing"]


def wide_aspect(rng):
    """Aspect drawn to match the eval set: median ~2.6 with a tail past 3:1."""
    return rng.choice([2.0, 2.2, 2.5, 2.7, 3.0, 3.4, 3.4, 3.8, 4.2, 4.6])


def plan_aspect(rng):
    return rng.choice([1.3, 1.4, 1.5, 1.5, 1.7, 2.0])


def family_count(rng):
    """How many material families this drawing carries.

    Drawn from the eval set rather than from an idea of a rich sheet. The real
    28 hold 1.8 families on average -- 13 of 28 have exactly ONE -- against the
    2.7 the first v4 list asked for. The difficulty there comes from faintness,
    clutter and materials that differ barely, not from stacking families, and a
    two-family image is also a third less to label.
    """
    roll = rng.random()
    return 1 if roll < 0.45 else 2 if roll < 0.80 else 3


def materials_clause(rng, pool, count):
    """`patterns` text for however many families were drawn."""
    picked = rng.sample(pool, count)
    body = "; ".join(f"{name}, drawn as {how}" for name, how in picked)
    word = {1: "one material family", 2: "two material families",
            3: "three material families"}[count]
    return f"{word}: {body}"


def difficulty_clause(rng, count):
    """A one-family image cannot have a confusable pair; it needs the other trap."""
    if count == 1:
        return rng.choice([
            "the single family recurs in several separated regions and must be "
            "told apart from plain unfilled surfaces that look similar at a glance",
            "every other surface in the drawing is plain or unfilled, so the trap "
            "is deciding where the one filled material stops",
            "the one filled material appears in both large and very small "
            "regions, some only a few lines wide"])
    return rng.choice(CONFUSERS)


def contrast_level(rng, faint_p, light_p):
    """none / light / faint.

    v4's first draft made 61 of 100 specs flat-out faint, which put the pool's
    median ink at 0.025 against the real pool's 0.111 -- the whole round sat at
    the sparse extreme instead of spanning it, and roofs came back so pale they
    read as empty. `light` is the middle the real set is actually full of.
    """
    roll = rng.random()
    if roll < faint_p:
        return "faint"
    if roll < faint_p + light_p:
        return "light"
    return "none"


def elevation(rng, house, presentation, faint):
    count = family_count(rng)
    return {"subject": house, "layout": choose_views(rng),
            "patterns": materials_clause(rng, SIDING, count),
            "confuser": difficulty_clause(rng, count),
            "presentation": presentation, "faint": faint,
            "aspect": wide_aspect(rng)}


def build(seed):
    rng = random.Random(seed)
    specs = []
    seen = set()

    def add(category, **fields):
        """Keep the draw only if this subject is new; the caller redraws."""
        key = (category, fields["subject"], fields["layout"], fields["patterns"])
        if key in seen:
            return False
        seen.add(key)
        specs.append(dict(id=0, category=category, **fields))
        return True

    # --- elevations, colourised, with markup (16) ---------------------------
    # A few is enough: 3 of the 14 acceptance images carry review colour, but
    # the pool does not need to match that share to teach robustness to it, and
    # green images are the least representative thing to over-produce.
    while sum(s['category'] == 'elev_colour_markup' for s in specs) < 6:
        house = rng.choice(HOUSE)
        row = elevation(rng, house, "colourised",
                        faint=contrast_level(rng, 0.10, 0.35))
        row["colour"] = rng.choice(COLOURS)
        row["markup"] = rng.choice(MARKUP)
        row["ui_chrome"] = rng.choice(UI_CHROME) if rng.random() < 0.3 else ""
        row["clutter"] = (rng.choice(ELEV_CLUTTER_SPARSE) if rng.random() < 0.7
                          else "dimension strings, level marks and material "
                               "callout leaders naming each finish")
        row["artifacts"] = rng.choice(ARTIFACTS)
        add("elev_colour_markup", **row)

    # --- elevations, colourised, no markup (12) -----------------------------
    while sum(s['category'] == 'elev_colour' for s in specs) < 18:
        row = elevation(rng, rng.choice(HOUSE), "colourised",
                        faint=contrast_level(rng, 0.05, 0.30))
        row["colour"] = rng.choice(COLOURS)
        row["markup"] = ""
        row["ui_chrome"] = ""
        row["clutter"] = (rng.choice(ELEV_CLUTTER_SPARSE) if rng.random() < 0.7
                          else "dimension strings, level marks, window tags and "
                               "material callout leaders")
        row["artifacts"] = rng.choice(ARTIFACTS)
        add("elev_colour", **row)

    # --- elevations, faint monochrome: the image-14 case (22) ---------------
    while sum(s['category'] == 'elev_faint' for s in specs) < 22:
        row = elevation(rng, rng.choice(HOUSE), "monochrome",
                        faint=contrast_level(rng, 0.42, 0.48))
        row["colour"] = ""
        row["markup"] = ""
        row["ui_chrome"] = ""
        row["clutter"] = (rng.choice(ELEV_CLUTTER_SPARSE) if rng.random() < 0.7
                          else "thin callout leaders naming each material in "
                               "small text, level marks, and a property-line symbol")
        row["artifacts"] = rng.choice(ARTIFACTS[:2] + ARTIFACTS[-1:])
        add("elev_faint", **row)

    # --- elevations, ordinary monochrome (8) --------------------------------
    while sum(s['category'] == 'elev_mono' for s in specs) < 12:
        row = elevation(rng, rng.choice(HOUSE), "monochrome", faint="none")
        row["colour"] = ""
        row["markup"] = ""
        row["ui_chrome"] = ""
        row["clutter"] = (rng.choice(ELEV_CLUTTER_SPARSE) if rng.random() < 0.7
                          else "dimension strings, level marks, window and door tags")
        row["artifacts"] = rng.choice(ARTIFACTS)
        add("elev_mono", **row)

    # --- floor plans under an MEP overlay: the image-12 case (14) -----------
    while sum(s['category'] == 'plan_mep' for s in specs) < 14:
        finish, how = rng.choice(PLAN_FINISH)
        add("plan_mep", subject=rng.choice(HOUSE),
            layout="a complete floor plan of the whole footprint",
            patterns=("two families. First, the walls: drawn as a SOLID FILLED "
                      "dark gray poche about six inches thick -- filled in, not "
                      "a pair of empty outlines -- repeating around every room "
                      "and broken into many disconnected runs by door and window "
                      "openings. Second, {finish} on the floor of the wet rooms "
                      "and the garage, drawn as {how}. Both must be plainly "
                      "visible: a plan whose walls are empty outlines has nothing "
                      "in it to select").format(finish=finish, how=how),
            confuser=("furniture outlines, cabinet runs and the dashed overlay "
                      "must not read as wall fill or as floor finish"),
            presentation="monochrome", faint=contrast_level(rng, 0.25, 0.55),
            colour="", markup="", ui_chrome="",
            clutter=rng.choice(MEP_CLUTTER),
            artifacts=rng.choice(ARTIFACTS), aspect=plan_aspect(rng))

    # --- floor plans with room finishes (6) ---------------------------------
    while sum(s['category'] == 'plan_finish' for s in specs) < 6:
        count = max(2, family_count(rng))        # a finish plan needs two to differ
        add("plan_finish", subject=rng.choice(HOUSE),
            layout="a complete floor plan with the room finishes filled in",
            patterns=materials_clause(rng, PLAN_FINISH, count).replace(
                "material famil", "floor finish famil"),
            confuser=difficulty_clause(rng, count), presentation="monochrome",
            faint="none", colour="", markup="", ui_chrome="",
            clutter=("furniture, fixtures, appliance symbols and room name and "
                     "area labels interrupting the fills"),
            artifacts=rng.choice(ARTIFACTS), aspect=plan_aspect(rng))

    # --- roof plans (8) ------------------------------------------------------
    while sum(s['category'] == 'roof_plan' for s in specs) < 8:
        count = family_count(rng)
        add("roof_plan", subject=rng.choice(HOUSE),
            layout="a complete roof plan over the whole footprint",
            patterns=materials_clause(rng, ROOF_FILL, count).replace(
                "material famil", "roof surface famil"),
            confuser=difficulty_clause(rng, count), presentation="monochrome",
            faint=contrast_level(rng, 0.15, 0.45), colour="", markup="",
            ui_chrome="",
            clutter=("ridge and valley lines, slope arrows with pitch labels, "
                     "vents, skylights and overflow scuppers"),
            artifacts=rng.choice(ARTIFACTS), aspect=plan_aspect(rng))

    # --- sparse sections and framing sheets: the image-7 case (14) ----------
    while sum(s['category'] == 'section_sparse' for s in specs) < 14:
        count = min(2, family_count(rng))        # a sparse sheet holds one or two
        add("section_sparse", subject=rng.choice(HOUSE),
            layout=rng.choice(SPARSE_SUBJECT),
            patterns=materials_clause(rng, SIDING, count),
            confuser=("the two families must be told apart only by their fill, "
                      "not by where they sit in the drawing") if count == 2 else
                     difficulty_clause(rng, 1),
            presentation="monochrome", faint=contrast_level(rng, 0.35, 0.45),
            colour="", markup="", ui_chrome="",
            clutter=("two or three blocks of small specification note text and "
                     "a few leader lines, with large areas of empty page"),
            artifacts=rng.choice(ARTIFACTS[:2] + ARTIFACTS[-1:]),
            aspect=rng.choice([2.0, 2.5, 2.7, 3.0, 3.4, 4.0, 5.0]))

    rng.shuffle(specs)
    for index, spec in enumerate(specs, 1):
        spec["id"] = index
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="image_generation/specs_v4.jsonl")
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()

    specs = build(args.seed)
    if len(specs) != 100:
        raise SystemExit(f"expected 100 specs, built {len(specs)}")
    keys = {(s["category"], s["subject"], s["layout"], s["patterns"]) for s in specs}
    if len(keys) != 100:
        raise SystemExit(f"only {len(keys)} distinct subjects; raise --seed or "
                         f"widen the vocabulary")
    Path(args.out).write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in specs))

    counts = Counter(s["category"] for s in specs)
    faint = sum(1 for s in specs if s["faint"] == "faint")
    light = sum(1 for s in specs if s["faint"] == "light")
    wide = sum(1 for s in specs if s["aspect"] > 3)
    coloured = sum(1 for s in specs if s["presentation"] == "colourised")
    print(f"wrote {len(specs)} specs to {args.out}")
    for category, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {category:<20} {n}")
    def n_fam(spec):
        return (1 if spec["patterns"].startswith("one") else
                2 if spec["patterns"].startswith("two") else 3)
    import statistics as _st
    counts = Counter(n_fam(s) for s in specs)
    print(f"families/image: mean {_st.mean([n_fam(s) for s in specs]):.1f}  "
          f"dist {dict(sorted(counts.items()))}   (real 28: mean 1.8, 13 single)")
    views = Counter("four" if "four" in s["layout"] else
                    "one" if "on its own" in s["layout"] else "two/plan"
                    for s in specs)
    print(f"views: {dict(views)}")
    print(f"faint: {faint}   light: {light}   colourised: {coloured}   "
          f"wider than 3:1: {wide}   with markup: "
          f"{sum(1 for s in specs if s['markup'])}")


if __name__ == "__main__":
    main()
