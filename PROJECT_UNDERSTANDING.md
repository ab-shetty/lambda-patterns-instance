# Floz Project Understanding

## User interaction

1. A user draws a rectangular reference selection inside a pattern/material
   region in one plan image.
2. The model uses the visual content of that rectangle as its query.
3. The model returns the complete region or regions in that same image which
   have the same pattern/material as the selected rectangle.

The output is therefore **reference-conditioned region selection**, not
class-agnostic instance detection.

## Meaning of pattern labels

Labels such as `pattern1`, `pattern2`, etc. are grouping IDs local to a single
image. They mean only that regions carrying the same ID within that image match
one another.

- `pattern1` in image A has no relationship to `pattern1` in image B.
- Pattern numbers must never be trained or evaluated as global material classes.
- Given a reference inside one annotation, the target is every annotation with
  the same local grouping ID in that image.

Roboflow `remove` polygons are not patterns. They are holes cut out of enclosing
pattern masks during import.

## Primary evaluation

For each heldout user reference selection:

1. Form `target_union`: the union of all ground-truth masks with the selected
   region's image-local grouping ID.
2. Form `prediction_union`: all pixels the model selects in response to that
   rectangle.
3. Compute `IoU = intersection(prediction_union, target_union) /
   union(prediction_union, target_union)`.
4. Average IoU across heldout reference selections.

This **reference-conditioned union IoU** penalizes all three relevant errors:

- missing a matching region;
- selecting a region with a different local pattern;
- selecting excess pixels outside the matching regions.

Every labelled heldout instance is evaluated as a possible source rectangle so
the result measures whether different valid user selections lead to the correct
region union. The fixed HF14 heldout image indices are:

`12,16,27,7,11,25,23,1,18,2,0,3,14,24`

## Acceptance target

Achieve **reference-conditioned union IoU >= 0.80** on the real HF14 heldout
set after a 10-epoch training run using a combination of the strongest
synthetic data and cleaned labelled real data, including Roboflow.

## Metrics that do not prove success

The following may be useful diagnostics, but are not the primary task metric:

- class-agnostic IoU over all annotated regions;
- per-GT best-candidate IoU that ignores false positives;
- an "oracle" that uses heldout labels to choose predictions;
- Mask R-CNN predictions made without consuming the user rectangle;
- global classification accuracy for `pattern1`, `pattern2`, etc.

The class-agnostic Mask R-CNN experiment reported as 0.8545 did **not** implement
the user interaction and is invalid as task-performance evidence.

## Required visual audit

Heldout visualizations must show, for each evaluated reference selection:

1. the input image with the user rectangle;
2. the expected union of matching image-local regions;
3. the model's reference-conditioned predicted union;
4. an error view distinguishing overlap, excess selection, and missed pixels.

The current corrected baseline artifacts are in
`data/visualizations/reference_conditioned_hf14_k2_q200`.
