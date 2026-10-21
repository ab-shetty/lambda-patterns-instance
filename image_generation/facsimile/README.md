# Facsimiles: a Gemini sheet in, a near-same synthetic sheet with exact labels out

Three routes, from most faithful to most general. All write the pool format (`images/`, `annotations/`).

## 1. Trace twin of an UNLABELLED sheet — the production route

```bash
GEMINI_API_KEY=... python3 scripts/facsimile_trace.py --images DIR_OF_PNGS --out data/synthetic/trace_NAME
```

Per sheet:

1. **Families from Gemini** (`gemini-2.5-pro`): every distinct material pattern with a generator kind and 1-3 small
   boxes inside it. A pattern in a different colour / tone / spacing / direction is a separate family.
2. **Masks from our model** (`data/hf_ckpt/xblend-kq30-kms30.pth`, CPU, 2048): one segmentation per box, a family is
   the max over its boxes. A box its own segmentation does not cover (it sat on callout text) is dropped; a family
   keeps its best box if that scores >= 0.15.
3. **Second pass**: Gemini sees the drawing with the families found so far outlined and lists the patterns still
   missing (or lumped into one outline). They are segmented and added; one that re-lists an existing family (same
   kind and colour, or its mask overlaps one at IoU >= 0.4) is merged into it.
4. **Winner per pixel** from the Gaussian-blurred maps. On colour-filled sheets a family whose fills form two
   different hues is split (chroma only, so shadows do not split), and pixels move between families to follow the
   fills (they are never dropped).
5. **Repaint and label**: each family's mask is repainted with the generator's texture for its kind
   (`facsimile_relabel.family_style`: the fill colour is the family's own median colour, the sheet's paper on
   line-only sheets); the sheet's own linework, openings and text stay. The labels ARE the repainted masks, one
   instance per connected piece with holes, so a twin's labels are exact by construction: a segmentation miss makes
   the twin differ from the original, never a twin label wrong. The one residual risk is pattern area no family
   claims -- it keeps Gemini's texture without a label ("unclaimed" below).

Outputs: `families.json` (Gemini's answers, cached -- re-runs reuse them), `compare.jpg` (original | twin | labels).
Already-traced sheets are skipped unless `--force`. About 2 min of CPU per sheet plus two Gemini calls.

### How good (round 4, 22 hand-labelled elevations, traced WITHOUT their labels)

`scripts/trace_eval.py --trace OUT --pool data/roboflow/floz-gen-gemini-r4-clean` matches each true family to the
traced family that overlaps it most. "In-label" IoU ignores traced pixels outside every hand label (roofs and trim
the labeller skipped). "Unclaimed" = labelled pattern the trace left unlabelled.

| version | family IoU | in-label | unclaimed |
|---|---|---|---|
| oracle: true families, boxes inside them (ceiling of steps 2-5) | 0.879 | 0.944 | 0.024 |
| Gemini families, one pass | 0.650 | 0.730 | 0.118 |
| + clearer prompt, second pass, duplicate merge, box self-check | 0.682 | 0.726 | 0.076 |

The gap to the oracle is Gemini's family list (lumped colours, boxes on callout text, missed families), not the
segmentation. Tried and dropped: family discovery from a grid of model references (slow, worse), SLIC superpixel
snapping (square tiles), re-sampled in-distribution boxes (`--refine-boxes 1`, 0.726 -> 0.680, merges families).

## 2. Label-shape twin of a LABELLED sheet

```bash
GEMINI_API_KEY=... python3 scripts/facsimile_kinds.py --pool POOL --out MAP.json     # family -> kind, from Gemini
python3 scripts/facsimile_relabel.py --map MAP.json --pool POOL --boxes '' --out OUT
```

Same repaint as step 5, on the hand labels: the most faithful twin when labels exist.

## 3. Parametric facsimile from a description

`scripts/describe_gemini_specs.py` → `scripts/facsimile_eval.py specs` → `generate_synthetic_fc.py --spec`. Rebuilds the
house in the generator from a description: new geometry, the generator's own drawing. Limited to what the generator's
vocabulary can express (see synth_progress.md, facsimile v1-v13).
