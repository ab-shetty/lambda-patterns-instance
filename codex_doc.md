# Handoff

Last updated: 2026-11-09 (edge-weight test + vertex curve, GH200)

Where the work stands and what to do next -- nothing else. `PROJECT_UNDERSTANDING.md` defines the task and the metric;
`startup.md` is the reference manual (current best, numbers, gate, setup, commands); `synth_progress.md` logs every experiment.
Finished handoffs are in `archive/startup_archive.md` and git history. Rewrite this file at the end of each session.

## 2026-11-09: vertices are now a primary metric

The user: polygons with ~3x the labeller's vertices are a product failure even at 0.92 IoU (PROJECT_UNDERSTANDING.md). Report
vertices / label (`scripts/pp_probe/vertcount.py`, or `curve.py --summary`) next to IoU.
- **Training for vertices failed**: `--edge-weight 4` and `10` (gem_s7 recipe) leave vertex count and boundary error unchanged
  (synth_progress 2026-11-09). Do not retry loss reweighting near the true boundary.
- **The trade-off curve** (`scripts/pp_probe/curve.py`): Douglas-Peucker eps is the lever -- eps 4 cuts the median ratio 30-40%
  for ~-0.002 IoU, eps 8 gets ~1.5x on every split for -0.006..-0.008. The p90 tail (10-26x, notches / holes) needs something
  else: hole filling and closing help real sheets but hurt Gemini30.
- **The user's aspirational target: <= 1.2x vertices / label at >= 0.95 mIoU.**
- **~1.2x is reached, and holds on unseen sheets**: fill irregular holes and inlets (convex-hull gaps) the model was unsure about
  (mean P >= 0.05), Douglas-Peucker 12 px + axis snap, drop pieces < 0.5% (`curve.py` "e .05 inlet+eps12+minf.005"): gemdev (xblend,
  unseen) 1.23x / -0.0082, Gemini30 1.17x / -0.0065, val 1.00x / -0.0089, HF14 1.15x / -0.0061. The tail was dropouts inside
  correct regions, not edges. (The adaptive-eps variant looked better on the eval sheets but did not transfer to gemdev.)
- **Next**: (1) the user picks the trade; (2) port to `scripts/polygonize.py` behind a flag and re-score the full stack (ms + cc); (3) win back the ~0.005-0.009 IoU: a
  better gap classifier than shape + mean P (e.g. ink / texture match to the reference inside the gap); (4) 0.95 mIoU is a model
  problem -- ignoring 2 px around every true edge still gives Gemini30 0.91, so the rest is wrong / missing regions.

## Where we were (2026-10-22): closed out

Best: `abshetty/floz-refunet-swint-gem2s-x` (0.25 gem_s7 + 0.25 gem_s8 + 0.5 xblend-kq30-kms30), with `-gk-x` and `-gke-x` published
beside it (all pass the gate; within noise of each other). Raw @4096: fix v2 0.9320, Gemini30 0.8891, val 0.8990, HF14 0.9235; with the
product stack (polygons + multi-scale + colour cut, 0.66 s / query) fix v2 0.9365, Gemini30 0.9122. The lever that worked: hand-labelled
Gemini sheets outside Gemini30 (+0.019 Gemini30, +0.022 fix v2 at seed 7; seed 8 agrees). Hard mining of those sheets, model-in-the-loop
post-processing (reciprocity, reference ensembles) and TTA (latency) did not survive. The user's 0.95 Gemini30 target was not reached.

## Next, if the project resumes

1. **Label more Gemini sheets** -- round 5 (200 generated, `image_generation/prompts_r5.jsonl`, unlabelled) -- and retrain the gem recipe
   (`run_gemini_holdout.sh` pool + `run_single_pass.sh`, two seeds, blend with the current best). This is the only lever measured to move
   Gemini30 by more than seed noise.
2. **Self-training with automatic boxes** on unlabelled Gemini sheets (parked 2026-10-20; worth it once round 5 exists). Tile each sheet
   with candidate boxes; keep a box only if it lies wholly inside its own answer at P > 0.9, the answer is a sensible size with texture,
   and the box is not mostly text. Note the 2026-10-22 reciprocity result: the model does not answer back reliably from a second box on
   the same material, so do not use reference-swap consistency as a filter. Validate on val's labelled sheets first; Gemini30 never in the pool.
3. The 2026-10-20d / 21a generator defaults (cross gables, wall splits, panels, pattern drift, base bands) are unscreened.

## Open problems

- Gemini30 26 and 19 (both < 0.5) did not move with the Gemini data; the rest of the loss is spread over ~15 sheets at 0.80-0.93.
- Gemini30 17 (hip-roof shingle plan) is fragile across arms; Gemini30 1 swings 0.30-0.92 between models.
- Scale sensitivity: 2048 vs 4096 swings single sheets by +-0.17-0.29.
- Gemini30 0: a tile reference takes the sofas / chairs. HF14 18: the two roofs merge under label fix v2.
