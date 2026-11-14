# Handoff

Last updated: 2026-11-14 (foreground A/B, GH200)

Where the work stands and what to do next -- nothing else. `PROJECT_UNDERSTANDING.md` defines the task and the metric;
`startup.md` is the reference manual (current best, numbers, gate, setup, commands); `synth_progress.md` logs every experiment.
Finished handoffs are in `archive/startup_archive.md` and git history. Rewrite this file at the end of each session.

## 2026-11-14: `--foreground` A/B passes the gate; default 0.4 (GEN 2026-11-14a)

One seed (7), restarts of gem2s-x, product v3: hf14fix4 0.9473 -> **0.9504** (+0.0031, p 0.23), valfix4 0.9517 -> 0.9514,
Gemini30 0.8990 -> 0.9068 (p 0.046). **But hf14fix4 vertices / label 1.17x -> 1.46x**, and the gain sits on 14 / 12 / 27, not
on the targeted sheet 02 (synth_progress 2026-11-14). Next: (1) seed 8 of both arms (`SEED=8 ./run_foreground_ab.sh`) before
trusting +0.003; (2) find why fg outlines carry more vertices on HF14 (render the questions that rose most); (3) the
deployment latency check on a 16-20 GB card. FreeCAD on this VM: user-local micromamba env, `FREECADCMD=~/fc/bin/freecadcmd`
(no sudo for /opt/fc).

## 2026-11-11: 0.95 within reach on val from inference changes; hf14fix2 needs label review + model work

Target (the user): >= 0.95 mIoU on valfix3 and hf14fix2 at <= 1.2x vertices / label, on the product output; the product follows
the real-sheet edge convention.
- **Refinement head (`--refine-head 32`, base frozen, +- edge weight) fails the gate**: product IoU within +-0.0014. Do not
  unfreeze it without a new reason (synth_progress 2026-11-10/11).
- **Best product output (gem2s-x, no retraining): 5120+4096+3072 equal blend, threshold 0.35, low-vertex gap fills, eps 6,
  pieces < 0.5%, polygons grown 1 px: valfix3 0.9495 / 1.13x, hf14fix2 0.9407 / 1.12x** (current ms+cc_lv: 0.9338 / 1.00x,
  0.9308 / 1.05x). Costs ~2.5x the compute of today's two scales (the user has not decided on latency). The scale part holds on
  gemdev; the offset loses on Gemini (opposite label convention) and eps 6 is ~1.5x vertices on Gemini sheets.
- **Label fixes v4 (2026-11-11)**: Roboflow `floz-eval28-reference` is the single source again -- v2 pushed into it
  (`scripts/push_v2_to_roboflow.py`, backups `eval_labels/roboflow_backup_2026-11-11/`), sheet 14's stale pattern2 rectangle on
  the window-09 panel removed, and the user made HF14 18's right roof an ignore region. `make_real_fixes_v3.py --roboflow-only`
  -> `eval_labels/real_fixes_v4.json`; pp_probe splits `valfix4` / `hf14fix4` (50 q: 18's right-roof question is dropped).
  Product v2 (`stack.py` ms3+cc_v2): **valfix4 0.9495 / 1.13x, hf14fix4 0.9457 / 1.17x**. hf14 14 q0 is a model miss (the
  right elevation's top band, same hatch as the reference), not a label question.
- **PRODUCT_V3 (cell snap, `polygonize.snap_to_cells`)**: valfix4 **0.9511 / 1.06x** (target met), hf14fix4 0.9473 / 1.24x
  (0.0027 short, vertices just over); +0.0002 Gemini30, +0.0019 gemdev. A per-question eps (val has vertex headroom, hf14 not)
  is the open knob.
- **GPU checks 2026-11-11**: flip TTA does not beat the 3-scale blend at equal compute; v3 holds on gk-x / gke-x (valfix4 0.9511
  on all three); latency ~1.16 s -> ~1.47 s per query (CPU post-processing dominates); a high-resolution fine-tune (3072-4096)
  fails on HF14 (blend -0.011) like `--hires-crop` before -- do not retry training at large sizes without a new idea.
- **Next**: (1) the user reviews val 5 q2 (darker porch roof, same ambiguity as val 9 / 10) in Roboflow;
  (2) port the configuration into `polygonize.py` (eps / offset options) and the evaluator (three scales) and score end to end;
  (3) model work for the rest of hf14fix2 (cast shadows on 2 q1; 27 q2) -- labelling more real / Gemini sheets is the only
  training lever that has ever moved results beyond seed noise; (4) a per-sheet eps if Gemini-like sheets must also stay <= 1.2x.
- Probes: `scripts/pp_probe/{lvdecomp,edgeoff,msweep,ensemble,tune3s,render_q}.py`; maps 3072 / 5120 / 6144 for gem2s_x and
  gemdev 2048-5120 for xblend on floz-diag.

## 2026-11-09: vertices are now a primary metric

The user: polygons with ~3x the labeller's vertices are a product failure even at 0.92 IoU (PROJECT_UNDERSTANDING.md). Report
vertices / label (`scripts/pp_probe/vertcount.py`, or `curve.py --summary`) next to IoU.
- **Training for vertices failed**: `--edge-weight 4` and `10` (gem_s7 recipe) leave vertex count and boundary error unchanged
  (synth_progress 2026-11-09). Do not retry loss reweighting near the true boundary.
- **The trade-off curve** (`scripts/pp_probe/curve.py`): Douglas-Peucker eps is the lever -- eps 4 cuts the median ratio 30-40%
  for ~-0.002 IoU, eps 8 gets ~1.5x on every split for -0.006..-0.008. The p90 tail (10-26x, notches / holes) needs something
  else: hole filling and closing help real sheets but hurt Gemini30.
- **The target (the user, 2026-11-09): >= 0.95 mIoU on the real sheets (val, HF14 / fix v2) AND <= 1.2x vertices / label**, on
  the product output. Gemini30 is a check, no longer the target (PROJECT_UNDERSTANDING.md).
- **~1.2x is reached, and holds on unseen sheets**: fill irregular holes and inlets (convex-hull gaps) the model was unsure about
  (mean P >= 0.05), Douglas-Peucker 12 px + axis snap, drop pieces < 0.5% (`curve.py` "e .05 inlet+eps12+minf.005"): gemdev (xblend,
  unseen) 1.23x / -0.0082, Gemini30 1.17x / -0.0065, val 1.00x / -0.0089, HF14 1.15x / -0.0061. The tail was dropouts inside
  correct regions, not edges. (The adaptive-eps variant looked better on the eval sheets but did not transfer to gemdev.)
- **Ported, not yet scored** (VM ended): `scripts/polygonize.py` `probs_to_polygons(..., low_vertex=True)` (off by default; `fill_gaps`
  + `LOW_VERTEX` settings; identical to curve.py's candidate on 23 / 23 val questions). `scripts/pp_probe/stack.py` now also reports
  `poly_lv` and `ms+cc_lv` (gap fill before the colour cut) with vertices / label for every variant. First thing to run on a new VM:
  ```bash
  set -a; . ~/.env; set +a
  python3 -c "from huggingface_hub import snapshot_download as d; d('abshetty/floz-diag', repo_type='dataset', allow_patterns=['gem2s_x/*'], local_dir='data/floz-diag')"
  for s in gemini30 val hf14 hf14fix2; do PYTHONPATH=. python3 scripts/pp_probe/stack.py $s data/floz-diag/gem2s_x & done; wait
  ```
  Compare `ms+cc` vs `ms+cc_lv` (IoU and vertices / label); if the user takes the trade, flip `low_vertex` to the default.
- **Next**: (1) the user picks the trade from that table; (2) win back the ~0.005-0.009 IoU: a better gap classifier than shape + mean P
  (e.g. ink / texture match to the reference inside the gap); (3) 0.95 mIoU is a model problem -- ignoring 2 px around every true
  edge still gives Gemini30 0.91, so the rest is wrong / missing regions.

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
