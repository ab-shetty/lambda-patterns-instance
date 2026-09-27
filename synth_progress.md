# Synthetic Dataset Progress

Working summary. Every dated entry (2026-08-06 to 2026-09-26) is kept verbatim
in `synth_progress_archive.md`, Part 1, under its original heading; other docs'
references to "`synth_progress.md` (DATE)" resolve there. Append new results
below "Log".

## Current state (2026-09-27)

- **Best model:** a completed restart (`run_restart_swa.sh`, epoch 13) of
  `abshetty/floz-refunet-swint-mixr4-e8`, inference at 4096 (chosen on
  validation; 4096-5120 is the plateau): **HF14 0.8170**, vs 0.7887 for the
  published e8 at 2048. TTA x4 (+0.0015) and multi-scale inference (0.8123) not
  adopted.
- **Synthetic source: v6d** (`generate_synthetic_v6.py`, build seed 6,
  deterministic per `(seed, image_id)`). No generator change has beaten it on
  HF14: v6e look-alike pairs (negative), v7 Gemini-style retune (-0.023, one
  seed), `--hardscape-plan` / `--same-fill-subtle` (null, 3 seeds),
  `--mottle` (null, one seed), v5 confusable pairs (-0.104).
- **Best synthetic-only:** v6d 100k single pass, RefUNet at 2048, HF14 0.7130;
  swin_t on v6d 1,995: 0.7066 (`abshetty/floz-refunet-swint-v6d-e8`).
- **Realism probe** (vs val 14): best pool r8 0.909 against v6d 0.948 +- 0.013.
  FreeCAD geometry 0.954, Revit-style 0.960. None of these pools has been
  trained on.

## How to test a synthetic change

- **Synth-only screen** (`run_synth_only_screen.sh`, `run_vocab2_screen.sh`):
  swin_t at 1024, 1,600 sheets, 1+8 epochs, scored at epoch 8 with 2048
  inference; three runs concurrently take ~20 min on a GH200. Use seeds 7/31/99
  and deltas against the same-seed v6d arm (`so_v6d_s{7,31,99}`). The v6d
  control alone spans 0.640-0.694 on HF14 across seeds, so only a consistent
  3-seed gain counts.
- **Don't** use `run_synth_ft_ab.sh`: a gentle fine-tune of a converged model
  leaves 48/52 selections tied and cannot detect a pixel-only change. A full
  retrain (`run_nogray_mix.sh`-style) takes ~80 min.
- Choose epochs and inference size on validation
  (`select_epoch_on_val.py`); read HF14 once. HF14 is 52 fixed questions
  (reference boxes identical across runs), so compare **paired** per
  selection, and only against a same-recipe control: two base runs differing
  only by seed pair at p = 0.0001.
- Checkpoint averaging reads ~0.04 low when a run is still climbing at the last
  epoch; check that it turned over first.
- Transfer: use `scripts/question_difficulty.py`, not `fresh_synth_iou.py`
  ratios (different question mix). `fresh_synth_iou.py` decides "seen" by
  filename, and merged pools are all `item_XXXXXX`: pass an empty
  `--trained-pool` for known-disjoint splits.
- **Realism probe** (`scripts/synth_realism_probe.py`): frozen DINOv2 plus
  logistic regression on crops inside labelled regions, reference val 14 at
  3168 px (indistinguishable from HF14: AUC 0.423), mean of 3 seeds, 320 synth
  sheets (seed 6, ids 200000+). Differences under ~0.04 are noise. It measures
  "looks like these ~12 documents", not "looks like a real drawing" (real
  Boise Revit sheets score 0.998), so it ranks changes and its crop sheets
  show what's off, but it is not a training-value metric.

## Findings that still bind

- **Volume:** v6d-only at 2048: 1,600 -> 0.640, 12,000 -> 0.693, 100,000
  seen once -> 0.713. Gains shrink ~60% per 8x. With enough unique data, one
  pass beats many epochs; a second pass already lowers fresh-synthetic IoU.
- **Real-plan penalty:** net of question difficulty, every source (Gemini
  included) pays ~-0.14 to -0.17 on real plans. At matched 148 sources Gemini
  and v6d tie on HF14.
- **Where the error is:** target thickness at input resolution (thin < 74 px at
  2048: 0.437, mid 0.716, thick 0.825; the thin third is about half the missing
  IoU). Region identification, not boundary placement: boundary methods and
  threshold sweeps (flat, peak 0.35) give ~0. HF14 image 7's reference box sits
  on printed text (worth ~+0.011; don't chase).
- **Resolution:** `--domain-random` shrinks training sheets 0.4-1.0x, so
  inference above the nominal size helps (+0.009 on v6d-only, +0.028 on the
  real mix at 4096). Training at 2560 beat 2048 once (0.7834,
  `abshetty/floz-refunet-res2560-e4`); a second seed was never run.
- **Distinct labelled real sources** are the binding data constraint:
  re-augmenting the same 114 sources 18x/36x/72x, synthetic volume and mix
  ratio at mix scale are all null.
- **Closed levers** (all null or negative, detail in the archive):
  conditioning mechanisms (`--corr-grid`, `--scale-matched-ref`, ranking loss,
  self/dynamic prototypes, anchor; the anchor-plane bug is fixed, default
  `--anchor-ref-plane 0.0`), decoder width, confusable/look-alike synthetic
  pairs, same-fill recolouring, grayscale-free augmentation (the model does use
  colour), contrast normalisation, two-pass inference, small-reference
  training.
- **Backbone:** swin transfers best and fits worst, stable across LRs;
  material separability of frozen features does not predict transfer.

## Generator flags (`generate_synthetic_v6.py`, all default-off, v6d byte-identical when off)

| flag | status |
|---|---|
| `--hardscape-plan`, `--same-fill-subtle` | null, 3 seeds |
| `--same-fill-new-colour` | negative on the real mix (band worse) |
| `--mottle` | null, one seed |
| `--vocab2` | untested (`run_vocab2_screen.sh` ready) |
| `--gemini-colour`, `--muted-palette` | probe only; r8 uses `--neutral-palette` instead |
| **r8 set:** `--val-details --material-mix --val-fills --mode-weights 86,7,7 --res-degrade 0.15 --neutral-palette --fill-scale --real-labelling` | probe 0.909; untested on HF14 |
| `--tight-crop` | framing fix (ink box + pad); untested |

FreeCAD pilot (`scripts/generate_synthetic_fc.py`, `scripts/fc_massing.py`,
`scripts/revit_render.py --revit`): same houses as r8 with 3D-correct
geometry, optionally Revit-style rendering. Untested on HF14.

## Open

- GPU screen, synth-only, 3 seeds: v6d / r8 / r8 + ~40% Revit-style elevations
  / Revit-only. Pools follow `v6d_1600`'s recipe (`--n 2000 --seed 6 --start 0`,
  first 1,600, as for `--vocab2`) and take ~10 min each on CPU. FreeCAD must be installed where the pools are
  built.
- `--vocab2` screen; second seed at 2560.

## Log

### 2026-09-26 — realism probe, r8, FreeCAD, Revit (detail: archive Part 1)

- Probe v6d vs Gemini: 0.999. Candidate references are all far apart (eval 28
  vs scraped 86: 0.965 at 640; eval vs Gemini 0.954), so val 14 became the
  reference. It looks like modern CAD/BIM exports with colour markup.
- Flag rounds against val 14 (3-seed means): v6d 0.948, r2 0.919, r4 0.932,
  r6 0.964, r7 0.945, **r8 0.909**. The biggest single gain came from labelling
  like real annotators (`--real-labelling`). Blurring crops to 28 px only
  takes r4 from 0.93 to 0.86, so the gap is coarse content.
- `--tight-crop` came from the user spotting white space: val drawings fill
  92% of the image, v6 62%.
- Boise public-domain ADU sets (`data/reference/boise_adu/`, 6 bid sets): the
  six A201 sheets (elevations plus sections, one firm) score 0.998 against val
  14. Another real document family is trivially separable, so AUC < 0.7 would
  mean overfitting to the eval documents.
- FreeCAD geometry on r8's houses: 0.954. Geometry is correct but the probe
  looks at local rendering. Revit-style renderer (cast shadows from the solids,
  line-weight hierarchy, level datums, view titles): 0.960. Looks like a Revit
  set, but val 14's most typical crops are hand-drafted shingle and colour
  markup, not Revit. The user prefers the Revit look; train on it as a mix arm
  and a Revit-only arm.
- FreeCAD install (container-local, ~5 min): micromamba from
  `conda.anaconda.org/conda-forge`, then
  `micromamba create -p /opt/fc -c conda-forge freecad`.
