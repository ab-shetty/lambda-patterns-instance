# Handoff

Last updated: 2026-10-06

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds
every number, command and reproduction path. This file holds only what changed
and where to pick up — if a fact appears in one of those two, it is not repeated
here.

## Pick up here (2026-10-06, GH200 session -- read this first)

One seed (7), swin_t, 2048 training, ROI add, hand boxes @4096 unless noted. Detail: `synth_progress.md`
"2026-10-06 (GH200)". All checkpoints below are on the Hub (private, round-trip verified).

- **Best synthetic-only: HF14 0.8779** (`abshetty/floz-refunet-swint-longswiss-swa37-41`): 20-epoch restart
  of `...-longv1004-swa27-31` (HF14 0.8564) on `revitfailswiss_hard2000` (new defaults + Swiss floor plans,
  mined by `...-revitfailswiss2k-e8`), SWA 37-41 picked on val (0.8766). +0.0215 paired vs its start (p=0.21).
  It matches or beats every fine-tuned model on HF14 -- best fine-tuned is still 0.8734.
- **Real-mix fine-tune of it loses on HF14:** `...-longswiss-ft-realmixtx-e54` val 0.9004, HF14 0.8578 (-0.020 vs
  its start); it drops 23 / 25 / 27 (0.91/0.92/0.83 -> 0.73/0.82/0.72). Weight interpolation 30-50% toward
  the fine-tune gives HF14 0.883-0.884 = noise over 0.8779, and val rises monotonically toward the fine-tune,
  so val cannot pick the mix. **Conclusion: stop spending GPU on fine-tune variants; HF14 moves with synthetic.**
- **Gemini eval set (new):** 30 random Gemini sheets (`pool_manifests/gemini30_s20261006.json`), boxes hand-
  reviewed by the user (`eval_boxes/gemini30_v1.json`: 35 moved, 7 dropped, 194 questions);
  `evaluate_refunet_selection.py --local-pool data/eval_pools/gemini30_s20261006 --boxes eval_boxes/gemini30_v1.json`.
  Best synthetic 0.8152 @4096 (previous long 0.7972, 2k screens ~0.72). These sheets are IN the real mix: valid
  for synthetic-only models only. Review artifacts: Gemini Box Review (claude.ai/artifact/KugeRR3u46HnwocmYgmWfv),
  Gemini Failure Review with per-question notes (claude.ai/artifact/W5FqgEoTubFgJZwwiGDHPb) -- read the notes.
- **Where synthetic fails on Gemini:** lines over tone. r4 #078 (tan vs olive lap = 2 families) and r4 #041
  (grey-washed seam fill vs bare joist lines) both select by line texture and ignore fill tone / colour. Label
  issue: r4 #091 lumped roof + walls; the user fixed it in Roboflow but it needs a new r4 version (v2) to arrive.
- **Defaults changed:** `augment_local_dataset.py --gray-prob 0` (was 0.15; the user ruled grayscale copies out:
  they make colour-only families identical). Synthetic training never had grayscale.
- **New, opt-in:** `TX=1 ./run_build_realmix.sh` (Tesseract text regions -> `ref_exclude` for real / Gemini; the
  augmenter carries the rings; catches room tags / callouts, NOT fixtures / windows / door swings / railings --
  no automatic method found, AUC 0.58 for backbone features); generator `--paper` (keep 0: the user says the
  photographed look is unintended), `--pale-ink`, `--revit-view-weights`; `--tight-crop` now screened:
  tight-crop + pale-ink 0.25 + views 50,45,5 = Gemini -0.013 (auto boxes), HF14 +0.010, val +0.014 (noise) --
  framing is not the lever. `run_hard_mine.sh SRC=failswiss NOTRAIN=1`; `publish_refunet.py` parallel-safe.
- **Unfair training questions in the mined pool (inspected 2026-10-06):** training-style reference boxes on 200 base
  vs 200 mined sheets of `revitfailswiss_hard2000`: pieces whose largest box is < 32 px @2048 are 27.0% (base) vs
  35.7% (mined), < 16 px 10.3% vs 12.0%. In 40 random mined boxes ~1 in 7 is a question no user would ask: gable /
  porch slivers (mostly blank paper + one line, s1795 q1, s1755 q0, s1390 q34), boxes on trim or blank band edges
  (s1765 q5, s1000 q8), a generator room tag inside the box (s1040 q4 "BATH 2"). Fixes, before the next long run:
  (a) train with `--ref-min-side 24` (exists, off by default: references only from pieces fitting a 24 px square
  when the family has one; slivers stay in the targets); (b) generator writes its own text (room tags, callouts)
  into `ref_exclude`, as Tesseract does for real data; (c) optional: a box-size filter in `build_hard_pool.py`
  so slivers don't count as "hard" (not implemented).
- **Next, in order:** (0) fixes (a)-(b) above for every new pool / run; (1) generator: same line pattern, different fill tone / colour as separate labelled
  families on one sheet (measure how often pools have them; `--colour-pairs`, `distinct_looks`); grey-wash vs
  bare-line framing plans; (2) another long restart of longswiss-swa37-41 on a freshly mined pool (steps have
  paid every time); (3) done: r4 v2 generated (only #091 changed), wired into run_build_realmix.sh and the gemini30 pool; best synthetic on Gemini30 now 0.8255.

## Pick up here (2026-10-03 to 10-06, CPU session -- read this first; supersedes the GH200 "Next" list below)

Nothing trained. Evaluation and generator defaults changed; every trained pool predates them.

**What changed**
- **Hand boxes are the default** in `evaluate_refunet_selection.py` and the scoring inside
  `train_refunet.py` / `select_epoch_on_val.py` / `average_checkpoints.py` (`eval_boxes/hand_v1.json`;
  boxes not moved or dropped stay automatic). `--boxes auto` = the old protocol. Best model:
  **HF14 0.8738 hand** (0.8816 auto); shipped 0.8466; table in `startup.md`.
- **Eval speed-up:** `--q-batch 8` (default) runs the swin image backbone once per sheet / TTA view and
  batches questions (`RefSwinUNet.forward_cached`); identical IoUs to `--q-batch 0`, CPU 1.6-1.8x.
- **Generator defaults** (`GEN_VERSION 2026-10-04`, details `synth_progress.md` "Current state"):
  `--cut-standing 0` (posts stay in wall labels), `--plan-v2 2` (furniture; floor labels without
  fixed pieces), `--downspouts 0.05`, `--ref-exclude 1` (reference boxes never on posts / railings /
  stairs / downspouts / solid furniture / rugs; hidden pieces never references). Old pools:
  `--plan-v2 1 --cut-standing 1 --downspouts 0 --ref-exclude 0`.
- **Pool guard:** pools carry `generator_version`; `run_revit_2k.sh`, `run_hard_mine.sh`,
  `run_hard_long.sh`, `run_build_realmix.sh`, `run_fit_sweep.sh` STOP on a pool from older defaults
  (`scripts/pool_guard.sh`; `ALLOW_STALE_POOLS=1` to reuse one on purpose). The pools on the GPU
  disk (`revitfail_*`, `revitcur_*`, `realmix_revithard`, `fit_fresh150_*`) are all stale: move them
  aside first, e.g. `for p in data/synthetic/revit{fail,cur}_* data/synthetic/fit_fresh150_* data/mixed/realmix_revithard; do mv $p ${p}_pre1004; done`.

**Next, in order**
1. Pull; re-score the best model once (`startup.md` command, now hand boxes, expect 0.8738) and time
   `--q-batch 8` vs `0` on two sheets at 4096 (lower `--q-batch` on OOM).
2. **The 0.8738 recipe on new-default pools** (one bundled change: label + reference corrections),
   same flags as before: `run_revit_2k.sh 7 revitfailroi` (screen + miner), `SRC=fail NFRESH=4000
   SHARDS=8 ./run_hard_mine.sh 7`, `run_hard_long.sh` on the new `revitfail_hard2000` (as
   `ck_long_hardfail`, SWA 16-20), `run_build_realmix.sh` + `run_restart_swa.sh` with `--lr 5e-5`
   (`startup.md`). Compare late-epoch averages / SWA vs 0.8738 (hand boxes), not single checkpoints.
3. Then, one change each, paired: `--plan-source` (Swiss layouts, opt-in); a mining filter for
   tiny reference boxes (`synth_progress.md` "Tiny reference boxes": mining doubled sliver
   questions; not implemented); `--gray-prob 0` in the real mix.
4. HF14 25 / 27 (shingle roof vs unpainted vertical boards): the **real-mix fine-tune** loses what the
   synthetic start had (swa16-20, hand boxes: 27 q3 0.906 -> 0.479, 25 q3 0.949 -> 0.725, 25 q1
   0.899 -> 0.538). Worth trying in the step-2 fine-tune: a larger synthetic share in the mix, or
   fewer fine-tune epochs; a targeted generator option is secondary.
   Labels checked (real 86 + Gemini 168): they don't teach it, and no 25 / 27-type sheet is in the
   mix (drift; generated 28 checked too). Real / Gemini / generated training
   reference boxes are often unrealistic (slivers, labels over doors / posts) -- unaddressed.

The GH200 list's item (1) is superseded: `--masonry-base` is a default since 10-01, and
`--roof-lines` / `--railings` were already in the best model's pool.

## Pick up here (2026-10-03, GH200 -- read this first)

Detail: `synth_progress.md` "2026-10-02/03". **New best HF14 0.8816** (`startup.md`): synthetic
pretraining on mined Revit sheets with a long schedule, then a 5e-5 real-mix fine-tune.

- Synthetic-only best: 0.8668 (`...-longhardfail-swa16-20`); shipped 0.8440 re-scored exactly.
- **Hard mining** helps at 2k / 9 epochs (+0.037 averaged) and is null at a 30-epoch schedule;
  steps are the bigger lever. The four failure options help at 2k (+0.025) and don't stack with mining.
- **Noise at 0.85:** single-checkpoint differences sd ~0.024 (a few flip-prone questions);
  compare late-epoch averages / SWA weights (se ~0.0075), not single checkpoints.
- **Fit:** 64 mined sheets memorise to 0.843 in 60 passes and **0.906 in 240** (ceiling 0.95);
  wrong-family picks go to ~0, the rest is boundary. Grouping is steps / coverage, not the
  architecture. Width / selfattn / corr-grid were tested before (negative or parity).
- **Next, in order:** (1) targeted synthetic for the shared failures -- labelled masonry base next
  to a labelled shingle roof (18 / val 17, now ~2 in 120 sheets), vertical boards next to a
  horizontal-line roof and railings over lap (25 / 27); (2) `--gray-prob 0` real mix + the 5e-5
  fine-tune, paired vs 0.8816 (one clean change; the 09-24 no-gray run was confounded);
  (3) longer / larger synthetic pretraining; (4) 3072-4096 training for thin targets.
  Mined pools rebuild from `pool_manifests/`.
- Remaining failures: 18's roof vs masonry base and 27 / 25 vertical boards vs horizontal-line
  roof (every model); the fine-tune trades synthetic wins on line-only markup for real ones.
- Ops (cloud CPU sessions): Roboflow is authenticated by the session proxy -- call the REST API with
  NO api_key (`curl https://api.roboflow.com/perceive-ai/<project>/<ver>/coco-segmentation`, then the
  export link); passing a key gets 400 and the `roboflow` package fails. Hugging Face works the same.
- Ops: queue GPU jobs on files a job writes last, never on `$!` after `setsid` (it is the
  wrapper); one 2048 training fills the 96 GB GPU. FreeCAD massing now runs on `--workers`.

## Pick up here (2026-10-02, CPU session -- read this first)

Detail: `synth_progress.md` "Synthetic-only failures on HF14 (2026-10-02)". Nothing trained this
session; generator changes only, all on the branch.

- **Where synthetic-only loses to shipped** (Model Results artifact, hand boxes, 4096): sheets 27,
  14, 25, 2, 12. Four generator gaps, each now an opt-in flag (off = byte-identical): `--roof-lines`
  (plain horizontal-line roofs over vertical boards, 25/27), `--railings` (decks / balconies /
  stairs in front of siding, 27), `--faint-lines` (pale line-only textures, 14), `--soft-shadows`
  (lighter, neutral, two-level shade from a second sun through `wall_shadows`, 02).
- **Default changed:** `--openings-per-face 1` (windows / doors no longer cross a bay's corners;
  23% of Revit elevations had one). `0` reproduces older pools.
- **Opt-in, untrained:** `--plan-source` (real Swiss Dwellings floor layouts, CC BY 4.0 -- credit
  Archilyse AG); floor labels now leave out fitted pieces (counters, tubs, vanities, stairs) like
  the Gemini plans.
- **GPU runbook, in order** (one seed, ROI add, the 10k recipe, P2=5):
  1. `EXTRA="--roi-ref --roi-ref-mode add" P2=5 ./run_revit_2k.sh 7 "revit10kcurroi revit10kfailroi"`
     builds both 10k pools if missing (64 workers). `curroi` = today's defaults, the control;
     `failroi` = + `--roof-lines 0.3 --railings 0.15 --soft-shadows 0.5 --faint-lines 0.2`.
     The script prints both comparisons (failroi vs curroi = the four options; curroi vs
     `revit10k-roi-e5`, pulled from the Hub and scored = the default changes since 09-30) and
     per-sheet 4096 HF14 means for 25 / 27 / 14 / 02. Scores use the automatic boxes; add
     `--boxes eval_boxes/hand_v1.json` runs if comparing to the Model Results artifact.
  2. If the bundle helps: drop-one arms to see which option carries it. The values are judgment,
     not tuned: per measured 300 sheets they change roof-lines 30% of sheets, railings 15% of views
     (27% of sheets), faint-lines 18% of line-only (10% of sheets), soft-shadows 48% (two-level
     grey on 19%). Colour-pairs showed 0.5 vs 1.0 is within one-seed noise; tune only if needed.
  3. Then the 2026-10-01 lever below (longer schedule, gentle DR) on the winning pool.
- **HF14 label fix v1** (`eval_labels/hf14_fixes_v1.json`, `evaluate_refunet_selection.py
  --label-fixes`): sheet 14's lap panel under window 09 was missed in labelling (labeller
  confirmed). The fix adds target pixels only -- every box and the question count are unchanged,
  and scores without the flag are exactly the old ones. `run_revit_2k.sh` now also writes
  `*_hf14fix.json`; report both for a while. Re-scored numbers: see `synth_progress.md`.

## Pick up here (2026-10-01, GH200 -- read this first)

Detail: `synth_progress.md` Log 2026-10-01. Goal still 0.90 on HF14, synthetic-only.

- **Training longer is a real lever.** 500 Revit sheets (>= 3 families) trained ~11k steps
  (1+60 epochs + a 30-epoch restart, swin_t, ROI add) reach HF14 0.822-0.830 @4096, above
  `revit10k-roi-e5` (0.8126), though val is lower (0.81-0.83 vs 0.886). On the Hub (private,
  round-trip verified): `abshetty/floz-refunet-swint-fit500-nodr-e60` (the restarts' start,
  HF14 0.7548 @2048) and `...-fit500-dr07-e90` (gentle-DR arm, HF14 0.8258 @4096).
- **Next:** that schedule with `--domain-random --dr-scale-min 0.7` (best val of the three
  arms) on more sheets: 2k from scratch (~4.3 h) or continue e90 on 2k (~1.5 h).
- `--ref-sample instance` (off by default) trains on the evaluator's per-instance question
  mix; it closes the synthetic seen/fresh gap but does nothing on HF14 (~1 piece per family).
- Hand-placed boxes (`--boxes eval_boxes/hand_v1.json`) move no model significantly on HF14.
- Box setup: `pip install --user -r requirements.txt`, FreeCAD via
  `/opt/miniconda/bin/conda create -p /opt/fc -c conda-forge --override-channels freecad`
  (aarch64 works), and `set -a; . ~/.env; set +a` before training (the per-epoch HF14
  diagnostic needs `HF_TOKEN`). Never `pgrep -f` / `pkill -f` a pattern that appears in the
  calling command; it matches the calling shell. Wait on PIDs.

## Pick up here (2026-09-30, CPU session -- read this first)

Detail: `synth_progress.md` Log "2026-09-30 (CPU session)". Goal for now: 0.90 on
HF14 with synthetic-only training (best: `revit10k-roi-e5`, 0.8126 @4096).

- **Where that model loses HF14:** markup sheets 18, 27, 25 (48% together), line-only 14
  (17%), plan 12 (14%). The markup-sheet failures are look-alike textures (unpainted
  vertical siding vs a horizontal-line roof; grey roof vs brick band), not the paint.
  (An earlier version of this note said "half the loss is markup" and put markup
  training first; markup sheets are 46% of the questions and paint is not the failure.
  `--markup` exists, off by default, not recommended.)
- **Inference levers are exhausted** (threshold, 2048+4096, synthetic-only ensembles,
  shipped+10k ensemble): none helps on val or HF14. 0.90 needs training changes.
- ~~The 10k model underfits multi-family sheets~~ -- **wrong (2026-10-01)**: on training-style
  questions it scores 0.948 on its own sheets. The gap was the evaluator asking once per
  piece (Revit families are split into many slivers). HF14 has ~1 piece per family.
- **GPU runbook, in order** (one seed, paired vs the published control):
  0. (Premise withdrawn 2026-10-01; `run_fit_sweep.sh` not needed as written.) Fit checks on the 10k ROI recipe: 1+10 epochs instead of 1+5, then `--model swin_s` /
     `swin_b`; score seen (pool ids 0-399) and fresh sheets by family count
     (`evaluate_refunet_selection.py --local-pool`).
  1. `--roi-ref-mode replace` on the 2k screen (line-only look-alikes, item 3 below).
  2. `SYN=v6fix ./run_revit_mix.sh 7` (+ stage 2 as in the script header) -- does v6d's
     real-mix edge on line-only survive fixing its labels (no trim questions)?
  3. 4096 training for thin features (12's L band, 14's eave band under-selected).
- CPU tooling: `evaluate_refunet_selection.py --save-probs / --local-pool`,
  `ensemble_selection.py`, `failure_panels.py`; `train_refunet.py` falls back to CPU
  (smoke tests / tiny fine-tunes only, ~10 s/step at 1024).

## Pick up here (2026-09-30)

Detail: `synth_progress.md` Log 2026-09-29/30. Goal is still 0.90 on HF14; shipped
is still 0.8440. The VM is gone; models are on the Hub (all private):
`...-revit10k-roi-e5` (best synthetic-only, val 0.886 @4096), `...-revit2k-roi-e8`
(screen control), `...-revitmix-roiadd-e10` (Revit in the real mix: HF14 0.8367,
not adopted). Pools rebuild byte-identically from the committed generator.

1. **Reference-jitter fix is the default** (728def3): the reference gets the
   sheet's `--domain-random` brightness/contrast. Old runs: `--legacy-ref-jitter`.
2. **Revit synthetic helps synthetic-only a lot but not the real mix** at a
   1,600-sheet swap (floor plans up, roof plans and line-only elevations down).
   Untried: more Revit sheets in the mix, ROI from stage 1.
3. **Next lever: the reference path on line-only look-alikes.** ROI (add) beat the
   crop-only model on those questions (0.606 -> 0.695 on fresh plans); try
   `--roi-ref-mode replace` on the 2k screen (keep `--domain-random`).
4. Screen hygiene learned the hard way: read val 17 over epochs 4-8; one-seed
   synthetic screens swing ~0.05-0.07 on HF14; never run GPU evals beside training
   (OOM); launch long chains with `setsid nohup`.

## Pick up here (2026-09-28)

Detail: `synth_progress.md` Log 2026-09-28. Goal is 0.90 on REAL (HF14).

1. **HF14 0.8440, one seed** (`abshetty/floz-refunet-swint-mixr4-roiadd-swa15`,
   infer 4096, round-trip verified): the shipped restart recipe + `--roi-ref
   --roi-ref-mode add` (reference prototype also pooled from the image's own
   features inside the user's box). +0.027 paired over 0.8170, p=0.017.
   No seed replicate: one seed per arm from here (user's call). Recipe: `CK=data/runs/ck_mix_roiadd_dr
   EXTRA="--roi-ref --roi-ref-mode add" ./run_restart_swa.sh` on
   `v6dmix_plus_r4` (build it with `run_best_mix.sh`'s data section).
   - **Keep `--domain-random`**: the same run without it val-selects a val-17
     spike (HF14 0.8583 by protocol) but loses to DR at all 12 candidates once
     sheet 17 is excluded, and degrades over the restart.
   - **Six-question training on the mix is negative** (0.8165, 7 better / 24
     worse vs 0.8440).
2. The ~0.74 own-plan "wall" was the magnified crop reference (tone matching)
   and the DR shrink, not the model's capacity: ROI + no DR fits 64 plans to
   0.901 and 2k plans to 0.887. No DR breaks small real boxes, though.
3. Remaining real loss is over-selection (about half on both splits) and val
   17. The box-size probe rules out reference information: bigger / longer
   boxes inside the same instance leave the 15 failing questions at 0.278
   (vs 0.277). It is look-alike discrimination -- next session's lever. Small boxes are NOT a loss source for the shipped model (0.827 on
   the 21 under 50 px): do not add a minimum box size to the UI.

## Pick up here (2026-09-27)

Detail: `synth_progress.md` ("Current state", "Generator flags", Log
2026-09-27). The VM is gone; pools rebuild from committed scripts (FreeCAD:
see the 2026-09-26 log entry for the install).

1. **Revit-style synthetic is the first generator change to beat v6d** on
   swin_t: 2k synth-only @2048, HF14 0.7564 vs 0.7066, val 0.7568 vs 0.6702
   (`abshetty/floz-refunet-swint-revit2k-e8`). It is the renderer, not the
   FreeCAD geometry. One seed, and at 4096 inference the lead is only +0.016.
   With `--refs-per-image 6` (4 epochs): val 0.7816 / HF14 0.7607, the best
   synthetic-only result (`abshetty/floz-refunet-swint-revit2k-sixq-e4`).
   **Next: swap it into the real + Gemini mix** -- the user deferred this.
2. **Screen synthetic changes at 2048, not 1024.** The 1024 screen flipped the
   sign of r8 on validation. `run_revit_2k.sh` is the recipe.
3. **r8 and `--real-labelling` hurt.** r8 -0.076 HF14; `--real-labelling`
   unlabels the foundation band val 17 asks about (-0.187 val). The realism
   probe ranked r8 best and Revit worst -- it does not predict training value.
4. **swin_t underfits grouping at ~0.74** (own training plans, HF14 protocol,
   ceiling 0.904); v6d volume flattens toward ~0.77. `--refs-per-image 6`
   reaches the 16k level with 2k plans but not past the wall, and doubling
   the steps adds only +0.02 (own-plan fit 0.765). The limit is the model:
   `--swin-decoder selfattn` (code in, never trained) is the next test
   (`DEC=selfattn TAG=selfattn ./run_multiref.sh 7 v6d`).

## Pick up here (2026-09-24)

Detail and every number: `synth_progress.md` (2026-09-23 and 2026-09-24
entries). The machine is gone; everything below is either on the Hub or
rebuildable from committed scripts.

**New best: HF14 0.8170**, trained at 2048, validation-selected:
`abshetty/floz-refunet-swint-mixr4-restart-e13` (published and round-trip
verified). It is the 0.7887 model (`...-swint-mixr4-e8`) plus one completed
fresh-cosine restart (`run_restart_swa.sh`), evaluated at **inference 4096**.
Validation chose both the epoch (e13 of the restart) and the size (4096 is the
plateau: 2048 .785 / 3072 .786 / 4096 .801 / 5120 .799 on the published e8).
Most of the +0.028 over 0.7887 is inference size: `--domain-random` shrinks
training sheets to 0.4-1.0x, so a "2048" model is trained on ~1430 px sheets
and thin targets need the larger input. **Every HF14 number in this repo is at
2048 inference unless it says otherwise -- compare like with like.**

**The user's direction: improve real mIoU by changing the synthetic data,
driven by looking at failures, not by training tricks.** What looking found
this session, and what did not work:
- Realistic-looking v7 (`generate_synthetic_v7.py`): null; train->real ratio
  identical to v6d. The "universal ~0.85 ratio" was a protocol artifact of
  `fresh_synth_iou.py`; use `scripts/question_difficulty.py` (HF14 protocol on
  synthetic plans, regression net of difficulty): every source, Gemini
  included, pays ~-0.15 on real plans.
- Target thickness predicts failure (thin < 74 px at input: 0.44 vs 0.83).
- Three hypotheses were built on and then disproved by LOOKING at the actual
  reference crops / a 2-min intervention probe -- do the probe first: tiny
  blurry references (they are sharp), colour blindness (the model uses colour;
  val 17's teal band and grey roof are just close), grayscale augmentation.
- Fill-vocabulary audit (one crop per real family vs every v6 fill) is the
  most productive tool found: v6 `stone` looks like running bond (taught HF14
  0's flood), no fill looked like random ashlar, real masonry/roofing is
  TEXTURED per unit with light mortar, real lap siding has 3D shadow bands,
  basketweave/parquet absent.
- Synthetic knobs added to `generate_synthetic_v6.py`, all default-off (v6d
  byte-identical, verified after each): `--same-fill-new-colour`,
  `--same-fill-subtle`, `--hardscape-plan` (with new `ashlar` fill and
  direction-turning decks), `--mottle`. Tested by `run_synth_ft_ab.sh`
  (gentle fine-tune from restart e13, control vs treatment differing only in
  the synthetic quarter): hardscape+subtle null, ashlar-hardscape dose 1.0
  -0.013, `--mottle` indistinguishable (48/52 selections tied). **That
  fine-tune A/B is too weak to detect a synthetic change** -- none of these
  knobs has had a fair test yet.

**Next:** finish the fill-vocabulary fixes the audit listed (3D lap shadows,
basketweave/parquet), and test each ALONE with a FULL retrain against a same-recipe
control (or synth-only at 1024, three arms concurrently, ~30 min) -- not
`run_synth_ft_ab.sh`, which barely moves predictions. `./run_synth_only_screen.sh`
(synth-only, 1024, arms concurrent, ~20 min) is the fast fair screen: results:
mottle null; hardscape-only and subtle-band-only both null on HF14 over three
seeds (-0.003 / +0.003 mean). Synth-only seed noise is ~0.05 on HF14 -- screen
with >= 3 seeds or not at all.

## Pick up here (2026-09-21)

Numbers in `startup.md` ("2026-09-21 — swin_t meets real data"). What to know:

**Never cite a number without checking which backbone produced it.** RefUNet and
swin_t respond to data in opposite directions (real data: +0.1235 vs +0.0337;
synthetic training raises RefUNet's HF14 and lowers swin_t's). Four separate
errors this session came from extrapolating RefUNet results onto swin_t — the
volume curve, the 16-epoch null, the August realism table, and the r4 mix null.

**New best: swin_t, 282-source mix, 2048 = 0.7887** (`data/mixed/v6dmix_plus_r4`,
`run_v6r_ab.sh`-style recipe, one seed). **Published: `abshetty/floz-refunet-swint-mixr4-e8`**
(`scripts/hf_ckpt_to_pth.py`; reproduces 0.7887 exactly, verified 2026-09-23) --
check the Hub before rebuilding anything; every `floz-refunet-*` repo is listed by
`HfApi().list_models(author="abshetty")`. Over the 0.7834 record. On the
194-source mix the same backbone scored 0.7403 and looked worse than RefUNet, so
the mix was the deficit. **START HERE: val-selection picked the final epoch with
HF14 still rising — warm restart from `epoch_8.pth`, then a second seed.**

**Realism-by-metric is 5-for-5 null.** Region scale was the largest measured gap
(2.7x) and matching it to 2.4% bought +0.0024. The user's read, which the record
supports: the only synthetic source that ever paid (Gemini) was made to look
right, not to match statistics. Next realism attempt should be driven by visual
inspection, not by matching a measured distribution.

**Gemini source count is the lever.** 80 → 168 sources: +0.112 on identical
training fit. 194 → 282 in the mix: null on RefUNet, **+0.048 on swin_t**. It has
paid at every scale tested on the right backbone. Value tracks the source ratio,
so the next +0.04 needs ~409 total sources (~127 more plans). Labelling more
Gemini is the highest-confidence use of hours; the user's plan is to drive the
next generation round by visual inspection rather than metric-matching.

**Tooling.** `./watch.sh [-w]` tracks every run with no arguments (finds them via
live process cmdlines, not log names). `generate_synthetic_v6.py` has
`--view-count-weights` / `--max-label-fams`. `fresh_synth_iou.py` reports a
valid-pixel number — quote that one. Use all 64 vCPUs; `augment_local_dataset.py`
is serial and must be chunked (`startup.md` "Use all 64 vCPUs").

## Pick up here (2026-09-20)

Five facts and one command. Detail in `synth_progress.md` (2026-09-20);
per-arm numbers in `startup.md`.

**1. A vision-transformer backbone beats ResNet50 — the biggest lever found in
months.** `refmask2former/ref_swin_unet.py` swaps the backbone and changes
nothing else. `run_backbone_ab.sh`, v6d-only 1,995 plans @2048, documented
9-epoch schedule, unfrozen, val-selected epoch, only `--model` differs:

| arm | hard train IoU | HF14 |
|---|---:|---:|
| RefUNet (ResNet50, 28.0M) | 0.6811 | 0.6359 |
| **RefSwinUNet (swin_t, 31.4M)** | **0.7826** | **0.7066** |

Paired over the 52 fixed selections: **+0.0707, t(51)=2.20, p=0.032**. The
control reproduces the documented `v6d-only @2048 = 0.6400`. **0.7066 with zero
real plans** beats the best synthetic-only on record (0.686) and matches
`mix5092 @1280` (0.7097). It wins where the deficit is: image 14 **+0.142**
(35% of the total) and image 7 **+0.726**; it loses on 2 (−0.171) and 27
(−0.100). **One seed — replicate before adopting.** Published:
`abshetty/floz-refunet-swint-v6d-e8`, control `...-resnet50-v6d-e8`.

**2. swin_b is worse than swin_t, and the gap widens with budget.** 248 steps:
0.4362 vs 0.4385 (tie). 744 steps: 0.5343 vs **0.5744**. Not slow convergence —
90.8M buys nothing over 31.4M and costs ~1.7x per step. Scale swin_t.

**3. The residual is region-level, not boundary-level.**
`scripts/residual_decomp.py` (reproduces the published HF14 mean to 4 decimals).
Boundary is ≤21% of all error everywhere; wrong-region-selected plus
unfilled-interior is 46–56%. Output stride, mask threshold and boundary methods
are all dead ends — optimised stride-4 and stride-1 both score 0.9798 on HF14,
and the threshold sweep is flat at 0.35. **Attack `false_region`.**

**4. Cheap screening ranks TRANSFER, not FIT.** Cache the frozen backbone once
(`cache_backbone_features.py`), then decoder/backbone variants train in 40–90 s
(`decoder_search.py`). It predicted the unfrozen A/B to within 0.012. But five
identical runs give train sd **0.0032** and HF14 sd **0.0243** — the freeze buys
speed, not statistical power, and its fit column inverts once you unfreeze
(frozen: swin fits worse; unfrozen: swin fits better).

**5. RETRACTED: "the existing architecture reaches 0.9866 train IoU, so
capacity is not the limit."** That was 200 fixed images, augmentation off,
frozen backbone, 120 passes — a CNN memorising a static pool. Real pipeline,
`--domain-random`, hard 0.35: 2,000 plans/7,500 steps → **0.5400**; 10,000
plans/7,500 steps → **0.6954**; 100,000/25,000 steps → 0.8022 (on record).
Train ≈ fresh under augmentation, so this is function-learning, not
memorisation — which is why *more* data at a matched budget *raises* it, and
why RefUNet's 0.80 is where its budget ran out, not a proven wall.
**A fit experiment on a few hundred images with augmentation off cannot answer
a 100k question.**

**The open question, and the command that settles it:** can any architecture
hold a high hard train IoU on 100k v6d? `./run_fit_100k.sh` — generates the
pool, gives `unet`/`swin_t`/`swin_b` a matched 25,000-step budget at 1024, and
prints hard train IoU. ~1 h per arm. The `unet` arm must land near 0.8022 or
something is broken.

**Also:** `abshetty/floz-refunet-synth100k-e1` is a byte-identical duplicate of
`floz-refunet-res2560-e4` — the 100k checkpoint behind the 2026-09-18
underfitting entry was never published and is gone. `publish_refunet.py` had
the same class of bug (stamped a Swin checkpoint "RefUNet"); fixed.

**Do not re-open:** decoder width/depth as a capacity lever (w256/w384 fit
*worse*); output stride, mask threshold, boundary methods; `run_capacity_probe.sh`
as written.

## Pick up here (2026-09-18)

An overnight session on a fresh GH200, rebuilt from a clean clone (every
pipeline count matched `startup.md`). The machine was killed at ~06:50; every
`data/runs/*` path below is gone. Two checkpoints were published:
`abshetty/floz-refunet-res2560-e4` and `abshetty/floz-refunet-synth100k-e1`.
**Correction (2026-09-20): the second upload carries the FIRST one's weights,
byte for byte. The 100k checkpoint was never published and is gone; every
number in this entry that came from it is unreproducible. See the 2026-09-20
entry.**
Full detail and tables: `synth_progress.md` (2026-09-18). **One seed per arm —
nothing below is adopted into the documented recipe.**

**The three things that matter, in order:**

1. **The binding term is synthetic→real transfer, ~0.11.** RefUNet on 100,000
   procedural plans, measured on 300 of the 1,401 held-out draws:

   | | fresh-synthetic | HF14 | train |
   |---|---:|---:|---:|
   | after 1 pass (e0) | **0.8250** | 0.7111 | — |
   | after 2 passes (e1) | 0.7980 | 0.7130 | 0.837 |

   Fresh-synthetic is already 0.83 after ONE pass: the generator is close to
   solved on unseen draws, so more of it cannot help. The ~0.11 gap to real is
   domain transfer and that is the ceiling on the synthetic route.
   **Note the second pass made fresh-synthetic WORSE (0.825 → 0.798) while
   train rose to 0.837** — overfitting onset at 2 repeats, so single-pass is
   the right regime and the train/fresh gap of 0.039 at e1 should NOT be read
   as pure underfitting (an earlier draft of this entry did; corrected).
   Whether capacity also binds is untested: at `--width 128` only ~2.4M of
   28.0M parameters sit outside the ResNet50 backbone. **`run_capacity_probe.sh`
   (width 128/256/384) was queued when the machine died — run it first.**

2. **Training at 2560 gave 0.7834 val-selected**, +0.038 over the identical run
   at 2048 and above the recorded best of 0.7594. One seed, under 2× sd, so not
   a result until replicated — but it is the strongest single lever found.
   It selects an early epoch (4), so schedule it short.

3. **Procedural volume pays past 2.5×, then decelerates.** v6d-only 1,600 →
   12,000 → 100,000 gave 0.6400 → 0.6934 → 0.7130 (+0.053 then +0.020 per ~8×).
   The recorded volume nulls tested 2–2.5× and were under-powered, not wrong.
   With enough unique data you need **steps, not epochs**: 100k seen twice beat
   12k seen eighteen times at fewer steps, and warm restarts are a small-data
   patch. Do not scale past ~100k — item 1 says the ceiling is now fit and
   transfer, not supply.

**Two methodological findings worth more than any single number:**

- **The 52 evaluation reference boxes are byte-identical across every run**
  (verified across epochs, runs and training sets). The eval is 52 fixed
  questions, so arms can be compared **paired per selection** rather than by
  means with sd ~0.02. That is much more power at the same compute and nothing
  in this repo has used it.
- **77% of the HF14 deficit is four images** (14, 12, 18, 7); the other ten sit
  at 0.83–0.99 where only boundary precision remains and four boundary methods
  already measured ~0.000. Median selection IoU is 0.816 against a 0.745 mean —
  a tail problem. Image 7 is the known reference-box-on-text artefact (~+0.011,
  do not chase).

**Open, in priority order (revised 2026-09-18):**

1. **Model capacity has never been tested, and it is the top suspect.**
   Train IoU on 100,000 unique plans with deterministic labels tops out at
   ~0.84. A correctly-sized model should fit far higher — the only irreducible
   floor is the look-alike families v6d injects on purpose. Two facts make
   capacity the obvious candidate:

   - At `--width 128` the model is 28.0M parameters of which ~25.6M is the
     ResNet50 backbone, leaving **~2.4M task-specific** for dense
     reference-conditioned prediction at 2048².
   - **MEASURED 2026-09-18, hard threshold 0.35, 300 training images**
     (`fresh_synth_iou.py` pointed at the training pool). This is the first
     valid train IoU in the project:

     | epoch-1 checkpoint | train | fresh synthetic | HF14 |
     |---|---:|---:|---:|
     | RefUNet (28.0M) | **0.8022** | 0.7980 | 0.7130 |
     | crossattn (27.9M) | **0.7668** | 0.7715 | 0.7042 |

     **Train ~= fresh for both** (+0.004, -0.005): zero generalization gap
     within synthetic, on 100,000 unique plans. Absolute fit is ~0.80 where a
     correctly-sized model on deterministic labels should reach ~0.95.
     **This is unambiguous underfitting and it is not a data problem.**
     Note this supersedes an intermediate "overfitting onset" reading, which
     came from comparing a soft-dice train number against a hard fresh number.

   - **Caveat on every OTHER "train IoU" in this repo (the ~0.84 figure, and
     all earlier entries):**
     it is derived from the SOFT dice term (`mask_loss`, `train_refunet.py`
     ~line 195, which uses `logits.sigmoid()` un-thresholded), so it is NOT
     comparable to the hard threshold-0.35 union IoU that every reported
     result uses. `startup.md`'s "Train IoU (derived from the Dice term)" and
     the 2026-09-17 "train mIoU 0.85->0.88" carry the same defect. **A valid
     hard train IoU has never been measured.** Measure it first — point
     `scripts/fresh_synth_iou.py` at training images — before concluding
     anything from a train/fresh or train/real gap.
   - **`RefCrossAttnUNet` is NOT a transformer and adds NO capacity**: 27.9M
     against RefUNet's 28.0M — *smaller*. It swaps cross-attention in for the
     conditioning block at the two coarsest scales of the same ResNet50+FPN
     CNN. Every "crossattn ties/loses" result in this repo is a conditioning-
     mechanism result at constant, very small capacity. **The transformer
     hypothesis has never actually been tested here.**
     **Partly retired 2026-09-20**: a hierarchical vision-transformer BACKBONE
     (Swin-B, frozen, ImageNet) now beats frozen ResNet50 by +0.0842 on real
     plans (5.5 sd, n=5). Still untested: a plain non-hierarchical ViT
     (DINOv2-class), and ANY transformer backbone finetuned rather than frozen
     -- which is where the "a ViT needs far more than 200 images" objection
     actually bites. See the 2026-09-20 entry.

   `./run_capacity_probe.sh 7` (width 128/256/384 → 28.0/39.6/58.3M, one epoch
   each at 1024 on the 100k pool, ~1h) is written and unrun. It is only a first
   step: it scales the decoder, not the backbone, so even width 384 leaves the
   ResNet50 untouched. A real test of "is the model the limit" wants a larger
   or attention-native backbone (ViT/Swin) at 100k+ single-pass, judged on
   **train IoU reaching ~0.95** before anything else is concluded about data.
2. **Second seed at 2560**, and whether 3072 continues the trend.
3. **Close the 0.085 fresh-synthetic → real transfer gap.** **Reframed
   2026-09-20: it is backbone-dependent, not purely a generator problem --
   Swin arms show ~0.000 on the same pool where ResNet50 shows +0.104. Fix the
   representation before spending another round on generator realism.**
   The original framing follows. This is the binding term on the synthetic
   route, and it is a generator-realism problem:
   the 2026-09-17 audit's open items (texture irregularity, implausible
   material colours, the 2-story label-collision bug) are unaddressed.
4. **crossattn on 100k ties overall but WINS where it matters — chase this.**
   Paired per-selection on the 52 fixed questions (same pool, steps, seed):
   mean diff +0.0088 for RefUNet, **t = +0.36 — a tie**. But the per-image
   split is large and structured:

   | crossattn better | | crossattn worse | |
   |---|---:|---|---:|
   | img 7 (1 sel) | **+0.616** | **img 0 (2 sel)** | **−0.677** |
   | **img 14 (9 sel)** | **+0.100** | img 18 (10 sel) | −0.046 |
   | img 16, 2, 12, 24, 1 | +0.001..+0.034 | img 27, 25, 11, 23 | −0.009..−0.042 |

   It gains on **image 14 — 36% of the entire HF14 deficit** and the
   long-range/wrong-material failure — and recovers image 7's text-crop
   reference. The whole tie is paid for by **one catastrophic collapse on
   image 0** (0.861 → 0.185). Diagnose that single failure and the
   architecture is ahead. This is the first evidence that attention conditioning
   helps precisely where the propagation diagnosis says it should.

   Headline numbers, for the record: HF14 val-selected 0.7069 vs 0.7130;
   fresh-synthetic 0.7951 vs 0.8250 (1 pass), 0.7715 vs 0.7980 (2 passes) —
   so it fits the generator worse while matching on real plans.

   Superseded note: crossattn at matched data — after one pass
   each: fresh-synthetic 0.7951 vs 0.8250, HF14 0.7067 vs 0.7111, train_loss
   0.5360 vs 0.5072. It fits *worse*, consistent with a weak-prior model still
   being under-served at 100k rather than with an architecture win. (An earlier
   draft claimed it fit better, from a single noisy tqdm batch — wrong.) Its
   epoch-1 number did not finish. A fair test needs a larger data regime again,
   or more capacity.
5. More labelled real sources — still the structural constraint for 0.9.

**Do not re-open:** everything in the 2026-09-17 list, plus — new — synthetic
volume beyond ~100k plans, and warm restart #2 (both val-selection and
checkpoint-averaging chose windows inside restart 1 on every run that had two).

## Pick up here (2026-09-17)

Four things happened this session, on a machine that no longer exists — every
`data/runs/*` checkpoint named below is gone; only what's written here and in
`synth_progress.md` survives. **Nothing below is adopted into the documented
recipe yet.** `synth_progress.md` was also reorganized: the pre-2026-08-06 /
retired-query-model material is now `synth_progress_archive.md`, actually moved
out this time rather than relabeled in place.

**1. A fourth Gemini round (`floz-gen-gemini-r4`, 88 images) is a clean null.**
Merged with r2+r3 into `floz-gen-gemini-r234-clean` (168 images), augmented 18x
(3,024 records), substituted for r23 in the documented mix ->
`mix6676_with_r4` (6,676 records) vs. baseline `mix5092`, 2 seeds each, 2048px:

| protocol | mix5092 | mix6676_with_r4 | Δ |
|---|---:|---:|---:|
| val-selected | 0.7443 ± 0.0263 | 0.7366 ± 0.0000 | −0.0077 |
| averaged (SWA) | 0.7559 ± 0.0147 | 0.7702 ± 0.0172 | +0.0143 |

Sign flips between protocols, both gaps under 1x sd — noise, not signal. r4 is
stylistically indistinguishable from r2/r3 by `pool_style_stats.py` (same
generator, same prompt family) and the model already fits r2/r3/r4 equally well
in-sample (0.850/0.870/0.844 mean IoU) — it isn't under-fit, it's redundant.
Consistent with the existing finding that the Gemini pool's marginal value
already shrank to +0.010 (noise) at 2048px before r4 existed. **Do not re-run
this exact test; a fifth Gemini round would need to look different, not just be
more of the same.**

**2. The documented 9-epoch schedule stops before the real ceiling — found by
warm-restarting, not by a longer single schedule.** The existing "16-epoch
schedule is null" finding (`startup.md`) used one continuously-annealing
cosine. Instead: reset the optimizer and give a **fresh** cosine restart from a
converged checkpoint, repeatedly. On `mix6676_with_r4` seed 31 (one seed):

| epoch | train mIoU (full-mix sample) | val-complement IoU (14 img / 77 sel) |
|---|---:|---:|
| 8 (documented recipe's last epoch) | — | 0.7924 |
| 18 (+1 restart, 10 epochs) | 0.8704 | 0.7993 |
| **20 (+2 restarts, peak)** | 0.8615 | **0.8127** |
| 28 (+2 restarts, final) | 0.8762 | 0.8064 |

The epoch-20 checkpoint is published: `abshetty/floz-refunet-warmrestart-e20`
(private HF Hub repo, `scripts/publish_refunet.py`) — the machine that trained
it is gone, this is the only surviving copy.

Real gain through epoch 20 (+0.033 over the documented recipe's own epoch 8),
then a plateau: epochs 22-28 kept climbing on train (0.85->0.88) without val
following past the epoch-20 peak — the overfitting signature, just not a
blowup. A further low-LR (5e-6, well under the ~1e-5 floor every restart cosine
bottoms out at) sustained continuation from epoch 28 also plateaued (HF14 diag
0.69-0.75, no trend). **One seed. Needs replication before this changes the
documented recipe**, but it means the 9-epoch number in `startup.md` is
probably not this setup's ceiling, and checkpoint-averaging's "check whether the
run has turned over" caution (2026-09-08 entry below) applies to schedule length
too, not just pool composition.

**3. A cross-attention variant of `RefUNet` exists and ties it at both the
documented budget AND under extended training.** `refmask2former/ref_attn_unet.py`:
same ResNet50 backbone, same FPN, same direct-union training objective — only
the two *coarsest* scales' conditioning changes from global-average-pooled-
vector-plus-conv (`ConditionBlock`) to multi-head cross-attention (image tokens
as queries, reference tokens as keys/values, so a location gets a learned blend
of the reference's actual features instead of one broadcast vector). `--model
crossattn` on `train_refunet.py`. On `mix6676_with_r4` seed 31:

| | 9 epochs (documented budget) | +1 warm restart (epochs 9-18) |
|---|---:|---:|
| RefUNet | val-sel 0.7366 | val-complement 0.7993 (ep 18) |
| crossattn | val-sel 0.7314 | val-complement 0.8004 (ep 17) / 0.7992 (ep 18) |

Gaps of −0.005 and +0.001 respectively — a clean tie under two different
training regimes now, not just one. RefUNet went on to a second restart (peak
0.8127 at epoch 20); crossattn's second restart wasn't run — that's the natural
next step. The epoch-17 checkpoint is published:
`abshetty/floz-refunet-crossattn-e17` (same reason as above); the machine's
`epoch_18.pth` to continue from is not.

This is NOT the same territory as `--corr-grid` (screened negative, monotonic
with matching precision): corr-grid only ever produced similarity *scores* as
extra channels, never aggregated the reference's feature *values*. Full
attention did not regress the way corr-grid did — parity, not harm — so the
hypothesis that value-aggregation avoids corr-grid's overfitting-to-spatial-
correspondence failure is not falsified, just not yet confirmed as a win either.

**4. Visual quality audit of the synthetic generator — new, and it changes how
much weight the "generator quality" deprioritization should carry.** Full
write-up and 8 example images: `synth_progress.md` (2026-09-17 entry),
`synth_quality_audit/`. Every prior synth-vs-real comparison in this project
(ink, saturation, contrast, aspect, `pool_style_stats.py`) is an aggregate
statistic; nobody had looked at the images next to real ones until this
session. v6 (`generate_synthetic_v6.py`) is a real composition improvement over
v5 (the pool still in the documented mix) — but has a **confirmed bug** (2-story
level-mark label collision, ~line 1451, one-line fix) and systematically
implausible material colours (blue clay tile, blue cultured stone, pure-green
brick — real instances of these materials don't look like that). v6d's own
+0.028 val-selected (2026-09-08 entry) was measured on a generator carrying
these defects. **The "source count, not the generator" deprioritization in
priority #1 below was reached from statistics that miss this — it should be
read as weaker than it was written.** Cheap next step: fix the label-collision
bug and constrain colourisation to per-material plausible hue ranges, then
re-measure v6d before drawing a stronger conclusion either way.

**Open, in priority order (revised 2026-09-17):**

1. **Replicate the warm-restart extended-training finding** (item 2 above) at a
   second seed, and on `mix5092` too (not just the r4-inclusive mix) — this is
   the single biggest number this session produced and it is one seed.
2. **More labelled sources**, still the structural constraint — but see item 4
   above before treating "generator quality doesn't matter" as settled.
3. **Fix the two confirmed/likely v6 generator defects** (label collision,
   material-colour plausibility) and re-measure v6d on the mix. Cheap, and the
   prior null may not have been a fair test of the generator.
4. **Push resolution past 2048.** Unchanged from 2026-09-08: 2560 is +0.05 at
   one seed with averaging, Gemini's median long side is 3168px.
5. **Run crossattn's second warm restart** (item 3 above) — ties RefUNet
   through the first restart; RefUNet's peak needed a second.
6. Re-measure the Gemini pool's per-pool value at 2048 (unresolved, +0.033 at
   1280 vs. +0.010 at 2048) — today's r4 null is consistent with "already near
   its ceiling at 2048" but doesn't fully resolve it.
7. Regularization — train/HF14 gap widens with epochs, though item 2's finding
   complicates "fitting is not the constraint" somewhat: val DID follow train up
   for 10 extra epochs before plateauing, further than the 9-epoch recipe alone
   showed.

**Do not re-open:** connectivity, tile-similarity confusables, v6e look-alike
pairs, orientation snapping in the polygon regularizer, `--anchor`, SAM 3 as a
product-task model without fine-tuning it on that task, and — new this
session — adding a further same-style Gemini round without a genuinely
different generator or source (see item 1 above). MixUp (pixel-blending two
scenes) was reasoned through and not tried: it directly conflicts with the
project's repeated finding that fine local texture is the signal, so pixel
blending is expected to actively hurt, not just be neutral.

Older handoffs (2026-09-08, 2026-08-08) removed 2026-09-17: their substance was
fully duplicated in `synth_progress.md` (v6d, checkpoint-averaging-plateau
caution, connectivity/disconnection findings, ranking loss, template matching)
and `labeling_assist.md` (SAM 3 fine-tuning, its encoder-resolution and
zero-shot limits), and their priority/do-not-reopen lists were superseded by
the 2026-09-17 ones above. The two facts that weren't preserved elsewhere —
image 14's outsized weight in HF14 (9/52 selections, 0.291 mean) and a 2026-08
A10 GPU memory baseline — are now in `synth_progress.md`; the A10 note was
dropped as stale (measured at 1280px/19.3GiB, the documented recipe is now
2048px/~51GB).

## Facts that live elsewhere

Single copies, so they cannot drift. Do not restate them here.

| what | where |
|---|---|
| task semantics, metric definition, what does *not* count as evidence | `PROJECT_UNDERSTANDING.md` |
| the `sample_reference_box` fix and why pre-2026-08-06 numbers are incomparable | `PROJECT_UNDERSTANDING.md`, mechanism in `startup.md` |
| current result, both selection protocols, the checkpoint and its artifacts | `startup.md` |
| every reproduction command, the data rebuild, the validation split | `startup.md` |
| evaluation rules and the fixed HF14 indices | `startup.md` |
| per-experiment history and the screened levers | `synth_progress.md` (pre-2026-08-06 / retired query-model material: `synth_progress_archive.md`) |
| the labelling-assist model, its numbers and its negatives | `labeling_assist.md` |
| the synthetic generator's visual quality audit and example images | `synth_progress.md` (2026-09-17), `synth_quality_audit/` |

Two standing traps: the reference resize in `refmask2former/dataset.py` looks
like a bug and "fixing" it costs 0.153, and `--corr-grid` / `--scale-matched-ref`
are implemented but screened negative — read the warnings before touching either.

## Relevant implementation

- `refmask2former/ref_unet.py` — shared backbone, conditioning blocks, FPN mask.
- `refmask2former/ref_attn_unet.py` — cross-attention conditioning variant
  (2026-09-17, item 3 above); `--model crossattn` on `train_refunet.py`.
- `scripts/publish_refunet.py` — publish a `RefUNet`/`RefCrossAttnUNet`
  checkpoint to the HF Hub as safetensors, mirroring `publish_sam3.py`'s
  rationale: `data/runs/` doesn't survive the machine, a Hub repo does.
- `refmask2former/dataset.py` — `sample_reference_box` (fixed) and
  `sample_reference_box_legacy`; `render_instance_mask` hole convention.
- `scripts/train_refunet.py` — union targets, BCE + Dice, continuation
  checkpoints, per-epoch HF14 diagnostics, explicit schedule length and
  optimizer reset.
- `scripts/evaluate_refunet_selection.py` — authoritative 52-selection eval.
- `scripts/visualize_refunet_selection.py` — the required visual audit.
- `scripts/hf_to_local.py` — HF parquet → local-data format.
- `scripts/roboflow_to_local.py`, `augment_local_dataset.py`,
  `merge_local_datasets.py` — deterministic real-data pipeline.
- `scripts/family_similarity_probe.py` — per-family appearance similarity:
  `--metrics` (per-image confusability, intra/inter margin, correlations),
  `--tiles` (tile-pool pairs; writes the table `--tile-sim` consumes),
  `--local-data` (pool distribution at training resolution).
- `scripts/generate_images_openai.py` — resumable generation of the unlabelled
  realistic pool on `gpt-image-2`, arbitrary `--size`, receipts carrying the
  billed token usage, and a contact sheet for the visual gates.
- `scripts/upload_unlabelled_roboflow.py` — validates a batch whole, uploads it
  unlabelled to a named project (`--create` to make one), and disposes of a
  review project again with `--delete-project`.
- `scripts/render_image_generation_prompts.py` — `--version v4` is the current
  prompt set, rendered from `specs_v4.jsonl`; v1-v3 are kept so their results
  stay reproducible and v1's recorded SHA still validates.
- `scripts/build_specs_v4.py` — the v4 subject list, built to the evaluation
  set's measured distribution and weighted toward the failure traits.
- `scripts/pool_style_stats.py` — aspect, colour, ink, contrast and fill
  regularity for a pool, so a generated round can be compared with the real 28
  instead of judged by eye.
- `generate_synthetic_v6.py` — the v6 synthetic generator: a house grammar
  (blocks, roofs, dormers, porches, townhouse rows) projected into elevations,
  roof plans and floor plans, with ~20 materials drawn as continuous ruled
  fields in feet. Defaults are the v6d round; the v6e knobs that measured
  negative are kept as named constants with their results in comments.
- `run_synth_v6.sh` — reproduces any synthetic-only arm end to end (pool,
  merge, two-phase train, val-selection, averaging). `./run_synth_v6.sh
  headline 7` is the 0.686 result.
- `scripts/cache_backbone_features.py` — runs a frozen backbone ONCE over a
  pool and stores `c1..c4` for image and reference, padded to one grid with a
  validity mask. `--backbone {resnet50,convnext_base,convnext_small,swin_b,
  swin_t}` via `StagedBackbone`, which exposes all of them as the same
  stride-4/8/16/32 pyramid; `--source {local,hf14,val14}` so real plans cache
  as the 52 fixed questions.
- `scripts/decoder_search.py` — trains reference-conditioned decoder variants
  against that cache (seconds each, not hours): width, depth, dense
  correlation, cross-attention, real self-attention. Reports hard train IoU,
  fresh-synthetic and cached-HF14 transfer.
- `scripts/label_ceiling.py` — what the best possible model with this output
  geometry and resize chain could score, naive and convex-optimised. Run it
  before treating any fit target as reachable.
- `scripts/residual_decomp.py` — splits missing IoU into boundary / missed
  region / missed interior / wrong region / fringe, plus a threshold sweep.
  Says WHICH axis to spend on.
- `scripts/paired_compare.py` — compares two arms PAIRED over the 52 fixed
  selections, with a paired t, a sign test and a per-image share-of-total
  breakdown. `startup.md` has recommended this since 2026-09-18 and nothing
  had used it; it resolves an epoch-to-epoch step at t=7.3 where unpaired
  means at sd 0.026 would need many seeds.
- `scripts/material_separability.py` — within-image same-vs-different family
  separability of a frozen backbone's features. No training; an upper bound on
  what any decoder reading those features can group.
- `scripts/hf_ckpt_to_pth.py` — rebuilds a loadable `.pth` from a published
  safetensors + config.json pair.
- `scripts/average_checkpoints.py` — averages the last epochs of a run and
  ranks the window on validation; the selection protocol as of 2026-09-07.
- `scripts/select_epoch_on_val.py` — val-selected protocol; one process per run,
  run them in parallel. `--tta {1,2,4,8}` for dihedral test-time augmentation
  (measured +0.004, not worth 8x inference).
- `scripts/train_status.sh` — one-screen progress for every run under
  `data/runs`, including live epoch and running peak.
- `scripts/run_*.sh`, `run_*.sh` — the experiment drivers, one per ablation.

## Data limitation and where to push next

The mix holds **194 unique real-ish source plans** (86 scraped + 28 generated v1
+ 80 Gemini r2/r3); everything else is synthetic or deterministic offline
variants. Source count is still the binding constraint, and generated sources
are now the best-value supply: 1.65x a scraped plan each, and they pick their own
resolution, which the natively-640px scraped pool never can.

**Before generating more, read the status section at the top of
`image_generation/README.md`.** The v1 100-prompt scheme was framed wrongly --
whole sheets with legends, then fragments -- and yielded 28 of 98. v3/v4 fixed it
(72-88% yield) by aiming at the eval set's measured distribution. There is no
labelling backlog worth mining; more data means a better round.

**Resolved 2026-09-06**: the standing hypothesis that image 12's thin wall poche
was a 1280px downscaling artifact is supported -- training at 2048 is worth +0.05
to +0.07 overall. Whether image 12 specifically recovers has not been checked
per-image.

## Evaluation rules

In `startup.md` — single copy, so the two cannot drift.

## Generated artifacts

**A clone has none of this.** Datasets, checkpoints, logs and evaluation JSON
live under git-ignored `data/` and `logs*`, so every `data/...` path quoted in
these docs is a provenance record, not a file you have. Preserve them between VMs
for byte-identical artifacts, or rebuild: `startup.md` covers the headline recipe
and `synth_progress.md` ("Rebuilding what these findings used") covers the
ablation pools, which `startup.md` does not. The v6 synthetic pools and every
synth-only arm rebuild from `./run_synth_v6.sh <arm> <seed>`. Source, scripts and documentation
are committed; credentials are never stored in the repo.
