# Handoff

Last updated: 2026-10-22 (close-out, GH200)

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds every number, command and the publish gate;
`synth_progress.md` holds every experiment. This file says only where to pick up and how to set a machine up. Older
runbooks and "pick up here" sections (2026-09-17 .. 2026-10-14) were removed on 2026-10-15; they are all done, their results
are in `synth_progress.md`, and the text is in git history (`git show 7bc6aa2:codex_doc.md`).

## Pick up here (2026-10-17)

**Best model:** `abshetty/floz-refunet-swint-xblend-kq30-kms30` (synthetic only, no training of its own): HF14 0.9163, HF14 fix v2
0.9290, val per-sheet 0.9364, Gemini30 0.8740 @4096, hand boxes. It is the float 0.5 / 0.5 weight average of `single18kq-blend30`
(0.3 x `-single18kq-dr` + 0.7 x `-single18kq-nodr`) and `single18kms-blend30` (same with `-single18kms-{dr,nodr}`, the
multi-resolution arms); all arms are one-epoch restarts of `longswiss-swa37-41` and all are on the Hub. Details and the gate:
`startup.md`. **Always score and gate HF14 with label fix v2 alongside plain HF14** (the user, 2026-10-17).

**The standard experiment** is `TAG=<name> [GEN="..."] [POOL=...] [TRAINX="..."] [BATCH=n] ./run_single_pass.sh` (~8 min pool,
~20 min per arm at batch 8, ~3 min scoring on a GH200; ids 600000+, seed 6, so runs at different generator versions are paired sheet
for sheet). It blends 50/50 and scores without `--save-probs`; also build the 0.3 / 0.7 (DR / no-DR) blend, and -- the cheap win of
2026-10-17 -- the 0.5 / 0.5 blend of that with the current best's components. Compare with `scripts/paired_compare.py` against
`eval_baselines/xblend-kq30-kms30_*`, apply the gate, and publish (Hub + HF14 / val / Gemini30 `--save-probs` maps on
`abshetty/floz-diag` `<tag>/<split>_4096/{metrics.json,probs/}`) only if it passes. Publish the arms too, so blends can be rebuilt.

**Cross-run weight blends are the current lever** (`synth_progress.md` 2026-10-17): kms30 and kw30 each failed the gate alone
(Gemini30 17 collapses in both), but 0.5 kq30 + 0.5 kms30 passes with Gemini30 +0.006. 0.5 kq30 + 0.5 kw30 ties (Gemini30 +0.0000,
sheet 1 -0.075). Untried: other weights (e.g. 0.6 / 0.4), kq30 + kms30 + kf / kp arms, soups of more one-epoch runs.

**Post-processing / polygons:** the frontend receives polygons, not masks (the user, 2026-10-17). Linear upsample + fill holes <= 4096
px is +0.006 val / +0.002 HF14 / +0.006 Gemini30 (23 sheets up, 1 down 0.001), not adopted (protocol change); speck drop + small-hole
fill take predictions from ~12-44 components to ~4. Next: a polygon stage (speck drop, hole fill, contour trace, simplify) scored as
the model output -- IoU of the rasterised polygons. `scripts/postproc_probe.py` (hf14 / val / gemini30).

**User's submission target:** "0.95 on real" (protocol -- HF14 vs val, per question vs per sheet -- not pinned down yet).

**Pick up here (2026-10-20, end of GH200 session; details in `synth_progress.md` 2026-10-20 entries).** Best model unchanged
(xblend-kq30-kms30). Today: seed noise ~0.016 (judge at two seeds, blend vs blend); late-checkpoint averaging, Hub soups,
hires-crop, CascadePSP, per-question / family-aware BCE all null or negative. **Inference / post-processing candidates, held for
the user's decision:** (1) 0.7 x 4096 + 0.3 x 2048 multi-scale (never hurts: val +0.0015, Gemini30 +0.0067, -0.25 fixes / q;
~+25% compute); (2) + TTA x8 (fix v2 0.9429, +0.0117, p=0.07; 8x compute); (3) polygon eps 2 (half the vertices, same IoU);
(4) BPR refiner (`scripts/bpr/bpr.py`, band 4) + sliver revert (refined pieces < 50% on the coarse mask go back to P):
HF14 +0.0046 (p=0.002), fix v2 +0.0033 (p=0.04), val +0.0016, extra deletes +0.06 / +0.02 per q -- small; next for BPR: maps
for the 40% of `bpr_s900000` lost to OOM, real-like patch augmentation, a P-only control, stacking with (1). New tools:
`scripts/editcost.py` (fix counts / vertices), `train_refunet.py --save-every`, `--bce-other-weight`, `--bce-per-question`.
On the Hub (`abshetty/floz-diag`): `bpr/refiner_2026-10-20.pth`, `xblend-kq30-kms30/{split}_2048` (2048 maps),
`xblend-kq30-kms30/bpr_b4/`, `xblend-kq30-kms30/tta/`, `checkpoints/kqtrace_s8_nodr_epoch0.pth` (the seed-8 kq no-DR arm, for
two-seed blends) + `checkpoints/kqtrace_s{7,8}_nodr_order.json`. Parked: TracIn, self-training with automatic boxes (more Gemini
images). **Twin test done (2026-10-20, `synth_progress.md`): Gemini30 1 / 26 recover on their twins (0.65 -> 0.98,
0.48 -> 0.82) = appearance failures; no question fails on both; twins 16 / 20 fail on their own -- inspect those twin images.**
Multi-scale + BPR stack: val +0.0023, fix v2 +0.0031, Gemini30 +0.0089, no added fixes (held for the user too).

**Seed noise (2026-10-20, `synth_progress.md`):** the same seed reproduces (per-sheet sd <= 0.012), but seed 8 vs seed 7 on the kq
no-DR arm moves HF14 / Gemini30 by ~0.016 and 5 / 30 Gemini30 sheets by > 0.05 -- the gate's sheet rule fires on seed noise, and
every single-pass run so far used seed 7. Late-checkpoint averaging within a run does not help; a 0.5 / 0.5 two-seed blend beats
both seeds on every set. Soups of the Hub arms and adding the seed-8 arm to the best fail the gate. `--save-every N` records step
checkpoints and the batch order. **Judge changes at two seeds (blend vs blend)**; revise the sheet rule to "beyond seed noise".
`--hires-crop 0.5` was screened at seed 7 only (fails the gate: Gemini30 -0.016, inside seed noise); no seed-8 replicate.

**Parked until the user has more Gemini images (2026-10-20): self-training with automatic boxes.** Tile each unlabelled sheet
with candidate boxes; ask the best model each one; keep a box only if (a) it lies wholly inside its own answer at P > 0.9, (b) a
second box elsewhere in that answer returns the same answer (reference-swap consistency), (c) the answer is a sensible size with
texture, (d) the box is not mostly text. Survivors + answers are training questions (uncertain pixels ignored), trained at hard
views (other scale / blur / line weight). First validate on val's labelled sheets: share of accepted boxes wholly inside one
family and their answers' accuracy, plus a contact sheet for the user. HF14 / val decide, not Gemini30; Gemini30 never in the pool.

**Synthetic facsimiles of Gemini images (2026-10-20) -- done: `--spec` twins too different (the user), label-shape twins
(`scripts/facsimile_relabel.py`) built and twin-tested on the GPU (see Pick up here). Original plan, kept for reference:** Add `generate_synthetic_fc.py --spec file.json`
that renders a given house + appearance (the dicts `G.make_house` / `G.make_appearance` now sample, `house_for` line 49)
instead of drawing them; describe each Gemini image in that vocabulary (Claude Code views the image; at scale, Gemini Batch
API writes the JSON), clamped to what the generator can draw. Uses: (1) labelled twins of Gemini content with exact masks and
`ref_exclude` (controlled boxes); (2) a gap list of what specs cannot express (e.g. corner boards); (3) the **twin test** on
Gemini30 -- good on the twin but bad on the original = appearance failure (line weight / scale / style), bad on both =
content / layout -- splits Gemini30's loss by cause. Twins of Gemini30 sheets are diagnostic only, never in a training pool.
Copies content, not appearance. Pilot: 10 user Gemini images + 10 Gemini30 twins, then score the twins (GPU, minutes).

**Open problems** (`synth_progress.md` 2026-10-14 .. 17):
- Gemini30 17 (hip-roof shingle plan, native 2400 px) collapses to blotchy half-confidence in both kms and kw (0.89 -> 0.68-0.70);
  only kq gets it right. Fragile, not specific to multi-res.
- Scale sensitivity: 2048 vs 4096 swings sheets +-0.17-0.29; multi-res training at [2048, 3072] did not shrink them (batch 8 OOMs
  with the range; no grad accumulation / checkpointing in `train_refunet.py`).
- HF14 12 q1 / q3 (turned deck boards) and HF14 2 / Gemini30 8 (elevations) move with the plan pairs; Gemini30 1 swings 0.30-0.69.
- Gemini30 0: tile reference takes the sofas / chairs. HF14 18: the two roofs merge under label fix v2.

## Pick up here (2026-10-22): closed out -- three Gemini-data blends published, `gem2s-x` the default

Best: `abshetty/floz-refunet-swint-gem2s-x` (0.25 gem_s7 + 0.25 gem_s8 + 0.5 xblend-kq30-kms30), with `-gk-x` and `-gke-x` published
beside it (all pass the gate; within noise of each other). Raw @4096: fix v2 0.9320, Gemini30 0.8891, val 0.8990, HF14 0.9235; with the
product stack (polygons + multi-scale + colour cut, 0.66 s / query) fix v2 0.9365, Gemini30 0.9122. Numbers and rebuild: `startup.md`;
every run: `synth_progress.md` 2026-10-22. The lever that worked: hand-labelled Gemini sheets outside Gemini30 (+0.019 Gemini30 at one
seed). Next if the project resumes: label round 5 (200 Gemini sheets, `image_generation/prompts_r5.jsonl`) and retrain the gem recipe.
Do not use TTA (latency budget ~2 s).

## Pick up here (2026-10-21): `./run_gemini_holdout.sh` -- does real Gemini appearance move Gemini30?

The twin test says Gemini30's worst sheets fail on how Gemini DRAWS materials, not on layout. The 139 hand-labelled
Gemini r2-r4 sheets that are not in Gemini30 (`scripts/build_gemini_holdout_pool.py`; Gemini30 is a random 30 of r2-r4,
so earlier mixes with all of r2-r4 leaked it) have never been trained on with a clean Gemini30 read. The script trains the
standard one-epoch restart (run_single_pass.sh) on the current best's own kq pool (generator at 2026-10-15b, rebuilt from
commit `6b25130^`) with and without those 139 sheets x18, at seeds 7 and 8, blends the seeds, scores HF14 / fix v2 / val /
Gemini30 @4096 and writes `logs/gemho_compare.txt` (ctl vs gem, paired; plus ctl seed 7 vs single18kq-blend50, which
should match). ~5 h. Read: Gemini30 up well beyond seed noise (~0.016) with HF14 / val not down = labelled Gemini data
is the lever (label more, e.g. round 5); flat = the remaining gap is not appearance data, measure the label ceiling.
The 2026-10-20d / 21a generator defaults (cross gables, wall splits, panels, pattern drift, base bands) are still
unscreened; this run deliberately does not use them.

## Machine setup (fresh GH200, ~15 min)

```bash
git pull; set -a; . ~/.env; set +a                    # HF_TOKEN, ROBOFLOW_API_KEY; never print them
pip install --user -r requirements.txt roboflow "networkx>=3.0"
sudo /opt/miniconda/bin/conda create -y -p /opt/fc -c conda-forge --override-channels freecad   # generator needs /opt/fc/bin/freecadcmd
mkdir -p data/reference/swiss_dwellings data/runs/ck_best logs
python3 -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('abshetty/floz-assets', 'swiss_dwellings/layouts.pkl.gz', repo_type='dataset'), 'data/reference/swiss_dwellings/layouts.pkl.gz')"
PYTHONPATH=. python3 scripts/hf_ckpt_to_pth.py --repo abshetty/floz-refunet-swint-longswiss-swa37-41 --out data/runs/ck_best/longswiss-swa37-41.pth
python3 -c "from datasets import load_dataset; print(len(load_dataset('abshetty/floz-synth-v5','real-world-test',split='test',cache_dir='./data')))"   # 28
python3 scripts/build_gemini30_pool.py               # data/eval_pools/gemini30_s20261006, 30 sheets
```

Sanity check before training: re-score the best on HF14 @4096 and match `eval_baselines/single18kq-blend30_hf14_4096.json`.
Two training arms at once OOM the GH200's 94.5 GB; run them one after the other (the script does). Four scoring jobs at
4096 side by side are fine. `git commit` needs `-c user.name=... -c user.email=...` on a fresh VM.

## Facts that live elsewhere

Single copies, so they cannot drift. Do not restate them here.

| what | where |
|---|---|
| task semantics, metric definition, what does *not* count as evidence | `PROJECT_UNDERSTANDING.md` |
| the `sample_reference_box` fix and why pre-2026-08-06 numbers are incomparable | `PROJECT_UNDERSTANDING.md`, mechanism in `startup.md` |
| current best, publish gate, reproduction commands, validation split, evaluation rules | `startup.md` |
| per-experiment history and generator defaults (GEN_VERSION) | `synth_progress.md` (older: `synth_progress_archive.md`) |
| the labelling-assist model | `labeling_assist.md` |
| image generation (Gemini, always the Batch API) | `image_generation/README.md` |

Two standing traps: the reference resize in `refmask2former/dataset.py` looks like a bug and "fixing" it costs 0.153, and
`--corr-grid` / `--scale-matched-ref` are implemented but screened negative -- read the warnings before touching either.

## Relevant implementation

- `refmask2former/ref_unet.py` -- the model (shared backbone, conditioning, FPN mask); `dataset.py` -- reference sampling
  (`sample_reference_box`, `reference_region` with `ref_exclude`), mask rendering.
- `scripts/generate_synthetic_fc.py` -- the current generator (Revit-style elevations from FreeCAD massing, `revit_plans.py`
  roof / floor plans, Swiss Dwellings layouts); `GEN_VERSION` names its defaults, `scripts/pool_guard.sh` refuses stale pools.
- `scripts/train_refunet.py` -- training; `--init-from` + `--reset-optimizer` for restarts, `--roi-ref --roi-ref-mode add`.
- `scripts/evaluate_refunet_selection.py` -- the authoritative eval: hand boxes by default, `--local-pool` (Gemini30, synthetic),
  `--label-fixes`, `--save-probs`.
- `scripts/paired_compare.py` -- paired t / sign test with per-image share of the difference; `scripts/select_epoch_on_val.py`
  `val_score` -- the per-sheet val mean.
- `scripts/publish_refunet.py` / `scripts/hf_ckpt_to_pth.py` -- Hub safetensors out / `.pth` back; always round-trip check.
- `scripts/probes/` -- counterfactual probe (`make_counterfactual_pool.py`, `counterfactual_summary.py`).
- `scripts/visualize_refunet_selection.py` -- the required four-panel visual audit.
- `scripts/label_ceiling.py`, `scripts/residual_decomp.py`, `scripts/per_question_breakdown.py` (sheet types) -- diagnostics.
- `run_single_pass.sh` -- the current experiment; `run_hard_mine.sh`, `run_hard_long.sh`, `run_revit_2k.sh` -- mining, long
  restarts, 2k screens; other `run_*.sh` are past ablations.

## Generated artifacts

**A clone has none of this.** Pools, checkpoints, logs and evaluation JSON live under git-ignored `data/` and `logs/`, so
every `data/...` path in these docs is a provenance record, not a file you have. Pools rebuild deterministically from their
ids / seed / GEN; checkpoints that matter are on the Hub (`abshetty/floz-refunet-swint-*`), maps on `abshetty/floz-diag`, and
per-question results are committed in `eval_baselines/`. Credentials are never stored in the repo.
