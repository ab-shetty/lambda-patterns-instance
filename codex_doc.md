# Handoff

Last updated: 2026-10-15

`PROJECT_UNDERSTANDING.md` defines the task and the metric. `startup.md` holds every number, command and the publish gate;
`synth_progress.md` holds every experiment. This file says only where to pick up and how to set a machine up. Older
runbooks and "pick up here" sections (2026-09-17 .. 2026-10-14) were removed on 2026-10-15; they are all done, their results
are in `synth_progress.md`, and the text is in git history (`git show 7bc6aa2:codex_doc.md`).

## Pick up here (2026-10-15)

**Best model:** `abshetty/floz-refunet-swint-single18kq-blend30` (synthetic only): HF14 0.9156, val per-sheet 0.9357, Gemini30
0.8680 @4096, hand boxes. It is 0.3 x `-single18kq-dr` + 0.7 x `-single18kq-nodr` (both on the Hub): one-epoch restarts of
`longswiss-swa37-41` on 18k fresh sheets at GEN 2026-10-15b, with / without `--domain-random`. `single18kf-blend50` (HF14 0.9157)
is kept as the HF14 reference. Details and the gate: `startup.md`.

**The standard experiment** is `TAG=<name> [GEN="..."] ./run_single_pass.sh` (~8 min pool, ~20 min per arm, ~3 min scoring on a
GH200; ids 600000+, seed 6, so runs at different generator versions are paired sheet for sheet). It blends 50/50; the best blend
is now DR 0.3 / no-DR 0.7, so score that ratio too (float `0.3 * dr + 0.7 * nodr` of the `model` tensors, eval only, ~3 min).
Compare with `scripts/paired_compare.py` against `eval_baselines/single18kq-blend30_*`, apply the gate in `startup.md`, and
publish (Hub + HF14 / val / Gemini30 `--save-probs` maps on `abshetty/floz-diag` `<tag>/<split>_4096/{metrics.json,probs/}`)
only if it passes.

**Unfinished:** `single18kw` (wing colour pairs back on: `GEN="--plan-source data/reference/swiss_dwellings/layouts.pkl.gz
--near-pairs 0.6"`). Only its nodr arm exists (`abshetty/floz-refunet-swint-single18kw-nodr`, not scored). To finish: fetch it to
`data/runs/ck_single18kw_nodr/epoch_0.pth`, `touch data/runs/ck_single18kw_nodr/.done`, run the script with that TAG / GEN (it
rebuilds the pool, trains only dr, blends, scores), then score the 0.3 / 0.7 blend; compare with single18kq.

**Open problems** (`synth_progress.md` 2026-10-14 / 15):
- HF14 12 q1 (turned deck boards) 0.93 in kf, 0.83-0.84 since the plan direction pairs.
- HF14 2 and Gemini30 8 (elevations) fell > 0.05 in every pool with the plan pairs (kp, kq) -- unexplained, since the pairs
  touch only plans.
- Gemini30 1 (pale board-and-batten) swings 0.30-0.65 between models; grey / flat-background / soften rescue it.
- Gemini30 0: tile reference takes the sofas / chairs. HF14 18: the two roofs merge under label fix v2.
- More no-DR weight raises Gemini30 every time; DR 0.3 also raised HF14 for kq. The ratio has only been tried at 0.3 / 0.5 / 0.7.

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
