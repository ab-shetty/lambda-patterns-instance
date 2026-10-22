# Startup Guide

The reference manual: what the current best is, how to rebuild and score it on a fresh VM, and the rules every number follows.
Read `PROJECT_UNDERSTANDING.md` first (the task, the metric, what does not count as evidence). `codex_doc.md` is the handoff
(where we left off, what is next); `synth_progress.md` logs every experiment. Past bests, superseded recipes and old comparison
tables are in `archive/startup_archive.md`. This file holds the single copy of current numbers and commands: overwrite them when a new
best lands, and move the old text to the archive.

Other tracks: the labelling-assist model is in `labeling_assist.md`; image generation in `image_generation/README.md`.
Gemini image generation always uses the Batch API (`scripts/generate_images_gemini.py --batch submit|status|fetch`), even for a few images.

## Read this before quoting any number

Every number measured before 2026-08-06 (`0.6127`, `0.5506`, `0.4646`, `0.3494`,
`0.6432`) used the broken `sample_reference_box`, as did the `>= 0.65` target,
and is **not comparable** to anything measured afterwards. Always state which
sampler a number came from. `PROJECT_UNDERSTANDING.md` explains why the fix
matters; "The reference-box fix" below has the mechanism and its effect.

**Reference jitter (2026-09-29).** `--domain-random`'s brightness/contrast jitter
(x0.75-1.25, +-18) used to reach the sheet only: the reference is cropped before it,
so in training the correct target could look up to ~20 L* lighter or darker than the
reference, while at inference both come from one image. It now applies the same jitter
to the reference (no extra RNG draws; sheet and masks unchanged). `--realism-aug` now
also draws one set of tone/blur/noise-strength/JPEG settings for the sheet and all its
references instead of one per image (fewer RNG draws, so realism-aug runs are not
byte-comparable to older ones). **Every run before
2026-09-29, 0.8440 included, was trained sheet-only**: reproduce those with
`train_refunet.py --legacy-ref-jitter`. The effect of the fix is not yet measured.

## Current best (2026-10-22)

**Best since 2026-10-22: three published Gemini-data blends, `gem2s-x` the default** (`abshetty/floz-refunet-swint-{gem2s-x,gk-x,gke-x}`,
private, round-trip identical 243 / 243). All pass the gate against xblend-kq30-kms30 and sit within noise of each other; the user
chose to publish all three. `gem2s-x` is the default because HF14 fix v2 is the priority split (the user, 2026-10-22). Components on the
Hub: `-gemho-gem-s7`, `-gemho-gem-s8`, `-gemkms-s7`, `-geme2-s7` (each a one-epoch restart of longswiss-swa37-41 on the kq pool + the 139
hand-labelled Gemini r2-r4 sheets outside Gemini30 x18, 50/50 DR / no-DR blend). Rebuild a blend with `scripts/blend_ckpts.py`.

**Product output = polygons + multi-scale (0.7 x 4096 + 0.3 x 2048) + chromaticity colour cut** (`scripts/pp_probe/stack.py`, ms+cc;
fixed before scoring). 0.66 s per query on a GH200 (median, warm; `scripts/latency_probe.py`; post-processing 0.48 s of it, CPU):
| model + stack (ms+cc) | HF14 fix v2 | Gemini30 | val | HF14 |
|---|---:|---:|---:|---:|
| `gem2s-x` | **0.9365** | 0.9122 | 0.9064 | **0.9268** |
| `gk-x` | 0.9349 | 0.9136 | 0.9067 | 0.9248 |
| `gke-x` | 0.9340 | **0.9150** | **0.9068** | 0.9254 |
| xblend-kq30-kms30 (previous best) | -- | 0.8977 | 0.9061 | 0.9187 |
Adding the paper cut and the adaptive colour threshold (both designed on Gemini30 sheets 19 / 26) gives Gemini30 ~0.928 and no change
elsewhere -- optimistic, report separately. No TTA: it breaks the ~2 s latency budget. Maps (4096 + 2048, HF14 / val / Gemini30) on
`abshetty/floz-diag` `{gem2s_x,gk_x,gke_x}/`. Details: `synth_progress.md` 2026-10-22.

Raw model @4096, hand boxes (no post-processing), with reference points:

| model | HF14 fix v2 | Gemini30 | val | HF14 |
|---|---:|---:|---:|---:|
| **`gem2s-x`** (default) | **0.9320** | 0.8891 | 0.8990 | **0.9235** |
| `gke-x` | 0.9293 | **0.8931** | **0.8999** | 0.9219 |
| `gk-x` | 0.9301 | 0.8913 | 0.8995 | 0.9212 |
| `xblend-kq30-kms30` (previous best, synthetic only, 2026-10-17) | 0.9290 | 0.8740 | 0.8988 | 0.9163 |
| `longswiss-swa37-41` (the restart point of every arm above, synthetic only) | -- | -- | 0.8766 | 0.8779 |
| `mixr4-roiadd-swa15` (shipped, 2026-09-28) | -- | -- | 0.8193 | 0.8466 |

Rebuild and score (any of the three; per-question baselines for the previous best are in `eval_baselines/`):

```bash
PYTHONPATH=. python3 scripts/hf_ckpt_to_pth.py --repo abshetty/floz-refunet-swint-gem2s-x --out data/runs/ck_best/gem2s-x.pth
scripts/score4.sh data/runs/ck_best/gem2s-x.pth data/evaluations/gem2s-x      # HF14, fix v2, val, Gemini30 @4096
scripts/gpu_maps.sh data/runs/ck_best/gem2s-x.pth data/floz-diag/gem2s-x      # 4096 + 2048 maps for the stack
for s in gemini30 val hf14 hf14fix2; do PYTHONPATH=. python3 scripts/pp_probe/stack.py $s data/floz-diag/gem2s-x; done
PYTHONPATH=. python3 scripts/latency_probe.py data/runs/ck_best/gem2s-x.pth 6
```

**Publish gate (the user's rule, revised 2026-10-15):** publish a new candidate (Hub + HF14 / val / Gemini30 `--save-probs`
maps on `abshetty/floz-diag`) and make it the best only if, against the current best at @4096 with hand boxes,
**(1) Gemini30 goes up; (2) neither val per-sheet (`val_score`) nor HF14 drops significantly (paired p < 0.1 counts as a drop);
(3) no single sheet (HF14 / val / Gemini30) falls by more than 0.05 because of the change itself.** For (3), a sheet the
changed generator feature cannot touch (e.g. an elevation under a plan-only change) does not count, but say so. Earlier
gates: val per-sheet + Gemini30 >= best (single18kp-blend50 passed it with HF14 -0.017, p=0.027); then Gemini30 up and
neither val per-sheet nor HF14 down at all (single18kq-blend50 failed it on HF14 -0.007, p=0.35).

**HF14 fix v2 is the priority real split** (the user, 2026-10-22): report it first and weigh it over plain HF14. **Seed noise is
~0.016** on HF14 / Gemini30 and moves single sheets by > 0.05 (2026-10-20): judge a training change at two seeds (blend vs blend), and
read the sheet rule as "beyond seed noise".

## Evaluation protocol

### Splits

- **HF14** (acceptance): `12,16,27,7,11,25,23,1,18,2,0,3,14,24` of the HF `real-world-test` config (28 images); 51 questions with
  hand boxes. **HF14 fix v2** adds missed target pixels (`--label-fixes eval_labels/hf14_fixes_v2.json`); score both, lead with fix v2.
- **val** (tuning): the other 14, 72 questions with hand boxes; `val_score` in `scripts/select_epoch_on_val.py` is the per-sheet mean.
- **Gemini30**: 30 hand-labelled Gemini sheets (`scripts/build_gemini30_pool.py` -> `data/eval_pools/gemini30_s20261006`, boxes
  `eval_boxes/gemini30_v1.json`), 195 questions. A random 30 of rounds 2-4, so any pool holding those rounds leaks it: train on
  `scripts/build_gemini_holdout_pool.py` (the other 139).

### Boxes: hand-placed by default (2026-10-03)

`evaluate_refunet_selection.py` (and the per-epoch / val-selection scoring in `train_refunet.py`,
`select_epoch_on_val.py`, `average_checkpoints.py`) now uses `eval_boxes/hand_v1.json` by default:
the 88 moved boxes and 6 dropped questions from the Reference Box Editor; every box not moved or
dropped is the automatic box, unchanged. HF14 = 51 questions, val = 72. `--boxes auto` gives the
older automatic-only protocol (most numbers in `archive/startup_archive.md` used it); `--local-pool` uses
automatic boxes unless `--boxes` names a file made for that pool (Gemini30: `eval_boxes/gemini30_v1.json`).

### Validation split — tune here, never on HF14

The HF `real-world-test` config has 28 images. HF14 is the acceptance holdout;
its complement is **never trained on and never scored for acceptance**, so it is
the tuning set:

```text
validation indices: 4,5,6,8,9,10,13,15,17,19,20,21,22,26   (14 images, 72 questions with hand boxes)
```

It correlates with HF14 at Pearson r=0.876 / Spearman 0.888 across 27
checkpoints, and reads systematically higher (~0.75 vs ~0.68) because it is an
easier set — use it for *ranking* configurations, not for absolute numbers.
Several of its plans share source PDFs with HF14 (Las Huertas, Ceilhunt, W
Felton, Maple Rd), so it is correlated rather than independent; treat a
validation gain as suggestive, then confirm once on HF14.

### Checkpoint selection is part of the metric

Taking the best epoch *by HF14 score* uses the acceptance set to choose the
checkpoint. Choosing the epoch on validation instead (measured 2026-09, `mix3652`, automatic boxes):

| protocol | HF14 |
|---|---:|
| epoch = HF14 peak (project's stated "any checkpoint within ten epochs") | 0.6860 ± 0.0175 |
| epoch chosen on validation | **0.6783 ± 0.0081** |

The +0.0077 gap is selection bias. Both are honest under their own protocol;
quote the val-selected number when you need one that will hold up.

## Fresh VM setup (GH200, ~15 min)

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

Sanity check before training: re-score the best on HF14 @4096 and match its recorded number. Two training arms at once OOM the
GH200's 94.5 GB; run them one after the other (the scripts do). Four GPU scoring jobs at 4096 side by side are fine. `git commit`
needs `-c user.name=... -c user.email=...` on a fresh VM.

Only for `scripts/generate_images_openai.py`: `pip install --user openai`.

Verified working stack: Python 3.10.12, torch 2.7.0, torchvision 0.22.0,
datasets 5.0.1, OpenCV 4.10.0, Pillow 12.3.0, numpy 1.26.4, shapely 2.1.2, on one
NVIDIA GH200 with 64 CPUs. Minor nondeterminism can change the last decimals.

**Use all 64 vCPUs for every CPU job.** Check with `ps -eo pcpu,args` before
walking away: a job at 100% is using ONE core, not all of them.

- `generate_synthetic_v6.py` defaults `--workers` to `os.cpu_count() // 2`; pass
  `--workers 64`. At 64 workers it is 0.01 s/image -- 2,000 plans in 14 s,
  100,000 in ~12 min -- so pool size is never a reason to compromise a design.
- `augment_local_dataset.py` has **no workers flag and is a serial loop.** Split
  the source dir into N chunks, augment them in parallel, and `merge_local_datasets.py`
  the outputs. On native-resolution Gemini (2,752 px) serial augmentation of 88
  sources takes ~1 h and the 11-way split takes ~3 min. Chunking changes each
  chunk's RNG stream, so the result is not byte-identical to a serial run --
  fine for a new pool, not for rebuilding one a recorded number came from.
- `--num-workers` on the dataloader is tuned against GPU feed rate (16), not a
  CPU limit; leave it.

The same applies to anything else single-threaded: parallelism here is free and
the box is otherwise 98% idle.

**Install `roboflow` together with `requirements.txt`, not separately** (the
command above already does this). `roboflow` alone pulls numpy 2.x, which
breaks the system `torch`/`scipy` install (compiled against numpy 1.x) with
`ValueError: numpy.dtype size changed`. If this happens, rerun
`pip install --user -r requirements.txt` to re-pin numpy `<2.0.0`.

**`--compile` also needs a newer `networkx` than the system one** (found on a
clean 2026-09-17 rebuild; not in `requirements.txt`). `torch.compile` imports
`functorch` -> `torch._functorch.partitioners` -> `networkx`, and Ubuntu's
system `networkx` still calls `np.int`, which the pinned numpy 1.26 removed:

```text
AttributeError: module 'numpy' has no attribute 'int'.
```

So the repo's own numpy pin and `--compile` are mutually incompatible out of
the box -- training dies in the first seconds, before any epoch. Fix once per
machine, which shadows the system copy and leaves numpy alone:

```bash
pip install --user "networkx>=3.0"          # 3.4.2; numpy stays 1.26.4
```

An eager (non-`--compile`) run is unaffected, which is why a smoke test without
`--compile` will not catch this.

**Host RAM (2026-10-22).** Everything on the VM runs in one systemd unit, so a host OOM kills tmux, training and the agent session
together. Six CPU evaluations at 4096 (with `--tta 8`) beside training did exactly that. Run CPU scoring two at a time
(`scripts/cpu_score_queue.sh`, `scripts/cpu_maps.sh`) and keep `scripts/memguard.sh` running (kills CPU evaluations under 50 GB free).

## The current workflow

- **One experiment** = `TAG=<name> [GEN="..."] [POOL=...] [TRAINX="..."] [BATCH=n] [SEED=n] ./run_single_pass.sh`: a fresh
  synthetic pool (ids 600000+, seed 6, so pools at different generator versions are paired sheet for sheet), two one-epoch restarts of
  `longswiss-swa37-41` (with and without `--domain-random --dr-scale-min 0.7`), their 50/50 weight blend, scored @4096 on all splits
  (~8 min pool, ~20 min per arm at batch 8, ~3 min scoring).
- **Gemini data**: `run_gemini_holdout.sh` builds the 139-sheet holdout pool x18 and the gem mix; `POOL=data/mixed/gemho_mix` trains on it.
- **Blends** are free and are the usual winners: `scripts/blend_ckpts.py OUT.pth A.pth:w B.pth:w ...`.
- **Compare** with `scripts/paired_compare.py` against the current best's per-question JSON, apply the gate, publish with
  `scripts/publish_refunet.py` (+ round-trip with `scripts/hf_ckpt_to_pth.py`), maps to `abshetty/floz-diag` `<tag>/<split>_<size>/`.
- **Latency budget ~2 s per query**: no TTA; time any new inference stage with `scripts/latency_probe.py`.

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
- `scripts/blend_ckpts.py` (weight blends), `scripts/score4.sh` (four splits on the GPU), `scripts/gpu_maps.sh` /
  `scripts/cpu_maps.sh` (maps for the stack), `scripts/pp_probe/stack.py` (product post-processing), `scripts/latency_probe.py`,
  `scripts/memguard.sh`.
- `run_single_pass.sh` -- the current experiment; `run_gemini_holdout.sh` -- the Gemini holdout pool and gem mix; `run_hard_mine.sh`, `run_hard_long.sh`, `run_revit_2k.sh` -- mining, long
  restarts, 2k screens; other `run_*.sh` are past ablations.

Two standing traps: the reference resize in `refmask2former/dataset.py` looks like a bug and "fixing" it costs 0.153, and
`--corr-grid` / `--scale-matched-ref` are implemented but screened negative -- read the warnings before touching either.

## Generated artifacts

**A clone has none of this.** Pools, checkpoints, logs and evaluation JSON live under git-ignored `data/` and `logs/`, so
every `data/...` path in these docs is a provenance record, not a file you have. Pools rebuild deterministically from their
ids / seed / GEN; checkpoints that matter are on the Hub (`abshetty/floz-refunet-swint-*`), maps on `abshetty/floz-diag`, and
per-question results are committed in `eval_baselines/`. Credentials are never stored in the repo.

## Evaluate

```bash
PYTHONPATH=. python3 scripts/evaluate_refunet_selection.py --checkpoint data/runs/ck_best/gem2s-x.pth \
  --indices 12,16,27,7,11,25,23,1,18,2,0,3,14,24 --label-fixes eval_labels/hf14_fixes_v2.json \
  --image-max-size 4096 --ref-size 224 --mask-thresh 0.35 --metrics-out data/evaluations/gem2s-x/hf14fix2_4096.json
# Gemini30: --local-pool data/eval_pools/gemini30_s20261006 --indices all --boxes eval_boxes/gemini30_v1.json
```

Render the required visual audit (input + rectangle, expected union, predicted
union, overlap/excess/missed) with:

```bash
PYTHONPATH=. python3 scripts/visualize_refunet_selection.py \
  --checkpoint data/runs/ck_best/gem2s-x.pth --image-max-size 4096 \
  --mask-thresh 0.35 --out data/visualizations/gem2s-x
```

It reproduces the evaluator's RNG, resize, and threshold exactly, names files by
IoU so failures sort first, and writes `manifest.csv` + `summary.txt`.

## The reference-box fix

`sample_reference_box` treated `min_size` (128px) as a floor without checking a
box that large could fit, so thin regions exhausted all 450 attempts and fell
through to an unchecked `centroid ± 16` box. On a ring-shaped mask — a wall with
a window punched out — that centroid sits **in the hole**, so the crop showed
none of the pattern, then upscaled 7× to 224px.

Incidence before the fix: **57%** of real-plan crops and **36%** of generated-plan
crops lay partly or entirely outside their own mask, but **0%** of synthetic ones
(those images are 2–5k px wide, so 128px boxes fit trivially). That is why it hid
— it only bit the real data.

The fix uses a Chebyshev `cv2.distanceTransform` for the largest square that fits
at every pixel: size is drawn from what actually fits, the centre only from
pixels deep enough to hold it, sides odd so the box is symmetric. `min_size` is
now a preference; containment is asserted across all pools.

Effect on HF14 (`mix3652`, 3 seeds): **0.6331 → 0.6860 (+0.053, t=4.18,
p=0.014)**, with complete separation between arms. Selections scoring below 0.25
dropped from 8 to 1 while the median barely moved — the fix removed impossible
questions rather than making the model broadly better.

## Evaluation rules

- **The reference boxes are deterministic and identical across every run** (hand boxes, else the seeded automatic box), so each
  split is a set of *fixed questions*: compare arms **paired per question**, not as means ± sd. The paired test removes
  question-difficulty variance, which dominates.
- HF `real-world-test` images are evaluation-only.
- Fixed indices: `12,16,27,7,11,25,23,1,18,2,0,3,14,24`.
- Report reference-conditioned union IoU only, with the sampler and box protocol named.
- Do not report the legacy training `real_iou`, per-GT best coverage, an oracle,
  class-agnostic Mask R-CNN scores, or probability-averaged checkpoint ensembles as product mIoU (a weight blend is one model
  and is fine).
- Threshold 0.35 is fixed. Do not sweep it, the inference resolution, or any
  other knob against HF14 — that is fitting the acceptance set.
- HF14 images 24, 25, and 27 carry human highlighter markup because they came
  from real markup PDFs. That is the real input distribution; keep them in, and
  treat robustness to markup as a model requirement.
