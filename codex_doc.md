# Handoff: closing the synth→real gap (for Codex / GPT-5.4)

You are picking up a synthetic-data realism project. Your job: **make synthetic
architectural-plan images cover the real distribution well enough that a
segmentation model trained on synth performs as well on *real* plans as it does
on held-out synth.** You have a strong vision model — use it to *look* at synth
vs real and drive a generator-tuning loop. But (read the Dead Ends section)
eyeballing alone has already burned us; every change must be validated with the
multi-seed instrument, judged by **divergence**, not by how pretty synth looks.

This doc is the orientation layer. The deep lab notebook is `synth_progress.md`
(start at the `STATE OF PLAY` header, ~line 1288). Hard-won facts are also in
the Claude memory dir referenced at the bottom.

---

## 0. UPDATE 2026-06-22 — READ FIRST (supersedes §3-bottom, §4 hypotheses, §7 ceiling)

A full autonomous search ran 2026-06-18→22. Net result for anyone trying to make
synth better: **no generator edit has moved held-out real at 5-seed rigor.** The
two confirmed levers are (a) more DISTINCT labelled real, (b) a better TRAINING
regime — neither is a generator change. Specifics:

- **The real lever is distinct-plan COVERAGE, not image count.** 14 distinct real
  plans → real ≈ 0.275; 40 distinct → ≈ 0.32. Adding COPIES/augs of the same
  plans, scaling synth, or richer augmentation are all NULL (the old "~10% real →
  0.32" framing was about fraction-of-the-same-reals and is misleading — it's the
  *number of distinct plans* that matters). New real arrives via Roboflow now:
  `scripts/roboflow_to_local.py` → `scripts/build_mix.py --real-extra-dir …`
  (handles the different resolution). **Labelling more of the 374 unannotated
  `floz-real-pool` images is the single highest-value action.**
- **CONFIRMED TRAINING-REGIME WIN (now the repo default):**
  `--batch-size 1 --grad-accum 4 --freeze-backbone-bn` lifts real **+0.03–0.08**
  vs batch-8 (t=4.4, 10 seeds, replicated at 1024 & 2048). It is a regularization
  effect, not a domain fix (raises synth AND real equally; divergence unchanged).
  It **raised the real ceiling 0.32 → ~0.35** and FLATTENED the real-fraction curve
  (you reach ~0.35 with as little as 10% real; more of the same real doesn't stack).
  Every pre-06-22 number in this doc was measured at the old batch-8 regime and is
  ~0.03 low. This regime STACKS with the data lever.
- **Generator edits TESTED this session and all NULL-or-HARMFUL at ≥5 seeds:**
  floorplan-heavy mode-weights + clutter "combo" (looked like a +0.014 win at 3
  seeds, **washed out to 0.324±0.019 = tied with base** at 5); kill-the-lollipop
  markup (`--markup-overlay-prob 0`, +0.008 at 3 seeds, **neutral** at 5);
  clutter-boost (neutral); textured negatives (`--negative-texture-prob`, **HURT**
  −0.023 — same-material ambiguity); resolution 1024 vs 2048 (NULL, closes the
  doc's "tiny instances" model-side lever); LLM/eyeball appearance realism (HURT,
  see §3-#1). Making synth "harder / cover floorplans" lowers synth_iou and drives
  divergence→0 **without moving real** — that is the lightly-trained-equilibrium
  trap (§2), not a win.
- **Hardened rigor rule (two false-positive "wins" this session forced it):**
  trust no real_iou delta < ~0.03; use ≥5 seeds (10 to confirm); judge by
  real_iou DIRECTLY via STANDARD ERROR of the two arms, not ±std-overlap and not
  divergence. 3-seed reads at real≈0.32 have a true ±0.02 band and have fooled us
  twice.

**If you still want to try generator improvements (the §4 vision job):** it's not
forbidden — but the bar is the rigor rule above, the target must be a genuine
COVERAGE hole (an image-type/structure real has and synth never generates), never
appearance/style, and you are competing against a lever (label more real) that is
known to work. Gate every change on held-out real at 5 seeds before believing it.

---

## 1. The model & the task

`RefMask2Former` (`refmask2former/`): a Mask2Former variant with a
reference-matching head. **Reference-conditioned, class-agnostic instance
segmentation** — given a plan image and a small reference patch of a
material/pattern, it segments every region of that material. Targets are
**filled material/pattern REGIONS** (area objects: roof fields, wall siding,
stone/tile fills), *not* 1px lines.

- Backbone ResNet50 (ImageNet) → FPN pixel decoder → transformer decoder.
- Eval metric: `mean_gt_iou` = recall-of-best-prediction per GT mask. It is
  **reference-INDEPENDENT** (uses objectness prob > 0.5, not ref similarity).

## 2. The north star (do not lose this)

- End goal: **0.95 IoU on real plans.**
- Operational metric every experiment is judged by:
  **`DIVERGENCE = synth_val_iou − real_iou` → 0.**
  The thesis (proven, see Dead Ends): if synth covered the real distribution,
  the two IoUs wouldn't diverge. A model that hits 0.95 on synth but 0.2 on
  real has a *coverage* gap, not a capacity gap.
- **Watch held-out real, not just divergence.** Divergence near 0 with both
  IoUs low is a lightly-trained equilibrium, not success.

## 3. Dead ends — DO NOT repeat these (all multi-seed validated)

1. **LLM/eyeball realism HURT.** A loop that made synth subjectively "look more
   real" (clapboard, roof cues) moved real **0.206 → 0.154**, divergence
   **+0.133 → +0.188** (3-seed, non-overlapping). Pretty ≠ covering. This is
   *the* trap for a vision model: do not optimize appearance for its own sake.
2. **Scaling synth 10× does NOT lift real.** 10× images @ 10 epochs left
   held-out real stuck ~0.31 and blew divergence to **+0.37** (synth_val soared
   to 0.68 from 10× gradient steps). More synth volume is not the lever.
3. **Backbone is not the lever.** A "line-art" blur-pool stem (provably
   preserves thin strokes max-pool erases) *slightly hurt* held-out real
   (0.317 → 0.289), divergence unchanged. Feature-level fixes don't move it →
   confirms the gap is OOD coverage, not low-level features. (`--backbone-stem-pool
   {max,blur,avg}`, default `max`.)
4. **Single hard real image (idx0) is uncoverable by pure synth.** 8 code-based
   iterations all left it ~0.02–0.05 while synth_val ~0.5–0.74. Oracle
   best-of-100 masks = 0.04 even trained on its exact stone texture at its exact
   scale. Two walls: model can't localize its real OOD context; synth_val is
   pinned by layout-inference.
5. **The eval instrument is NOISY.** 10ep / 28-real / single-seed runs have
   ~0.01–0.07 run-to-run noise — bigger than most effects. A "+0.101 floor"
   that looked real didn't reproduce. **Always use `probe_multiseed.py`, ≥3
   seeds, and only trust an effect that clears the reported ± band.** Single-run
   reads have repeatedly fooled us.

What *did* move held-out real: **adding genuinely distinct real PLANS** (count of
distinct plans, not image volume). 0 real → 0.24 ceiling; 14 distinct → 0.275; 40
distinct → 0.32; 100% real → memorization. (Superseded detail in §0: the ceiling
is ~0.35 under the new default training regime, and adding copies/augs of the
*same* plans does NOT help — only distinct plans do.)

## 4. The current visual gap (your vision job)

I rendered real vs synth (see `review_pairs/*.png`; regenerate with the snippet
in §6). The tells that currently give synth away, ranked by impact:

1. **Floating colored lollipop markers** — red/blue/magenta circles on leader
   sticks scattered over synth elevations. Real plans never have these; they're
   reference-tile annotation pins leaking into the raster. Biggest tell.
2. **Flat single-color fills with a tiled swatch hatch** — synth stamps a whole
   mass one desaturated teal/green + uniform speckle. Real renders *per surface*:
   cream lap siding (fine horizontal lines), brown shingle roof (mottled), muted
   earth tones with lighting. No material identity in synth.
3. **No roof/wall material differentiation** — synth paints the whole silhouette
   one color; real has distinct roof vs wall materials.
4. **Crude windows** — synth = black rectangle + 1 mullion; real = frame,
   muntins, sill, casing in thin precise CAD linework, with depth.
5. **Missing detail/depth** — real has eaves, fascia, soffits, drop shadows,
   dimension ticks, vanishing-point construction lines. Synth is bare.
6. **Palette** — synth reaches for saturated arbitrary colors; real is muted
   naturals (cream, taupe, brown, grey).
7. **Floorplan coverage thin** — synth modes are ~65% elevation / 30% roof_plan
   / 5% `freeform` (= *sparse* code-based floor plans). Real includes **dense
   construction floorplans** (double-line walls, fixtures, room labels, curved
   callout leaders everywhere). This is a *coverage* gap (matters most per the
   thesis), not a cosmetic one.

**STATUS (2026-06-22, see §0): #1 and #7 have now BOTH been tested at 5 seeds and
are NULL.** Killing the lollipops (`--markup-overlay-prob 0`) is neutral; dense
floorplans (`--mode-weights 40,40,20`, ±clutter) looked like a win at 3 seeds but
washed out to tied-with-base at 5. They lower synth_iou and zero divergence
without moving real (the §2 trap). #2–#6 are appearance → expect Dead-End-#1.
Treat the list below as catalogued tells, not an open to-do list — anything you
retry here must clear the §0 rigor bar on held-out real.

## 5. Environment setup (fresh container)

```bash
pip install -r requirements.txt   # torch, torchvision, datasets, pillow, opencv, numpy
# Datasets (HuggingFace, public under abshetty/):
#   abshetty/floz-synth-v5 : default config = synth train; config "real-world-test"
#       split "test" = the 28 REAL plans (eval set).
#   abshetty/floz-assets   : reference tiles for synth generation.
# First load auto-downloads to ~/.cache/huggingface. Find the tiles dir:
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('abshetty/floz-assets', repo_type='dataset'))"
# tiles live under <snapshot>/reference_tiles_curated
```

Hardware we ran on: **RTX 5070 12GB, 28 vCPU, 15GB RAM, WSL2.** Constraints:
- 12GB GPU → train at `--image-max-size 1024` with the default regime
  (`--batch-size 1 --grad-accum 4 --freeze-backbone-bn`, all on by default as of
  2026-06-22) — also the confirmed-best recipe for real_iou, not just a memory
  workaround (+0.03-0.08 vs batch-8; see synth_progress.md). bf16 autocast on by default.
- **15GB RAM is the tight one.** Local-data loading is lazy (stores image PATHS,
  decodes per-item) — keep it that way or you OOM at a few thousand images.

## 6. Tooling & exact commands

**Generate canonical synth** (modes `['elevation','freeform','roof_plan']`):
```bash
TILES=<assets-snapshot>/reference_tiles_curated
python generate_synthetic_v5.py --n 5000 --seed 5858 --tiles "$TILES" \
  --out /tmp/synth_broad5000 --workers 28 \
  --dense-fill-scope instance --dense-fill-frac 0.45 --dense-fill-opacity 0.18 \
  --mode-weights 65,5,30 --markup-overlay-prob 0.35 --mono-image-prob 0.5 --realstyle-prob 1.0
```

**The eval instrument** (THE way to judge any change — never trust 1 seed):
```bash
HELD="12,16,27,7,11,25,23,1,18,2,0,3,14,24"   # held-out 14 real (split = random.Random(1234) over the 28)
python scripts/probe_multiseed.py --local-data /tmp/<dataset> --seeds 0,1,2 --epochs 10 \
  --tag MYRUN --extra="--domain-random --real-eval-indices $HELD"
# NOTE: --extra MUST use the = form (leading -- otherwise confuses argparse).
# Reports per-epoch mean±std + plateau (last 3 epochs) + a ± trust band.
```
`--extra` is appended verbatim to `train.py`, so add any train flag there
(e.g. `--backbone-stem-pool blur`).

**The "best-of-both" parity setup** (where divergence ≈ 0, real ≈ 0.32): broad
synth + ~10% augmented real (4 photometric augs/scene × the 14 train reals),
merged. Build it reproducibly:
```bash
python scripts/build_mix.py --synth-dir /tmp/synth_500 --out /tmp/mix_best --aug-per-scene 4
# prints the held-out 14 indices to pass to --real-eval-indices, and real_frac.
# Real FRACTION scales with synth size: ~500 synth + 56 aug = ~10%% real.
```

**View images (your strength).** Files in the container aren't host-mounted;
render to PNG and read them directly, or push to GitHub:
```python
import io, glob, random
from PIL import Image
from refmask2former import load_parquet_records
recs = load_parquet_records("abshetty/floz-synth-v5", cache_dir="./data",
                            config="real-world-test", split="test")  # 28 real
img = Image.open(io.BytesIO(recs[4]["image"])).convert("RGB")        # a real elevation
# synth images: /tmp/synth_broad5000/images/*.png + matching annotations/*.json
```
In JupyterLab you can also `scripts/build_review_pairs.py` for a browsable
real-vs-synth folder with GT overlays.

## 7. The loop to run

1. Pick ONE genuine COVERAGE hypothesis (an image-type/structure real has that
   synth never generates) — NOT #1/#7 (tested null, §0) and NOT appearance (§3-#1).
2. Implement it in `generate_synthetic_v5.py` behind a default-OFF flag.
3. Generate ~500 synth with the change; build the mix (`build_mix.py`, include the
   Roboflow reals via `--real-extra-dir`).
4. `probe_multiseed.py`, **≥5 seeds**, 10 epochs, eval held-out 14, in the default
   regime (bs1/ga4/freeze-bn — now train.py's default).
5. **Accept only if held-out real_iou improves by ≥0.03 with the two arms' STANDARD
   ERROR separating them** (judge real_iou DIRECTLY, not divergence — a change that
   only shrinks divergence by making synth harder is the §2 trap, not a win).
   Record the result (pos or neg) in `synth_progress.md`.
6. Reality check: as of 2026-06-22 NO generator edit has cleared this bar; the
   working levers are labelling more distinct real and the training regime (§0).
   If you can't clear it, that's the expected outcome — say so and stop, don't
   chase noise.

## 8. Pointers

- Lab notebook: `synth_progress.md` (STATE OF PLAY header first).
- Generator: `generate_synthetic_v5.py` (modes, realstyle, dense-fill, markup).
- Model: `refmask2former/{backbone,model,pixel_decoder,transformer_decoder,
  reference_encoder,dataset,criterion,matcher}.py`.
- Train: `train.py`. Eval instrument: `scripts/probe_multiseed.py`.
- Prior findings/memory (Claude): closing-divergence-method, divergence-north-star,
  divergence-is-ood-coverage, synth-divergence-dead-ends, eval-instrument-too-noisy,
  line-art-backbone-dead-end.

**One sentence to keep in front of you:** the gap is distribution *coverage*, not
realism-as-appearance and not model capacity — find what real contains that synth
doesn't *generate at all*, and generate it.
