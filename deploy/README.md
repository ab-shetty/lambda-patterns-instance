# Box-and-see app on Runpod Serverless

`deploy/handler.py` is a Runpod Serverless worker running the shipped product (fgavg78 + `scripts/product.py`) for one box;
`deploy/app.py` + `app.html` is the front end (pick one of the 28 real sheets, drag a box, see the polygons with the human
label for comparison, download JSON). `deploy/segment.py` is the shared inference code (checked against the evaluator: HF14
sheet 12 q0 / q1 0.9311 / 0.9490 vs 0.9323 / 0.9472 GPU-scored, CPU float32 vs GPU bf16).

## Create the endpoint (Runpod console, once)

1. Settings -> connect GitHub; give Runpod access to `ab-shetty/lambda-patterns-instance`.
2. Serverless -> New Endpoint -> Import Git Repository: branch `reference-conditioned-mixed-real-2026-07-16`,
   Dockerfile path `deploy/Dockerfile` (build context: repository root).
3. Settings:
   - GPU: 24 GB (RTX 4090 / A5000 class); if workers fail with CUDA out-of-memory at the 5120 px scale, use 48 GB.
   - Workers: max 1, active (always-on) 0; idle timeout ~120 s, so boxes in one session reuse a warm worker; FlashBoot on.
   - Execution timeout 600 s; container disk 20 GB.
   - Advanced -> allowed CUDA versions: 12.8 and newer (the image is built on CUDA 12.8; older host drivers can't run it).
   - Environment: `HF_TOKEN` = a Hugging Face read token for the private model repo
     (`abshetty/floz-refunet-swint-fgavg78`), preferably as a Runpod secret.
4. Copy the endpoint ID.

Later code changes reach the endpoint through a new GitHub release (Runpod rebuilds on releases).

## Run the front end (your machine; no GPU or torch needed)

    pip install -r deploy/requirements-app.txt
    export HF_TOKEN=...          # read access to the private dataset abshetty/floz-synth-v5 (the 28 sheets)
    export RUNPOD_API_KEY=...    # Runpod key with access to the endpoint
    python3 deploy/app.py --endpoint 5crvpzv67a8ed1
    open http://127.0.0.1:7860

Checked 2026-11-24 on the endpoint (RTX PRO 6000 Blackwell MIG 24 GB slice): HF14 sheet 12 q0/q1/q2 IoU 0.9316/0.9475/0.9121
(eval 0.9323/0.9472), ~1.6 s model time, 2.5-5 s per box warm. The first box after the worker has been idle waits for it to
start. `python3 deploy/app.py --local data/runs/ck_best/fgavg78.pth` runs the model in-process instead (needs torch; GPU
machine; 20-80 s per box on CPU).
