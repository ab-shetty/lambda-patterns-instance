# Runpod Serverless worker for the pattern-segmentation product (deploy/handler.py). Build context = the repository root:
# in the Runpod console, New Endpoint -> Import Git Repository, this branch, Dockerfile path deploy/Dockerfile.
# The model is downloaded from the private Hub repo at start-up: set HF_TOKEN in the endpoint's environment (a Runpod secret).
# CUDA 12.8 build: torch 2.5/cu124 has no kernels for Blackwell GPUs (sm_120), which Runpod's 24 GB pool can hand out.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY deploy/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY refmask2former/ refmask2former/
COPY scripts/ scripts/
COPY deploy/ deploy/
# torchvision's swin_t is built with weights=None (pretrained=False in load_refunet), so nothing else is downloaded.
WORKDIR /app/deploy
CMD ["python", "-u", "handler.py"]
