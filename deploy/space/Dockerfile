# Hugging Face Space (Docker SDK) serving the box-and-see front end (deploy/app.py); the model runs on the Runpod endpoint.
# Space secrets: HF_TOKEN (read access to abshetty/floz-synth-v5), RUNPOD_API_KEY. Uploaded by deploy/space/push.py.
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app
COPY --chown=user requirements-app.txt .
RUN pip install --user -r requirements-app.txt
COPY --chown=user deploy/ deploy/
EXPOSE 7860
CMD ["python", "deploy/app.py", "--endpoint", "5crvpzv67a8ed1", "--host", "0.0.0.0", "--port", "7860"]
