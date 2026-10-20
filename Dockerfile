# GPU-ready image for training + inference + web UI.
# Base image ships PyTorch with CUDA 12.8 (works with the NVIDIA Container Toolkit; falls back to CPU without a GPU).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NOCHMAL_CHECKPOINTS=/app/checkpoints \
    NOCHMAL_BOARDS=/app/boards

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nochmal ./nochmal
COPY web ./web
COPY boards ./boards
COPY tests ./tests

EXPOSE 8000
CMD ["python", "-m", "nochmal.server"]
