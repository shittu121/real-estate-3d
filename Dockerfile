# ── Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04 ────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── Environment ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JOBS_DIR=/jobs

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        # OpenCV runtime dependencies
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        # FFmpeg for video encoding
        ffmpeg \
        # Miscellaneous
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3 / pip3 point to 3.10 (already the default on 22.04, but be explicit)
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10

# ── Python dependencies (cached layer — only rebuilds when requirements change)
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# ── PyTorch CUDA wheel (installed after base requirements to override CPU build)
# Pick the wheel that matches your CUDA version (12.1 → cu121)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ── Project source ────────────────────────────────────────────────────────────
COPY . .

# ── Persistent storage for jobs and HuggingFace model cache ──────────────────
RUN mkdir -p /jobs /root/.cache/huggingface

# Expose the API port
EXPOSE 8000

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["bash", "start.sh"]
