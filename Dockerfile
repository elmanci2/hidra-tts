# Base image with CUDA 12.1 and Ubuntu 22.04 runtime (lightweight and safe)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HuggingFace cache directory (mount as volume for persistence)
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system dependencies and Python 3.11 via deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    sox \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*


# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set python3.11 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ============================================================
# PASO 1: Copiar SOLO archivos de dependencias (cambian poco)
# Docker cachea esta capa. Si no tocas pyproject.toml,
# no reinstala nada — rebuild instantáneo.
# ============================================================
COPY pyproject.toml setup.cfg* setup.py* README.md ./
COPY qwen_tts/ ./qwen_tts/

# Para garantizar la compatibilidad exacta con la rueda (.whl) de flash-attn,
# fijamos la versión de PyTorch ANTES de instalar el resto de la app
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Instalamos la rueda PRE-COMPILADA exacta (v2.7.4.post1) para PyTorch 2.6 y CUDA 12 + Python 3.11
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install the project and all remaining dependencies
RUN pip install --no-cache-dir .


# ============================================================
# PASO 2: Copiar el código de la app (cambia frecuentemente)
# Solo esta capa se rehace cuando modificas tu código.
# ============================================================
COPY src/ ./src/
COPY main.py ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "main.py"]
