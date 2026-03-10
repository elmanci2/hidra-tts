# Base image with CUDA 12.8 devel (includes nvcc for compiling flash-attn)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HuggingFace cache directory (mount as volume for persistence)
ENV HF_HOME=/app/.cache/huggingface

# CUDA paths for flash-attn compilation
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /app

# Install system dependencies and Python 3.11 via deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    sox \
    ninja-build \
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

# Copy project files
COPY . .

# Install the project dependencies (without flash-attn, se instala aparte)
RUN pip install --no-cache-dir .

# Install flash-attn separately (requires nvcc from devel image)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "main.py"]
