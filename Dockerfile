# Base image with CUDA 12.1 and Ubuntu 22.04
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

# Copy project files
COPY . .

# Install the project and all dependencies
RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "main.py"]
