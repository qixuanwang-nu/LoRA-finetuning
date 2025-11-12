# Dockerfile for LoRA Fine-tuning with Qwen Model
# Base image with CUDA support for GPU acceleration
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY *.py .
COPY *.sh .
COPY *.md .

# Make shell scripts executable
RUN chmod +x *.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_HOME=/workspace/.cache/huggingface

# Create cache directory
RUN mkdir -p /workspace/.cache/huggingface

# Default command: run unit test
CMD ["python3", "unit_test.py"]

# To run full training, use:
# docker run --gpus all -v $(pwd)/output:/workspace/output your-image python3 lora_finetuning.py

