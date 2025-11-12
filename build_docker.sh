#!/bin/bash

# Build Docker image for LoRA fine-tuning
# Usage: ./build_docker.sh [tag]

TAG=${1:-latest}
IMAGE_NAME="lora-finetuning"

echo "======================================================================"
echo "Building Docker Image: ${IMAGE_NAME}:${TAG}"
echo "======================================================================"

# Build the image
docker build -t ${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Docker image built successfully!"
    echo "======================================================================"
    echo ""
    echo "To run unit test:"
    echo "  docker run --rm ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To run full training (GPU):"
    echo "  docker run --gpus all --rm -v \$(pwd)/output:/workspace/output ${IMAGE_NAME}:${TAG} python3 lora_finetuning.py"
    echo ""
    echo "To run full training (CPU):"
    echo "  docker run --rm -v \$(pwd)/output:/workspace/output ${IMAGE_NAME}:${TAG} python3 lora_finetuning.py"
    echo ""
    echo "To push to Docker Hub:"
    echo "  docker tag ${IMAGE_NAME}:${TAG} YOUR_USERNAME/${IMAGE_NAME}:${TAG}"
    echo "  docker push YOUR_USERNAME/${IMAGE_NAME}:${TAG}"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ Docker build failed!"
    echo "======================================================================"
    exit 1
fi

