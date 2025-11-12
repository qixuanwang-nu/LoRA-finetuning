#!/bin/bash

# LoRA Fine-tuning Training Script
# This script sets up the environment and runs the LoRA fine-tuning

echo "=========================================="
echo "LoRA Fine-tuning with PEFT"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
if ! python3 -c "import torch, transformers, peft, datasets" 2>/dev/null; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
    echo "Dependencies installed successfully!"
else
    echo "All dependencies are already installed."
fi
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run the training script
echo "Starting LoRA fine-tuning..."
echo "=========================================="
echo ""

python3 lora_finetuning.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - lora_finetuned_model/          (Fine-tuned model)"
    echo "  - original_model_results.json    (Original model evaluation)"
    echo "  - finetuned_model_results.json   (Fine-tuned model evaluation)"
    echo "  - comparison_results.json        (Performance comparison)"
else
    echo ""
    echo "=========================================="
    echo "Training failed with errors"
    echo "=========================================="
    exit 1
fi
