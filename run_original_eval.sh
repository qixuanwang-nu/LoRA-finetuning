#!/bin/bash

# Script to evaluate the original Qwen3-0.6B model without training

echo "=========================================="
echo "Evaluating Original Qwen3-0.6B Model"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
if ! python3 -c "import torch, transformers, datasets" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

echo "Starting evaluation..."
echo "=========================================="
echo ""

python3 evaluate_original_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: original_model_evaluation.json"
else
    echo ""
    echo "Error: Evaluation failed"
    exit 1
fi
