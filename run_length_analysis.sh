#!/bin/bash

# Script to analyze token lengths in the MetaMathQA dataset

echo "=========================================="
echo "Analyzing Dataset Token Lengths"
echo "=========================================="
echo ""
echo "This will help you determine optimal values for:"
echo "  - max_length (for training)"
echo "  - max_new_tokens (for generation)"
echo ""
echo "Currently set to 2048, which may be excessive..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "Installing numpy..."
    pip install numpy
fi

python3 analyze_dataset_lengths.py

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Review the recommendations above"
echo "2. Update config.py with suggested values"
echo "3. Update the values in all .py files accordingly"
echo ""
