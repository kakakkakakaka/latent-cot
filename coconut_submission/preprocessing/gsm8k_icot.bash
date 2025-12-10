#!/bin/bash

# Preprocessing script for GSM8K dataset with iCoT enhancement (controllable size)
# Downloads iCoT-enhanced GSM8K data and converts to coconut training format
# Supports sampling to control dataset size (default: 70k-100k for training set)

set -e

echo "=========================================="
echo "GSM8K Dataset Preprocessing (iCoT Enhanced)"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Download iCoT-enhanced GSM8K dataset"
echo "  2. Convert to coconut training format"
echo "  3. Sample training set to target size (default: 85000)"
echo "  4. Save to data/ directory"
echo ""
echo "iCoT Enhancement:"
echo "  - Original training set: ~7,473 examples"
echo "  - iCoT-enhanced training set: ~385,620 examples"
echo "  - After sampling: TARGET_SIZE examples (default: 85000)"
echo "  - Validation set: ~1,319 examples (using test split)"
echo ""

# Check if datasets library is installed (for HuggingFace download)
if ! python -c "import datasets" 2>/dev/null; then
    echo "Warning: datasets library not found. Will use direct download from GitHub."
    echo "  To install: pip install datasets"
    echo ""
fi

# Check if requests is installed (for GitHub download)
if ! python -c "import requests" 2>/dev/null; then
    echo "Error: requests library not found. Please install it:"
    echo "  pip install requests"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Set target size for training set
# With 5-8 samples per question (7,473 questions):
#   Minimum: 7,473 * 5 = 37,365 samples
#   Maximum: 7,473 * 8 = 59,784 samples
#   Recommended: 45,000-55,000 samples
TARGET_SIZE=${TARGET_SIZE:-50000}  # Default: 50,000 (about 6.7 samples per question)
echo "Target training set size: ${TARGET_SIZE} examples"
echo "Each question will have 5-8 samples (sufficient for Stage 1)"
echo ""

echo "Processing training set..."
python preprocessing/gsm8k_icot.py \
    --split train \
    --output-dir data \
    --target-size ${TARGET_SIZE}

echo ""
echo "Processing validation set (test split, no sampling)..."
python preprocessing/gsm8k_icot.py \
    --split test \
    --output-dir data \
    --target-size None

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data/gsm8k_train.json (${TARGET_SIZE} samples)"
echo "  - data/gsm8k_validation.json (~1,319 samples)"
echo ""
echo "You can now use these files for training:"
echo "  train_path: data/gsm8k_train.json"
echo "  val_path: data/gsm8k_validation.json"
echo ""
echo "To change target size, set TARGET_SIZE environment variable:"
echo "  TARGET_SIZE=70000 bash preprocessing/gsm8k_icot.bash"
echo "  TARGET_SIZE=100000 bash preprocessing/gsm8k_icot.bash"
echo ""

