#!/bin/bash
# Continue training from checkpoint_1
# This will continue through Stage 1, Stage 2, and Stage 3 (curriculum learning)

set -e

echo "=========================================="
echo "Continue Training from Checkpoint_1"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Starting from: checkpoint_1 (Epoch 1 completed)"
echo "  - Curriculum Learning:"
echo "    - Stage 1: Epochs 1-2 (epochs_per_stage=2)"
echo "    - Stage 2: Epochs 3-4"
echo "    - Stage 3: Epochs 5-6"
echo "  - Max epochs: 20"
echo "  - Validation: 400 samples per epoch"
echo "  - Logs: Training and validation results will be saved"
echo ""

# Check if checkpoint_1 exists
CHECKPOINT_PATH="/tmp/checkpoints/gsm8k-coconut-qwen3-a100-fresh/checkpoint_1"
if [ ! -f "$CHECKPOINT_PATH" ] && [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint_1 not found!"
    echo "   Expected: $CHECKPOINT_PATH"
    exit 1
fi

echo "✅ Checkpoint_1 found"
echo ""

# Check if data exists
if [ ! -f "data/gsm8k_train.json" ]; then
    echo "❌ Error: Training data not found!"
    echo "   Please run: TARGET_SIZE=50000 bash preprocessing/gsm8k_icot.bash"
    exit 1
fi

echo "✅ Training data found"
echo ""

# Create log directory
LOG_DIR="/tmp/checkpoints/gsm8k-coconut-qwen3-a100-fresh"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training.log"

echo "Training output will be saved to: $LOG_FILE"
echo "Validation examples will be saved to: $LOG_DIR/validation_examples.txt"
echo "Training history will be saved to: $LOG_DIR/gsm8k-coconut-qwen3-a100-fresh_history.json"
echo ""
echo "Starting training..."
echo ""

# Run training with output to log file
# Using nohup to ensure it continues even if connection is lost
nohup torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm8k_coconut_qwen3_a100_fresh_continue.yaml > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "Training started in background. PID: $TRAIN_PID"
echo ""
echo "=========================================="
echo "Training initiated!"
echo "=========================================="
echo ""
echo "Monitor progress using:"
echo "  - tail -f $LOG_FILE"
echo "  - bash monitor_training_fresh.sh"
echo ""
echo "Check results:"
echo "  - Training log: $LOG_FILE"
echo "  - Validation examples: $LOG_DIR/validation_examples.txt"
echo "  - Training history: $LOG_DIR/gsm8k-coconut-qwen3-a100-fresh_history.json"
echo "  - Checkpoints: $LOG_DIR/checkpoint_*"
echo ""
echo "Note: Each epoch's validation will save the first 5 samples' complete inference results"
echo ""

