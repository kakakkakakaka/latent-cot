#!/bin/bash
# Fresh start training script for GSM8K Coconut with corrected iCoT data
# Training only 1 epoch to test if accuracy improves with corrected data

set -e

echo "=========================================="
echo "Fresh Start Training (1 Epoch)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: Qwen3-4B-Instruct (base model, fresh start)"
echo "  - Data: Corrected iCoT data with full reasoning steps"
echo "  - Epochs: 1 (test run)"
echo "  - Learning rate: 0.00001"
echo "  - Batch size: 18 (effective: 108 with gradient accumulation)"
echo ""

# Check if data exists
if [ ! -f "data/gsm8k_train.json" ]; then
    echo "❌ Error: Training data not found!"
    echo "   Please run: TARGET_SIZE=50000 bash preprocessing/gsm8k_icot.bash"
    exit 1
fi

# Check data format (should have multiple steps)
echo "Checking data format..."
if python3 -c "
import json
from collections import defaultdict

data = json.load(open('data/gsm8k_train.json'))
multi_step = [s for s in data if len(s.get('steps', [])) > 1]
print(f'Total samples: {len(data)}')
print(f'Samples with multiple steps: {len(multi_step)} ({len(multi_step)/len(data)*100:.1f}%)')

if len(multi_step) < len(data) * 0.5:
    print('⚠️  Warning: Less than 50% of samples have multiple steps')
    print('   This may indicate data format issue')
    exit(1)
else:
    print('✅ Data format looks correct')
    exit(0)
" 2>/dev/null; then
    echo "✅ Data check passed"
else
    echo "❌ Data format check failed"
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

# Run training with output redirection for logging
LOG_DIR="/tmp/checkpoints/gsm8k-coconut-qwen3-a100-fresh"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training.log"

echo "Training output will be logged to: $LOG_FILE"
echo "You can monitor with: tail -f $LOG_FILE"
echo ""

# Run training with output redirection
torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm8k_coconut_qwen3_a100_fresh_start.yaml 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  - Checkpoint: /tmp/checkpoints/gsm8k-coconut-qwen3-a100-fresh/checkpoint_1"
echo "  - History: /tmp/checkpoints/gsm8k-coconut-qwen3-a100-fresh/gsm8k-coconut-qwen3-a100-fresh_history.json"
echo ""

