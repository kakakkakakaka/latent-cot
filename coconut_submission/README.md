# Coconut Training and Evaluation Project

This project implements the Coconut method for training language models with continuous thoughts on GSM8K dataset using Qwen3-4B-Instruct.

## Project Structure

```
coconut_dependencies/
├── coconut.py              # Core Coconut model implementation
├── dataset.py              # Dataset processing and latent token handling
├── run.py                  # Main training and evaluation script
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
├── args/                   # Configuration files
│   ├── gsm8k_coconut_qwen3_a100_fresh_start.yaml      # Fresh start training config
│   ├── gsm8k_coconut_qwen3_a100_fresh_continue.yaml   # Continue training config
│   └── gsm8k_coconut_qwen3_a100_fresh_eval.yaml        # Evaluation config
├── preprocessing/          # Data preprocessing scripts
│   ├── gsm8k_icot.py      # iCoT data processing
│   └── gsm8k_icot.bash    # Preprocessing bash script
├── data/                   # Data files (should be provided separately)
│   ├── gsm8k_train.json
│   └── gsm8k_validation.json
├── train_fresh_start.sh    # Script to start fresh training
└── train_continue_fresh.sh # Script to continue training
```

## Quick Start

### 1. Installation

```bash
cd coconut_dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place the following files in `coconut_dependencies/data/`:
- `gsm8k_train.json` - Training data with iCoT format
- `gsm8k_validation.json` - Validation data

Or preprocess from scratch:
```bash
cd coconut_dependencies/preprocessing
TARGET_SIZE=50000 bash gsm8k_icot.bash
```

### 3. Training

**Fresh Start:**
```bash
cd coconut_dependencies
bash train_fresh_start.sh
```

**Continue Training:**
```bash
cd coconut_dependencies
# Update args/gsm8k_coconut_qwen3_a100_fresh_continue.yaml first
bash train_continue_fresh.sh
```

### 4. Evaluation

```bash
cd coconut_dependencies
# Update args/gsm8k_coconut_qwen3_a100_fresh_eval.yaml:
# - Set load_model_path to your checkpoint
# - Set resume to checkpoint epoch number
WANDB_DISABLED=true torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm8k_coconut_qwen3_a100_fresh_eval.yaml
```

## Key Parameters

### Training Configuration

- `c_thought: 2` - Number of latent tokens per stage (c ∈ {0, 1, 2} in paper)
- `epochs_per_stage: 2` - Number of epochs per stage
- `max_latent_stage: 3` - Maximum stage number
- `batch_size_training: 16` - Training batch size
- `gradient_accumulation_steps: 6` - Effective batch = 16 × 6 = 96
- `max_seq_length: 512` - Maximum sequence length
- `lr: 1.0e-05` - Learning rate
- `num_epochs: 20` - Total training epochs

### Stage Mechanism

The training uses curriculum learning with stages:

- **Stage 0** (Epochs 1): 0 latent tokens, learns all reasoning steps
- **Stage 1** (Epochs 2-3): 2 latent tokens (c_thought=2), skips 1 step
- **Stage 2** (Epochs 4-5): 4 latent tokens, skips 2 steps
- **Stage 3** (Epochs 6+): 6 latent tokens, skips 3 steps

Stage is calculated as: `stage = epoch // epochs_per_stage`

## Model Checkpoints

Checkpoints are saved at https://huggingface.co/sleepyyZ/coconut_qwen3_4B_Instruct_checkpoint/tree/main

## Results

Evaluation results include:
- Accuracy on validation set
- CoT (Chain-of-Thought) match rate
- Validation examples saved to `validation_examples.txt`

## Hardware Requirements

- **GPU**: A100 80GB (or similar with sufficient memory)
- **CUDA**: 11.8+ (compatible with PyTorch 2.5.1)
- **Memory**: ~40GB GPU memory for training
- **Storage**: ~9GB per checkpoint

## Notes

- The code uses FSDP (Fully Sharded Data Parallel) for distributed training
- Wandb logging can be disabled by setting `WANDB_DISABLED=true`
- For evaluation, use `batch_size=1` in the DataLoader for consistency
- Checkpoints should be provided separately due to size (~9GB each)
