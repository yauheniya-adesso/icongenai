#!/usr/bin/env bash
# =============================================================================
#  06_train.sh — QLoRA fine-tune Qwen3-Coder-30B-A3B on H100 80 GB
#
#  Prerequisites (on the GPU node):
#    pip install torch transformers peft trl bitsandbytes accelerate flash-attn
#    python scripts/05_prepare.py        # produces data/{train,valid,test}.jsonl
#    mkdir -p models logs
#
#  Usage:
#    bash scripts/06_train.sh                      # full run
#    bash scripts/06_train.sh --num_train_epochs 1 # quick test
#
#  All extra arguments are forwarded verbatim to 06_train.py.
# =============================================================================
set -euo pipefail

# ── Model ────────────────────────────────────────────────────────────────────
# Load the base (bf16) model; bitsandbytes quantises to 4-bit at runtime.
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR="data"
ADAPTER_DIR="models/icongenai-qlora"
mkdir -p "$ADAPTER_DIR" logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE=4          # H100 80 GB + 4-bit: 4 is safe
GRAD_ACCUM=4          # effective batch = 16
LORA_RANK=64
LORA_ALPHA=128
MAX_SEQ_LEN=4096
EPOCHS=3
LR=2e-4
SAVE_STEPS=500
WARMUP_RATIO=0.05

echo "============================================================"
echo "  IconGenAI QLoRA training"
echo "  Model    : $MODEL"
echo "  Data     : $DATA_DIR/"
echo "  Adapter  : $ADAPTER_DIR/"
echo "  Log      : $LOG_FILE"
echo "  Batch    : $BATCH_SIZE × $GRAD_ACCUM accum = $(($BATCH_SIZE * $GRAD_ACCUM)) effective"
echo "  LoRA     : rank=$LORA_RANK  alpha=$LORA_ALPHA"
echo "  Epochs   : $EPOCHS   LR: $LR   Max seq: $MAX_SEQ_LEN"
echo "  Extra args: $*"
echo "============================================================"
echo ""

python scripts/06_train.py \
    --model_name        "$MODEL"       \
    --data_dir          "$DATA_DIR"    \
    --output_dir        "$ADAPTER_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE"  \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lora_r            "$LORA_RANK"   \
    --lora_alpha        "$LORA_ALPHA"  \
    --max_seq_length    "$MAX_SEQ_LEN" \
    --num_train_epochs  "$EPOCHS"      \
    --learning_rate     "$LR"          \
    --save_steps        "$SAVE_STEPS"  \
    --warmup_ratio      "$WARMUP_RATIO" \
    "$@" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete."
echo "  Adapter : $ADAPTER_DIR"
echo "  Log     : $LOG_FILE"
echo ""
echo "Next step:"
echo "  python scripts/07_generate.py --adapter $ADAPTER_DIR"
