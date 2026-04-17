#!/usr/bin/env bash
# =============================================================================
#  04_train.sh — LoRA fine-tune Qwen3.5-9B on the IconGenAI dataset with mlx-lm
#
#  Prerequisites:
#    uv run scripts/03_prepare.py      # produces data/{train,valid,test}.jsonl
#    mkdir -p models logs
#
#  Usage:
#    bash scripts/04_train.sh                  # full run with defaults below
#    bash scripts/04_train.sh --iters 500      # quick feasibility check (~5 min)
#    bash scripts/04_train.sh --batch-size 2   # if OOM on smaller hardware
#
#  All extra arguments are forwarded verbatim to mlx_lm.lora, so any flag
#  accepted by that command can be passed here.
# =============================================================================
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL="mlx-community/Qwen3.5-9B-MLX-4bit"
DATA_DIR="data"
ADAPTER_DIR="models/icongenai-lora"
mkdir -p "$ADAPTER_DIR" logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"

# ── Default hyperparameters (override via command-line args passed through) ──
# These are baked into the call below; use the forwarded $@ for overrides.
BATCH_SIZE=2           # M4 24 GB: 2 is safe; use 1 if still OOM
GRAD_ACCUM=4           # effective batch = BATCH_SIZE × GRAD_ACCUM = 8
LORA_LAYERS=8          # 8 layers keeps memory reasonable on 24 GB
MAX_SEQ_LEN=2048       # keeps activation memory tractable on M4 24 GB
                       # most icon SVGs tokenize to <400 tokens; raise for cloud GPU
ITERS=10000            # ~1 epoch on ~30 K augmented records
LR=2e-4
SAVE_EVERY=500
VAL_BATCHES=25

echo "============================================================"
echo "  IconGenAI LoRA training"
echo "  Model    : $MODEL"
echo "  Data     : $DATA_DIR/"
echo "  Adapter  : $ADAPTER_DIR/"
echo "  Log      : $LOG_FILE"
echo "  -- default hyperparameters (may be overridden by extra args below) --"
echo "  Iters    : $ITERS"
echo "  Batch    : $BATCH_SIZE × $GRAD_ACCUM accum = $(($BATCH_SIZE * $GRAD_ACCUM)) effective"
echo "  LR       : $LR"
echo "  LoRA layers: $LORA_LAYERS"
echo "  Max seq  : $MAX_SEQ_LEN tokens"
echo "  Save every: $SAVE_EVERY"
echo "  Extra args: $*"
echo "============================================================"
echo ""

# caffeinate keeps the Mac awake for the full training run
caffeinate -i uv run mlx_lm.lora \
    --model                    "$MODEL"       \
    --train                                   \
    --data                     "$DATA_DIR"    \
    --batch-size               "$BATCH_SIZE"  \
    --grad-accumulation-steps  "$GRAD_ACCUM"  \
    --num-layers               "$LORA_LAYERS" \
    --max-seq-length           "$MAX_SEQ_LEN" \
    --iters                    "$ITERS"       \
    --learning-rate            "$LR"          \
    --val-batches              "$VAL_BATCHES" \
    --save-every               "$SAVE_EVERY"  \
    --adapter-path             "$ADAPTER_DIR" \
    --grad-checkpoint                         \
    "$@"                                      \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete."
echo "  Adapter : $ADAPTER_DIR"
echo "  Log     : $LOG_FILE"
echo ""
echo "Next step:"
echo "  uv run scripts/05_generate.py --adapter $ADAPTER_DIR"
