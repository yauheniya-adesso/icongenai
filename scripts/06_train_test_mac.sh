#!/usr/bin/env bash
# =============================================================================
#  06_train_test_mac.sh — Full pilot fine-tuning on Apple Silicon (M4 24 GB)
#
#  Runs a complete LoRA fine-tune of Qwen2.5-Coder-1.5B on a representative
#  subset of the training data, then generates SVGs and evaluates metrics.
#  Produces concrete before/after numbers (VR, RSR, MPC) suitable for a
#  GPU access application.
#
#  Prerequisites:
#    uv run scripts/05_prepare.py     # produces data/{train,valid,test}.jsonl
#
#  Usage:
#    bash scripts/06_train_test_mac.sh                     # full pilot (~3-4 h total)
#    bash scripts/06_train_test_mac.sh --iters 500         # override iters
#    bash scripts/06_train_test_mac.sh \
#        --resume-adapter-file models/pilot-lora/adapters.npz
#
#  Generation is resumable — safe to interrupt and restart.
# =============================================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
TRAIN_N=100000          # training records in pilot subset
VAL_N=500              # validation records (used during training)
TEST_N=100            # test records for generation + evaluation (~2-3 h to generate)
EPOCHS=3
BATCH=1
GRAD_ACCUM=8           # effective batch = 8
LORA_LAYERS=16         # out of 28 total
MAX_SEQ=4096
LR=5e-5
VAL_BATCHES=100
SAVE_EVERY=500
STEPS_PER_EVAL=$SAVE_EVERY   # align val loss with checkpoint saves

DATA_DIR="data"
PILOT_DIR="data/pilot"
ADAPTER_DIR="models/pilot-lora"
RESULTS_DIR="results/pilot"

EFFECTIVE_BATCH=$((BATCH * GRAD_ACCUM))

mkdir -p "$PILOT_DIR" "$ADAPTER_DIR" "$RESULTS_DIR" logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_mac_${TIMESTAMP}.log"

# ── Step 1: Create pilot data subset ─────────────────────────────────────────
if [[ -f "$PILOT_DIR/train.jsonl" ]]; then
    echo "Pilot data already exists at $PILOT_DIR — skipping subset creation."
else
    echo "Creating pilot data subset (${TRAIN_N} train / ${VAL_N} val / ${TEST_N} test)…"
    python3 << EOF
import json, random
def sample(src, n, seed=42):
    rows = [json.loads(l) for l in open(src) if l.strip()]
    random.seed(seed)
    random.shuffle(rows)
    return rows[:min(n, len(rows))]
for split, n in [("train", $TRAIN_N), ("valid", $VAL_N), ("test", $TEST_N)]:
    rows = sample(f"$DATA_DIR/{split}.jsonl", n)
    with open(f"$PILOT_DIR/{split}.jsonl", "w") as f:
        [f.write(json.dumps(r) + "\n") for r in rows]
    print(f"  {split}: {len(rows):,} records  →  $PILOT_DIR/{split}.jsonl")
EOF
fi

# ── Step 2: Compute iters for EPOCHS full passes ──────────────────────────────
ACTUAL_TRAIN=$(wc -l < "$PILOT_DIR/train.jsonl" | tr -d ' ')
ITERS=$(python3 -c "import math; print(math.ceil($ACTUAL_TRAIN / $EFFECTIVE_BATCH) * $EPOCHS)")
EST_MIN=$(python3 -c "print(round($ITERS / 90))")   # ~1.5 iters/sec on M4

echo ""
echo "============================================================"
echo "  IconGenAI — pilot fine-tuning"
echo "  Model    : $MODEL"
echo "  Data     : ${ACTUAL_TRAIN} records × ${EPOCHS} epochs = ${ITERS} iters"
echo "  Batch    : ${BATCH} × grad_accum ${GRAD_ACCUM} = effective ${EFFECTIVE_BATCH}"
echo "  LoRA     : ${LORA_LAYERS} layers  lr=${LR}"
echo "  Train ETA: ~${EST_MIN} min  (+2-3 h generation for ${TEST_N} test icons)"
echo "  Adapter  : $ADAPTER_DIR"
echo "  Log      : $LOG_FILE"
echo "============================================================"
echo ""

# ── Step 3: Train ─────────────────────────────────────────────────────────────
caffeinate -i uv run mlx_lm.lora \
    --model                   "$MODEL"       \
    --train                                  \
    --data                    "$PILOT_DIR"   \
    --batch-size              $BATCH         \
    --grad-accumulation-steps $GRAD_ACCUM    \
    --num-layers              $LORA_LAYERS   \
    --max-seq-length          $MAX_SEQ       \
    --iters                   $ITERS         \
    --learning-rate           $LR            \
    --mask-prompt                            \
    --val-batches             $VAL_BATCHES   \
    --steps-per-eval          $STEPS_PER_EVAL \
    --save-every              $SAVE_EVERY    \
    --adapter-path            "$ADAPTER_DIR" \
    "$@"                                     \
    2>&1 | grep -Eav "(Calculating loss|Fetching [0-9]+ files|[0-9]+%\|)" \
         | tee "$LOG_FILE"

echo ""
echo "Training complete."

# ── Step 4: Generate — fine-tuned model ──────────────────────────────────────
echo ""
echo "Generating SVGs (fine-tuned)…"
uv run scripts/07_generate.py \
    --model   "$MODEL" \
    --adapter "$ADAPTER_DIR" \
    --test    "$PILOT_DIR/test.jsonl" \
    --output  "$RESULTS_DIR/generated.jsonl"

# ── Step 5: Generate — zero-shot baseline ────────────────────────────────────
echo ""
echo "Generating SVGs (zero-shot baseline)…"
uv run scripts/07_generate.py \
    --model      "$MODEL" \
    --no-adapter \
    --test       "$PILOT_DIR/test.jsonl" \
    --output     "$RESULTS_DIR/zeroshot.jsonl"

# ── Step 6: Evaluate both ────────────────────────────────────────────────────
echo ""
echo "Evaluating (VR, RSR, MPC)…"
uv run scripts/08_evaluate.py \
    --input "$RESULTS_DIR/generated.jsonl" \
    --skip-clip --skip-fid

uv run scripts/08_evaluate.py \
    --input "$RESULTS_DIR/zeroshot.jsonl" \
    --skip-clip --skip-fid

# ── Step 7: Compare ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
uv run scripts/09_compare.py \
    "Qwen2.5-1.5B zero-shot:$RESULTS_DIR/zeroshot.metrics.json" \
    "Qwen2.5-1.5B + LoRA (pilot):$RESULTS_DIR/generated.metrics.json"

echo ""
echo "============================================================"
echo "  Pilot run complete."
echo "  Adapter : $ADAPTER_DIR"
echo "  Results : $RESULTS_DIR/generated.metrics.json"
echo "  Log     : $LOG_FILE"
echo "============================================================"
