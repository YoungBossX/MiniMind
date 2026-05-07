#!/usr/bin/env bash
# MiniMind 一键评估示例脚本
#
# 用法:
#   bash scripts/run_eval.sh
#   bash scripts/run_eval.sh /path/to/checkpoint.pth

set -euo pipefail

CHECKPOINT="${1:-out/dpo_512.pth}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== MiniMind Full Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

# 1. Language Model Evaluation
echo "--- 1/4: Language Model Evaluation ---"
python evals/eval_lm.py \
  --checkpoint_path "$CHECKPOINT" \
  --data_path evals/data/lm_eval_sample.txt \
  --output_path outputs/evals/lm_eval.json \
  --batch_size 4 \
  --max_length 512

# 2. QA Evaluation
echo ""
echo "--- 2/4: QA Evaluation ---"
python evals/eval_qa.py \
  --checkpoint_path "$CHECKPOINT" \
  --data_path evals/data/qa_eval_sample.jsonl \
  --output_path outputs/evals/qa_eval.json \
  --predictions_path outputs/evals/qa_predictions.jsonl

# 3. Generation Constraint Evaluation
echo ""
echo "--- 3/4: Generation Constraint Evaluation ---"
python evals/eval_generation.py \
  --checkpoint_path "$CHECKPOINT" \
  --data_path evals/data/generation_eval_sample.jsonl \
  --output_path outputs/evals/generation_eval.json \
  --predictions_path outputs/evals/generation_predictions.jsonl

# 4. Speed Evaluation
echo ""
echo "--- 4/4: Speed Evaluation ---"
python evals/eval_speed.py \
  --checkpoint_path "$CHECKPOINT" \
  --output_path outputs/evals/speed_eval.json

echo ""
echo "=== All evaluations complete ==="
echo "Results saved to outputs/evals/"

# Generate summary report (optional — run_all.py also does this)
if command -v python &> /dev/null; then
  python evals/run_all.py --checkpoint_path "$CHECKPOINT" 2>/dev/null || true
fi
