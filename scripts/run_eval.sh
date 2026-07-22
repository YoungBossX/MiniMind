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

# run_all owns stage orchestration, stale-output isolation, and failure propagation.
python evals/run_all.py --checkpoint_path "$CHECKPOINT"
