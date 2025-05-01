#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# Initialize single log file
: > logs/judge_all_hotpotqa.log

BASE_DIR="dataset/hotpotqa"

for model_dir in "$BASE_DIR"/*; do
  [ -d "$model_dir" ] || continue
  model=$(basename "$model_dir")

  for csv in "$model_dir"/*.csv; do
    [ -f "$csv" ] || continue
    method=$(basename "$csv" .csv)

    echo "========================================"
    echo "執行模型：${model} 方法：${method}"
    echo "----------------------------------------"

    python reeval.py \
      -d "$csv" \
      -p "hotpotqa_llm_as_a_judge" \
      2>&1 | tee -a logs/judge_all_hotpotqa.log

    echo ""
  done
done