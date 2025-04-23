#!/bin/bash

# 指定 pipeline 方法: vanilla, rag, direct, long, coa, ragcoa-algo1, ragcoa-algo2
PIPELINE_METHOD="rag"  

# Dataset name
DATASET_NAME="narrativeqa"  # 可選擇的 dataset: hotpotqa, narrativeqa, triviaqa...
TOP_K_VALUES=(20 40 60 80 100) # for RAG

# LLM/tokenizer model
LLM_MODEL="gpt-4o"
TOKENIZER="gpt-4o"

# 指定 Weave logging
# 取出模型名稱部分，自動清理掉 prefix 與特殊字元
MODEL_NAME_CLEAN="${LLM_MODEL##*/}"        # "Llama-3.3-70B-Instruct"
MODEL_NAME_CLEAN="${MODEL_NAME_CLEAN//./-}" # "Llama-3-3-70B-Instruct"
WEAVE_PROJECT_RAW="rag-${DATASET_NAME}-${MODEL_NAME_CLEAN}"
WEAVE_PROJECT="${WEAVE_PROJECT_RAW,,}" # lowercase

SERVER_PORT=8002

API=http://localhost:$SERVER_PORT/query

mkdir -p logs

for TOP_K in "${TOP_K_VALUES[@]}"; do
    # --------- (1) setsid 啟動 server ------------
    setsid python server.py \
        -m "$PIPELINE_METHOD" -k "$TOP_K" -w "$WEAVE_PROJECT" \
        -p "$SERVER_PORT" -l "$LLM_MODEL" -t "$TOKENIZER" \
        > "logs/server_${SERVER_PORT}.log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID (== PGID): $SERVER_PID"

    # 等待 server 啟動
    sleep 10

    # --------- eval.py ---------------------------
    EXPERIMENT_NAME="${PIPELINE_METHOD}-top${TOP_K}"
    python eval.py -project "$WEAVE_PROJECT" -name "$EXPERIMENT_NAME" \
                   -api "$API" -d "$DATASET_NAME"

    # --------- (2) 結束 server：對 -SERVER_PID 送訊號 -----
    echo "Stopping server (process‑group $SERVER_PID)…"
    kill -TERM -"$SERVER_PID" 2>/dev/null || true   # ⚑ 這行取代原先 PGID
    sleep 5
    kill -KILL -"$SERVER_PID" 2>/dev/null || true   # ⚑ 這行取代原先 PGID

    # 等待 port 釋放
    sleep 10
done