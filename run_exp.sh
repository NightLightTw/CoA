#!/bin/bash

# 指定要跑的 pipeline 方法
PIPELINE_METHODS=("vanilla" "rag" "direct" "long" "coa")

# Dataset name
DATASET_NAME="hotpotqa"

# LLM/tokenizer model
LLM_MODEL="gpt-4.1"
TOKENIZER="gpt-4.1"

# 取出模型名稱部分，自動清理掉 prefix 與特殊字元
MODEL_NAME_CLEAN="${LLM_MODEL##*/}"
MODEL_NAME_CLEAN="${MODEL_NAME_CLEAN//./-}"

# 保留原本的命名規則
WEAVE_PROJECT_RAW="${DATASET_NAME}-${MODEL_NAME_CLEAN}"
WEAVE_PROJECT="${WEAVE_PROJECT_RAW,,}" # lowercase

SERVER_PORT=8003
API=http://localhost:$SERVER_PORT/query

# 建立 logs 資料夾（如果不存在）
mkdir -p logs

for PIPELINE_METHOD in "${PIPELINE_METHODS[@]}"
do
    echo "============================="
    echo "Running Pipeline: $PIPELINE_METHOD"
    echo "============================="

    # 啟動 server.py 並寫入 logs
    python server.py -m $PIPELINE_METHOD -w $WEAVE_PROJECT -p $SERVER_PORT -l $LLM_MODEL -t $TOKENIZER > logs/server_${SERVER_PORT}_${PIPELINE_METHOD}.log 2>&1 &

    # 等待 server 啟動
    echo "Waiting for server to start..."
    sleep 5

    # 執行 eval.py
    python eval.py -project $WEAVE_PROJECT -name $PIPELINE_METHOD -api $API -d $DATASET_NAME

    # 結束 server
    echo "Stopping server..."
    kill $(lsof -t -i:$SERVER_PORT)

    sleep 3
done

echo "All experiments done!"
