#!/bin/bash

# 指定 pipeline 方法: vanilla, rag, direct, long, coa
PIPELINE_METHOD="rag"  

# Dataset name
DATASET_NAME="narrativeqa"  # 可選擇的 dataset: hotpotqa, narrativeqa, triviaqa...

# LLM/tokenizer model
LLM_MODEL="gpt-4.1"
TOKENIZER="gpt-4.1"

# 指定 Weave logging
# 取出模型名稱部分，自動清理掉 prefix 與特殊字元
MODEL_NAME_CLEAN="${LLM_MODEL##*/}"        # "Llama-3.3-70B-Instruct"
MODEL_NAME_CLEAN="${MODEL_NAME_CLEAN//./-}" # "Llama-3-3-70B-Instruct"
WEAVE_PROJECT_RAW="${DATASET_NAME}-${MODEL_NAME_CLEAN}"
WEAVE_PROJECT="${WEAVE_PROJECT_RAW,,}" # lowercase

SERVER_PORT=8002

API=http://localhost:$SERVER_PORT/query

mkdir -p logs

# 啟動 server.py
if [ -z "$WEAVE_PROJECT" ]; then
    python server.py -m $PIPELINE_METHOD -p $SERVER_PORT -l $LLM_MODEL -t $TOKENIZER > logs/server_$SERVER_PORT.log 2>&1 &
else
    python server.py -m $PIPELINE_METHOD -w $WEAVE_PROJECT -p $SERVER_PORT -l $LLM_MODEL -t $TOKENIZER > logs/server_$SERVER_PORT.log 2>&1 &
fi

# 等待 server 啟動
echo "Waiting for server to start..."
sleep 10

# 執行 eval.py 進行評估
python eval.py -project $WEAVE_PROJECT -name $PIPELINE_METHOD -api $API -d $DATASET_NAME

# 結束 server
echo "Stopping server..."
kill $(lsof -t -i:$SERVER_PORT)