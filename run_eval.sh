#!/bin/bash

# 指定 pipeline 方法: vanilla, rag, direct, long, coa, ragcoa-algo1
PIPELINE_METHOD="coa"  

# Dataset name
DATASET_NAME="narrativeqa"  # 可選擇的 dataset: hotpotqa, narrativeqa, triviaqa...
CHUNKSIZE=30000 # 32k-2k

# LLM/tokenizer model
LLM_MODEL="gpt-4o-mini"
TOKENIZER="gpt-4o-mini"

# 指定 Weave logging
# 取出模型名稱部分，自動清理掉 prefix 與特殊字元
MODEL_NAME_CLEAN="${LLM_MODEL##*/}"        # "Llama-3.3-70B-Instruct"
MODEL_NAME_CLEAN="${MODEL_NAME_CLEAN//./-}" # "Llama-3-3-70B-Instruct"
WEAVE_PROJECT_RAW="new-${DATASET_NAME}-${MODEL_NAME_CLEAN}"
WEAVE_PROJECT="${WEAVE_PROJECT_RAW,,}" # lowercase

SERVER_PORT=8000

API=http://localhost:$SERVER_PORT/query

mkdir -p logs

# 啟動 server.py
python server.py -d $DATASET_NAME -m $PIPELINE_METHOD -cs $CHUNKSIZE -w $WEAVE_PROJECT -p $SERVER_PORT -l $LLM_MODEL -t $TOKENIZER > logs/server_$SERVER_PORT.log 2>&1 &

# 等待 server 啟動
echo "Waiting for server to start..."
sleep 10

# 執行 eval.py 進行評估
EXPERIMENT_NAME="${PIPELINE_METHOD}_${CHUNKSIZE}"
python eval.py -project $WEAVE_PROJECT -name $EXPERIMENT_NAME -api $API -d $DATASET_NAME

# 結束 server
echo "Stopping server..."
kill $(lsof -t -i:$SERVER_PORT)