#!/bin/bash

# 指定 pipeline 方法: rag, coa, vanilla, direct, long
PIPELINE_METHOD="long"  

# Dataset name
DATASET_NAME="narrativeqa"  # 可選擇的 dataset: hotpotqa, narrativeqa, triviaqa...

# LLM/tokenizer model
LLM_MODEL="gpt-4o-mini"

# 指定 Weave logging（可選，若不用則留空）
WEAVE_PROJECT="narrativeqa-4o-mini"

SERVER_PORT=8002

API=http://localhost:$SERVER_PORT/query


# 啟動 server.py
if [ -z "$WEAVE_PROJECT" ]; then
    python server.py -m $PIPELINE_METHOD -p $SERVER_PORT -l LLM_MODEL > server_$SERVER_PORT.log 2>&1 &
else
    python server.py -m $PIPELINE_METHOD -w $WEAVE_PROJECT -p $SERVER_PORT -l LLM_MODEL > server_$SERVER_PORT.log 2>&1 &
fi

# 等待 server 啟動
echo "Waiting for server to start..."
sleep 5

# 執行 eval.py 進行評估
python eval.py -project $WEAVE_PROJECT -name $PIPELINE_METHOD -api $API -d $DATASET_NAME

# 結束 server
echo "Stopping server..."
kill $(lsof -t -i:$SERVER_PORT)