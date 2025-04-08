#!/bin/bash

# 指定 pipeline 方法: rag, coa 或 vanilla
PIPELINE_METHOD="rag"  

# Dataset name
DATASET_NAME="narrativeqa"  # 可選擇的 dataset: hotpotqa, narrativeqa, triviaqa...

# 指定 Weave logging（可選，若不用則留空）
WEAVE_PROJECT="narrativeqa-eval"

API=http://localhost:8000/query

# 啟動 server.py
if [ -z "$WEAVE_PROJECT" ]; then
    python server.py -m $PIPELINE_METHOD > server.log 2>&1 &
else
    python server.py -m $PIPELINE_METHOD -w $WEAVE_PROJECT > server.log 2>&1 &
fi

# 等待 server 啟動
echo "Waiting for server to start..."
sleep 5  # 可以調整等待時間，視 server 啟動速度而定

# 執行 eval.py 進行評估
python eval.py -project $WEAVE_PROJECT -name $PIPELINE_METHOD -api $API -d $DATASET_NAME

# 結束 server
echo "Stopping server..."
kill $(lsof -t -i:8000)