CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8096 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8001