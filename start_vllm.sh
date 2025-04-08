CUDA_VISIBLE_DEVICES=1,3 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8001