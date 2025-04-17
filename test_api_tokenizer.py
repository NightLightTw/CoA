import os
import requests
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

# 載入 .env
load_dotenv(override=True)

# 設定參數
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1"
model = "google/gemma-3-27b-it"
tokenizer_name = "google/gemma-3-27b-it"

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# 載入資料
dataset = load_dataset('THUDM/LongBench', "hotpotqa", split='test')
data_sample = dataset[3]
question = data_sample["input"]
context = data_sample["context"]

# 組合 payload
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "http://localhost",
    "X-Title": "test_request"
}

url = f"{base_url}/chat/completions"
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": f"Context: {context}"},
        {"role": "user", "content": question}
    ],
    "max_tokens": 1024,
    "temperature": 0,
}

# 發送請求
response = requests.post(url, headers=headers, json=payload)

# 顯示速率資訊
print("--- Rate Limit Info ---")
print("X-RateLimit-Remaining:", response.headers.get("X-RateLimit-Remaining"))
print("X-RateLimit-Reset:", response.headers.get("X-RateLimit-Reset"))

# 解析回應
if response.status_code == 200:
    answer = response.json()["choices"][0]["message"]["content"].strip()
    print("Answer:", answer)
else:
    print("Request failed with status code:", response.status_code)
    print(response.text)
