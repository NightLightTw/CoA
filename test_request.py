import requests
from datasets import load_dataset

url = "http://localhost:8000/query"

dataset = load_dataset('THUDM/LongBench', "hotpotqa", split='test')
data_sample = dataset[3]
question = data_sample["input"]
context = data_sample["context"]

payload = {
    "question": f"{question}",
    "context": f"{context}"
}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Response Text:", response.text)
result = response.json()
answer = result["choices"][0]["message"]["content"].strip()
print("Answer:", answer)

