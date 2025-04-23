import requests
import argparse
from datasets import load_dataset
from src.metrics import qa_f1_score
from tqdm import tqdm
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-project", "-p", type=str, default="hotpotqa-eval", help="WandB project name")
    parser.add_argument("-name", "-n", type=str, default="vanilla", help="WandB run name")
    parser.add_argument("-api", type=str, default="http://localhost:8000/query", help="API endpoint URL")
    parser.add_argument("-dataset", "-d", type=str, default="hotpotqa", help="Specify the dataset to use")
    args = parser.parse_args()

    wandb.init(project=args.project, name=args.name)
    API_URL = args.api
    dataset_name = args.dataset
    dataset = load_dataset('THUDM/LongBench', dataset_name, split='test')
    # dataset = dataset.select(range(1)) # Test few examples
    predictions = []
    ground_truths = []

    for example in tqdm(dataset):
        question = example['input']
        context = example['context']
        ground_truth = example['answers'][0][0] if isinstance(example['answers'][0], list) else example['answers'][0]

        payload = {
            "question": question,
            "context": context
        }

        # 發送至 API 端取得回應
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            # print("Result:", result)
            if result.get("choices"):
                prediction = result["choices"][0]["message"]["content"].strip()
            else:
                print("API error:", result.get("error", {}).get("message"))
                prediction = ""
        else:
            print(f"API Error: {response.status_code}")
            prediction = ""

        predictions.append(prediction)
        ground_truths.append(ground_truth)

    f1_scores = []
    for pred, truth in zip(predictions, ground_truths):
        f1 = qa_f1_score([pred], [truth])
        f1_scores.append(f1)

    average_f1 = sum(f1_scores) / len(f1_scores)
    print(f"average_f1: {average_f1:.3f}")
    
    table = wandb.Table(columns=["index", "question", "ground_truth", "prediction", "f1_score"])
    for i, (ex, pred, truth, f1) in enumerate(zip(dataset, predictions, ground_truths, f1_scores)):
        table.add_data(i, ex["input"], truth, pred, f1)
    wandb.log({"average_f1": average_f1})
    wandb.log({args.name: table})
    wandb.finish()