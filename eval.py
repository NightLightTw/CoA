import os
import requests
import argparse
from datasets import load_dataset
from src.metrics import qa_f1_score
from tqdm import tqdm
import wandb
import weave
from openai import OpenAI
from dotenv import load_dotenv
from src.prompt import (
    JUDGE_RULES
)

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@weave.op()
def request_llm(client, prompt):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-project", "-p", type=str, default="hotpotqa-eval", help="WandB project name")
    parser.add_argument("-name", "-n", type=str, default="vanilla", help="WandB run name")
    parser.add_argument("-api", type=str, default="http://localhost:8000/query", help="API endpoint URL")
    parser.add_argument("-dataset", "-d", type=str, default="hotpotqa", help="Specify the dataset to use")
    args = parser.parse_args()

    wandb.init(project=args.project, name=args.name)
    weave.init(project=args.project)
    
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

    # F1 score
    f1_scores = []
    questions = dataset["input"]
    for pred, truth in zip(predictions, ground_truths):
        f1 = qa_f1_score([pred], [truth])
        f1_scores.append(f1)

    average_f1 = sum(f1_scores) / len(f1_scores)
    print(f"average_f1: {average_f1:.3f}")
    
    # LLM as a judge
    llm_as_a_judge_scores = []
    for idx, question in enumerate(tqdm(questions, desc="Evaluating with LLM as a judge")):
        try:
            prompt = JUDGE_RULES.format(
            question=question,
            answer=ground_truths[idx],
            pred=predictions[idx]
            )
            response = request_llm(client, prompt)
            score_str = response.choices[0].message.content.strip()
            llm_score = float(score_str)
            
        except Exception as e:
            print(f"Warning: LLM returned invalid score for index {idx}: '{score_str}'. Error: {e}")
            llm_score = 0.0
            
        llm_as_a_judge_scores.append(llm_score)

    table = wandb.Table(columns=["index", "question", "ground_truth", "prediction", "f1_score", "llm_as_a_judge_score"])
    for i, (question, pred, truth, f1, llm_as_a_judge_score) in enumerate(zip(questions, predictions, ground_truths, f1_scores, llm_as_a_judge_scores)):
        table.add_data(i, question, truth, pred, f1, llm_as_a_judge_score)
    
    # Logging
    wandb.log({"average_f1": average_f1})
    wandb.log({args.name: table})
    wandb.finish()