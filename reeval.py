import os
import pandas as pd
import weave
import wandb
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from src.prompt import (
    JUDGE_RULES
)

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@weave.op()
def request(prompt):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0
    )

@weave.op()
def main(args, name):
    
    table = pd.read_csv(args.dataPath)
    questions = table["question"]
    ground_truths = table["ground_truth"]
    predictions = table["prediction"]
    f1_scores = table["f1_score"]

    llm_as_a_judge_scores = []
    for idx, question in enumerate(tqdm(questions, desc="Evaluating with LLM as a judge")):
        try:
            prompt = JUDGE_RULES.format(
            question=question,
            answer=ground_truths[idx],
            pred=predictions[idx]
            )
            response = request(prompt)
            score_str = response.choices[0].message.content.strip()
            llm_score = float(score_str)
            
        except Exception as e:
            print(f"Warning: LLM returned invalid score for index {idx}: '{score_str}'. Error: {e}")
            llm_score = 0.0
            
        llm_as_a_judge_scores.append(llm_score)

    table = wandb.Table(columns=["index", "question", "ground_truth", "prediction", "f1_score", "llm_as_a_judge_score"])
    for i, (question, pred, truth, f1, llm_as_a_judge_score) in enumerate(zip(questions, predictions, ground_truths, f1_scores, llm_as_a_judge_scores)):
        table.add_data(i, question, truth, pred, f1, llm_as_a_judge_score)
    
    average_f1 = sum(f1_scores) / len(f1_scores)
    wandb.log({"average_f1": average_f1})
    average_llm_as_a_judge_score = sum(llm_as_a_judge_scores) / len(llm_as_a_judge_scores)
    wandb.log({"average_llm_as_a_judge_score": average_llm_as_a_judge_score})
    wandb.log({name: table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-project", "-p", type=str, default="test_llm_as_a_judge", help="WandB project name")
    parser.add_argument("-dataPath", "-d", type=str, default="dataset/table/gemma-3-27b-it/coa.csv", help="Path to the dataset")
    args = parser.parse_args()
    
    basename = os.path.basename(args.dataPath) # e.g., "coa.csv"
    filename_no_ext = os.path.splitext(basename)[0] # "coa"
    parent_folder = os.path.basename(os.path.dirname(args.dataPath)) # "gemma-3-27b-it"
    name = f"{parent_folder}-{filename_no_ext}" # "gemma-3-27b-it-coa"
    
    weave.init(args.project)
    wandb.init(project=args.project, name=name)
    main(args, name)
    wandb.finish()