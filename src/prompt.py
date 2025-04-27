# Specific Requirement prompt from the paper "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
# https://arxiv.org/abs/2308.14508

HotpotQA_specific_requirement = """
Answer the question based on the given passages. Only give me the answer and do not output any other words.
"""

NarrativeQA_specific_requirement = """
You are given a story, which can be either a novel or a movie script, and a question.
Answer the question as concisely as you can, using a single phrase if possible. Do not provide any
explanation.
"""

COA_WORKER_PROMPT = """
{Input_Chunk_ci}
Here is the summary of the previous source text: {Previous_Communication_Unit}\n
Question: {Question_q}\n
You need to read current source text and summary of previous source text (if any) and generate a summary to include them both.\n
Later, this summary will be used for other agents to answer the Query, if any.\n
So please write the summary that can include the evidence for answering the Query:
"""

COA_MANAGER_PROMPT = """
{Task_specific_requirement}\n
The following are given passages.\n
However, the source text is too long and has been summarized.\n
You need to answer based on the summary: \n
{Previous_Communication_Unit}\n
Question: {Question_q}\n
Answer:
"""

VANILLA_PROMPT = """
{Task_specific_requirement}\n
{Source_Input_x_with_truncation_if_needed}\n
Question: {Question_q}\n
Answer:
"""

RAG_PROMPT = """
{Task_specific_requirement}\n
{Retrieved_Chunks_of_Source_Input_x}\n
Question: {Question_q}\n
Answer:
"""

DIRECT_PROMPT = """
Only give me the answer and do not output any other words.\n
Question: {Question_q}\n
Answer:
"""

# LLM-as-a-judge prompt from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/mmmu/utils.py
JUDGE_RULES = """
You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
0 or 1
"""