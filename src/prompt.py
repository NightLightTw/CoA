# Specific Requirement prompt from the paper "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
# https://arxiv.org/abs/2308.14508

HotpotQA_specific_requirement = """
Answer the question based on the given passages. Only give me the answer and do not output any other words.
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