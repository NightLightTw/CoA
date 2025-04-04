from src.prompt import (
    COA_WORKER_PROMPT,
    COA_MANAGER_PROMPT,
    VANILLA_PROMPT,
    RAG_PROMPT
)
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import weave

# WorkerAgent
@weave.op()
def worker_agent(client, model, input_chunk, previous_cu, query):
    prompt = COA_WORKER_PROMPT.format(
        Input_Chunk_ci=input_chunk,
        Previous_Communication_Unit=previous_cu or "",
        Question_q=query
    )
    
    logger.info("Input words of worker: %d", len(prompt.split()))
    # logger.info("Worker prompt:\n%s", prompt)
    
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0
    )

# ManagerAgent
@weave.op()
def manager_agent(client, model, task_requirement, previous_cu, query):
    prompt = COA_MANAGER_PROMPT.format(
        Task_specific_requirement=task_requirement,
        Previous_Communication_Unit=previous_cu or "",
        Question_q=query
    )
    
    logger.info("Input words of manager: %d", len(prompt.split()))
    logger.info("Manager prompt:\n%s", prompt)
    
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0
    )
    
# VanillaAgent
@weave.op()
def vanilla_agent(client, model, task_requirement, input_chunk, query):
    prompt = VANILLA_PROMPT.format(
        Source_Input_x_with_truncation_if_needed=input_chunk,
        Task_specific_requirement=task_requirement,
        Question_q=query
    )
    
    logger.info("Input words of vanilla: %d", len(prompt.split()))
    logger.info("Input prompt:\n%s", prompt)
    
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0
    )

# RAGAgent
@weave.op()
def RAG_agent(client, model, task_requirement, input_chunk, query):
    prompt = RAG_PROMPT.format(
        Retrieved_Chunks_of_Source_Input_x=input_chunk,
        Task_specific_requirement=task_requirement,
        Question_q=query
    )
    
    print("Input words of RAG:", len(prompt.split()))
    print(f"Input prompt:\n{prompt}")
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0
    )

    return response.choices[0].message.content.strip()