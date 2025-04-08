import os
import logging
import weave
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from FlagEmbedding import FlagModel
from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import *
from src.agent import RAG_agent, worker_agent, manager_agent, vanilla_agent
from src.prompt import HotpotQA_specific_requirement

load_dotenv(override=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    context: str

class RAGPipeline:
    def __init__(
        self,
        tokenizer,
        model_name="BAAI/bge-large-en",
        max_chunk_size=300,
        max_total_words=5000,
        client=None,
        gen_model_name="meta-llama/llama-3.3-70b-instruct:free",
        task_requirement=None
    ):
        self.retriever = FlagModel(model_name, use_fp16=True)
        self.max_chunk_size = max_chunk_size
        self.max_total_words = max_total_words
        self.client = client
        self.gen_model_name = gen_model_name
        self.tokenizer = tokenizer
        self.task_requirement = task_requirement

    def embed_chunks(self, chunks):
        return self.retriever.encode(chunks, batch_size=32, max_length=512)

    def retrieve_relevant_chunks(self, query, chunks, embeddings):
        query_embedding = self.retriever.encode([query])[0]
        similarities = np.dot(embeddings, query_embedding)
        sorted_indices = np.argsort(similarities)[::-1]

        relevant_chunks = []
        total_words = 0
        for idx in sorted_indices:
            chunk = chunks[idx]
            chunk_word_count = len(chunk.split())
            if total_words + chunk_word_count <= self.max_total_words:
                relevant_chunks.append(chunk)
                total_words += chunk_word_count
            else:
                break
        return relevant_chunks

    def retrieve(self, context, question):
        # chunks = split_into_chunks_with_word(context, max_chunk_size=self.max_chunk_size)
        chunks = split_into_chunks_with_token(context, max_chunk_size=self.max_chunk_size, tokenizer=self.tokenizer, model=self.gen_model_name)
        embeddings = self.embed_chunks(chunks)
        relevant_chunks = self.retrieve_relevant_chunks(question, chunks, embeddings)
        input_chunk = " ".join(relevant_chunks)
        
        input_chunk = split_into_chunks_with_token(input_chunk, max_chunk_size=self.max_total_words, tokenizer=self.tokenizer, model=self.gen_model_name)
        input_chunk = input_chunk[0] # Truncate to the first chunk
        logger.info("Truncated input chunk to :%d words", len(input_chunk.split()))
        return input_chunk

    @weave.op()
    def run(self, question, context):
        input_chunk = self.retrieve(context, question)
        return RAG_agent(
            client=self.client,
            model=self.gen_model_name,
            input_chunk=input_chunk,
            task_requirement=self.task_requirement,
            query=question
        )

class ChainOfAgentsPipeline:
    def __init__(self, client, model, tokenizer, task_requirement, max_chunk_size=5000):
        self.client = client
        self.model = model
        self.tokenizer = tokenizer
        self.task_requirement = task_requirement
        self.max_chunk_size = max_chunk_size

    @weave.op()
    def run(self, question, context):
        # chunks = split_into_chunks_with_word(context, max_chunk_size=self.max_chunk_size)
        chunks = split_into_chunks_with_token(context, max_chunk_size=self.max_chunk_size, tokenizer=self.tokenizer, model=self.model)

        previous_cu = None

        for idx, chunk in enumerate(chunks):
            logger.info("Worker %d/%d 處理中...", idx + 1, len(chunks))
            logger.info("Words of chunk: %d", len(chunk.split()))
            response = worker_agent(
                client=self.client,
                model=self.model,
                input_chunk=chunk,
                previous_cu=previous_cu,
                query=question
            )
            previous_cu = response.choices[0].message.content.strip()

        logger.info("Manager 最終整合...")
        return manager_agent(
            client=self.client,
            model=self.model,
            task_requirement=self.task_requirement,
            previous_cu=previous_cu,
            query=question
        )

class VanillaPipeline:
    def __init__(self, client, model, tokenizer, task_requirement, max_chunk_size=5000):
        self.client = client
        self.model = model
        self.tokenizer = tokenizer
        self.task_requirement = task_requirement
        self.max_chunk_size = max_chunk_size

    @weave.op()
    def run(self, question, context):
        # input_chunk = split_into_chunks_with_word(context, max_chunk_size=self.max_chunk_size)
        input_chunk = split_into_chunks_with_token(context, max_chunk_size=self.max_chunk_size, tokenizer=self.tokenizer, model=self.model)

        input_chunk = input_chunk[0]  # Truncate to the first chunk
        logger.info("Truncated input chunk to :%d words", len(input_chunk.split()))

        return vanilla_agent(
            client=self.client,
            model=self.model,
            input_chunk=input_chunk,
            task_requirement=self.task_requirement,
            query=question
        )

PIPELINE_METHOD = None

@app.post("/query")
def query_pipeline(request: QueryRequest):
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        # base_url="https://openrouter.ai/api/v1",
        base_url="http://localhost:8001/v1", # Local vllm server
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Chain_of_agents_DEMO"
        }
    )
    # model_name = "meta-llama/llama-3.3-70b-instruct:free"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    task_requirement = HotpotQA_specific_requirement

    if PIPELINE_METHOD == "rag":
        pipeline = RAGPipeline(
            model_name="BAAI/bge-large-en",
            tokenizer=tokenizer,
            max_chunk_size=300,
            max_total_words=6000,
            client=client,
            gen_model_name=model_name,
            task_requirement=task_requirement
        )
    elif PIPELINE_METHOD == "coa":
        pipeline = ChainOfAgentsPipeline(
            client=client,
            model=model_name,
            tokenizer=tokenizer,
            task_requirement=task_requirement,
            max_chunk_size=6000
        )
    elif PIPELINE_METHOD == "vanilla":
        pipeline = VanillaPipeline(
            client,
            model_name,
            tokenizer=tokenizer,
            task_requirement=task_requirement,
            max_chunk_size=6000
            )
    else:
        return {"error": "Invalid method"}

    return pipeline.run(request.question, request.context)

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("-weave", "-w", type=str, help="Use weave for logging")
    parser.add_argument("-method", "-m", type=str, choices=["rag", "coa", "vanilla"], required=True, help="Specify the pipeline method")
    args = parser.parse_args()

    PIPELINE_METHOD = args.method

    if args.weave:
        weave.init(args.weave)

    uvicorn.run(app, host="0.0.0.0", port=8000)
