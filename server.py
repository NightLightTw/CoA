import os
import logging
import weave
import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from FlagEmbedding import FlagModel
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import tiktoken

from src.utils import *
from src.agent import *
from src.prompt import *

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
        embed_model_name="BAAI/bge-large-en",
        max_chunk_size=300,
        client=None,
        gen_model_name=None,
        task_requirement=None,
        top_k=None
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = FlagModel(embed_model_name, use_fp16=True)
        self.retriever.model = self.retriever.model.to(device) # Move model to GPU if available
        self.max_chunk_size = max_chunk_size
        self.client = client
        self.gen_model_name = gen_model_name
        self.tokenizer = tokenizer
        self.task_requirement = task_requirement
        self.top_k = top_k

    def embed_chunks(self, chunks):
        return self.retriever.encode(chunks, batch_size=32, max_length=512)

    def retrieve_relevant_chunks(self, query, chunks, embeddings):
        query_embedding = self.retriever.encode([query])[0]
        similarities = np.dot(embeddings, query_embedding)
        sorted_indices = np.argsort(similarities)[::-1]

        relevant_chunks = [chunks[idx] for idx in sorted_indices[:self.top_k]]
        return relevant_chunks

    def retrieve(self, context, question):
        # chunks = split_into_chunks_with_word(context, max_chunk_size=self.max_chunk_size)
        chunks = split_into_chunks_with_token(context, max_chunk_size=self.max_chunk_size, tokenizer=self.tokenizer, model=self.gen_model_name)
        embeddings = self.embed_chunks(chunks)
        relevant_chunks = self.retrieve_relevant_chunks(question, chunks, embeddings)
        input_chunk = " ".join(relevant_chunks)

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
    def __init__(self, client, model, tokenizer, task_requirement, max_chunk_size=6000):
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
    def __init__(self, client, model, tokenizer, task_requirement, max_chunk_size=6000):
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

class DirectPipeline:
    def __init__(self, client, model):
        self.client = client
        self.model = model
        
    @weave.op()
    def run(self, question, context):
        return direct_agent(
            client=self.client,
            model=self.model,
            query=question
        )

class RAGCoAAlgo1Pipeline:
    def __init__(
        self,
        tokenizer,
        embed_model_name="BAAI/bge-large-en",
        max_chunk_size=300,  # 300 tokens per chunk as specified
        chunks_per_worker=20,  # 20 chunks per worker as specified
        client=None,
        gen_model_name="meta-llama/llama-3.3-70b-instruct:free",
        task_requirement=None,
        top_k=None
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = FlagModel(embed_model_name, use_fp16=True)
        self.retriever.model = self.retriever.model.to(device)  # Move model to GPU if available
        self.max_chunk_size = max_chunk_size
        self.chunks_per_worker = chunks_per_worker
        self.client = client
        self.gen_model_name = gen_model_name
        self.tokenizer = tokenizer
        self.task_requirement = task_requirement
        self.top_k = top_k

    def embed_chunks(self, chunks):
        return self.retriever.encode(chunks, batch_size=32, max_length=512)

    def rank_chunks(self, query, chunks, embeddings):
        query_embedding = self.retriever.encode([query])[0]
        similarities = np.dot(embeddings, query_embedding)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Return all sorted chunks and their indices
        return sorted_indices, similarities

    @weave.op()
    def run(self, question, context):
        # Step 1: Split source into 300-token chunks
        chunks = split_into_chunks_with_token(context, max_chunk_size=self.max_chunk_size, 
                                             tokenizer=self.tokenizer, model=self.gen_model_name)
        logger.info("Total chunks: %d", len(chunks))
        
        # Step 2: BGE retriever ranks all chunks
        embeddings = self.embed_chunks(chunks)
        sorted_indices, similarities = self.rank_chunks(question, chunks, embeddings)
        
        # Step 3: Select top-k chunks for processing
        if self.top_k:
            sorted_indices = sorted_indices[:self.top_k]
            
        # Step-by-step processing with workers
        previous_cu = None
        
        # Determine how many worker groups we need based on chunks_per_worker
        num_workers = len(sorted_indices) // self.chunks_per_worker
        if len(sorted_indices) % self.chunks_per_worker > 0:
            num_workers += 1
            
        logger.info("Using %d workers for %d chunks", num_workers, len(sorted_indices))
        
        # Process chunks in groups of chunks_per_worker
        for worker_idx in range(num_workers):
            start_idx = worker_idx * self.chunks_per_worker
            end_idx = min((worker_idx + 1) * self.chunks_per_worker, len(sorted_indices))
            
            # Skip if no more chunks to process
            if start_idx >= len(sorted_indices):
                break
                
            worker_chunk_indices = sorted_indices[start_idx:end_idx]
            
            # Combine worker's assigned chunks
            worker_text = ""
            for idx in worker_chunk_indices:
                worker_text += chunks[idx] + " "
                
            logger.info("Worker %d/%d processing chunks %d-%d", 
                       worker_idx + 1, num_workers, start_idx + 1, end_idx)
            logger.info("Worker text length: %d tokens", len(worker_text.split()))
            
            # Process with worker agent
            response = worker_agent(
                client=self.client,
                model=self.gen_model_name,
                input_chunk=worker_text,
                previous_cu=previous_cu,
                query=question
            )
            
            previous_cu = response.choices[0].message.content.strip()
        
        # Step 4: Manager receives last Communication Unity and answers the question
        logger.info("Manager integrating final answer...")
        return manager_agent(
            client=self.client,
            model=self.gen_model_name,
            task_requirement=self.task_requirement,
            previous_cu=previous_cu,
            query=question
        )

class RAGCoAAlgo2Pipeline:
    def __init__(
        self,
        tokenizer,
        embed_model_name="BAAI/bge-large-en",
        max_chunk_size=300,  # 300 tokens per chunk as specified
        chunks_per_worker=20,  # 20 chunks per worker as specified
        client=None,
        gen_model_name="meta-llama/llama-3.3-70b-instruct:free",
        task_requirement=None,
        top_k=None
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = FlagModel(embed_model_name, use_fp16=True)
        self.retriever.model = self.retriever.model.to(device)  # Move model to GPU if available
        self.max_chunk_size = max_chunk_size
        self.chunks_per_worker = chunks_per_worker
        self.client = client
        self.gen_model_name = gen_model_name
        self.tokenizer = tokenizer
        self.task_requirement = task_requirement
        self.top_k = top_k

    def embed_chunks(self, chunks):
        return self.retriever.encode(chunks, batch_size=32, max_length=512)

    def rank_chunks(self, query, chunks, embeddings):
        query_embedding = self.retriever.encode([query])[0]
        similarities = np.dot(embeddings, query_embedding)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Return all sorted chunks and their indices
        return sorted_indices, similarities

    @weave.op()
    def run(self, question, context):
        # Step 1: Split source into 300-token chunks
        chunks = split_into_chunks_with_token(context, max_chunk_size=self.max_chunk_size, 
                                             tokenizer=self.tokenizer, model=self.gen_model_name)
        logger.info("Total chunks: %d", len(chunks))
        
        # Step 2: BGE retriever ranks all chunks
        embeddings = self.embed_chunks(chunks)
        sorted_indices, similarities = self.rank_chunks(question, chunks, embeddings)
        
        # Step 3: Select top-k chunks for processing
        if self.top_k:
            sorted_indices = sorted_indices[:self.top_k]
        
        sorted_indices = sorted_indices[::-1] # Reverse the order
        
        # Step-by-step processing with workers
        previous_cu = None
        # keep total length for inverse‑rank logging
        total_selected = len(sorted_indices)
        
        # Determine how many worker groups we need based on chunks_per_worker
        num_workers = len(sorted_indices) // self.chunks_per_worker
        if len(sorted_indices) % self.chunks_per_worker > 0:
            num_workers += 1
            
        logger.info("Using %d workers for %d chunks", num_workers, len(sorted_indices))
        
        # Process chunks in groups of chunks_per_worker
        for worker_idx in range(num_workers):
            start_idx = worker_idx * self.chunks_per_worker
            end_idx = min((worker_idx + 1) * self.chunks_per_worker, len(sorted_indices))
            
            # Skip if no more chunks to process
            if start_idx >= len(sorted_indices):
                break
                
            worker_chunk_indices = sorted_indices[start_idx:end_idx]
            
            # Combine worker's assigned chunks
            worker_text = ""
            for idx in worker_chunk_indices:
                worker_text += chunks[idx] + " "
                
            logger.info(
                "Worker %d/%d processing inverse‑ranked chunks %d‑%d",
                worker_idx + 1,
                num_workers,
                total_selected - start_idx,
                total_selected - end_idx + 1,
            )
            logger.info("Worker text length: %d tokens", len(worker_text.split()))
            
            # Process with worker agent
            response = worker_agent(
                client=self.client,
                model=self.gen_model_name,
                input_chunk=worker_text,
                previous_cu=previous_cu,
                query=question
            )
            
            previous_cu = response.choices[0].message.content.strip()
        
        # Step 4: Manager receives last Communication Unity and answers the question
        logger.info("Manager integrating final answer...")
        return manager_agent(
            client=self.client,
            model=self.gen_model_name,
            task_requirement=self.task_requirement,
            previous_cu=previous_cu,
            query=question
        )

PIPELINE_METHOD = None
pipeline = None
args = None

@app.on_event("startup")
def startup_event():
    global pipeline, PIPELINE_METHOD, args
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        # api_key=os.getenv("OPENROUTER_API_KEY"),
        # base_url="http://10.8.4.128:8001/v1", # Local vllm server
        # base_url="https://openrouter.ai/api/v1",
        # default_headers={ # for OpenRouter
        #     "HTTP-Referer": "http://localhost",
        #     "X-Title": "Chain_of_agents"
        # }
    )
    model_name = args.llm
    tokenizer = tiktoken.encoding_for_model(model_name)
    # tokenizer = tiktoken.get_encoding("cl100k_base")
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    task_requirement = None
    if args.dataset == "hotpotqa":
        task_requirement = HotpotQA_specific_requirement
    elif args.dataset == "narrativeqa":
        task_requirement = NarrativeQA_specific_requirement

    if PIPELINE_METHOD == "rag":
        pipeline = RAGPipeline(
            embed_model_name="BAAI/bge-large-en",
            tokenizer=tokenizer,
            max_chunk_size=300,
            client=client,
            gen_model_name=model_name,
            task_requirement=task_requirement,
            top_k=args.top_k
        )
    elif PIPELINE_METHOD == "coa":
        pipeline = ChainOfAgentsPipeline(
            client=client,
            model=model_name,
            tokenizer=tokenizer,
            task_requirement=task_requirement,
            max_chunk_size=args.chunk_size #6000
        )
    elif PIPELINE_METHOD == "vanilla":
        pipeline = VanillaPipeline(
            client,
            model_name,
            tokenizer=tokenizer,
            task_requirement=task_requirement,
            max_chunk_size=6000
        )
    elif PIPELINE_METHOD == "direct":
        pipeline = DirectPipeline(
            client,
            model_name,
        )
    elif PIPELINE_METHOD == "long":
        pipeline = VanillaPipeline(
            client,
            model_name,
            tokenizer=tokenizer,
            task_requirement=task_requirement,
            max_chunk_size=126000
        )
    elif PIPELINE_METHOD == "ragcoa-algo1":
        pipeline = RAGCoAAlgo1Pipeline(
            embed_model_name="BAAI/bge-large-en",
            tokenizer=tokenizer,
            max_chunk_size=300,  # 300 tokens per chunk
            chunks_per_worker=20,  # 20 chunks per worker (6000 tokens)
            client=client,
            gen_model_name=model_name,
            task_requirement=task_requirement,
            top_k=args.top_k
        )
    elif PIPELINE_METHOD == "ragcoa-algo2":
        pipeline = RAGCoAAlgo2Pipeline(
            embed_model_name="BAAI/bge-large-en",
            tokenizer=tokenizer,
            max_chunk_size=300,
            chunks_per_worker=20,
            client=client,
            gen_model_name=model_name,
            task_requirement=task_requirement,
            top_k=args.top_k
        )
    else:
        raise ValueError(f"Invalid method: {PIPELINE_METHOD}")

@app.post("/query")
def query_pipeline(request: QueryRequest):
    global pipeline
    return pipeline.run(request.question, request.context)

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "-d", type=str, default="hotpotqa", help="Dataset name")
    parser.add_argument("-weave", "-w", type=str, help="Use weave for logging")
    parser.add_argument("-method", "-m", type=str, 
                       choices=["rag", "coa", "vanilla", "direct", "long", "ragcoa-algo1", "ragcoa-algo2"], 
                       required=True, help="Specify the pipeline method")
    parser.add_argument("-port", "-p", type=int, default=8000, help="Port number for the server")
    parser.add_argument("-llm", "-l", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("-tokenizer_name", "-t", type=str, default="meta-llama/llama-3.3-70b-instruct", help="Tokenizer model name")
    parser.add_argument("-top_k", "-k", type=int, default=20, help="Top k chunks to retrieve")
    parser.add_argument("-chunk_size", "-cs", type=int, default=300, help="Chunk size for splitting the context")
    args = parser.parse_args()

    PIPELINE_METHOD = args.method

    if args.weave:
        weave.init(args.weave)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
