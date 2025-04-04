import os
import logging
import argparse
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from FlagEmbedding import FlagModel

from src.utils import split_into_chunks_with_word
from src.agent import RAG_agent, worker_agent, manager_agent, vanilla_agent
from src.prompt import HotpotQA_specific_requirement

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class RAGPipeline:
    def __init__(
        self,
        model_name="BAAI/bge-large-en",
        max_chunk_size=300,
        max_total_words=5000,
        client=None,
        gen_model_name="meta-llama/llama-3.3-70b-instruct:Together",
        task_requirement=None
    ):
        self.retriever = FlagModel(model_name, use_fp16=True)
        self.max_chunk_size = max_chunk_size
        self.max_total_words = max_total_words
        self.client = client
        self.gen_model_name = gen_model_name
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
        chunks = split_into_chunks_with_word(context, max_chunk_size=self.max_chunk_size)
        embeddings = self.embed_chunks(chunks)
        relevant_chunks = self.retrieve_relevant_chunks(question, chunks, embeddings)
        input_chunk = " ".join(relevant_chunks)
        logger.info("Truncated input chunk to :%d words", len(input_chunk.split()))
        return input_chunk

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
    def __init__(self, client, model, task_requirement, max_chunk_size=5000):
        self.client = client
        self.model = model
        self.task_requirement = task_requirement
        self.max_chunk_size = max_chunk_size

    def run(self, query, long_text):
        chunks = split_into_chunks_with_word(long_text, max_chunk_size=self.max_chunk_size)
        previous_cu = None

        for idx, chunk in enumerate(chunks):
            logger.info("Worker %d/%d 處理中...", idx + 1, len(chunks))
            logger.info("Words of chunk: %d", len(chunk.split()))
            previous_cu = worker_agent(
                client=self.client,
                model=self.model,
                input_chunk=chunk,
                previous_cu=previous_cu,
                query=query
            )

        logger.info("Manager 最終整合...")
        return manager_agent(
            client=self.client,
            model=self.model,
            task_requirement=self.task_requirement,
            previous_cu=previous_cu,
            query=query
        )


class VanillaPipeline:
    def __init__(self, client, model, task_requirement, max_chunk_size=5000):
        self.client = client
        self.model = model
        self.task_requirement = task_requirement
        self.max_chunk_size = max_chunk_size

    def run(self, question, combined_context):
        input_chunk = split_into_chunks_with_word(combined_context, max_chunk_size=self.max_chunk_size)
        input_chunk = input_chunk[0]  # Truncate to the first chunk
        logger.info("Truncated input chunk to :%d words", len(input_chunk.split()))

        return vanilla_agent(
            client=self.client,
            model=self.model,
            input_chunk=input_chunk,
            task_requirement=self.task_requirement,
            query=question
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-method",
        "-m",
        type=str,
        choices=["rag", "coa", "vanilla"],
        required=True,
        help="Choose which method to run: rag, coa, or vanilla"
    )
    args = parser.parse_args()

    # 載入資料
    dataset = load_dataset('THUDM/LongBench', "hotpotqa", split='test')
    data_sample = dataset[3]
    question = data_sample["input"]
    answer = data_sample["answers"]
    combined_context = data_sample['context']

    logger.info("Question:\n%s", question)
    logger.info("Ground Truth:\n%s", answer)

    # 共用參數初始化
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Chain_of_agents_DEMO"
        }
    )
    model_name = "meta-llama/llama-3.3-70b-instruct:Together"
    task_requirement = HotpotQA_specific_requirement

    # Pipeline 初始化
    if args.method == "rag":
        pipeline = RAGPipeline(
            model_name="BAAI/bge-large-en",
            max_chunk_size=300,
            max_total_words=5000,
            client=client,
            gen_model_name=model_name,
            task_requirement=task_requirement
        )
    elif args.method == "coa":
        pipeline = ChainOfAgentsPipeline(
            client=client,
            model=model_name,
            task_requirement=task_requirement,
            max_chunk_size=5000
        )
    elif args.method == "vanilla":
        pipeline = VanillaPipeline(client, model_name, task_requirement)

    final_answer = pipeline.run(question, combined_context)

    print(f"\n\nFinal Answer:\n{final_answer}")
    print(f"Ground Truth:\n{answer}")
    if final_answer == answer:
        print("✅ The final answer matches the ground truth!")
    else:
        print("❌ The final answer does not match the ground truth.")

if __name__ == "__main__":
    main()
