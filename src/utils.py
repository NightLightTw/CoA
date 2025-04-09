from typing import List
import logging
from transformers import AutoTokenizer
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_into_chunks_with_word(text: str, max_chunk_size: int, model: str = "meta-llama/Meta-Llama-3-8B") -> List[str]:
    """
    Split text into chunks based on word count.
    
    Args:
        text: The input text to split
        chunk_size: Maximum number of words per chunk
        model: Not used, kept for compatibility
        
    Returns:
        List[str]: List of text chunks
    """
    # Split by paragraphs first to maintain context
    paragraphs = text.split('\n\n')
    words = []
    current_chunk = []
    chunks = []
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        paragraph_words = paragraph.split()
        
        # If adding this paragraph exceeds chunk size, save current chunk and start new one
        if len(current_chunk) + len(paragraph_words) > max_chunk_size:
            if current_chunk:  # Save current chunk if it exists
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            
            # Handle paragraphs larger than chunk_size
            while len(paragraph_words) > max_chunk_size:
                chunks.append(' '.join(paragraph_words[:max_chunk_size]))
                paragraph_words = paragraph_words[max_chunk_size:]
            
            current_chunk = paragraph_words
        else:
            current_chunk.extend(paragraph_words)
    
    # Add any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def split_into_chunks_with_token(text: str, max_chunk_size: int, tokenizer, model: str = "meta-llama/Meta-Llama-3-8B") -> List[str]:
    """
    Split text into chunks based on token count using a tokenizer.
    
    Args:
        text: The input text to split
        max_chunk_size: Maximum number of tokens per chunk
        model: Model name to use for tokenization
        
    Returns:
        List[str]: List of text chunks
    """

def split_into_chunks_with_token(text: str, max_chunk_size: int, tokenizer, model: str = "gpt-4o-mini") -> List[str]:
    """
    用 tiktoken 將文字依 token 數估算切成多個文字 chunk（無 decode）
    
    Args:
        text: 原始文字
        max_chunk_size: 每段最大 token 數
        model: 用於取得對應 tokenizer 的模型名稱（如 gpt-4o-mini）
        
    Returns:
        List[str]: List of text chunks
    """

    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        tokens = tokenizer.encode(paragraph)
        paragraph_token_count = len(tokens)

        if current_chunk_tokens + paragraph_token_count > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_tokens = 0

            # 當段落太長時，切成多個 token 塊，估算對應的原文字串
            if paragraph_token_count > max_chunk_size:
                approx_chars_per_token = len(paragraph) / paragraph_token_count
                start = 0
                while start < paragraph_token_count:
                    end = min(start + max_chunk_size, paragraph_token_count)
                    char_start = int(start * approx_chars_per_token)
                    char_end = int(end * approx_chars_per_token)
                    chunk_text = paragraph[char_start:char_end].strip()
                    chunks.append(chunk_text)
                    start = end
                current_chunk = ""
                current_chunk_tokens = 0
            else:
                current_chunk = paragraph
                current_chunk_tokens = paragraph_token_count
        else:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
            current_chunk_tokens = len(tokenizer.encode(current_chunk))

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"Split text into {len(chunks)} chunks using tiktoken")
    return chunks

def split_into_chunks_with_token_AutoTokenizer(text: str, max_chunk_size: int, tokenizer, model: str = "meta-llama/Meta-Llama-3-8B") -> List[str]:
    """
    Split text into chunks based on token count using a tokenizer.
    
    Args:
        text: The input text to split
        max_chunk_size: Maximum number of tokens per chunk
        model: Model name to use for tokenization
        
    Returns:
        List[str]: List of text chunks
    """

    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        paragraph_tokens = tokenizer(paragraph).input_ids
        paragraph_token_count = len(paragraph_tokens)

        if current_chunk_tokens + paragraph_token_count > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_tokens = 0
                
            # Handle paragraphs larger than chunk_size
            while paragraph_token_count > max_chunk_size:
                partial_tokens = paragraph_tokens[:max_chunk_size]
                partial_text = tokenizer.decode(partial_tokens, skip_special_tokens=True).strip()
                chunks.append(partial_text)

                paragraph_tokens = paragraph_tokens[max_chunk_size:]
                paragraph_token_count = len(paragraph_tokens)
                paragraph = tokenizer.decode(paragraph_tokens, skip_special_tokens=True).strip()

            current_chunk = paragraph
            current_chunk_tokens = paragraph_token_count
        else:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
            current_chunk_tokens = len(tokenizer(current_chunk).input_ids)

    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"Split text into {len(chunks)} chunks using tokenization")
    return chunks

def count_words(text: str, model: str = "meta-llama/Meta-Llama-3-8B") -> int:
    """
    Count the number of words in a text string.
    
    Args:
        text: The input text
        model: Not used, kept for compatibility
        
    Returns:
        int: Number of words
    """
    return len(text.split())

def count_tokens(text: str, model: str = "meta-llama/Meta-Llama-3-8B") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The input text
        model: Model to use for tokenization
    Returns:
        int: Number of tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    return len(tokenizer(text).input_ids)

def get_default_prompts() -> tuple[str, str]:
    """
    Get default system prompts for worker and manager agents.
    
    Returns:
        tuple[str, str]: (worker_prompt, manager_prompt)
    """
    worker_prompt = """You are a worker agent responsible for analyzing a portion of a document.
Your task is to identify key information related to the user's query and provide clear, concise analysis."""

    manager_prompt = """You are a manager agent responsible for synthesizing information from multiple workers.
Your task is to combine their analyses into a coherent, comprehensive response that directly answers the user's query."""

    return worker_prompt, manager_prompt 

def calculate_word_counts(dataset):
    """
    Calculate the number of words in questions, answers, and contexts.
    
    Args:
        dataset: The dataset containing the samples
    """
    # Initialize counters
    total_question_words = 0
    total_context_words = 0
    sample_count = len(dataset)

    # Process each sample
    for i in range(sample_count):
        data_sample = dataset[i]
        
        # Get question and count words
        question = data_sample["input"]
        question_words = count_words(question)
        total_question_words += question_words
        
        # Combine all sentences in the context
        combined_context = data_sample["context"]
        context_words = count_words(combined_context)
        total_context_words += context_words

    # Calculate averages
    avg_question_words = total_question_words / sample_count
    avg_context_words = total_context_words / sample_count

    # Print results
    print(f"Total samples: {sample_count}")
    print(f"Average question words: {avg_question_words:.2f}")
    print(f"Average context words: {avg_context_words:.2f}")
