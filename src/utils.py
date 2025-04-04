from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_into_chunks_with_word(text: str, max_chunk_size: int, model: str = "llama-3.3-70b-versatile") -> List[str]:
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

def count_words(text: str, model: str = "llama-3.3-70b-versatile") -> int:
    """
    Count the number of words in a text string.
    
    Args:
        text: The input text
        model: Not used, kept for compatibility
        
    Returns:
        int: Number of words
    """
    return len(text.split())

def count_tokens(text: str, model: str = "llama-3.3-70b-versatile") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The input text
        model: Not used, kept for compatibility
    Returns:
        int: Number of tokens
    """
    

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
