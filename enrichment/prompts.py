"""
Prompt templates for document enrichment.

Contains prompt generation functions for enriching document chunks
with summaries, keywords, questions, and table data.
"""

from typing import Dict, Any


def generate_enrichment_prompt(chunk_text: str, is_table: bool) -> str:
    """
    Generate a prompt for LLM to generate enrichment for a document chunk.
    
    Args:
        chunk_text: The text content of the chunk
        is_table: Whether the chunk is a table
        
    Returns:
        Formatted prompt string for the LLM
    """
    table_prompt = """
    Summarize the following TABLE chunk in a few sentences. Your summary should describe the key insights and data points in the table.
    """ if is_table else ""
    
    prompt = f"""
    You are a helpful assistant that generates the specified enrichment for the following document chunk:
    {table_prompt}
    Chunk Content:
    _________________________
     {chunk_text}
    _________________________
    """
    return prompt


def get_table_enrichment_prompt() -> str:
    """
    Get the table-specific enrichment prompt.
    
    Returns:
        Table enrichment prompt template
    """
    return """
    Summarize the following TABLE chunk in a few sentences. Your summary should describe the key insights and data points in the table.
    """


def get_text_enrichment_prompt() -> str:
    """
    Get the text-specific enrichment prompt.
    
    Returns:
        Text enrichment prompt template
    """
    return """
    You are a helpful assistant that generates the specified enrichment for the following document chunk:
    """


def create_custom_enrichment_prompt(
    chunk_text: str, 
    is_table: bool, 
    custom_instructions: str = ""
) -> str:
    """
    Create a custom enrichment prompt with additional instructions.
    
    Args:
        chunk_text: The text content of the chunk
        is_table: Whether the chunk is a table
        custom_instructions: Additional custom instructions
        
    Returns:
        Custom formatted prompt string
    """
    base_prompt = generate_enrichment_prompt(chunk_text, is_table)
    
    if custom_instructions:
        base_prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
    
    return base_prompt
