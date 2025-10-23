"""
Prompt generation functions for logic extraction.

This module provides functions to generate prompts for LLMs to extract logical and causal
structure from document chunks, including claims, relationships, assumptions, constraints,
and open questions.
"""

from typing import Optional


def generate_logic_extraction_prompt(chunk_text: str, is_table: bool = False) -> str:
    """
    Generate a prompt for LLM to extract logic from a document chunk.
    
    Args:
        chunk_text: The text content of the chunk to analyze
        is_table: Whether the chunk is a table (affects prompt instructions)
        
    Returns:
        Formatted prompt string for the LLM
    """
    if is_table:
        prompt = f"""
You are a Logic Extraction Agent. This chunk is a table. 
Use the table summary or text to extract logical and causal structure.
Identify claims, relationships, assumptions, constraints, and open questions.
Represent your output in valid JSON following the LogicExtractionSchemaLiteChunk format.

Table Summary / Content:
_________________________
{chunk_text}
_________________________

Instructions:
1. Extract factual claims from the table data
2. Identify logical relationships between data points
3. Note any assumptions underlying the data presentation
4. Identify constraints or limitations mentioned
5. Generate open questions that arise from the data
6. Use unique IDs for claims (claim_1, claim_2, etc.)
7. Ensure all claim references in relations are valid
"""
    else:
        prompt = f"""
You are a Logic Extraction Agent. Your task is to read a text chunk and extract its logical and causal structure. 
Identify claims, relationships, assumptions, constraints, and open questions.
Represent your output in valid JSON following the LogicExtractionSchemaLiteChunk format.

Chunk Content:
_________________________
{chunk_text}
_________________________

Instructions:
1. Identify atomic claims (factual statements, inferences, speculations, assumptions)
2. Determine logical relationships between claims (causal, inferential, supportive, contradictory)
3. Extract unstated assumptions that underlie the reasoning
4. Identify constraints or limitations mentioned or implied
5. Generate open questions that arise from the content
6. Use unique IDs for claims (claim_1, claim_2, etc.)
7. Ensure all claim references in relations are valid
8. Set appropriate confidence levels for claims and certainty levels for relations
"""
    
    return prompt.strip()


def get_table_logic_extraction_prompt(table_content: str) -> str:
    """
    Generate a specialized prompt for table logic extraction.
    
    Args:
        table_content: The table content (HTML or text)
        
    Returns:
        Formatted prompt string for table logic extraction
    """
    prompt = f"""
You are a Logic Extraction Agent specializing in tabular data analysis.
Extract logical and causal structure from the following table content.

Table Content:
_________________________
{table_content}
_________________________

Focus on:
1. Data patterns and trends that represent claims
2. Comparative relationships between data points
3. Causal relationships suggested by the data
4. Assumptions about data collection or methodology
5. Constraints or limitations in the data
6. Questions that arise from gaps or patterns in the data

Provide your analysis in the LogicExtractionSchemaLiteChunk format.
"""
    
    return prompt.strip()


def get_text_logic_extraction_prompt(text_content: str) -> str:
    """
    Generate a specialized prompt for text logic extraction.
    
    Args:
        text_content: The text content to analyze
        
    Returns:
        Formatted prompt string for text logic extraction
    """
    prompt = f"""
You are a Logic Extraction Agent specializing in textual content analysis.
Extract logical and causal structure from the following text content.

Text Content:
_________________________
{text_content}
_________________________

Focus on:
1. Explicit claims and statements
2. Implicit claims and implications
3. Logical arguments and reasoning chains
4. Cause-effect relationships
5. Supporting evidence and premises
6. Assumptions underlying the arguments
7. Constraints or limitations mentioned
8. Open questions or uncertainties

Provide your analysis in the LogicExtractionSchemaLiteChunk format.
"""
    
    return prompt.strip()


def create_custom_logic_extraction_prompt(
    chunk_text: str, 
    is_table: bool = False,
    custom_instructions: Optional[str] = None
) -> str:
    """
    Create a custom logic extraction prompt with optional instructions.
    
    Args:
        chunk_text: The text content of the chunk
        is_table: Whether the chunk is a table
        custom_instructions: Optional custom instructions to add
        
    Returns:
        Customized prompt string
    """
    base_prompt = generate_logic_extraction_prompt(chunk_text, is_table)
    
    if custom_instructions:
        custom_section = f"""
Additional Instructions:
{custom_instructions}
"""
        return base_prompt + custom_section
    
    return base_prompt


def get_logic_extraction_schema_prompt() -> str:
    """
    Get a prompt that explains the LogicExtractionSchemaLiteChunk format.
    
    Returns:
        Prompt explaining the expected output format
    """
    return """
You must respond with a JSON object following this exact schema:

{
    "chunk_id": "string - unique identifier for this chunk",
    "claims": [
        {
            "id": "string - unique claim identifier (e.g., claim_1, claim_2)",
            "statement": "string - the actual claim text",
            "type": "string - one of: factual, inferential, speculative, assumptive",
            "confidence": "number - confidence level from 0.0 to 1.0",
            "derived_from": ["array of claim IDs this claim depends on, or null"]
        }
    ],
    "logical_relations": [
        {
            "premise": "string - claim ID serving as premise",
            "conclusion": "string - claim ID serving as conclusion", 
            "relation_type": "string - one of: causal, inferential, correlative, contradictory, supportive",
            "certainty": "number - certainty level from 0.0 to 1.0"
        }
    ],
    "assumptions": ["array of unstated but implied premises"],
    "constraints": ["array of contextual or technical constraints"],
    "open_questions": ["array of questions or uncertainties raised"]
}

Important rules:
- All claim IDs referenced in logical_relations must exist in the claims array
- Use consistent claim ID naming (claim_1, claim_2, etc.)
- Confidence and certainty values must be between 0.0 and 1.0
- Arrays can be empty but must be present
- All strings must be properly escaped for JSON
"""
