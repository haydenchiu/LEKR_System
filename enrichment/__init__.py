"""
LERK System - Document Enrichment Module

This module handles the enrichment of document chunks with summaries, keywords,
questions, and table data using LLM-based processing.
"""

from .models import ChunkEnrichment
from .enricher import DocumentEnricher, add_enrichment_to_chunk, add_enrichment_to_chunk_async
from .prompts import generate_enrichment_prompt
from .config import EnrichmentConfig, DEFAULT_ENRICHMENT_CONFIG
from .utils import (
    is_table_chunk,
    get_chunk_content,
    process_chunks_concurrently,
    enrich_and_extract_logic
)
from .exceptions import (
    EnrichmentError,
    PromptGenerationError,
    LLMInvocationError,
    ChunkProcessingError
)

__all__ = [
    # Main classes
    "ChunkEnrichment",
    "DocumentEnricher",
    
    # Convenience functions
    "add_enrichment_to_chunk",
    "add_enrichment_to_chunk_async",
    "generate_enrichment_prompt",
    "process_chunks_concurrently",
    "enrich_and_extract_logic",
    
    # Configuration
    "EnrichmentConfig",
    "DEFAULT_ENRICHMENT_CONFIG",
    
    # Utilities
    "is_table_chunk",
    "get_chunk_content",
    
    # Exceptions
    "EnrichmentError",
    "PromptGenerationError",
    "LLMInvocationError",
    "ChunkProcessingError"
]
