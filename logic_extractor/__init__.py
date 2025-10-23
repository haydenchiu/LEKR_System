"""
Logic Extractor Module

This module provides functionality for extracting logical and causal structure from document chunks.
It identifies claims, relationships, assumptions, constraints, and open questions from text content.

Main Components:
- Models: Pydantic models for structured logic extraction results
- Extractor: Core logic extraction functionality using LLMs
- Prompts: Template functions for generating LLM prompts
- Config: Configuration classes and presets
- Utils: Utility functions for logic processing
- Exceptions: Custom exception classes for error handling

Example Usage:
    from logic_extractor import LogicExtractor, DEFAULT_LOGIC_EXTRACTION_CONFIG
    
    extractor = LogicExtractor()
    result = extractor.extract_logic(chunk)
"""

from .config import (
    LogicExtractionConfig,
    DEFAULT_LOGIC_EXTRACTION_CONFIG,
    FAST_LOGIC_EXTRACTION_CONFIG,
    HIGH_QUALITY_LOGIC_EXTRACTION_CONFIG
)
from .exceptions import (
    LogicExtractionError,
    LLMInvocationError,
    InvalidChunkError,
    MissingAPIKeyError,
    ChunkProcessingError
)
from .models import Claim, LogicalRelation, LogicExtractionSchemaLiteChunk
from .prompts import generate_logic_extraction_prompt
from .extractor import (
    LogicExtractor,
    add_logic_extraction_to_chunk,
    add_logic_extraction_to_chunk_async
)
from .utils import process_chunks_concurrently, extract_logic_from_chunk

__version__ = "1.0.0"
__author__ = "LERK System Team"

# Public API
__all__ = [
    # Models
    "Claim",
    "LogicalRelation",
    "LogicExtractionSchemaLiteChunk",

    # Configuration
    "LogicExtractionConfig",
    "DEFAULT_LOGIC_EXTRACTION_CONFIG",
    "FAST_LOGIC_EXTRACTION_CONFIG",
    "HIGH_QUALITY_LOGIC_EXTRACTION_CONFIG",

    # Core functionality
    "LogicExtractor",
    "add_logic_extraction_to_chunk",
    "add_logic_extraction_to_chunk_async",

    # Prompts
    "generate_logic_extraction_prompt",

    # Utilities
    "process_chunks_concurrently",
    "extract_logic_from_chunk",

    # Exceptions
    "LogicExtractionError",
    "LLMInvocationError",
    "InvalidChunkError",
    "MissingAPIKeyError",
    "ChunkProcessingError",
]
