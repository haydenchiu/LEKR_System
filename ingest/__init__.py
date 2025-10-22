"""
LERK System - Document Ingestion Module

This module handles the ingestion and initial processing of documents,
including parsing, chunking, and preparing documents for enrichment.
"""

from .parsing import DocumentParser, parse_file
from .chunking import DocumentChunker, elements_to_chunks
from .orchestrator import DocumentIngestionOrchestrator
from .config import IngestionConfig, DEFAULT_CONFIG, LARGE_DOCUMENT_CONFIG, FAST_CONFIG, HIGH_QUALITY_CONFIG
from .utils import (
    get_file_type, 
    is_supported_file_type, 
    get_chunk_content, 
    is_table_chunk,
    get_chunk_metadata,
    validate_file_path,
    get_ingestion_summary
)
from .exceptions import (
    IngestionError,
    ParsingError,
    ChunkingError,
    UnsupportedFileTypeError,
    FileNotFoundError,
    ConfigurationError
)

__all__ = [
    # Main classes
    "DocumentParser",
    "DocumentChunker", 
    "DocumentIngestionOrchestrator",
    
    # Convenience functions
    "parse_file",
    "elements_to_chunks",
    
    # Configuration
    "IngestionConfig",
    "DEFAULT_CONFIG",
    "LARGE_DOCUMENT_CONFIG", 
    "FAST_CONFIG",
    "HIGH_QUALITY_CONFIG",
    
    # Utilities
    "get_file_type",
    "is_supported_file_type",
    "get_chunk_content",
    "is_table_chunk",
    "get_chunk_metadata", 
    "validate_file_path",
    "get_ingestion_summary",
    
    # Exceptions
    "IngestionError",
    "ParsingError",
    "ChunkingError", 
    "UnsupportedFileTypeError",
    "FileNotFoundError",
    "ConfigurationError"
]
