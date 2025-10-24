"""
Retriever Module

This module provides functionality for retrieving relevant knowledge from the LERK System.
It supports semantic search, hybrid search, and context-aware retrieval using LangChain
integration with Qdrant vector database.

Main Components:
- SemanticRetriever: Vector similarity search using embeddings
- HybridRetriever: Combines semantic and keyword search
- ContextRetriever: Context-aware retrieval with conversation history
- BaseRetriever: Custom base retriever with LERK-specific features
- Config: Configuration classes and presets
- Utils: Utility functions for retrieval processing
- Exceptions: Custom exception classes for error handling

Example Usage:
    from retriever import SemanticRetriever, DEFAULT_RETRIEVER_CONFIG
    
    retriever = SemanticRetriever()
    results = retriever.retrieve("machine learning concepts", limit=5)
"""

from .config import (
    RetrieverConfig,
    DEFAULT_RETRIEVER_CONFIG,
    FAST_RETRIEVER_CONFIG,
    HIGH_QUALITY_RETRIEVER_CONFIG,
    SEMANTIC_RETRIEVER_CONFIG,
    HYBRID_RETRIEVER_CONFIG
)
from .exceptions import (
    RetrievalError,
    VectorSearchError,
    InvalidQueryError,
    MissingEmbeddingModelError,
    DatabaseConnectionError,
    ConfigurationError
)
from .base_retriever import BaseRetriever
from .semantic_retriever import SemanticRetriever
from .hybrid_retriever import HybridRetriever
from .context_retriever import ContextRetriever
from .utils import (
    format_retrieval_results,
    calculate_relevance_score,
    filter_by_metadata,
    rank_by_quality,
    batch_retrieve,
    merge_retrieval_results
)

__version__ = "1.0.0"
__author__ = "LERK System Team"

# Public API
__all__ = [
    # Configuration
    "RetrieverConfig",
    "DEFAULT_RETRIEVER_CONFIG",
    "FAST_RETRIEVER_CONFIG", 
    "HIGH_QUALITY_RETRIEVER_CONFIG",
    "SEMANTIC_RETRIEVER_CONFIG",
    "HYBRID_RETRIEVER_CONFIG",
    
    # Core retrievers
    "BaseRetriever",
    "SemanticRetriever",
    "HybridRetriever", 
    "ContextRetriever",
    
    # Utilities
    "format_retrieval_results",
    "calculate_relevance_score",
    "filter_by_metadata",
    "rank_by_quality",
    "batch_retrieve",
    "merge_retrieval_results",
    
    # Exceptions
    "RetrievalError",
    "VectorSearchError",
    "InvalidQueryError",
    "MissingEmbeddingModelError",
    "DatabaseConnectionError",
    "ConfigurationError",
]
