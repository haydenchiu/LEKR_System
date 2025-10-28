"""
Retriever Module

This module provides functionality for retrieving relevant knowledge from the LERK System.
It supports semantic search, hybrid search, context-aware retrieval, and Qdrant indexing
using LangChain integration with Qdrant vector database.

Main Components:
- SemanticRetriever: Vector similarity search using embeddings
- HybridRetriever: Combines semantic and keyword search
- ContextRetriever: Context-aware retrieval with conversation history
- QdrantIndexer: Indexing service for Qdrant vector database
- BaseRetriever: Custom base retriever with LERK-specific features
- Config: Configuration classes and presets
- Utils: Utility functions for retrieval processing
- Exceptions: Custom exception classes for error handling

Example Usage:
    from retriever import SemanticRetriever, QdrantIndexer, DEFAULT_RETRIEVER_CONFIG
    
    # Indexing
    indexer = QdrantIndexer()
    indexer.index_processed_documents("data/processed")
    
    # Retrieval
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
from .indexing import QdrantIndexer, EmbeddingStrategy, create_indexer, index_processed_documents
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
    
    # Indexing
    "QdrantIndexer",
    "EmbeddingStrategy", 
    "create_indexer",
    "index_processed_documents",
    
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
