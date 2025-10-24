"""
Custom exception classes for the retriever module.

This module defines a hierarchy of custom exceptions to provide more specific
error handling for various scenarios that can occur during retrieval operations,
such as vector search failures, invalid queries, or missing embedding models.
"""


class RetrievalError(Exception):
    """Base exception for retriever module errors."""
    pass


class VectorSearchError(RetrievalError):
    """Raised when vector search operations fail."""
    pass


class InvalidQueryError(RetrievalError):
    """Raised when an invalid query is provided for retrieval."""
    pass


class MissingEmbeddingModelError(RetrievalError):
    """Raised when an embedding model is missing or cannot be loaded."""
    pass


class DatabaseConnectionError(RetrievalError):
    """Raised when there are issues connecting to the vector database."""
    pass


class ConfigurationError(RetrievalError):
    """Raised when there are configuration issues."""
    pass


class EmbeddingError(RetrievalError):
    """Raised when embedding generation fails."""
    pass


class FilteringError(RetrievalError):
    """Raised when metadata filtering fails."""
    pass


class RankingError(RetrievalError):
    """Raised when result ranking fails."""
    pass


class CacheError(RetrievalError):
    """Raised when caching operations fail."""
    pass


class TimeoutError(RetrievalError):
    """Raised when retrieval operations timeout."""
    pass


class BatchProcessingError(RetrievalError):
    """Raised when batch retrieval operations fail."""
    pass
