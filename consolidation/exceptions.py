"""
Custom exception classes for the consolidation module.

This module defines a hierarchy of custom exceptions to provide more specific
error handling for various scenarios that can occur during knowledge consolidation,
such as issues with document processing, subject aggregation, or storage operations.
"""


class ConsolidationError(Exception):
    """Base exception for consolidation module errors."""
    pass


class DocumentConsolidationError(ConsolidationError):
    """Raised when document-level consolidation fails."""
    pass


class SubjectConsolidationError(ConsolidationError):
    """Raised when subject-level consolidation fails."""
    pass


class StorageError(ConsolidationError):
    """Raised when storage operations fail."""
    pass


class InvalidKnowledgeError(ConsolidationError):
    """Raised when invalid knowledge data is provided."""
    pass


class MissingAPIKeyError(ConsolidationError):
    """Raised when an API key is missing for LLM initialization."""
    pass


class ConceptExtractionError(ConsolidationError):
    """Raised when concept extraction fails."""
    pass


class RelationExtractionError(ConsolidationError):
    """Raised when relation extraction fails."""
    pass


class KnowledgeValidationError(ConsolidationError):
    """Raised when knowledge validation fails."""
    pass


class StorageBackendError(StorageError):
    """Raised when storage backend operations fail."""
    pass


class VectorSearchError(StorageError):
    """Raised when vector search operations fail."""
    pass


class KnowledgeMergeError(ConsolidationError):
    """Raised when merging knowledge fails."""
    pass


class QualityValidationError(ConsolidationError):
    """Raised when quality validation fails."""
    pass
