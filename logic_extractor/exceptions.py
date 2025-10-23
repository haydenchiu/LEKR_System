"""
Custom exception classes for the logic extraction module.

This module defines a hierarchy of custom exceptions for robust error handling
in the logic extraction functionality.
"""


class LogicExtractionError(Exception):
    """Base exception for logic extraction module errors."""
    pass


class LLMInvocationError(LogicExtractionError):
    """Raised when an LLM call fails during logic extraction."""
    pass


class InvalidChunkError(LogicExtractionError):
    """Raised when an invalid chunk is provided for logic extraction."""
    pass


class MissingAPIKeyError(LogicExtractionError):
    """Raised when an API key is missing for LLM initialization."""
    pass


class ChunkProcessingError(LogicExtractionError):
    """Raised when there is an error processing a chunk during logic extraction."""
    pass