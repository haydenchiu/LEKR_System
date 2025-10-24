"""
Custom exception classes for the QA agent module.

This module defines a hierarchy of custom exceptions to provide more specific
error handling for various scenarios that can occur during question answering,
such as issues with retrieval, answer generation, or session management.
"""


class QAError(Exception):
    """Base exception for QA agent module errors."""
    pass


class RetrievalError(QAError):
    """Raised when an error occurs during document retrieval."""
    pass


class AnswerGenerationError(QAError):
    """Raised when an error occurs during answer generation."""
    pass


class ConfigurationError(QAError):
    """Raised when there is an issue with configuration."""
    pass


class SessionError(QAError):
    """Raised when there is an issue with session management."""
    pass


class TimeoutError(QAError):
    """Raised when an operation times out."""
    pass


class ValidationError(QAError):
    """Raised when input validation fails."""
    pass


class ContextError(QAError):
    """Raised when there is an issue with conversation context."""
    pass
