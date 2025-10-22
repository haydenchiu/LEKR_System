"""
Custom exceptions for the ingestion module.
"""


class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class ParsingError(IngestionError):
    """Exception raised when document parsing fails."""
    pass


class ChunkingError(IngestionError):
    """Exception raised when document chunking fails."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Exception raised when file type is not supported."""
    pass


class FileNotFoundError(IngestionError):
    """Exception raised when file is not found."""
    pass


class ConfigurationError(IngestionError):
    """Exception raised when configuration is invalid."""
    pass
