"""
Custom exceptions for the enrichment module.

Defines specific exception classes for enrichment-related errors
to provide better error handling and debugging.
"""


class EnrichmentError(Exception):
    """Base exception for enrichment-related errors."""
    
    def __init__(self, message: str, details: str = None):
        """
        Initialize enrichment error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self):
        """Return string representation of the error."""
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class PromptGenerationError(EnrichmentError):
    """Exception raised when prompt generation fails."""
    
    def __init__(self, message: str, chunk_info: str = None):
        """
        Initialize prompt generation error.
        
        Args:
            message: Error message
            chunk_info: Information about the chunk that caused the error
        """
        super().__init__(message, chunk_info)
        self.chunk_info = chunk_info


class LLMInvocationError(EnrichmentError):
    """Exception raised when LLM invocation fails."""
    
    def __init__(self, message: str, model_name: str = None, retry_count: int = 0):
        """
        Initialize LLM invocation error.
        
        Args:
            message: Error message
            model_name: Name of the model that failed
            retry_count: Number of retry attempts made
        """
        super().__init__(message)
        self.model_name = model_name
        self.retry_count = retry_count


class ChunkProcessingError(EnrichmentError):
    """Exception raised when chunk processing fails."""
    
    def __init__(self, message: str, chunk_id: str = None, chunk_type: str = None):
        """
        Initialize chunk processing error.
        
        Args:
            message: Error message
            chunk_id: Identifier of the chunk that failed
            chunk_type: Type of the chunk (text/table)
        """
        super().__init__(message)
        self.chunk_id = chunk_id
        self.chunk_type = chunk_type


class BatchProcessingError(EnrichmentError):
    """Exception raised when batch processing fails."""
    
    def __init__(self, message: str, batch_size: int = None, failed_chunks: int = 0):
        """
        Initialize batch processing error.
        
        Args:
            message: Error message
            batch_size: Size of the batch that failed
            failed_chunks: Number of chunks that failed in the batch
        """
        super().__init__(message)
        self.batch_size = batch_size
        self.failed_chunks = failed_chunks


class ConfigurationError(EnrichmentError):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, config_field: str = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_field: Name of the configuration field that caused the error
        """
        super().__init__(message)
        self.config_field = config_field


class ValidationError(EnrichmentError):
    """Exception raised when data validation fails."""
    
    def __init__(self, message: str, field_name: str = None, value: str = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field_name: Name of the field that failed validation
            value: Value that failed validation
        """
        super().__init__(message)
        self.field_name = field_name
        self.value = value
