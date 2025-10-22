"""
Unit tests for enrichment exceptions.

Tests the custom exception classes and error handling.
"""

import pytest

from enrichment.exceptions import (
    EnrichmentError,
    PromptGenerationError,
    LLMInvocationError,
    ChunkProcessingError,
    BatchProcessingError,
    ConfigurationError,
    ValidationError
)


class TestEnrichmentError:
    """Test cases for EnrichmentError base exception."""
    
    def test_init_basic(self):
        """Test EnrichmentError initialization with basic message."""
        error = EnrichmentError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details is None
    
    def test_init_with_details(self):
        """Test EnrichmentError initialization with details."""
        error = EnrichmentError("Test error message", "Additional details")
        
        assert str(error) == "Test error message: Additional details"
        assert error.message == "Test error message"
        assert error.details == "Additional details"
    
    def test_str_representation(self):
        """Test string representation of EnrichmentError."""
        error = EnrichmentError("Test error")
        assert str(error) == "Test error"
        
        error_with_details = EnrichmentError("Test error", "Details")
        assert str(error_with_details) == "Test error: Details"
    
    def test_inheritance(self):
        """Test that EnrichmentError inherits from Exception."""
        error = EnrichmentError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, EnrichmentError)


class TestPromptGenerationError:
    """Test cases for PromptGenerationError."""
    
    def test_init_basic(self):
        """Test PromptGenerationError initialization with basic message."""
        error = PromptGenerationError("Prompt generation failed")
        
        assert str(error) == "Prompt generation failed"
        assert error.message == "Prompt generation failed"
        assert error.chunk_info is None
    
    def test_init_with_chunk_info(self):
        """Test PromptGenerationError initialization with chunk info."""
        error = PromptGenerationError("Prompt generation failed", "Chunk 123")
        
        assert str(error) == "Prompt generation failed: Chunk 123"
        assert error.message == "Prompt generation failed"
        assert error.chunk_info == "Chunk 123"
    
    def test_inheritance(self):
        """Test that PromptGenerationError inherits from EnrichmentError."""
        error = PromptGenerationError("Test error")
        assert isinstance(error, EnrichmentError)
        assert isinstance(error, PromptGenerationError)
    
    def test_str_representation(self):
        """Test string representation of PromptGenerationError."""
        error = PromptGenerationError("Prompt error", "Chunk info")
        assert str(error) == "Prompt error: Chunk info"
        
        error_no_info = PromptGenerationError("Prompt error")
        assert str(error_no_info) == "Prompt error"


class TestLLMInvocationError:
    """Test cases for LLMInvocationError."""
    
    def test_init_basic(self):
        """Test LLMInvocationError initialization with basic message."""
        error = LLMInvocationError("LLM invocation failed")
        
        assert str(error) == "LLM invocation failed"
        assert error.message == "LLM invocation failed"
        assert error.model_name is None
        assert error.retry_count == 0
    
    def test_init_with_model_name(self):
        """Test LLMInvocationError initialization with model name."""
        error = LLMInvocationError("LLM invocation failed", "gpt-4o")
        
        assert str(error) == "LLM invocation failed"
        assert error.message == "LLM invocation failed"
        assert error.model_name == "gpt-4o"
        assert error.retry_count == 0
    
    def test_init_with_retry_count(self):
        """Test LLMInvocationError initialization with retry count."""
        error = LLMInvocationError("LLM invocation failed", "gpt-4o", 3)
        
        assert str(error) == "LLM invocation failed"
        assert error.message == "LLM invocation failed"
        assert error.model_name == "gpt-4o"
        assert error.retry_count == 3
    
    def test_inheritance(self):
        """Test that LLMInvocationError inherits from EnrichmentError."""
        error = LLMInvocationError("Test error")
        assert isinstance(error, EnrichmentError)
        assert isinstance(error, LLMInvocationError)
    
    def test_str_representation(self):
        """Test string representation of LLMInvocationError."""
        error = LLMInvocationError("LLM error", "gpt-4o", 2)
        assert str(error) == "LLM error"
        
        error_no_details = LLMInvocationError("LLM error")
        assert str(error_no_details) == "LLM error"


class TestChunkProcessingError:
    """Test cases for ChunkProcessingError."""
    
    def test_init_basic(self):
        """Test ChunkProcessingError initialization with basic message."""
        error = ChunkProcessingError("Chunk processing failed")
        
        assert str(error) == "Chunk processing failed"
        assert error.message == "Chunk processing failed"
        assert error.chunk_id is None
        assert error.chunk_type is None
    
    def test_init_with_chunk_id(self):
        """Test ChunkProcessingError initialization with chunk ID."""
        error = ChunkProcessingError("Chunk processing failed", "chunk_123")
        
        assert str(error) == "Chunk processing failed"
        assert error.message == "Chunk processing failed"
        assert error.chunk_id == "chunk_123"
        assert error.chunk_type is None
    
    def test_init_with_chunk_type(self):
        """Test ChunkProcessingError initialization with chunk type."""
        error = ChunkProcessingError("Chunk processing failed", "chunk_123", "text")
        
        assert str(error) == "Chunk processing failed"
        assert error.message == "Chunk processing failed"
        assert error.chunk_id == "chunk_123"
        assert error.chunk_type == "text"
    
    def test_inheritance(self):
        """Test that ChunkProcessingError inherits from EnrichmentError."""
        error = ChunkProcessingError("Test error")
        assert isinstance(error, EnrichmentError)
        assert isinstance(error, ChunkProcessingError)
    
    def test_str_representation(self):
        """Test string representation of ChunkProcessingError."""
        error = ChunkProcessingError("Chunk error", "chunk_123", "text")
        assert str(error) == "Chunk error"
        
        error_no_details = ChunkProcessingError("Chunk error")
        assert str(error_no_details) == "Chunk error"


class TestBatchProcessingError:
    """Test cases for BatchProcessingError."""
    
    def test_init_basic(self):
        """Test BatchProcessingError initialization with basic message."""
        error = BatchProcessingError("Batch processing failed")
        
        assert str(error) == "Batch processing failed"
        assert error.message == "Batch processing failed"
        assert error.batch_size is None
        assert error.failed_chunks == 0
    
    def test_init_with_batch_size(self):
        """Test BatchProcessingError initialization with batch size."""
        error = BatchProcessingError("Batch processing failed", 10)
        
        assert str(error) == "Batch processing failed"
        assert error.message == "Batch processing failed"
        assert error.batch_size == 10
        assert error.failed_chunks == 0
    
    def test_init_with_failed_chunks(self):
        """Test BatchProcessingError initialization with failed chunks count."""
        error = BatchProcessingError("Batch processing failed", 10, 3)
        
        assert str(error) == "Batch processing failed"
        assert error.message == "Batch processing failed"
        assert error.batch_size == 10
        assert error.failed_chunks == 3
    
    def test_inheritance(self):
        """Test that BatchProcessingError inherits from EnrichmentError."""
        error = BatchProcessingError("Test error")
        assert isinstance(error, EnrichmentError)
        assert isinstance(error, BatchProcessingError)
    
    def test_str_representation(self):
        """Test string representation of BatchProcessingError."""
        error = BatchProcessingError("Batch error", 10, 3)
        assert str(error) == "Batch error"
        
        error_no_details = BatchProcessingError("Batch error")
        assert str(error_no_details) == "Batch error"


class TestConfigurationError:
    """Test cases for ConfigurationError."""
    
    def test_init_basic(self):
        """Test ConfigurationError initialization with basic message."""
        error = ConfigurationError("Configuration error")
        
        assert str(error) == "Configuration error"
        assert error.message == "Configuration error"
        assert error.config_field is None
    
    def test_init_with_config_field(self):
        """Test ConfigurationError initialization with config field."""
        error = ConfigurationError("Configuration error", "temperature")
        
        assert str(error) == "Configuration error"
        assert error.message == "Configuration error"
        assert error.config_field == "temperature"
    
    def test_inheritance(self):
        """Test that ConfigurationError inherits from EnrichmentError."""
        error = ConfigurationError("Test error")
        assert isinstance(error, EnrichmentError)
        assert isinstance(error, ConfigurationError)
    
    def test_str_representation(self):
        """Test string representation of ConfigurationError."""
        error = ConfigurationError("Config error", "temperature")
        assert str(error) == "Config error"
        
        error_no_field = ConfigurationError("Config error")
        assert str(error_no_field) == "Config error"


class TestValidationError:
    """Test cases for ValidationError."""
    
    def test_init_basic(self):
        """Test ValidationError initialization with basic message."""
        error = ValidationError("Validation error")
        
        assert str(error) == "Validation error"
        assert error.message == "Validation error"
        assert error.field_name is None
        assert error.value is None
    
    def test_init_with_field_name(self):
        """Test ValidationError initialization with field name."""
        error = ValidationError("Validation error", "temperature")
        
        assert str(error) == "Validation error"
        assert error.message == "Validation error"
        assert error.field_name == "temperature"
        assert error.value is None
    
    def test_init_with_value(self):
        """Test ValidationError initialization with value."""
        error = ValidationError("Validation error", "temperature", "2.5")
        
        assert str(error) == "Validation error"
        assert error.message == "Validation error"
        assert error.field_name == "temperature"
        assert error.value == "2.5"
    
    def test_inheritance(self):
        """Test that ValidationError inherits from EnrichmentError."""
        error = ValidationError("Test error")
        assert isinstance(error, EnrichmentError)
        assert isinstance(error, ValidationError)
    
    def test_str_representation(self):
        """Test string representation of ValidationError."""
        error = ValidationError("Validation error", "temperature", "2.5")
        assert str(error) == "Validation error"
        
        error_no_details = ValidationError("Validation error")
        assert str(error_no_details) == "Validation error"


class TestExceptionHierarchy:
    """Test cases for exception hierarchy and inheritance."""
    
    def test_exception_inheritance(self):
        """Test that all custom exceptions inherit from EnrichmentError."""
        exceptions = [
            PromptGenerationError("Test"),
            LLMInvocationError("Test"),
            ChunkProcessingError("Test"),
            BatchProcessingError("Test"),
            ConfigurationError("Test"),
            ValidationError("Test")
        ]
        
        for exception in exceptions:
            assert isinstance(exception, EnrichmentError)
            assert isinstance(exception, Exception)
    
    def test_exception_specific_types(self):
        """Test that exceptions have correct specific types."""
        prompt_error = PromptGenerationError("Test")
        llm_error = LLMInvocationError("Test")
        chunk_error = ChunkProcessingError("Test")
        batch_error = BatchProcessingError("Test")
        config_error = ConfigurationError("Test")
        validation_error = ValidationError("Test")
        
        assert isinstance(prompt_error, PromptGenerationError)
        assert isinstance(llm_error, LLMInvocationError)
        assert isinstance(chunk_error, ChunkProcessingError)
        assert isinstance(batch_error, BatchProcessingError)
        assert isinstance(config_error, ConfigurationError)
        assert isinstance(validation_error, ValidationError)
    
    def test_exception_catching(self):
        """Test that exceptions can be caught by their base class."""
        try:
            raise PromptGenerationError("Test error")
        except EnrichmentError as e:
            assert str(e) == "Test error"
        except Exception:
            pytest.fail("Should have been caught by EnrichmentError")
    
    def test_exception_catching_specific(self):
        """Test that exceptions can be caught by their specific type."""
        try:
            raise LLMInvocationError("Test error")
        except LLMInvocationError as e:
            assert str(e) == "Test error"
        except EnrichmentError:
            pytest.fail("Should have been caught by LLMInvocationError")
        except Exception:
            pytest.fail("Should have been caught by LLMInvocationError")


class TestExceptionUsage:
    """Test cases for exception usage patterns."""
    
    def test_exception_with_context(self):
        """Test exception with context information."""
        try:
            raise ChunkProcessingError("Processing failed", "chunk_123", "text")
        except ChunkProcessingError as e:
            assert e.chunk_id == "chunk_123"
            assert e.chunk_type == "text"
    
    def test_exception_with_retry_info(self):
        """Test exception with retry information."""
        try:
            raise LLMInvocationError("LLM failed", "gpt-4o", 3)
        except LLMInvocationError as e:
            assert e.model_name == "gpt-4o"
            assert e.retry_count == 3
    
    def test_exception_with_batch_info(self):
        """Test exception with batch information."""
        try:
            raise BatchProcessingError("Batch failed", 10, 3)
        except BatchProcessingError as e:
            assert e.batch_size == 10
            assert e.failed_chunks == 3
    
    def test_exception_with_config_info(self):
        """Test exception with configuration information."""
        try:
            raise ConfigurationError("Config failed", "temperature")
        except ConfigurationError as e:
            assert e.config_field == "temperature"
    
    def test_exception_with_validation_info(self):
        """Test exception with validation information."""
        try:
            raise ValidationError("Validation failed", "temperature", "2.5")
        except ValidationError as e:
            assert e.field_name == "temperature"
            assert e.value == "2.5"
    
    def test_exception_chaining(self):
        """Test exception chaining with cause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ChunkProcessingError("Chunk processing failed") from e
        except ChunkProcessingError as e:
            assert str(e) == "Chunk processing failed"
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"
    
    def test_exception_with_details(self):
        """Test exception with additional details."""
        try:
            raise EnrichmentError("Enrichment failed", "Additional context")
        except EnrichmentError as e:
            assert e.message == "Enrichment failed"
            assert e.details == "Additional context"
            assert str(e) == "Enrichment failed: Additional context"
