"""
Tests for the logic_extractor exceptions module.
"""

import pytest
from logic_extractor.exceptions import (
    LogicExtractionError,
    LLMInvocationError,
    InvalidChunkError,
    MissingAPIKeyError,
    ChunkProcessingError
)


class TestLogicExtractionExceptions:
    """Test cases for logic extraction exceptions."""

    def test_logic_extraction_error(self):
        """Test LogicExtractionError base exception."""
        error = LogicExtractionError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_llm_invocation_error(self):
        """Test LLMInvocationError exception."""
        error = LLMInvocationError("LLM failed")
        assert str(error) == "LLM failed"
        assert isinstance(error, LogicExtractionError)

    def test_invalid_chunk_error(self):
        """Test InvalidChunkError exception."""
        error = InvalidChunkError("Invalid chunk")
        assert str(error) == "Invalid chunk"
        assert isinstance(error, LogicExtractionError)

    def test_missing_api_key_error(self):
        """Test MissingAPIKeyError exception."""
        error = MissingAPIKeyError("API key missing")
        assert str(error) == "API key missing"
        assert isinstance(error, LogicExtractionError)

    def test_chunk_processing_error(self):
        """Test ChunkProcessingError exception."""
        error = ChunkProcessingError("Processing failed")
        assert str(error) == "Processing failed"
        assert isinstance(error, LogicExtractionError)

    def test_exception_inheritance(self):
        """Test that all custom exceptions inherit from LogicExtractionError."""
        exceptions = [
            LLMInvocationError,
            InvalidChunkError,
            MissingAPIKeyError,
            ChunkProcessingError
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, LogicExtractionError)
            assert issubclass(exc_class, Exception)