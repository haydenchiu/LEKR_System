"""
Unit tests for the exceptions module.
"""

import pytest

from ingest.exceptions import (
    IngestionError,
    ParsingError,
    ChunkingError,
    UnsupportedFileTypeError,
    FileNotFoundError,
    ConfigurationError
)


class TestExceptionHierarchy:
    """Test cases for exception hierarchy."""
    
    def test_ingestion_error_base(self):
        """Test that IngestionError is the base exception."""
        error = IngestionError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_parsing_error_inheritance(self):
        """Test that ParsingError inherits from IngestionError."""
        error = ParsingError("Parsing failed")
        assert isinstance(error, IngestionError)
        assert isinstance(error, Exception)
        assert str(error) == "Parsing failed"
    
    def test_chunking_error_inheritance(self):
        """Test that ChunkingError inherits from IngestionError."""
        error = ChunkingError("Chunking failed")
        assert isinstance(error, IngestionError)
        assert isinstance(error, Exception)
        assert str(error) == "Chunking failed"
    
    def test_unsupported_file_type_error_inheritance(self):
        """Test that UnsupportedFileTypeError inherits from IngestionError."""
        error = UnsupportedFileTypeError("Unsupported file type")
        assert isinstance(error, IngestionError)
        assert isinstance(error, Exception)
        assert str(error) == "Unsupported file type"
    
    def test_file_not_found_error_inheritance(self):
        """Test that FileNotFoundError inherits from IngestionError."""
        error = FileNotFoundError("File not found")
        assert isinstance(error, IngestionError)
        assert isinstance(error, Exception)
        assert str(error) == "File not found"
    
    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from IngestionError."""
        error = ConfigurationError("Configuration error")
        assert isinstance(error, IngestionError)
        assert isinstance(error, Exception)
        assert str(error) == "Configuration error"


class TestExceptionCreation:
    """Test cases for exception creation."""
    
    def test_ingestion_error_no_message(self):
        """Test IngestionError creation without message."""
        error = IngestionError()
        assert str(error) == ""
    
    def test_ingestion_error_with_message(self):
        """Test IngestionError creation with message."""
        error = IngestionError("Custom error message")
        assert str(error) == "Custom error message"
    
    def test_parsing_error_with_message(self):
        """Test ParsingError creation with message."""
        error = ParsingError("Failed to parse document")
        assert str(error) == "Failed to parse document"
    
    def test_chunking_error_with_message(self):
        """Test ChunkingError creation with message."""
        error = ChunkingError("Failed to chunk document")
        assert str(error) == "Failed to chunk document"
    
    def test_unsupported_file_type_error_with_message(self):
        """Test UnsupportedFileTypeError creation with message."""
        error = UnsupportedFileTypeError("File type .xyz not supported")
        assert str(error) == "File type .xyz not supported"
    
    def test_file_not_found_error_with_message(self):
        """Test FileNotFoundError creation with message."""
        error = FileNotFoundError("File 'test.txt' not found")
        assert str(error) == "File 'test.txt' not found"
    
    def test_configuration_error_with_message(self):
        """Test ConfigurationError creation with message."""
        error = ConfigurationError("Invalid configuration parameter")
        assert str(error) == "Invalid configuration parameter"


class TestExceptionRaising:
    """Test cases for exception raising."""
    
    def test_raise_ingestion_error(self):
        """Test raising IngestionError."""
        with pytest.raises(IngestionError) as exc_info:
            raise IngestionError("Test error")
        
        assert str(exc_info.value) == "Test error"
    
    def test_raise_parsing_error(self):
        """Test raising ParsingError."""
        with pytest.raises(ParsingError) as exc_info:
            raise ParsingError("Parsing failed")
        
        assert str(exc_info.value) == "Parsing failed"
        assert isinstance(exc_info.value, IngestionError)
    
    def test_raise_chunking_error(self):
        """Test raising ChunkingError."""
        with pytest.raises(ChunkingError) as exc_info:
            raise ChunkingError("Chunking failed")
        
        assert str(exc_info.value) == "Chunking failed"
        assert isinstance(exc_info.value, IngestionError)
    
    def test_raise_unsupported_file_type_error(self):
        """Test raising UnsupportedFileTypeError."""
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            raise UnsupportedFileTypeError("Unsupported file type")
        
        assert str(exc_info.value) == "Unsupported file type"
        assert isinstance(exc_info.value, IngestionError)
    
    def test_raise_file_not_found_error(self):
        """Test raising FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            raise FileNotFoundError("File not found")
        
        assert str(exc_info.value) == "File not found"
        assert isinstance(exc_info.value, IngestionError)
    
    def test_raise_configuration_error(self):
        """Test raising ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Configuration error")
        
        assert str(exc_info.value) == "Configuration error"
        assert isinstance(exc_info.value, IngestionError)


class TestExceptionCatching:
    """Test cases for exception catching."""
    
    def test_catch_ingestion_error(self):
        """Test catching IngestionError."""
        try:
            raise IngestionError("Test error")
        except IngestionError as e:
            assert str(e) == "Test error"
        except Exception:
            pytest.fail("Should have caught IngestionError")
    
    def test_catch_parsing_error_as_ingestion_error(self):
        """Test catching ParsingError as IngestionError."""
        try:
            raise ParsingError("Parsing failed")
        except IngestionError as e:
            assert str(e) == "Parsing failed"
            assert isinstance(e, ParsingError)
        except Exception:
            pytest.fail("Should have caught ParsingError as IngestionError")
    
    def test_catch_chunking_error_as_ingestion_error(self):
        """Test catching ChunkingError as IngestionError."""
        try:
            raise ChunkingError("Chunking failed")
        except IngestionError as e:
            assert str(e) == "Chunking failed"
            assert isinstance(e, ChunkingError)
        except Exception:
            pytest.fail("Should have caught ChunkingError as IngestionError")
    
    def test_catch_specific_exception(self):
        """Test catching specific exception types."""
        try:
            raise ParsingError("Parsing failed")
        except ParsingError as e:
            assert str(e) == "Parsing failed"
        except ChunkingError:
            pytest.fail("Should have caught ParsingError, not ChunkingError")
        except Exception:
            pytest.fail("Should have caught ParsingError")
    
    def test_catch_multiple_exception_types(self):
        """Test catching multiple exception types."""
        exceptions_to_test = [
            ParsingError("Parsing failed"),
            ChunkingError("Chunking failed"),
            UnsupportedFileTypeError("Unsupported file type"),
            FileNotFoundError("File not found"),
            ConfigurationError("Configuration error")
        ]
        
        for exc in exceptions_to_test:
            try:
                raise exc
            except IngestionError as e:
                assert isinstance(e, IngestionError)
                assert str(e) == str(exc)
            except Exception:
                pytest.fail(f"Should have caught {type(exc).__name__} as IngestionError")


class TestExceptionMessages:
    """Test cases for exception messages."""
    
    def test_detailed_error_messages(self):
        """Test detailed error messages."""
        # Parsing error with file details
        error = ParsingError("Failed to parse document.pdf: Invalid format")
        assert "document.pdf" in str(error)
        assert "Invalid format" in str(error)
        
        # Chunking error with chunk details
        error = ChunkingError("Failed to chunk element: Text too long")
        assert "element" in str(error)
        assert "Text too long" in str(error)
        
        # File not found error with path
        error = FileNotFoundError("File not found: /path/to/document.pdf")
        assert "/path/to/document.pdf" in str(error)
        
        # Unsupported file type error with type
        error = UnsupportedFileTypeError("File type .xyz not supported")
        assert ".xyz" in str(error)
        
        # Configuration error with parameter
        error = ConfigurationError("Invalid configuration: max_characters must be positive")
        assert "max_characters" in str(error)
    
    def test_error_message_formatting(self):
        """Test error message formatting."""
        # Test with different message formats
        messages = [
            "Simple error",
            "Error with details: specific information",
            "Multi-line\nerror message",
            "Error with numbers: 123",
            "Error with special chars: !@#$%^&*()",
        ]
        
        for message in messages:
            error = IngestionError(message)
            assert str(error) == message


class TestExceptionUsage:
    """Test cases for exception usage patterns."""
    
    def test_exception_chain(self):
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ParsingError("Failed to parse document") from e
        except ParsingError as e:
            assert str(e) == "Failed to parse document"
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
    
    def test_exception_with_context(self):
        """Test exception with additional context."""
        def parse_document(filename):
            if not filename.endswith('.pdf'):
                raise UnsupportedFileTypeError(f"File {filename} is not a PDF")
            return "parsed content"
        
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            parse_document("document.txt")
        
        assert "document.txt" in str(exc_info.value)
        assert "not a PDF" in str(exc_info.value)
    
    def test_exception_handling_pattern(self):
        """Test common exception handling pattern."""
        def safe_parse(filename):
            try:
                if not filename:
                    raise ValueError("Filename cannot be empty")
                if not filename.endswith(('.pdf', '.txt', '.html')):
                    raise UnsupportedFileTypeError(f"Unsupported file type: {filename}")
                return "parsed content"
            except (ValueError, UnsupportedFileTypeError) as e:
                raise ParsingError(f"Failed to parse {filename}") from e
        
        # Test with valid file
        result = safe_parse("document.pdf")
        assert result == "parsed content"
        
        # Test with unsupported file type
        with pytest.raises(ParsingError) as exc_info:
            safe_parse("document.xyz")
        
        assert "document.xyz" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, UnsupportedFileTypeError)
        
        # Test with empty filename
        with pytest.raises(ParsingError) as exc_info:
            safe_parse("")
        
        assert "Failed to parse" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)
