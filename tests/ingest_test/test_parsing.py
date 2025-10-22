"""
Unit tests for the parsing module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ingest.parsing import DocumentParser, parse_file
from ingest.exceptions import ParsingError, FileNotFoundError


class TestDocumentParser:
    """Test cases for DocumentParser class."""
    
    def test_init_default(self):
        """Test DocumentParser initialization with default parameters."""
        parser = DocumentParser()
        
        assert parser.strategy == "hi_res"
        assert parser.skip_infer_table_types == []
        assert parser.max_partition is None
    
    def test_init_custom(self):
        """Test DocumentParser initialization with custom parameters."""
        parser = DocumentParser(
            strategy="fast",
            skip_infer_table_types=["pdf"],
            max_partition=100
        )
        
        assert parser.strategy == "fast"
        assert parser.skip_infer_table_types == ["pdf"]
        assert parser.max_partition == 100
    
    @patch('ingest.parsing.partition')
    def test_parse_file_success(self, mock_partition, temp_file):
        """Test successful file parsing."""
        # Setup mock
        mock_elements = [Mock(), Mock()]
        mock_partition.return_value = mock_elements
        
        parser = DocumentParser()
        result = parser.parse_file(temp_file)
        
        # Verify mock was called correctly
        mock_partition.assert_called_once_with(
            filename=temp_file,
            skip_infer_table_types=[],
            strategy="hi_res",
            max_partition=None
        )
        
        assert result == mock_elements
    
    @patch('ingest.parsing.partition')
    def test_parse_file_with_custom_params(self, mock_partition, temp_file):
        """Test file parsing with custom parameters."""
        mock_elements = [Mock()]
        mock_partition.return_value = mock_elements
        
        parser = DocumentParser(
            strategy="fast",
            skip_infer_table_types=["pdf"],
            max_partition=50
        )
        
        result = parser.parse_file(temp_file)
        
        mock_partition.assert_called_once_with(
            filename=temp_file,
            skip_infer_table_types=["pdf"],
            strategy="fast",
            max_partition=50
        )
        assert result == mock_elements
    
    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        parser = DocumentParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent.txt")
    
    @patch('ingest.parsing.partition')
    def test_parse_file_error(self, mock_partition, temp_file):
        """Test file parsing with error."""
        mock_partition.side_effect = Exception("Parsing failed")
        
        parser = DocumentParser()
        
        with pytest.raises(Exception) as exc_info:
            parser.parse_file(temp_file)
        
        assert "Parsing failed" in str(exc_info.value)
    
    @patch('ingest.parsing.partition')
    def test_parse_url_success(self, mock_partition):
        """Test successful URL parsing."""
        mock_elements = [Mock(), Mock()]
        mock_partition.return_value = mock_elements
        
        parser = DocumentParser()
        url = "https://example.com/article"
        result = parser.parse_url(url)
        
        mock_partition.assert_called_once_with(url=url)
        assert result == mock_elements
    
    @patch('ingest.parsing.partition')
    def test_parse_url_error(self, mock_partition):
        """Test URL parsing with error."""
        mock_partition.side_effect = Exception("URL parsing failed")
        
        parser = DocumentParser()
        
        with pytest.raises(Exception) as exc_info:
            parser.parse_url("https://example.com")
        
        assert "URL parsing failed" in str(exc_info.value)
    
    @patch('ingest.parsing.partition_text')
    def test_parse_text_success(self, mock_partition_text):
        """Test successful text parsing."""
        mock_elements = [Mock()]
        mock_partition_text.return_value = mock_elements
        
        parser = DocumentParser()
        text = "Sample text content"
        result = parser.parse_text(text, "text/plain")
        
        mock_partition_text.assert_called_once_with(text=text)
        assert result == mock_elements
    
    @patch('ingest.parsing.partition_text')
    def test_parse_text_error(self, mock_partition_text):
        """Test text parsing with error."""
        mock_partition_text.side_effect = Exception("Text parsing failed")
        
        parser = DocumentParser()
        
        with pytest.raises(Exception) as exc_info:
            parser.parse_text("Sample text", "text/plain")
        
        assert "Text parsing failed" in str(exc_info.value)
    
    def test_get_file_type(self, temp_file):
        """Test file type detection."""
        parser = DocumentParser()
        file_type = parser.get_file_type(temp_file)
        
        assert file_type == "text/plain"
    
    def test_get_file_type_unknown(self):
        """Test file type detection for unknown file."""
        parser = DocumentParser()
        file_type = parser.get_file_type("unknown.xyz")
        
        # The system might detect .xyz as chemical/x-xyz, so we check for either
        assert file_type in ["application/octet-stream", "chemical/x-xyz"]


class TestParseFileFunction:
    """Test cases for the parse_file convenience function."""
    
    @patch('ingest.parsing.DocumentParser')
    def test_parse_file_function(self, mock_parser_class, temp_file):
        """Test the parse_file convenience function."""
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_elements = [Mock(), Mock()]
        mock_parser.parse_file.return_value = mock_elements
        
        result = parse_file(temp_file)
        
        # Verify DocumentParser was created with correct parameters
        mock_parser_class.assert_called_once_with(
            strategy="hi_res",
            skip_infer_table_types=None,
            max_partition=None
        )
        
        # Verify parse_file was called
        mock_parser.parse_file.assert_called_once_with(temp_file)
        
        assert result == mock_elements
    
    @patch('ingest.parsing.DocumentParser')
    def test_parse_file_function_with_params(self, mock_parser_class, temp_file):
        """Test the parse_file function with custom parameters."""
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_elements = [Mock()]
        mock_parser.parse_file.return_value = mock_elements
        
        result = parse_file(
            temp_file,
            strategy="fast",
            skip_infer_table_types=["pdf"],
            max_partition=100
        )
        
        # Verify DocumentParser was created with custom parameters
        mock_parser_class.assert_called_once_with(
            strategy="fast",
            skip_infer_table_types=["pdf"],
            max_partition=100
        )
        
        assert result == mock_elements
