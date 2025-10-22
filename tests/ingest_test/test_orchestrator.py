"""
Unit tests for the orchestrator module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from ingest.orchestrator import DocumentIngestionOrchestrator
from ingest.exceptions import IngestionError, ParsingError, ChunkingError


class TestDocumentIngestionOrchestrator:
    """Test cases for DocumentIngestionOrchestrator class."""
    
    def test_init_default(self):
        """Test orchestrator initialization with default parameters."""
        orchestrator = DocumentIngestionOrchestrator()
        
        assert orchestrator.parser is not None
        assert orchestrator.chunker is not None
        assert orchestrator.parser.strategy == "hi_res"
        assert orchestrator.chunker.max_characters == 2048
    
    def test_init_custom(self):
        """Test orchestrator initialization with custom parameters."""
        orchestrator = DocumentIngestionOrchestrator(
            parsing_strategy="fast",
            skip_infer_table_types=["pdf"],
            max_partition=100,
            max_characters=1024,
            combine_text_under_n_chars=128,
            new_after_n_chars=900
        )
        
        assert orchestrator.parser.strategy == "fast"
        assert orchestrator.parser.skip_infer_table_types == ["pdf"]
        assert orchestrator.parser.max_partition == 100
        assert orchestrator.chunker.max_characters == 1024
        assert orchestrator.chunker.combine_text_under_n_chars == 128
        assert orchestrator.chunker.new_after_n_chars == 900
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_ingest_file_success(self, mock_chunker_class, mock_parser_class, temp_file, sample_elements, sample_chunks):
        """Test successful file ingestion."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_file.return_value = sample_elements
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk_elements.return_value = sample_chunks
        mock_chunker.get_chunk_statistics.return_value = {
            "total_chunks": 2,
            "text_chunks": 1,
            "table_chunks": 1
        }
        
        orchestrator = DocumentIngestionOrchestrator()
        result = orchestrator.ingest_file(temp_file)
        
        # Verify calls
        mock_parser.parse_file.assert_called_once_with(temp_file)
        mock_chunker.chunk_elements.assert_called_once_with(sample_elements)
        mock_chunker.get_chunk_statistics.assert_called_once_with(sample_chunks)
        
        # Verify result
        assert result["success"] is True
        assert result["file_path"] == temp_file
        assert result["elements"] == sample_elements
        assert result["chunks"] == sample_chunks
        assert result["statistics"]["total_chunks"] == 2
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_ingest_file_parsing_error(self, mock_chunker_class, mock_parser_class, temp_file):
        """Test file ingestion with parsing error."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_file.side_effect = Exception("Parsing failed")
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        
        orchestrator = DocumentIngestionOrchestrator()
        result = orchestrator.ingest_file(temp_file)
        
        # Verify result
        assert result["success"] is False
        assert result["file_path"] == temp_file
        assert result["elements"] == []
        assert result["chunks"] == []
        assert result["statistics"]["total_chunks"] == 0
        assert "Parsing failed" in result["error"]
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_ingest_file_chunking_error(self, mock_chunker_class, mock_parser_class, temp_file, sample_elements):
        """Test file ingestion with chunking error."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_file.return_value = sample_elements
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk_elements.side_effect = Exception("Chunking failed")
        
        orchestrator = DocumentIngestionOrchestrator()
        result = orchestrator.ingest_file(temp_file)
        
        # Verify result
        assert result["success"] is False
        assert "Chunking failed" in result["error"]
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_ingest_url_success(self, mock_chunker_class, mock_parser_class, sample_elements, sample_chunks):
        """Test successful URL ingestion."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_url.return_value = sample_elements
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk_elements.return_value = sample_chunks
        mock_chunker.get_chunk_statistics.return_value = {
            "total_chunks": 2,
            "text_chunks": 1,
            "table_chunks": 1
        }
        
        orchestrator = DocumentIngestionOrchestrator()
        url = "https://example.com/article"
        result = orchestrator.ingest_url(url)
        
        # Verify calls
        mock_parser.parse_url.assert_called_once_with(url)
        mock_chunker.chunk_elements.assert_called_once_with(sample_elements)
        
        # Verify result
        assert result["success"] is True
        assert result["url"] == url
        assert result["elements"] == sample_elements
        assert result["chunks"] == sample_chunks
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_ingest_url_error(self, mock_chunker_class, mock_parser_class):
        """Test URL ingestion with error."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_url.side_effect = Exception("URL parsing failed")
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        
        orchestrator = DocumentIngestionOrchestrator()
        result = orchestrator.ingest_url("https://example.com")
        
        # Verify result
        assert result["success"] is False
        assert "URL parsing failed" in result["error"]
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_ingest_text_success(self, mock_chunker_class, mock_parser_class, sample_elements, sample_chunks):
        """Test successful text ingestion."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_text.return_value = sample_elements
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk_elements.return_value = sample_chunks
        mock_chunker.get_chunk_statistics.return_value = {
            "total_chunks": 2,
            "text_chunks": 2,
            "table_chunks": 0
        }
        
        orchestrator = DocumentIngestionOrchestrator()
        text = "Sample text content"
        result = orchestrator.ingest_text(text, "text/plain")
        
        # Verify calls
        mock_parser.parse_text.assert_called_once_with(text, "text/plain")
        mock_chunker.chunk_elements.assert_called_once_with(sample_elements)
        
        # Verify result
        assert result["success"] is True
        assert result["content_type"] == "text"
        assert result["filetype"] == "text/plain"
        assert result["elements"] == sample_elements
        assert result["chunks"] == sample_chunks
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_ingest_text_error(self, mock_chunker_class, mock_parser_class):
        """Test text ingestion with error."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_text.side_effect = Exception("Text parsing failed")
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        
        orchestrator = DocumentIngestionOrchestrator()
        result = orchestrator.ingest_text("Sample text", "text/plain")
        
        # Verify result
        assert result["success"] is False
        assert "Text parsing failed" in result["error"]
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_ingest_multiple_files(self, mock_chunker_class, mock_parser_class, temp_file):
        """Test multiple file ingestion."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        # Make parse_file raise an exception for non-existent files
        def mock_parse_file(file_path):
            if file_path == "nonexistent.txt":
                raise FileNotFoundError(f"File not found: {file_path}")
            return []
        
        mock_parser.parse_file.side_effect = mock_parse_file
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk_elements.return_value = []
        mock_chunker.get_chunk_statistics.return_value = {
            "total_chunks": 0,
            "text_chunks": 0,
            "table_chunks": 0
        }
        
        orchestrator = DocumentIngestionOrchestrator()
        file_paths = [temp_file, "nonexistent.txt"]
        results = orchestrator.ingest_multiple_files(file_paths)
        
        # Verify results
        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
    
    def test_get_supported_file_types(self):
        """Test getting supported file types."""
        orchestrator = DocumentIngestionOrchestrator()
        file_types = orchestrator.get_supported_file_types()
        
        assert isinstance(file_types, list)
        assert "application/pdf" in file_types
        assert "text/html" in file_types
        assert "text/plain" in file_types
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in file_types


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator."""
    
    @patch('ingest.orchestrator.DocumentParser')
    @patch('ingest.orchestrator.DocumentChunker')
    def test_complete_ingestion_pipeline(self, mock_chunker_class, mock_parser_class, temp_file):
        """Test the complete ingestion pipeline."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_file.return_value = [Mock(), Mock()]
        
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk_elements.return_value = [Mock(), Mock(), Mock()]
        mock_chunker.get_chunk_statistics.return_value = {
            "total_chunks": 3,
            "text_chunks": 2,
            "table_chunks": 1
        }
        
        orchestrator = DocumentIngestionOrchestrator()
        result = orchestrator.ingest_file(temp_file)
        
        # Verify the complete pipeline
        mock_parser.parse_file.assert_called_once_with(temp_file)
        mock_chunker.chunk_elements.assert_called_once()
        mock_chunker.get_chunk_statistics.assert_called_once()
        
        assert result["success"] is True
        assert result["statistics"]["total_chunks"] == 3
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across all methods."""
        orchestrator = DocumentIngestionOrchestrator()
        
        # Test with non-existent file
        result = orchestrator.ingest_file("nonexistent.txt")
        assert result["success"] is False
        assert "error" in result
        
        # Test with invalid URL
        result = orchestrator.ingest_url("invalid-url")
        assert result["success"] is False
        assert "error" in result
        
        # Test with empty text
        result = orchestrator.ingest_text("", "text/plain")
        # This might succeed or fail depending on implementation
        assert "success" in result
