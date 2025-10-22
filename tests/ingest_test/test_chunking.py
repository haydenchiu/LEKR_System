"""
Unit tests for the chunking module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from ingest.chunking import DocumentChunker, elements_to_chunks
from ingest.exceptions import ChunkingError


class TestDocumentChunker:
    """Test cases for DocumentChunker class."""
    
    def test_init_default(self):
        """Test DocumentChunker initialization with default parameters."""
        chunker = DocumentChunker()
        
        assert chunker.max_characters == 2048
        assert chunker.combine_text_under_n_chars == 256
        assert chunker.new_after_n_chars == 1800
    
    def test_init_custom(self):
        """Test DocumentChunker initialization with custom parameters."""
        chunker = DocumentChunker(
            max_characters=4096,
            combine_text_under_n_chars=512,
            new_after_n_chars=3600
        )
        
        assert chunker.max_characters == 4096
        assert chunker.combine_text_under_n_chars == 512
        assert chunker.new_after_n_chars == 3600
    
    @patch('ingest.chunking.chunk_by_title')
    def test_chunk_elements_success(self, mock_chunk_by_title, sample_elements):
        """Test successful element chunking."""
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_chunk_by_title.return_value = mock_chunks
        
        chunker = DocumentChunker()
        result = chunker.chunk_elements(sample_elements)
        
        # Verify chunk_by_title was called with correct parameters
        mock_chunk_by_title.assert_called_once_with(
            sample_elements,
            max_characters=2048,
            combine_text_under_n_chars=256,
            new_after_n_chars=1800
        )
        
        assert result == mock_chunks
    
    @patch('ingest.chunking.chunk_by_title')
    def test_chunk_elements_with_custom_params(self, mock_chunk_by_title, sample_elements):
        """Test element chunking with custom parameters."""
        mock_chunks = [Mock()]
        mock_chunk_by_title.return_value = mock_chunks
        
        chunker = DocumentChunker(
            max_characters=1024,
            combine_text_under_n_chars=128,
            new_after_n_chars=900
        )
        
        result = chunker.chunk_elements(sample_elements)
        
        mock_chunk_by_title.assert_called_once_with(
            sample_elements,
            max_characters=1024,
            combine_text_under_n_chars=128,
            new_after_n_chars=900
        )
        assert result == mock_chunks
    
    @patch('ingest.chunking.chunk_by_title')
    def test_chunk_elements_error(self, mock_chunk_by_title, sample_elements):
        """Test element chunking with error."""
        mock_chunk_by_title.side_effect = Exception("Chunking failed")
        
        chunker = DocumentChunker()
        
        with pytest.raises(Exception) as exc_info:
            chunker.chunk_elements(sample_elements)
        
        assert "Chunking failed" in str(exc_info.value)
    
    def test_get_chunk_statistics(self, sample_chunks):
        """Test chunk statistics calculation."""
        chunker = DocumentChunker()
        stats = chunker.get_chunk_statistics(sample_chunks)
        
        assert stats["total_chunks"] == 2
        assert stats["text_chunks"] == 1
        assert stats["table_chunks"] == 1
    
    def test_get_chunk_statistics_empty(self):
        """Test chunk statistics with empty chunks."""
        chunker = DocumentChunker()
        stats = chunker.get_chunk_statistics([])
        
        assert stats["total_chunks"] == 0
        assert stats["text_chunks"] == 0
        assert stats["table_chunks"] == 0
    
    def test_get_chunk_statistics_mixed(self):
        """Test chunk statistics with mixed chunk types."""
        # Create mock chunks with different metadata
        text_chunk = Mock()
        text_chunk.metadata.to_dict.return_value = {"filetype": "text/plain"}
        
        table_chunk = Mock()
        table_chunk.metadata.to_dict.return_value = {
            "filetype": "text/html",
            "text_as_html": "<table></table>"
        }
        
        chunks = [text_chunk, text_chunk, table_chunk]
        
        chunker = DocumentChunker()
        stats = chunker.get_chunk_statistics(chunks)
        
        assert stats["total_chunks"] == 3
        assert stats["text_chunks"] == 2
        assert stats["table_chunks"] == 1


class TestElementsToChunksFunction:
    """Test cases for the elements_to_chunks convenience function."""
    
    @patch('ingest.chunking.DocumentChunker')
    def test_elements_to_chunks_function(self, mock_chunker_class, sample_elements):
        """Test the elements_to_chunks convenience function."""
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunks = [Mock(), Mock()]
        mock_chunker.chunk_elements.return_value = mock_chunks
        
        result = elements_to_chunks(sample_elements)
        
        # Verify DocumentChunker was created with correct parameters
        mock_chunker_class.assert_called_once_with(
            max_characters=2048,
            combine_text_under_n_chars=256,
            new_after_n_chars=1800
        )
        
        # Verify chunk_elements was called
        mock_chunker.chunk_elements.assert_called_once_with(sample_elements)
        
        assert result == mock_chunks
    
    @patch('ingest.chunking.DocumentChunker')
    def test_elements_to_chunks_function_with_params(self, mock_chunker_class, sample_elements):
        """Test the elements_to_chunks function with custom parameters."""
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunks = [Mock()]
        mock_chunker.chunk_elements.return_value = mock_chunks
        
        result = elements_to_chunks(
            sample_elements,
            max_characters=1024,
            combine_text_under_n_chars=128,
            new_after_n_chars=900
        )
        
        # Verify DocumentChunker was created with custom parameters
        mock_chunker_class.assert_called_once_with(
            max_characters=1024,
            combine_text_under_n_chars=128,
            new_after_n_chars=900
        )
        
        assert result == mock_chunks


class TestChunkingIntegration:
    """Integration tests for chunking functionality."""
    
    def test_chunking_pipeline(self, sample_elements):
        """Test the complete chunking pipeline."""
        chunker = DocumentChunker(
            max_characters=100,
            combine_text_under_n_chars=50,
            new_after_n_chars=80
        )
        
        # This would normally call the real chunk_by_title function
        # In a real test, you might want to use actual elements
        with patch('ingest.chunking.chunk_by_title') as mock_chunk:
            mock_chunk.return_value = sample_elements
            chunks = chunker.chunk_elements(sample_elements)
            
            assert chunks == sample_elements
            mock_chunk.assert_called_once_with(
                sample_elements,
                max_characters=100,
                combine_text_under_n_chars=50,
                new_after_n_chars=80
            )
    
    def test_chunk_statistics_calculation(self):
        """Test chunk statistics calculation with various chunk types."""
        # Create mock chunks with different characteristics
        text_chunk1 = Mock()
        text_chunk1.metadata.to_dict.return_value = {"type": "text"}
        
        text_chunk2 = Mock()
        text_chunk2.metadata.to_dict.return_value = {"type": "text"}
        
        table_chunk = Mock()
        table_chunk.metadata.to_dict.return_value = {
            "type": "table",
            "text_as_html": "<table></table>"
        }
        
        chunks = [text_chunk1, text_chunk2, table_chunk]
        
        chunker = DocumentChunker()
        stats = chunker.get_chunk_statistics(chunks)
        
        assert stats["total_chunks"] == 3
        assert stats["text_chunks"] == 2
        assert stats["table_chunks"] == 1
