"""
Unit tests for enrichment utilities.

Tests the utility functions for chunk processing and batch operations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from enrichment.utils import (
    is_table_chunk,
    get_chunk_content,
    get_chunk_metadata,
    is_enriched_chunk,
    get_enrichment_data,
    process_chunks_concurrently,
    enrich_and_extract_logic,
    validate_chunk,
    filter_valid_chunks,
    get_chunk_statistics
)


class TestIsTableChunk:
    """Test cases for is_table_chunk function."""
    
    def test_table_chunk_with_html(self, sample_table_chunk):
        """Test table chunk detection with HTML content."""
        result = is_table_chunk(sample_table_chunk)
        assert result is True
    
    def test_text_chunk_not_table(self, sample_chunk):
        """Test text chunk is not detected as table."""
        result = is_table_chunk(sample_chunk)
        assert result is False
    
    def test_chunk_without_metadata(self):
        """Test chunk without metadata."""
        chunk = Mock()
        chunk.metadata = None
        
        result = is_table_chunk(chunk)
        assert result is False
    
    def test_chunk_without_to_dict(self):
        """Test chunk with metadata but no to_dict method."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict = None
        
        result = is_table_chunk(chunk)
        assert result is False
    
    def test_chunk_metadata_error(self):
        """Test chunk with metadata error."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.side_effect = Exception("Metadata error")
        
        result = is_table_chunk(chunk)
        assert result is False
    
    def test_chunk_with_empty_metadata(self):
        """Test chunk with empty metadata."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}
        
        result = is_table_chunk(chunk)
        assert result is False
    
    def test_chunk_with_text_as_html(self):
        """Test chunk with text_as_html in metadata."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {"text_as_html": "<table>data</table>"}
        
        result = is_table_chunk(chunk)
        assert result is True


class TestGetChunkContent:
    """Test cases for get_chunk_content function."""
    
    def test_text_chunk_content(self, sample_chunk):
        """Test getting content from text chunk."""
        content = get_chunk_content(sample_chunk)
        assert content == sample_chunk.text
    
    def test_table_chunk_content(self, sample_table_chunk):
        """Test getting content from table chunk."""
        content = get_chunk_content(sample_table_chunk, is_table=True)
        assert content == sample_table_chunk.metadata.text_as_html
    
    def test_chunk_content_auto_detect_table(self, sample_table_chunk):
        """Test getting content with auto table detection."""
        content = get_chunk_content(sample_table_chunk)
        assert content == sample_table_chunk.metadata.text_as_html
    
    def test_chunk_content_auto_detect_text(self, sample_chunk):
        """Test getting content with auto text detection."""
        content = get_chunk_content(sample_chunk)
        assert content == sample_chunk.text
    
    def test_chunk_without_text_attribute(self):
        """Test chunk without text attribute."""
        chunk = Mock()
        chunk.text = None
        chunk.content = "Content from content attribute"
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}
        
        content = get_chunk_content(chunk)
        # The function should return the content attribute
        assert content == "Content from content attribute"
    
    def test_chunk_without_content_attributes(self):
        """Test chunk without text or content attributes."""
        chunk = Mock()
        chunk.text = None
        chunk.content = None
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}
        
        content = get_chunk_content(chunk)
        # The function should return the string representation of the chunk
        assert content == str(chunk)
    
    def test_chunk_content_error(self):
        """Test chunk content extraction with error."""
        chunk = Mock()
        chunk.text = None
        chunk.content = None
        chunk.metadata = Mock()
        chunk.metadata.to_dict.side_effect = Exception("Metadata error")
        
        content = get_chunk_content(chunk)
        assert content == str(chunk)
    
    def test_chunk_content_fallback(self):
        """Test chunk content fallback to string representation."""
        chunk = Mock()
        chunk.text = None
        chunk.content = None
        chunk.metadata = None
        
        content = get_chunk_content(chunk)
        assert content == str(chunk)

    def test_chunk_content_table_metadata_text_as_html(self):
        """Test chunk content for table with metadata.text_as_html attribute."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.text_as_html = "<table>Table content</table>"
        
        content = get_chunk_content(chunk, is_table=True)
        assert content == "<table>Table content</table>"

    def test_chunk_content_table_metadata_dict(self):
        """Test chunk content for table with metadata.to_dict() method."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {"text_as_html": "<table>Table from dict</table>"}
        # Ensure the chunk doesn't have text_as_html attribute directly
        delattr(chunk.metadata, 'text_as_html')
        
        content = get_chunk_content(chunk, is_table=True)
        assert content == "<table>Table from dict</table>"

    def test_chunk_content_table_metadata_dict_missing_key(self):
        """Test chunk content for table with metadata.to_dict() but missing text_as_html key."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}
        # Ensure the chunk doesn't have text_as_html attribute directly
        delattr(chunk.metadata, 'text_as_html')
        
        content = get_chunk_content(chunk, is_table=True)
        # When text_as_html is missing from metadata dict, it returns empty string
        assert content == ""

    def test_chunk_content_error_handling(self):
        """Test chunk content error handling."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.side_effect = Exception("Metadata error")
        # Ensure the chunk doesn't have text or content attributes
        chunk.text = None
        chunk.content = None
        
        content = get_chunk_content(chunk)
        assert content == str(chunk)

    def test_chunk_content_fallback_case(self):
        """Test chunk content fallback case when all methods fail."""
        chunk = Mock()
        # Remove all attributes that could provide content
        delattr(chunk, 'text')
        delattr(chunk, 'content')
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}
        # Ensure the chunk doesn't have text_as_html attribute
        delattr(chunk.metadata, 'text_as_html')
        
        content = get_chunk_content(chunk)
        assert content == str(chunk)

    def test_chunk_content_exception_handling(self):
        """Test chunk content exception handling in the outer try-catch."""
        # Create a simple object that will cause exceptions when accessed
        class ProblematicChunk:
            def __init__(self):
                pass
            
            @property
            def text(self):
                raise Exception("Access error")
            
            @property
            def content(self):
                raise Exception("Access error")
            
            @property
            def metadata(self):
                raise Exception("Metadata error")
            
            def __str__(self):
                return "<ProblematicChunk>"
        
        chunk = ProblematicChunk()
        content = get_chunk_content(chunk)
        
        # The function should return the string representation of the chunk
        # when all access methods fail
        assert content == "<ProblematicChunk>"
        assert isinstance(content, str)


class TestGetChunkMetadata:
    """Test cases for get_chunk_metadata function."""
    
    def test_chunk_with_metadata_to_dict(self, sample_chunk):
        """Test getting metadata from chunk with to_dict method."""
        metadata = get_chunk_metadata(sample_chunk)
        assert isinstance(metadata, dict)
        assert "filetype" in metadata
        assert "page_number" in metadata
    
    def test_chunk_with_metadata_dict(self):
        """Test getting metadata from chunk with dict metadata."""
        chunk = Mock()
        chunk.metadata = {"filetype": "text/plain", "page_number": 1}
        
        metadata = get_chunk_metadata(chunk)
        assert metadata == {"filetype": "text/plain", "page_number": 1}
    
    def test_chunk_without_metadata(self):
        """Test getting metadata from chunk without metadata."""
        chunk = Mock()
        chunk.metadata = None
        
        metadata = get_chunk_metadata(chunk)
        assert metadata == {}
    
    def test_chunk_metadata_error(self):
        """Test getting metadata with error."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.side_effect = Exception("Metadata error")
        
        metadata = get_chunk_metadata(chunk)
        assert metadata == {}


class TestIsEnrichedChunk:
    """Test cases for is_enriched_chunk function."""
    
    def test_enriched_chunk(self, sample_chunk):
        """Test enriched chunk detection."""
        sample_chunk.enrichment = Mock()
        
        result = is_enriched_chunk(sample_chunk)
        assert result is True
    
    def test_non_enriched_chunk(self, sample_chunk):
        """Test non-enriched chunk detection."""
        # Ensure the chunk doesn't have enrichment attribute
        if hasattr(sample_chunk, 'enrichment'):
            delattr(sample_chunk, 'enrichment')
        
        result = is_enriched_chunk(sample_chunk)
        assert result is False
    
    def test_chunk_with_none_enrichment(self, sample_chunk):
        """Test chunk with None enrichment."""
        sample_chunk.enrichment = None
        
        result = is_enriched_chunk(sample_chunk)
        assert result is False
    
    def test_chunk_without_enrichment_attribute(self, sample_chunk):
        """Test chunk without enrichment attribute."""
        delattr(sample_chunk, 'enrichment')
        
        result = is_enriched_chunk(sample_chunk)
        assert result is False


class TestGetEnrichmentData:
    """Test cases for get_enrichment_data function."""
    
    def test_enriched_chunk_with_model_dump(self, sample_chunk):
        """Test getting enrichment data with model_dump method."""
        enrichment = Mock()
        enrichment.model_dump.return_value = {"summary": "Test summary"}
        sample_chunk.enrichment = enrichment
        
        data = get_enrichment_data(sample_chunk)
        assert data == {"summary": "Test summary"}
    
    def test_enriched_chunk_with_dict_method(self, sample_chunk):
        """Test getting enrichment data with dict method."""
        enrichment = Mock()
        enrichment.dict.return_value = {"summary": "Test summary"}
        # Remove model_dump method to test dict method
        if hasattr(enrichment, 'model_dump'):
            delattr(enrichment, 'model_dump')
        sample_chunk.enrichment = enrichment
        
        data = get_enrichment_data(sample_chunk)
        assert data == {"summary": "Test summary"}
    
    def test_enriched_chunk_with_dict_conversion(self, sample_chunk):
        """Test getting enrichment data with dict conversion."""
        enrichment = {"summary": "Test summary"}
        sample_chunk.enrichment = enrichment
        
        data = get_enrichment_data(sample_chunk)
        assert data == {"summary": "Test summary"}
    
    def test_non_enriched_chunk(self):
        """Test getting enrichment data from non-enriched chunk."""
        chunk = Mock()
        chunk.enrichment = None
        data = get_enrichment_data(chunk)
        assert data == {}


class TestProcessChunksConcurrently:
    """Test cases for process_chunks_concurrently function."""
    
    @pytest.mark.asyncio
    async def test_process_chunks_success(self, sample_batch_data):
        """Test successful concurrent chunk processing."""
        chunks = sample_batch_data["chunks"]
        batch_size = sample_batch_data["batch_size"]
        
        async def mock_func(chunk):
            return chunk
        
        results = await process_chunks_concurrently(chunks, mock_func, batch_size)
        
        assert len(results) == len(chunks)
        assert results == chunks
    
    @pytest.mark.asyncio
    async def test_process_chunks_with_errors(self, sample_batch_data):
        """Test concurrent chunk processing with errors."""
        chunks = sample_batch_data["chunks"]
        batch_size = sample_batch_data["batch_size"]
        
        async def mock_func(chunk):
            if chunk == chunks[0]:  # First chunk fails
                raise Exception("Processing error")
            return chunk
        
        results = await process_chunks_concurrently(chunks, mock_func, batch_size)
        
        # Should return original chunks when processing fails
        assert len(results) == len(chunks)
        assert results == chunks
    
    @pytest.mark.asyncio
    async def test_process_chunks_single_batch(self):
        """Test processing chunks in a single batch."""
        chunks = [Mock() for _ in range(3)]
        batch_size = 10  # Larger than chunk count
        
        async def mock_func(chunk):
            return chunk
        
        results = await process_chunks_concurrently(chunks, mock_func, batch_size)
        
        assert len(results) == len(chunks)
        assert results == chunks
    
    @pytest.mark.asyncio
    async def test_process_chunks_empty_list(self):
        """Test processing empty chunk list."""
        chunks = []
        batch_size = 5
        
        async def mock_func(chunk):
            return chunk
        
        results = await process_chunks_concurrently(chunks, mock_func, batch_size)
        
        assert len(results) == 0
        assert results == []
    
    @pytest.mark.asyncio
    async def test_process_chunks_single_chunk(self):
        """Test processing single chunk."""
        chunks = [Mock()]
        batch_size = 5
        
        async def mock_func(chunk):
            return chunk
        
        results = await process_chunks_concurrently(chunks, mock_func, batch_size)
        
        assert len(results) == 1
        assert results == chunks


class TestEnrichAndExtractLogic:
    """Test cases for enrich_and_extract_logic function."""
    
    @pytest.mark.asyncio
    async def test_enrich_and_extract_logic_success(self, sample_chunk):
        """Test successful enrichment and logic extraction."""
        mock_enricher = Mock()
        mock_enricher.enrich_chunk_async = AsyncMock(return_value=sample_chunk)
        
        mock_logic_extractor = Mock()
        mock_logic_extractor.extract_logic_async = AsyncMock(return_value=sample_chunk)
        
        result = await enrich_and_extract_logic(
            sample_chunk, 
            enricher=mock_enricher, 
            logic_extractor=mock_logic_extractor
        )
        
        assert result == sample_chunk
        mock_enricher.enrich_chunk_async.assert_called_once_with(sample_chunk)
        mock_logic_extractor.extract_logic_async.assert_called_once_with(sample_chunk)
    
    @pytest.mark.asyncio
    async def test_enrich_and_extract_logic_without_enricher(self, sample_chunk):
        """Test enrichment and logic extraction without enricher."""
        with patch('enrichment.enricher.DocumentEnricher') as mock_enricher_class:
            mock_enricher = Mock()
            mock_enricher.enrich_chunk_async = AsyncMock(return_value=sample_chunk)
            mock_enricher_class.return_value = mock_enricher
            
            result = await enrich_and_extract_logic(sample_chunk)
            
            assert result == sample_chunk
            mock_enricher_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enrich_and_extract_logic_without_logic_extractor(self, sample_chunk):
        """Test enrichment and logic extraction without logic extractor."""
        mock_enricher = Mock()
        mock_enricher.enrich_chunk_async = AsyncMock(return_value=sample_chunk)
        
        result = await enrich_and_extract_logic(
            sample_chunk, 
            enricher=mock_enricher, 
            logic_extractor=None
        )
        
        assert result == sample_chunk
        mock_enricher.enrich_chunk_async.assert_called_once_with(sample_chunk)
    
    @pytest.mark.asyncio
    async def test_enrich_and_extract_logic_error(self, sample_chunk):
        """Test enrichment and logic extraction with error."""
        mock_enricher = Mock()
        mock_enricher.enrich_chunk_async = AsyncMock(side_effect=Exception("Enrichment error"))
        
        result = await enrich_and_extract_logic(
            sample_chunk, 
            enricher=mock_enricher, 
            logic_extractor=None
        )
        
        # Should return original chunk on error
        assert result == sample_chunk


class TestValidateChunk:
    """Test cases for validate_chunk function."""
    
    def test_valid_chunk(self, sample_chunk):
        """Test validation of valid chunk."""
        result = validate_chunk(sample_chunk)
        assert result is True
    
    def test_invalid_chunk_empty_content(self):
        """Test validation of chunk with empty content."""
        chunk = Mock()
        chunk.text = ""
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {"filetype": "text/plain"}
        
        result = validate_chunk(chunk)
        assert result is False
    
    def test_invalid_chunk_none_content(self):
        """Test validation of chunk with None content."""
        chunk = Mock()
        chunk.text = None
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {"filetype": "text/plain"}
        
        result = validate_chunk(chunk)
        assert result is False
    
    def test_invalid_chunk_no_metadata(self):
        """Test validation of chunk without metadata."""
        chunk = Mock()
        chunk.text = "Valid text"
        chunk.metadata = None
        
        result = validate_chunk(chunk)
        assert result is False
    
    def test_invalid_chunk_metadata_error(self):
        """Test validation of chunk with metadata error."""
        chunk = Mock()
        chunk.text = "Valid text"
        chunk.metadata = Mock()
        chunk.metadata.to_dict.side_effect = Exception("Metadata error")
        
        result = validate_chunk(chunk)
        assert result is False
    
    def test_invalid_chunk_validation_error(self):
        """Test validation of chunk with validation error."""
        chunk = Mock()
        chunk.text = "Valid text"
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {"filetype": "text/plain"}
        
        # Mock get_chunk_content to raise exception
        with patch('enrichment.utils.get_chunk_content') as mock_get_content:
            mock_get_content.side_effect = Exception("Content error")
            
            result = validate_chunk(chunk)
            assert result is False


class TestFilterValidChunks:
    """Test cases for filter_valid_chunks function."""
    
    def test_filter_valid_chunks_success(self, sample_chunks):
        """Test filtering valid chunks."""
        # Create proper mock chunks with required attributes
        mock_chunks = []
        for i in range(3):
            chunk = Mock()
            chunk.text = f"Sample text {i}"
            chunk.metadata = Mock()
            chunk.metadata.to_dict.return_value = {}
            mock_chunks.append(chunk)
        
        with patch('enrichment.utils.validate_chunk') as mock_validate:
            mock_validate.side_effect = [True, False, True]  # First and third are valid
            
            valid_chunks = filter_valid_chunks(mock_chunks)
            
            assert len(valid_chunks) == 2
            assert valid_chunks[0] == mock_chunks[0]
            assert valid_chunks[1] == mock_chunks[2]
    
    def test_filter_valid_chunks_all_valid(self, sample_chunks):
        """Test filtering when all chunks are valid."""
        with patch('enrichment.utils.validate_chunk') as mock_validate:
            mock_validate.return_value = True
            
            valid_chunks = filter_valid_chunks(sample_chunks)
            
            assert len(valid_chunks) == len(sample_chunks)
            assert valid_chunks == sample_chunks
    
    def test_filter_valid_chunks_none_valid(self, sample_chunks):
        """Test filtering when no chunks are valid."""
        with patch('enrichment.utils.validate_chunk') as mock_validate:
            mock_validate.return_value = False
            
            valid_chunks = filter_valid_chunks(sample_chunks)
            
            assert len(valid_chunks) == 0
            assert valid_chunks == []
    
    def test_filter_valid_chunks_empty_list(self):
        """Test filtering empty chunk list."""
        valid_chunks = filter_valid_chunks([])
        
        assert len(valid_chunks) == 0
        assert valid_chunks == []


class TestGetChunkStatistics:
    """Test cases for get_chunk_statistics function."""
    
    def test_chunk_statistics_basic(self, sample_chunks):
        """Test basic chunk statistics."""
        with patch('enrichment.utils.is_table_chunk') as mock_is_table:
            mock_is_table.side_effect = [False, True]  # First is text, second is table
            
            with patch('enrichment.utils.is_enriched_chunk') as mock_is_enriched:
                mock_is_enriched.return_value = True
                
                with patch('enrichment.utils.get_chunk_content') as mock_get_content:
                    mock_get_content.return_value = "Sample content"
                    
                    stats = get_chunk_statistics(sample_chunks)
                    
                    assert stats["total_chunks"] == 2
                    assert stats["table_chunks"] == 1
                    assert stats["text_chunks"] == 1
                    assert stats["enriched_chunks"] == 2
                    assert stats["enrichment_rate"] == 1.0
    
    def test_chunk_statistics_empty_list(self):
        """Test chunk statistics with empty list."""
        stats = get_chunk_statistics([])
        
        assert stats["total_chunks"] == 0
        assert stats["table_chunks"] == 0
        assert stats["text_chunks"] == 0
        assert stats["enriched_chunks"] == 0
        assert stats["enrichment_rate"] == 0
    
    def test_chunk_statistics_content_lengths(self, sample_chunks):
        """Test chunk statistics with content lengths."""
        with patch('enrichment.utils.is_table_chunk') as mock_is_table:
            mock_is_table.return_value = False
            
            with patch('enrichment.utils.is_enriched_chunk') as mock_is_enriched:
                mock_is_enriched.return_value = True
                
                with patch('enrichment.utils.get_chunk_content') as mock_get_content:
                    mock_get_content.side_effect = ["Short", "Much longer content here"]
                    
                    stats = get_chunk_statistics(sample_chunks)
                    
                    assert stats["total_chunks"] == 2
                    # Calculate expected average: (5 + 24) / 2 = 14.5
                    expected_avg = (5 + 24) / 2
                    assert stats["avg_content_length"] == expected_avg
                    assert stats["max_content_length"] == 24
                    assert stats["min_content_length"] == 5
    
    def test_chunk_statistics_content_error(self, sample_chunks):
        """Test chunk statistics with content extraction error."""
        with patch('enrichment.utils.is_table_chunk') as mock_is_table:
            mock_is_table.return_value = False
            
            with patch('enrichment.utils.is_enriched_chunk') as mock_is_enriched:
                mock_is_enriched.return_value = True
                
                with patch('enrichment.utils.get_chunk_content') as mock_get_content:
                    mock_get_content.side_effect = [Exception("Content error"), "Valid content"]
                    
                    stats = get_chunk_statistics(sample_chunks)
                    
                    assert stats["total_chunks"] == 2
                    # Calculate expected average: (0 + 13) / 2 = 6.5
                    expected_avg = (0 + 13) / 2
                    assert stats["avg_content_length"] == expected_avg
                    assert stats["max_content_length"] == 13
                    assert stats["min_content_length"] == 0
    
    def test_chunk_statistics_mixed_enrichment(self, sample_chunks):
        """Test chunk statistics with mixed enrichment status."""
        with patch('enrichment.utils.is_table_chunk') as mock_is_table:
            mock_is_table.return_value = False
            
            with patch('enrichment.utils.is_enriched_chunk') as mock_is_enriched:
                mock_is_enriched.side_effect = [True, False]  # First enriched, second not
                
                with patch('enrichment.utils.get_chunk_content') as mock_get_content:
                    mock_get_content.return_value = "Sample content"
                    
                    stats = get_chunk_statistics(sample_chunks)
                    
                    assert stats["total_chunks"] == 2
                    assert stats["enriched_chunks"] == 1
                    assert stats["enrichment_rate"] == 0.5
