"""
Tests for the logic_extractor utils module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from unstructured.documents.elements import Text, Table
from logic_extractor.utils import (
    is_table_chunk, get_chunk_content, get_chunk_metadata,
    is_logic_extracted_chunk, get_logic_data, process_chunks_concurrently,
    extract_logic_from_chunk
)
from logic_extractor.models import LogicExtractionSchemaLiteChunk, Claim
from logic_extractor.config import DEFAULT_LOGIC_EXTRACTION_CONFIG


class TestIsTableChunk:
    """Test cases for is_table_chunk function."""

    @pytest.fixture
    def table_chunk_with_html(self):
        """Fixture for a mock table chunk with text_as_html."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {"text_as_html": "<table>...</table>"}
        return chunk

    @pytest.fixture
    def text_chunk(self):
        """Fixture for a mock text chunk."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}
        return chunk

    def test_table_chunk_with_html(self, table_chunk_with_html):
        """Test table chunk detection with text_as_html in metadata."""
        assert is_table_chunk(table_chunk_with_html) is True

    def test_text_chunk_not_table(self, text_chunk):
        """Test text chunk is not detected as table."""
        assert is_table_chunk(text_chunk) is False

    def test_chunk_without_metadata(self):
        """Test chunk without metadata attribute."""
        chunk = Mock()
        delattr(chunk, 'metadata')
        assert is_table_chunk(chunk) is False

    def test_chunk_without_to_dict(self):
        """Test chunk with metadata but without to_dict method."""
        chunk = Mock()
        chunk.metadata = Mock()
        delattr(chunk.metadata, 'to_dict')
        assert is_table_chunk(chunk) is False

    def test_chunk_metadata_error(self):
        """Test chunk metadata access error handling."""
        chunk = Mock()
        chunk.metadata = Mock()
        chunk.metadata.to_dict.side_effect = Exception("Metadata error")
        assert is_table_chunk(chunk) is False


class TestGetChunkContent:
    """Test cases for get_chunk_content function."""

    @pytest.fixture
    def sample_chunk(self):
        """Fixture for a mock chunk."""
        chunk = Mock()
        chunk.text = "This is a sample text."
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}
        return chunk

    @pytest.fixture
    def sample_table_chunk(self):
        """Fixture for a mock table chunk."""
        chunk = Mock()
        chunk.text = "Table text summary."
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {"text_as_html": "<table>Table HTML content</table>"}
        chunk.metadata.text_as_html = "<table>Table HTML content</table>"
        return chunk

    def test_text_chunk_content(self, sample_chunk):
        """Test content extraction for a text chunk."""
        content = get_chunk_content(sample_chunk, is_table=False)
        assert content == "This is a sample text."

    def test_table_chunk_content(self, sample_table_chunk):
        """Test content extraction for a table chunk."""
        content = get_chunk_content(sample_table_chunk, is_table=True)
        assert content == "<table>Table HTML content</table>"

    def test_chunk_without_text_attribute(self):
        """Test chunk without 'text' attribute but with 'content'."""
        chunk = Mock()
        delattr(chunk, 'text')
        chunk.content = "Content from content attribute"
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}

        content = get_chunk_content(chunk)
        assert content == "Content from content attribute"

    def test_chunk_without_content_attributes(self):
        """Test chunk without 'text' or 'content' attributes."""
        chunk = Mock()
        delattr(chunk, 'text')
        delattr(chunk, 'content')
        chunk.metadata = Mock()
        chunk.metadata.to_dict.return_value = {}

        content = get_chunk_content(chunk)
        assert content == str(chunk)


class TestGetChunkMetadata:
    """Test cases for get_chunk_metadata function."""

    @pytest.fixture
    def chunk_with_to_dict(self):
        """Fixture for a mock chunk with metadata.to_dict()."""
        chunk = Mock()
        chunk.metadata.to_dict.return_value = {"source": "file.pdf", "page_number": 1}
        return chunk

    @pytest.fixture
    def chunk_with_dict_metadata(self):
        """Fixture for a mock chunk with dict metadata."""
        chunk = Mock()
        chunk.metadata = {"source": "web.html", "url": "http://example.com"}
        return chunk

    def test_chunk_with_metadata_to_dict(self, chunk_with_to_dict):
        """Test metadata extraction when to_dict method is present."""
        metadata = get_chunk_metadata(chunk_with_to_dict)
        assert metadata == {"source": "file.pdf", "page_number": 1}

    def test_chunk_with_metadata_dict(self, chunk_with_dict_metadata):
        """Test metadata extraction when metadata is a dictionary."""
        metadata = get_chunk_metadata(chunk_with_dict_metadata)
        assert metadata == {"source": "web.html", "url": "http://example.com"}

    def test_chunk_without_metadata(self):
        """Test metadata extraction when metadata attribute is missing."""
        chunk = Mock()
        delattr(chunk, 'metadata')
        metadata = get_chunk_metadata(chunk)
        assert metadata == {}


class TestIsLogicExtractedChunk:
    """Test cases for is_logic_extracted_chunk function."""

    @pytest.fixture
    def sample_chunk(self):
        """Fixture for a mock chunk."""
        chunk = Mock()
        return chunk

    def test_logic_extracted_chunk(self, sample_chunk):
        """Test logic extracted chunk detection."""
        sample_chunk.logic = Mock()
        assert is_logic_extracted_chunk(sample_chunk) is True

    def test_non_logic_extracted_chunk(self, sample_chunk):
        """Test non-logic extracted chunk detection."""
        # Ensure the chunk doesn't have logic attribute
        if hasattr(sample_chunk, 'logic'):
            delattr(sample_chunk, 'logic')

        result = is_logic_extracted_chunk(sample_chunk)
        assert result is False

    def test_chunk_with_none_logic(self, sample_chunk):
        """Test chunk with None logic."""
        sample_chunk.logic = None

        result = is_logic_extracted_chunk(sample_chunk)
        assert result is False


class TestGetLogicData:
    """Test cases for get_logic_data function."""

    @pytest.fixture
    def sample_chunk(self):
        """Fixture for a mock chunk."""
        chunk = Mock()
        return chunk

    def test_logic_extracted_chunk_with_model_dump(self, sample_chunk):
        """Test getting logic data when model_dump is available."""
        logic = Mock()
        logic.model_dump.return_value = {"claims": [{"id": "c1", "statement": "Test claim"}]}
        sample_chunk.logic = logic

        data = get_logic_data(sample_chunk)
        assert data == {"claims": [{"id": "c1", "statement": "Test claim"}]}

    def test_logic_extracted_chunk_with_dict_method(self, sample_chunk):
        """Test getting logic data when dict method is available."""
        logic = Mock()
        delattr(logic, 'model_dump')  # Ensure model_dump is not called
        logic.dict.return_value = {"claims": [{"id": "c1", "statement": "Test claim from dict"}]}
        sample_chunk.logic = logic

        data = get_logic_data(sample_chunk)
        assert data == {"claims": [{"id": "c1", "statement": "Test claim from dict"}]}

    def test_chunk_without_logic(self, sample_chunk):
        """Test getting logic data from chunk without logic."""
        delattr(sample_chunk, 'logic')

        data = get_logic_data(sample_chunk)
        assert data == {}


class TestProcessChunksConcurrently:
    """Test cases for process_chunks_concurrently function."""

    @pytest.mark.asyncio
    async def test_process_chunks_concurrently_success(self):
        """Test successful concurrent processing of chunks."""
        chunks = [Mock(), Mock(), Mock()]
        async_func = AsyncMock(side_effect=["result1", "result2", "result3"])

        results = await process_chunks_concurrently(chunks, async_func, batch_size=2)

        assert len(results) == 3
        assert results == ["result1", "result2", "result3"]
        assert async_func.call_count == 3

    @pytest.mark.asyncio
    async def test_process_chunks_concurrently_with_errors(self):
        """Test concurrent processing with some errors."""
        chunks = [Mock(), Mock(), Mock()]
        async_func = AsyncMock(side_effect=["result1", Exception("Error"), "result3"])

        results = await process_chunks_concurrently(chunks, async_func, batch_size=2)

        assert len(results) == 3
        assert results[0] == "result1"
        assert results[1] == chunks[1]  # Original chunk returned on error
        assert results[2] == "result3"


class TestExtractLogicFromChunk:
    """Test cases for extract_logic_from_chunk function."""

    @pytest.mark.asyncio
    @patch('logic_extractor.utils.LogicExtractor')
    async def test_extract_logic_from_chunk(self, mock_extractor_class):
        """Test extract_logic_from_chunk function."""
        mock_extractor = AsyncMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_logic_from_chunk_async.return_value = Mock(logic="extracted_logic")

        chunk = Mock()
        result = await extract_logic_from_chunk(chunk)

        mock_extractor_class.assert_called_once_with(config=DEFAULT_LOGIC_EXTRACTION_CONFIG)
        mock_extractor.extract_logic_from_chunk_async.assert_called_once_with(chunk)
        assert result.logic == "extracted_logic"