"""
Tests for the logic_extractor example module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from logic_extractor.example import (
    create_sample_chunks,
    demonstrate_sync_logic_extraction,
    demonstrate_async_logic_extraction
)


class TestExampleFunctions:
    """Test cases for example functions."""

    @patch('logic_extractor.example.partition')
    @patch('logic_extractor.example.chunk_by_title')
    @patch('logic_extractor.example.Table')
    @patch('logic_extractor.example.Text')
    def test_create_sample_chunks(self, mock_text, mock_table, mock_chunk_by_title, mock_partition):
        """Test create_sample_chunks function."""
        # Mock the elements
        mock_text_element = Mock()
        mock_text_element.text = "This is a sample text chunk about AI. It discusses machine learning and neural networks."
        mock_text.return_value = mock_text_element
        
        mock_table_element = Mock()
        mock_table_element.text = "Sample table summary."
        mock_table.return_value = mock_table_element
        
        mock_partition.return_value = [Mock()]
        mock_chunk_by_title.return_value = [Mock()]

        chunks = create_sample_chunks()

        assert len(chunks) == 3
        mock_text.assert_called_once()
        mock_table.assert_called_once()
        mock_partition.assert_called_once()
        mock_chunk_by_title.assert_called_once()

    @patch('logic_extractor.example.LogicExtractor')
    @patch('logic_extractor.example.create_sample_chunks')
    def test_demonstrate_sync_logic_extraction(self, mock_create_chunks, mock_extractor_class):
        """Test demonstrate_sync_logic_extraction function."""
        # Mock the chunks
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_create_chunks.return_value = mock_chunks
        
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_logic_from_chunks.return_value = [
            Mock(logic=Mock(claims=[Mock(statement="Test claim")], logical_relations=[]))
        ]

        # Should not raise any exceptions
        demonstrate_sync_logic_extraction()

        mock_create_chunks.assert_called_once()
        mock_extractor_class.assert_called_once()
        mock_extractor.extract_logic_from_chunks.assert_called_once()

    @pytest.mark.asyncio
    @patch('logic_extractor.example.LogicExtractor')
    @patch('logic_extractor.example.create_sample_chunks')
    async def test_demonstrate_async_logic_extraction(self, mock_create_chunks, mock_extractor_class):
        """Test demonstrate_async_logic_extraction function."""
        # Mock the chunks
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_create_chunks.return_value = mock_chunks
        
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_logic_from_chunks_async = AsyncMock(return_value=[
            Mock(logic=Mock(claims=[Mock(statement="Test claim")], logical_relations=[]))
        ])

        # Should not raise any exceptions
        await demonstrate_async_logic_extraction()

        mock_create_chunks.assert_called_once()
        mock_extractor_class.assert_called_once()
        mock_extractor.extract_logic_from_chunks_async.assert_called_once()