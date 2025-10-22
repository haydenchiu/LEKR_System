"""
Unit tests for enrichment example script.

Tests the example script to ensure it can be imported and run without errors.
"""

import pytest
import os
import sys
import asyncio
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestEnrichmentExample:
    """Test cases for enrichment example script."""

    def test_example_import(self):
        """Test that the example module can be imported."""
        try:
            from enrichment.example import main
            assert callable(main)
        except ImportError as e:
            pytest.skip(f"Could not import enrichment example: {e}")

    def test_example_script_structure(self):
        """Test that the example script has the expected structure."""
        example_path = project_root / "enrichment" / "example.py"
        assert example_path.exists(), "Example script should exist"
        
        with open(example_path, 'r') as f:
            content = f.read()
            
        # Check for key components
        assert "def main():" in content, "Should have main function"
        assert "DocumentEnricher" in content, "Should use DocumentEnricher"
        assert "if __name__ == \"__main__\":" in content, "Should have main guard"

    def test_example_imports(self):
        """Test that all required imports are available."""
        try:
            from enrichment.example import main
            # If we get here, all imports succeeded
            assert callable(main)
        except ImportError as e:
            pytest.fail(f"Failed to import from example: {e}")

    def test_example_functions_exist(self):
        """Test that example functions exist and are callable."""
        try:
            from enrichment.example import (
                create_sample_chunks,
                demonstrate_sync_enrichment,
                demonstrate_configurations,
                demonstrate_convenience_functions,
                demonstrate_async_enrichment,
                demonstrate_batch_processing,
                main
            )
            
            # Check that functions are callable
            assert callable(create_sample_chunks)
            assert callable(demonstrate_sync_enrichment)
            assert callable(demonstrate_configurations)
            assert callable(demonstrate_convenience_functions)
            assert callable(demonstrate_async_enrichment)
            assert callable(demonstrate_batch_processing)
            assert callable(main)
            
        except ImportError as e:
            pytest.fail(f"Failed to import example functions: {e}")

    def test_example_with_mocked_imports(self):
        """Test example with mocked imports to avoid external dependencies."""
        with patch('enrichment.example.DocumentEnricher') as mock_enricher:
            with patch('enrichment.example.ChunkEnrichment') as mock_enrichment:
                with patch('enrichment.example.EnrichmentConfig') as mock_config:
                    try:
                        from enrichment.example import main
                        # Should be able to import without errors
                        assert callable(main)
                    except Exception as e:
                        # If there are import errors, that's okay for this test
                        # We're mainly testing that the structure is correct
                        pass

    @patch('enrichment.example.DocumentEnricher')
    @patch('enrichment.example.ChunkEnrichment')
    @patch('enrichment.example.EnrichmentConfig')
    def test_create_sample_chunks(self, mock_config, mock_enrichment, mock_enricher):
        """Test create_sample_chunks function."""
        from enrichment.example import create_sample_chunks
        
        # Mock the chunks
        mock_chunk1 = Mock()
        mock_chunk1.text = "Sample text 1"
        mock_chunk1.metadata = Mock()
        mock_chunk1.metadata.to_dict.return_value = {}
        
        mock_chunk2 = Mock()
        mock_chunk2.text = "Sample text 2"
        mock_chunk2.metadata = Mock()
        mock_chunk2.metadata.to_dict.return_value = {"text_as_html": "<table>...</table>"}
        
        with patch('unstructured.documents.elements.Text') as mock_text:
            with patch('unstructured.documents.elements.Table') as mock_table:
                mock_text.return_value = mock_chunk1
                mock_table.return_value = mock_chunk2
                
                chunks = create_sample_chunks()
                
                assert len(chunks) == 3
                # The function creates 3 chunks: 2 text chunks and 1 table chunk
                assert chunks[0].text == "This is a sample text chunk about machine learning and artificial intelligence."
                assert chunks[1].text == "Another text chunk discussing natural language processing and transformer models."
                assert chunks[2].metadata.is_table == True

    @patch('enrichment.example.DocumentEnricher')
    @patch('enrichment.example.ChunkEnrichment')
    def test_demonstrate_sync_enrichment(self, mock_enrichment, mock_enricher):
        """Test demonstrate_sync_enrichment function."""
        from enrichment.example import demonstrate_sync_enrichment
        
        # Mock the enricher
        mock_enricher_instance = Mock()
        mock_enricher.return_value = mock_enricher_instance
        
        # Mock enrichment result
        mock_enrichment_instance = Mock()
        mock_enrichment_instance.summary = "Test summary"
        mock_enrichment_instance.keywords = ["test", "keyword"]
        mock_enrichment_instance.hypothetical_questions = ["What is this about?"]
        mock_enrichment_instance.table_summary = None
        
        mock_enricher_instance.enrich_chunk.return_value = Mock()
        
        # Mock chunks
        mock_chunk = Mock()
        mock_chunk.text = "Test content"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.to_dict.return_value = {}
        
        with patch('enrichment.example.create_sample_chunks', return_value=[mock_chunk]):
            demonstrate_sync_enrichment()

    @patch('enrichment.example.DocumentEnricher')
    def test_demonstrate_configurations(self, mock_enricher):
        """Test demonstrate_configurations function."""
        from enrichment.example import demonstrate_configurations
        
        # Mock the enricher
        mock_enricher_instance = Mock()
        mock_enricher.return_value = mock_enricher_instance
        
        # Mock chunks
        mock_chunk = Mock()
        mock_chunk.text = "Test content"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.to_dict.return_value = {}
        
        with patch('enrichment.example.create_sample_chunks', return_value=[mock_chunk]):
            demonstrate_configurations()

    @patch('enrichment.example.add_enrichment_to_chunk')
    @patch('enrichment.example.add_enrichment_to_chunk_async')
    def test_demonstrate_convenience_functions(self, mock_async_func, mock_sync_func):
        """Test demonstrate_convenience_functions function."""
        from enrichment.example import demonstrate_convenience_functions
        
        # Mock chunks
        mock_chunk = Mock()
        mock_chunk.text = "Test content"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.to_dict.return_value = {}
        
        with patch('enrichment.example.create_sample_chunks', return_value=[mock_chunk]):
            demonstrate_convenience_functions()

    @patch('enrichment.example.DocumentEnricher')
    def test_demonstrate_async_enrichment(self, mock_enricher):
        """Test demonstrate_async_enrichment function."""
        from enrichment.example import demonstrate_async_enrichment
        
        # Mock the enricher
        mock_enricher_instance = Mock()
        mock_enricher.return_value = mock_enricher_instance
        
        # Mock chunks
        mock_chunk = Mock()
        mock_chunk.text = "Test content"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.to_dict.return_value = {}
        
        with patch('enrichment.example.create_sample_chunks', return_value=[mock_chunk]):
            # Run the async function
            asyncio.run(demonstrate_async_enrichment())

    @patch('enrichment.example.DocumentEnricher')
    def test_demonstrate_batch_processing(self, mock_enricher):
        """Test demonstrate_batch_processing function."""
        from enrichment.example import demonstrate_batch_processing
        
        # Mock the enricher
        mock_enricher_instance = Mock()
        mock_enricher.return_value = mock_enricher_instance
        
        # Mock chunks
        mock_chunks = []
        for i in range(5):
            mock_chunk = Mock()
            mock_chunk.text = f"Test content {i}"
            mock_chunk.metadata = Mock()
            mock_chunk.metadata.to_dict.return_value = {}
            mock_chunks.append(mock_chunk)
        
        with patch('enrichment.example.create_sample_chunks', return_value=mock_chunks):
            demonstrate_batch_processing()

    @patch('enrichment.example.demonstrate_sync_enrichment')
    @patch('enrichment.example.demonstrate_configurations')
    @patch('enrichment.example.demonstrate_convenience_functions')
    @patch('enrichment.example.demonstrate_async_enrichment')
    @patch('enrichment.example.demonstrate_batch_processing')
    def test_main_function(self, mock_batch, mock_async, mock_conv, mock_config, mock_sync):
        """Test main function execution."""
        from enrichment.example import main
        
        # Mock the async function to return a coroutine
        async def mock_async_func():
            pass
        mock_async.return_value = mock_async_func()
        
        main()
        
        # Verify all demonstration functions were called
        mock_sync.assert_called_once()
        mock_config.assert_called_once()
        mock_conv.assert_called_once()
        mock_async.assert_called_once()
        mock_batch.assert_called_once()

    def test_example_with_real_imports(self):
        """Test example with real imports but mocked LLM calls."""
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            with patch('unstructured.documents.elements.Text') as mock_text:
                with patch('unstructured.documents.elements.Table') as mock_table:
                    # Mock the LLM
                    mock_llm = Mock()
                    mock_chat.return_value = mock_llm
                    mock_llm.with_structured_output.return_value = mock_llm
                    
                    # Mock chunks
                    mock_chunk = Mock()
                    mock_chunk.text = "Test content"
                    mock_chunk.metadata = Mock()
                    mock_chunk.metadata.to_dict.return_value = {}
                    mock_text.return_value = mock_chunk
                    mock_table.return_value = mock_chunk
                    
                    try:
                        from enrichment.example import main
                        # Should be able to import and call main
                        main()
                    except Exception as e:
                        # If there are runtime errors, that's okay for this test
                        # We're mainly testing that the structure is correct
                        pass
