"""
Tests for Qdrant indexing functionality.

This module contains comprehensive tests for the Qdrant indexing service,
including embedding strategies, chunk indexing, and index management.
"""

import pytest
import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from retriever.indexing import QdrantIndexer, EmbeddingStrategy
from retriever.config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG
from retriever.exceptions import VectorSearchError, DatabaseConnectionError


class TestEmbeddingStrategy:
    """Test cases for EmbeddingStrategy class."""
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        strategy = EmbeddingStrategy({})
        
        assert strategy.include_base_content is True
        assert strategy.include_enrichments is True
        assert strategy.include_logic_extractions is True
        assert strategy.combination_strategy == 'structured'
        assert strategy.max_text_length == 512
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            'include_base_content': False,
            'include_enrichments': True,
            'include_logic_extractions': False,
            'combination_strategy': 'concatenate',
            'max_text_length': 256
        }
        strategy = EmbeddingStrategy(config)
        
        assert strategy.include_base_content is False
        assert strategy.include_enrichments is True
        assert strategy.include_logic_extractions is False
        assert strategy.combination_strategy == 'concatenate'
        assert strategy.max_text_length == 256
    
    def test_create_embedding_text_content_only(self):
        """Test creating embedding text with content only."""
        strategy = EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': False,
            'include_logic_extractions': False,
            'combination_strategy': 'structured'
        })
        
        chunk_data = {
            'content': 'This is test content.',
            'enrichment': {'summary': 'Test summary'},
            'logic_extraction': {'claims': [{'text': 'Test claim'}]}
        }
        
        result = strategy.create_embedding_text(chunk_data)
        assert result == "Content: This is test content."
    
    def test_create_embedding_text_with_enrichments(self):
        """Test creating embedding text with enrichments."""
        strategy = EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': True,
            'include_logic_extractions': False,
            'combination_strategy': 'structured'
        })
        
        chunk_data = {
            'content': 'This is test content.',
            'enrichment': {
                'summary': 'Test summary',
                'keywords': ['test', 'keyword'],
                'table_summary': 'Table summary'
            }
        }
        
        result = strategy.create_embedding_text(chunk_data)
        expected_lines = [
            "Content: This is test content.",
            "Summary: Test summary",
            "Keywords: test, keyword",
            "Table: Table summary"
        ]
        
        for line in expected_lines:
            assert line in result
    
    def test_create_embedding_text_with_logic_extractions(self):
        """Test creating embedding text with logic extractions."""
        strategy = EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': False,
            'include_logic_extractions': True,
            'combination_strategy': 'structured'
        })
        
        chunk_data = {
            'content': 'This is test content.',
            'logic_extraction': {
                'claims': [{'text': 'Test claim 1'}, {'text': 'Test claim 2'}],
                'relations': [{'description': 'Test relation'}]
            }
        }
        
        result = strategy.create_embedding_text(chunk_data)
        assert "Content: This is test content." in result
        assert "Claims: Test claim 1, Test claim 2" in result
        assert "Relations: Test relation" in result
    
    def test_create_embedding_text_concatenate_strategy(self):
        """Test creating embedding text with concatenate strategy."""
        strategy = EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': True,
            'include_logic_extractions': False,
            'combination_strategy': 'concatenate'
        })
        
        chunk_data = {
            'content': 'This is test content.',
            'enrichment': {'summary': 'Test summary'}
        }
        
        result = strategy.create_embedding_text(chunk_data)
        assert "Content:" not in result
        assert "Summary:" not in result
        assert "This is test content." in result
        assert "Test summary" in result
    
    def test_create_embedding_text_max_length_truncation(self):
        """Test text truncation when exceeding max length."""
        strategy = EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': False,
            'include_logic_extractions': False,
            'max_text_length': 20
        })
        
        chunk_data = {
            'content': 'This is a very long test content that exceeds the maximum length limit.'
        }
        
        result = strategy.create_embedding_text(chunk_data)
        assert len(result) <= 23  # "Content: " + truncated content + "..."
        assert result.endswith("...")
    
    def test_create_embedding_text_empty_data(self):
        """Test creating embedding text with empty chunk data."""
        strategy = EmbeddingStrategy({})
        
        chunk_data = {}
        result = strategy.create_embedding_text(chunk_data)
        assert result == ""
    
    def test_create_embedding_text_missing_fields(self):
        """Test creating embedding text with missing optional fields."""
        strategy = EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': True,
            'include_logic_extractions': True,
            'combination_strategy': 'structured'
        })
        
        chunk_data = {
            'content': 'Test content',
            'enrichment': {
                'summary': 'Test summary',
                'keywords': None,
                'table_summary': None
            },
            'logic_extraction': {
                'claims': None,
                'relations': []
            }
        }
        
        result = strategy.create_embedding_text(chunk_data)
        assert "Content: Test content" in result
        assert "Summary: Test summary" in result
        assert "Keywords:" not in result
        assert "Table:" not in result
        assert "Claims:" not in result
        assert "Relations:" not in result


class TestQdrantIndexer:
    """Test cases for QdrantIndexer class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock retriever configuration."""
        return RetrieverConfig(
            collection_name='test_collection',
            host='localhost',
            port=6333,
            embedding_model='all-MiniLM-L6-v2',
            embedding_dimension=384
        )
    
    @pytest.fixture
    def mock_embedding_strategy(self):
        """Create a mock embedding strategy."""
        return EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': True,
            'include_logic_extractions': True,
            'combination_strategy': 'structured'
        })
    
    @pytest.fixture
    def sample_chunk_data(self):
        """Create sample chunk data for testing."""
        return {
            'id': 'test_chunk_1',
            'document_id': 'test_document',
            'content': 'This is test content for indexing.',
            'content_type': 'text',
            'chunk_index': 0,
            'enrichment': {
                'summary': 'Test chunk summary',
                'keywords': ['test', 'chunk', 'indexing']
            },
            'logic_extraction': {
                'claims': [{'text': 'This is a test claim', 'confidence': 0.9}]
            },
            'metadata': {'test': True}
        }
    
    @pytest.fixture
    def sample_processed_document(self):
        """Create sample processed document data."""
        return {
            'file_path': 'test_document.pdf',
            'processing_timestamp': datetime.utcnow().isoformat(),
            'chunks': [
                {
                    'id': 'chunk_1',
                    'content': 'First chunk content.',
                    'content_type': 'text',
                    'chunk_index': 0,
                    'enrichment': {'summary': 'First chunk summary'},
                    'logic_extraction': {'claims': [{'text': 'First claim'}]},
                    'metadata': {}
                },
                {
                    'id': 'chunk_2',
                    'content': 'Second chunk content.',
                    'content_type': 'text',
                    'chunk_index': 1,
                    'enrichment': {'summary': 'Second chunk summary'},
                    'logic_extraction': {'claims': [{'text': 'Second claim'}]},
                    'metadata': {}
                }
            ],
            'processing_stats': {
                'total_chunks': 2,
                'enriched_chunks': 2,
                'logic_chunks': 2
            }
        }
    
    @patch('retriever.indexing.QdrantClient')
    @patch('retriever.indexing.SentenceTransformer')
    def test_init(self, mock_transformer, mock_client, mock_config, mock_embedding_strategy):
        """Test QdrantIndexer initialization."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock collection operations
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.create_collection.return_value = None
        
        indexer = QdrantIndexer(mock_config, mock_embedding_strategy)
        
        assert indexer.config == mock_config
        assert indexer.embedding_strategy == mock_embedding_strategy
        assert indexer._client == mock_client_instance
        assert indexer._embedding_model == mock_transformer_instance
        assert indexer._initialized is True
    
    @patch('retriever.indexing.QdrantClient')
    @patch('retriever.indexing.SentenceTransformer')
    def test_index_chunk(self, mock_transformer, mock_client, mock_config, mock_embedding_strategy, sample_chunk_data):
        """Test indexing a single chunk."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock collection operations
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.create_collection.return_value = None
        mock_client_instance.upsert.return_value = None
        
        # Mock embedding
        mock_embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions
        mock_transformer_instance.encode.return_value = mock_embedding
        
        indexer = QdrantIndexer(mock_config, mock_embedding_strategy)
        
        point_id = indexer.index_chunk(sample_chunk_data)
        
        # Verify upsert was called
        mock_client_instance.upsert.assert_called_once()
        call_args = mock_client_instance.upsert.call_args
        
        assert call_args[1]['collection_name'] == 'test_collection'
        assert len(call_args[1]['points']) == 1
        
        point = call_args[1]['points'][0]
        assert point.id == point_id
        assert point.vector == mock_embedding
        assert point.payload['chunk_id'] == 'test_chunk_1'
        assert point.payload['content'] == 'This is test content for indexing.'
    
    @patch('retriever.indexing.QdrantClient')
    @patch('retriever.indexing.SentenceTransformer')
    def test_index_chunks_batch(self, mock_transformer, mock_client, mock_config, mock_embedding_strategy):
        """Test indexing multiple chunks in batch."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock collection operations
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.create_collection.return_value = None
        mock_client_instance.upsert.return_value = None
        
        # Mock embedding
        mock_embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions
        mock_transformer_instance.encode.return_value = mock_embedding
        
        indexer = QdrantIndexer(mock_config, mock_embedding_strategy)
        
        chunks_data = [
            {
                'id': 'chunk_1',
                'content': 'First chunk',
                'document_id': 'doc_1'
            },
            {
                'id': 'chunk_2',
                'content': 'Second chunk',
                'document_id': 'doc_1'
            }
        ]
        
        point_ids = indexer.index_chunks_batch(chunks_data)
        
        assert len(point_ids) == 2
        mock_client_instance.upsert.assert_called_once()
        
        call_args = mock_client_instance.upsert.call_args
        assert len(call_args[1]['points']) == 2
    
    def test_index_processed_document(self, mock_config, mock_embedding_strategy, sample_processed_document):
        """Test indexing a processed document file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_processed_document, f)
            temp_file = f.name
        
        try:
            with patch('retriever.indexing.QdrantClient'), \
                 patch('retriever.indexing.SentenceTransformer') as mock_transformer:
                
                mock_transformer_instance = Mock()
                mock_transformer.return_value = mock_transformer_instance
                mock_transformer_instance.encode.return_value = [0.1] * 384
                
                indexer = QdrantIndexer(mock_config, mock_embedding_strategy)
                
                # Mock the batch indexing method
                with patch.object(indexer, 'index_chunks_batch', return_value=['point_1', 'point_2']):
                    result = indexer.index_processed_document(temp_file)
                
                assert result['success'] is True
                assert result['indexed_count'] == 2
                assert result['file_path'] == temp_file
        
        finally:
            Path(temp_file).unlink()
    
    def test_index_processed_document_no_chunks(self, mock_config, mock_embedding_strategy):
        """Test indexing a processed document with no chunks."""
        sample_data = {
            'file_path': 'empty_document.pdf',
            'chunks': []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name
        
        try:
            indexer = QdrantIndexer(mock_config, mock_embedding_strategy)
            result = indexer.index_processed_document(temp_file)
            
            assert result['success'] is False
            assert 'No chunks found' in result['error']
            assert result['indexed_count'] == 0
        
        finally:
            Path(temp_file).unlink()
    
    @patch('retriever.indexing.QdrantClient')
    @patch('retriever.indexing.SentenceTransformer')
    def test_delete_chunk(self, mock_transformer, mock_client, mock_config, mock_embedding_strategy):
        """Test deleting a chunk from Qdrant."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock collection operations
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.create_collection.return_value = None
        
        # Mock scroll result with points found
        mock_point = Mock()
        mock_point.id = 'point_123'
        mock_client_instance.scroll.return_value = ([mock_point], None)
        mock_client_instance.delete.return_value = None
        
        indexer = QdrantIndexer(mock_config, mock_embedding_strategy)
        
        result = indexer.delete_chunk('test_chunk')
        
        assert result is True
        mock_client_instance.delete.assert_called_once()
    
    @patch('retriever.indexing.QdrantClient')
    @patch('retriever.indexing.SentenceTransformer')
    def test_get_collection_stats(self, mock_transformer, mock_client, mock_config, mock_embedding_strategy):
        """Test getting collection statistics."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        
        # Mock collection operations
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_client_instance.create_collection.return_value = None
        
        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.config.params.vectors.distance = 'Cosine'
        mock_collection_info.status = 'green'
        mock_client_instance.get_collection.return_value = mock_collection_info
        
        indexer = QdrantIndexer(mock_config, mock_embedding_strategy)
        
        stats = indexer.get_collection_stats()
        
        assert stats['collection_name'] == 'test_collection'
        assert stats['points_count'] == 100
        assert stats['vector_size'] == 384
        assert stats['distance'] == 'Cosine'
        assert stats['status'] == 'green'


if __name__ == "__main__":
    pytest.main([__file__])
