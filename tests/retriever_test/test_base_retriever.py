"""
Unit tests for the base retriever class.

This module tests the LERKBaseRetriever class and its core functionality,
including initialization, document retrieval, result processing, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from retriever.base_retriever import LERKBaseRetriever
from retriever.config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG
from retriever.exceptions import RetrievalError, InvalidQueryError, DatabaseConnectionError


class TestLERKBaseRetriever:
    """Test cases for the LERKBaseRetriever class."""
    
    def test_init_default(self):
        """Test initialization with default configuration."""
        retriever = LERKBaseRetriever()
        
        assert isinstance(retriever.config, RetrieverConfig)
        assert retriever.vector_store is None
        assert retriever.embedding_model is None
        assert retriever._client is None
        assert retriever._initialized == False
    
    def test_init_with_config(self):
        """Test initialization with custom configuration."""
        config = RetrieverConfig(max_results=20, similarity_threshold=0.8)
        retriever = LERKBaseRetriever(config=config)
        
        assert retriever.config == config
        assert retriever.config.max_results == 20
        assert retriever.config.similarity_threshold == 0.8
    
    def test_init_with_vector_store(self):
        """Test initialization with pre-initialized vector store."""
        mock_vector_store = Mock()
        retriever = LERKBaseRetriever(vector_store=mock_vector_store)
        
        assert retriever.vector_store == mock_vector_store
    
    def test_init_with_embedding_model(self):
        """Test initialization with pre-initialized embedding model."""
        mock_embedding_model = Mock()
        retriever = LERKBaseRetriever(embedding_model=mock_embedding_model)
        
        assert retriever.embedding_model == mock_embedding_model
    
    @patch('retriever.base_retriever.SentenceTransformer')
    @patch('retriever.base_retriever.QdrantClient')
    @patch('retriever.base_retriever.Qdrant')
    def test_initialize_success(self, mock_qdrant, mock_qdrant_client, mock_sentence_transformer):
        """Test successful initialization."""
        # Setup mocks
        mock_embedding_model = Mock()
        mock_sentence_transformer.return_value = mock_embedding_model
        
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        mock_vector_store = Mock()
        mock_qdrant.return_value = mock_vector_store
        
        # Test initialization
        retriever = LERKBaseRetriever()
        retriever._initialize()
        
        assert retriever._initialized == True
        assert retriever.embedding_model == mock_embedding_model
        assert retriever.vector_store == mock_vector_store
        assert retriever._client == mock_client
        
        # Verify method calls
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        mock_qdrant_client.assert_called_once_with(host="localhost", port=6333)
        mock_qdrant.assert_called_once()
    
    @patch('retriever.base_retriever.SentenceTransformer')
    def test_initialize_embedding_model_error(self, mock_sentence_transformer):
        """Test initialization failure due to embedding model error."""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        retriever = LERKBaseRetriever()
        
        with pytest.raises(RetrievalError, match="Failed to initialize retriever"):
            retriever._initialize()
    
    @patch('retriever.base_retriever.SentenceTransformer')
    @patch('retriever.base_retriever.QdrantClient')
    def test_initialize_vector_store_error(self, mock_qdrant_client, mock_sentence_transformer):
        """Test initialization failure due to vector store error."""
        mock_sentence_transformer.return_value = Mock()
        mock_qdrant_client.side_effect = Exception("Connection failed")
        
        retriever = LERKBaseRetriever()
        
        with pytest.raises(DatabaseConnectionError, match="Failed to initialize vector store"):
            retriever._initialize()
    
    def test_get_relevant_documents_not_initialized(self):
        """Test get_relevant_documents when not initialized."""
        retriever = LERKBaseRetriever()
        
        # Mock _initialize to prevent actual initialization
        retriever._initialize = Mock()
        
        # Mock _retrieve_documents
        retriever._retrieve_documents = Mock(return_value=[])
        
        results = retriever.get_relevant_documents("test query")
        
        # Should call _initialize
        retriever._initialize.assert_called_once()
        retriever._retrieve_documents.assert_called_once_with("test query", None)
    
    def test_get_relevant_documents_empty_query(self):
        """Test get_relevant_documents with empty query."""
        retriever = LERKBaseRetriever()
        retriever._initialized = True
        
        with pytest.raises(InvalidQueryError, match="Query cannot be empty"):
            retriever.get_relevant_documents("")
        
        with pytest.raises(InvalidQueryError, match="Query cannot be empty"):
            retriever.get_relevant_documents("   ")
    
    def test_get_relevant_documents_retrieval_error(self):
        """Test get_relevant_documents with retrieval error."""
        retriever = LERKBaseRetriever()
        retriever._initialized = True
        retriever._retrieve_documents = Mock(side_effect=Exception("Retrieval failed"))
        
        with pytest.raises(RetrievalError, match="Failed to retrieve documents"):
            retriever.get_relevant_documents("test query")
    
    def test_filter_by_similarity(self):
        """Test similarity threshold filtering."""
        retriever = LERKBaseRetriever()
        retriever.config.similarity_threshold = 0.7
        
        # Create test documents with different similarity scores
        docs = [
            Document(page_content="doc1", metadata={"similarity_score": 0.8}),
            Document(page_content="doc2", metadata={"similarity_score": 0.6}),
            Document(page_content="doc3", metadata={"similarity_score": 0.9}),
            Document(page_content="doc4", metadata={}),  # No similarity score
        ]
        
        filtered_docs = retriever._filter_by_similarity(docs)
        
        # Should only include docs with similarity >= 0.7
        assert len(filtered_docs) == 2
        assert filtered_docs[0].page_content == "doc1"
        assert filtered_docs[1].page_content == "doc3"
    
    def test_rank_results_relevance(self):
        """Test ranking by relevance score."""
        retriever = LERKBaseRetriever()
        retriever.config.ranking_method.value = "relevance"
        
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.6}),
            Document(page_content="doc2", metadata={"relevance_score": 0.9}),
            Document(page_content="doc3", metadata={"relevance_score": 0.3}),
        ]
        
        ranked_docs = retriever._rank_results(docs, "test query")
        
        # Should be sorted by relevance score (descending)
        assert ranked_docs[0].metadata["relevance_score"] == 0.9
        assert ranked_docs[1].metadata["relevance_score"] == 0.6
        assert ranked_docs[2].metadata["relevance_score"] == 0.3
    
    def test_rank_results_quality(self):
        """Test ranking by quality score."""
        retriever = LERKBaseRetriever()
        retriever.config.ranking_method.value = "quality"
        
        docs = [
            Document(page_content="doc1", metadata={"quality_score": 0.7}),
            Document(page_content="doc2", metadata={"quality_score": 0.9}),
            Document(page_content="doc3", metadata={"quality_score": 0.5}),
        ]
        
        # Mock rank_by_quality to return sorted docs
        with patch('retriever.base_retriever.rank_by_quality') as mock_rank:
            mock_rank.return_value = sorted(docs, key=lambda x: x.metadata["quality_score"], reverse=True)
            
            ranked_docs = retriever._rank_results(docs, "test query")
            
            mock_rank.assert_called_once_with(docs)
            assert len(ranked_docs) == 3
    
    def test_rank_results_combined(self):
        """Test combined ranking."""
        retriever = LERKBaseRetriever()
        retriever.config.ranking_method.value = "combined"
        retriever.config.relevance_weight = 0.6
        retriever.config.quality_weight = 0.4
        
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.8, "quality_score": 0.6}),
            Document(page_content="doc2", metadata={"relevance_score": 0.6, "quality_score": 0.9}),
        ]
        
        ranked_docs = retriever._rank_results(docs, "test query")
        
        # Check combined scores
        doc1_score = 0.6 * 0.8 + 0.4 * 0.6  # 0.72
        doc2_score = 0.6 * 0.6 + 0.4 * 0.9  # 0.72
        
        assert ranked_docs[0].metadata["combined_score"] == doc1_score
        assert ranked_docs[1].metadata["combined_score"] == doc2_score
    
    def test_rank_results_recency(self):
        """Test ranking by recency."""
        retriever = LERKBaseRetriever()
        retriever.config.ranking_method.value = "recency"
        
        docs = [
            Document(page_content="doc1", metadata={"timestamp": "2024-01-01T10:00:00Z"}),
            Document(page_content="doc2", metadata={"timestamp": "2024-01-02T10:00:00Z"}),
            Document(page_content="doc3", metadata={"timestamp": "2024-01-01T15:00:00Z"}),
        ]
        
        ranked_docs = retriever._rank_results(docs, "test query")
        
        # Should be sorted by timestamp (descending)
        assert ranked_docs[0].page_content == "doc2"  # Most recent
        assert ranked_docs[1].page_content == "doc3"
        assert ranked_docs[2].page_content == "doc1"  # Oldest
    
    def test_apply_diversity_filtering(self):
        """Test diversity filtering."""
        retriever = LERKBaseRetriever()
        retriever.config.enable_diversity = True
        retriever.config.diversity_threshold = 0.3
        retriever.embedding_model = Mock()
        retriever.embedding_model.encode = Mock(return_value=[[0.1, 0.2, 0.3]])
        
        docs = [
            Document(page_content="doc1", metadata={"embedding": [0.1, 0.2, 0.3]}),
            Document(page_content="doc2", metadata={"embedding": [0.1, 0.2, 0.3]}),  # Similar to doc1
            Document(page_content="doc3", metadata={"embedding": [0.9, 0.8, 0.7]}),  # Different
        ]
        
        with patch('retriever.base_retriever.np') as mock_np:
            mock_np.array.return_value = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]
            mock_np.dot.return_value = [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
            
            filtered_docs = retriever._apply_diversity_filtering(docs)
            
            # Should include diverse documents
            assert len(filtered_docs) >= 1
    
    def test_apply_diversity_filtering_single_doc(self):
        """Test diversity filtering with single document."""
        retriever = LERKBaseRetriever()
        retriever.config.enable_diversity = True
        
        docs = [Document(page_content="doc1", metadata={})]
        
        filtered_docs = retriever._apply_diversity_filtering(docs)
        
        # Should return the single document
        assert len(filtered_docs) == 1
        assert filtered_docs[0].page_content == "doc1"
    
    def test_apply_diversity_filtering_error(self):
        """Test diversity filtering with error."""
        retriever = LERKBaseRetriever()
        retriever.config.enable_diversity = True
        retriever.embedding_model = Mock()
        retriever.embedding_model.encode = Mock(side_effect=Exception("Encoding failed"))
        
        docs = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={}),
        ]
        
        # Should return original docs on error
        filtered_docs = retriever._apply_diversity_filtering(docs)
        assert len(filtered_docs) == 2
    
    def test_batch_retrieve(self):
        """Test batch retrieval functionality."""
        retriever = LERKBaseRetriever()
        retriever._initialized = True
        retriever.config.batch_size = 2
        
        # Mock get_relevant_documents
        retriever.get_relevant_documents = Mock(side_effect=[
            [Document(page_content="result1")],
            [Document(page_content="result2")],
            [Document(page_content="result3")],
        ])
        
        queries = ["query1", "query2", "query3"]
        results = retriever.batch_retrieve(queries)
        
        assert len(results) == 3
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert len(results[2]) == 1
        
        # Verify method calls
        assert retriever.get_relevant_documents.call_count == 3
    
    def test_batch_retrieve_with_errors(self):
        """Test batch retrieval with some errors."""
        retriever = LERKBaseRetriever()
        retriever._initialized = True
        retriever.config.batch_size = 2
        
        # Mock get_relevant_documents to fail for one query
        def mock_get_docs(query, **kwargs):
            if query == "query2":
                raise Exception("Retrieval failed")
            return [Document(page_content=f"result_{query}")]
        
        retriever.get_relevant_documents = Mock(side_effect=mock_get_docs)
        
        queries = ["query1", "query2", "query3"]
        results = retriever.batch_retrieve(queries)
        
        assert len(results) == 3
        assert len(results[0]) == 1  # Success
        assert len(results[1]) == 0  # Failed
        assert len(results[2]) == 1  # Success
    
    def test_get_retrieval_stats(self):
        """Test retrieval statistics."""
        retriever = LERKBaseRetriever()
        retriever._initialized = True
        retriever._client = Mock()
        retriever.embedding_model = Mock()
        
        stats = retriever.get_retrieval_stats()
        
        assert "config" in stats
        assert "initialized" in stats
        assert "vector_store_connected" in stats
        assert "embedding_model_loaded" in stats
        
        assert stats["initialized"] == True
        assert stats["vector_store_connected"] == True
        assert stats["embedding_model_loaded"] == True
    
    def test_str_representation(self):
        """Test string representation."""
        retriever = LERKBaseRetriever()
        retriever._initialized = True
        
        retriever_str = str(retriever)
        
        assert "LERKBaseRetriever" in retriever_str
        assert "initialized=True" in retriever_str
    
    def test_repr_representation(self):
        """Test detailed representation."""
        retriever = LERKBaseRetriever()
        retriever.vector_store = Mock()
        retriever.embedding_model = Mock()
        
        retriever_repr = repr(retriever)
        
        assert "LERKBaseRetriever" in retriever_repr
        assert "vector_store=True" in retriever_repr
        assert "embedding_model=True" in retriever_repr


class TestLERKBaseRetrieverEdgeCases:
    """Test edge cases for LERKBaseRetriever."""
    
    def test_rank_results_empty_list(self):
        """Test ranking with empty document list."""
        retriever = LERKBaseRetriever()
        
        ranked_docs = retriever._rank_results([], "test query")
        assert len(ranked_docs) == 0
    
    def test_rank_results_no_metadata(self):
        """Test ranking with documents lacking metadata."""
        retriever = LERKBaseRetriever()
        retriever.config.ranking_method.value = "relevance"
        
        docs = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={}),
        ]
        
        # Should not raise error
        ranked_docs = retriever._rank_results(docs, "test query")
        assert len(ranked_docs) == 2
    
    def test_rank_results_ranking_error(self):
        """Test ranking with error in ranking process."""
        retriever = LERKBaseRetriever()
        retriever.config.ranking_method.value = "quality"
        
        docs = [Document(page_content="doc1", metadata={})]
        
        # Mock rank_by_quality to raise error
        with patch('retriever.base_retriever.rank_by_quality') as mock_rank:
            mock_rank.side_effect = Exception("Ranking failed")
            
            # Should return original docs on error
            ranked_docs = retriever._rank_results(docs, "test query")
            assert len(ranked_docs) == 1
            assert ranked_docs[0].page_content == "doc1"
    
    def test_apply_diversity_filtering_no_embeddings(self):
        """Test diversity filtering with no embeddings."""
        retriever = LERKBaseRetriever()
        retriever.config.enable_diversity = True
        
        docs = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={}),
        ]
        
        # Should return original docs
        filtered_docs = retriever._apply_diversity_filtering(docs)
        assert len(filtered_docs) == 2
