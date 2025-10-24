"""
Unit tests for the semantic retriever.

This module tests the SemanticRetriever class and its functionality,
including semantic search, similarity scoring, and result processing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from retriever.semantic_retriever import SemanticRetriever
from retriever.config import RetrieverConfig, SearchStrategy
from retriever.exceptions import VectorSearchError, InvalidQueryError


class TestSemanticRetriever:
    """Test cases for the SemanticRetriever class."""
    
    def test_init_default(self):
        """Test initialization with default configuration."""
        retriever = SemanticRetriever()
        
        assert retriever.config.search_strategy == SearchStrategy.SEMANTIC
        assert retriever._search_method == "similarity"
    
    def test_init_with_config(self):
        """Test initialization with custom configuration."""
        config = RetrieverConfig(max_results=20, similarity_threshold=0.8)
        retriever = SemanticRetriever(config=config)
        
        assert retriever.config == config
        assert retriever.config.search_strategy == SearchStrategy.SEMANTIC
    
    def test_init_force_semantic_strategy(self):
        """Test that search strategy is forced to SEMANTIC."""
        config = RetrieverConfig(search_strategy=SearchStrategy.HYBRID)
        retriever = SemanticRetriever(config=config)
        
        assert retriever.config.search_strategy == SearchStrategy.SEMANTIC
    
    def test_retrieve_documents_success(self):
        """Test successful document retrieval."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        
        # Mock similarity search results
        mock_docs = [
            (Document(page_content="doc1", metadata={"id": "1"}), 0.2),
            (Document(page_content="doc2", metadata={"id": "2"}), 0.4),
            (Document(page_content="doc3", metadata={"id": "3"}), 0.1),
        ]
        retriever.vector_store.similarity_search_with_score.return_value = mock_docs
        
        results = retriever._retrieve_documents("test query")
        
        assert len(results) == 3
        assert results[0].metadata["similarity_score"] == 0.8  # 1.0 - 0.2
        assert results[0].metadata["search_score"] == 0.2
        assert results[0].metadata["search_method"] == "similarity"
        
        assert results[1].metadata["similarity_score"] == 0.6  # 1.0 - 0.4
        assert results[2].metadata["similarity_score"] == 0.9  # 1.0 - 0.1
    
    def test_retrieve_documents_high_scores(self):
        """Test retrieval with scores > 1.0."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        
        # Mock similarity search results with scores > 1.0
        mock_docs = [
            (Document(page_content="doc1", metadata={"id": "1"}), 2.0),
            (Document(page_content="doc2", metadata={"id": "2"}), 5.0),
        ]
        retriever.vector_store.similarity_search_with_score.return_value = mock_docs
        
        results = retriever._retrieve_documents("test query")
        
        assert len(results) == 2
        assert results[0].metadata["similarity_score"] == 1.0 / 3.0  # 1.0 / (1.0 + 2.0)
        assert results[1].metadata["similarity_score"] == 1.0 / 6.0  # 1.0 / (1.0 + 5.0)
    
    def test_retrieve_documents_with_filters(self):
        """Test retrieval with metadata filters."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        
        mock_docs = [(Document(page_content="doc1", metadata={"id": "1"}), 0.2)]
        retriever.vector_store.similarity_search_with_score.return_value = mock_docs
        
        filters = {"category": "AI"}
        results = retriever._retrieve_documents("test query", filters=filters)
        
        # Verify filters were passed to vector store
        retriever.vector_store.similarity_search_with_score.assert_called_once_with(
            query="test query",
            k=20,  # max_results * 2
            filter=filters
        )
    
    def test_retrieve_documents_error(self):
        """Test retrieval with error."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.vector_store.similarity_search_with_score.side_effect = Exception("Search failed")
        
        with pytest.raises(VectorSearchError, match="Semantic search failed"):
            retriever._retrieve_documents("test query")
    
    def test_similarity_search_with_threshold(self):
        """Test similarity search with custom threshold."""
        retriever = SemanticRetriever()
        retriever._retrieve_documents = Mock(return_value=[
            Document(page_content="doc1", metadata={"similarity_score": 0.9}),
            Document(page_content="doc2", metadata={"similarity_score": 0.6}),
            Document(page_content="doc3", metadata={"similarity_score": 0.8}),
        ])
        
        results = retriever.similarity_search_with_threshold("test query", threshold=0.7)
        
        # Should only include docs with similarity >= 0.7
        assert len(results) == 2
        assert results[0].metadata["similarity_score"] == 0.9
        assert results[1].metadata["similarity_score"] == 0.8
    
    def test_similarity_search_with_threshold_default(self):
        """Test similarity search with default threshold."""
        retriever = SemanticRetriever()
        retriever.config.similarity_threshold = 0.8
        retriever._retrieve_documents = Mock(return_value=[
            Document(page_content="doc1", metadata={"similarity_score": 0.9}),
            Document(page_content="doc2", metadata={"similarity_score": 0.6}),
        ])
        
        results = retriever.similarity_search_with_threshold("test query")
        
        # Should use config threshold (0.8)
        assert len(results) == 1
        assert results[0].metadata["similarity_score"] == 0.9
    
    def test_get_similar_documents_success(self):
        """Test getting similar documents."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        
        # Mock reference document
        ref_doc = Document(page_content="reference content", metadata={"document_id": "ref_1"})
        retriever.vector_store.similarity_search.return_value = [ref_doc]
        
        # Mock similar documents
        similar_docs = [
            (Document(page_content="similar1", metadata={"document_id": "sim_1"}), 0.2),
            (Document(page_content="similar2", metadata={"document_id": "sim_2"}), 0.4),
        ]
        retriever.vector_store.similarity_search_with_score.return_value = similar_docs
        
        results = retriever.get_similar_documents("ref_1", limit=2)
        
        assert len(results) == 2
        assert results[0].metadata["reference_document_id"] == "ref_1"
        assert results[0].metadata["similarity_score"] == 0.8  # 1.0 - 0.2
        assert results[1].metadata["similarity_score"] == 0.6  # 1.0 - 0.4
    
    def test_get_similar_documents_not_found(self):
        """Test getting similar documents when reference not found."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.vector_store.similarity_search.return_value = []
        
        results = retriever.get_similar_documents("nonexistent_id")
        
        assert len(results) == 0
    
    def test_get_similar_documents_error(self):
        """Test getting similar documents with error."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.vector_store.similarity_search.side_effect = Exception("Search failed")
        
        with pytest.raises(VectorSearchError, match="Failed to find similar documents"):
            retriever.get_similar_documents("ref_1")
    
    def test_semantic_search_by_concept_success(self):
        """Test semantic search by concept."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.config.similarity_threshold = 0.7
        
        # Mock search results
        concept_docs = [
            (Document(page_content="concept doc1", metadata={"id": "1"}), 0.2),
            (Document(page_content="concept doc2", metadata={"id": "2"}), 0.4),
            (Document(page_content="concept doc3", metadata={"id": "3"}), 0.5),
        ]
        retriever.vector_store.similarity_search_with_score.return_value = concept_docs
        
        results = retriever.semantic_search_by_concept(
            concept="artificial intelligence",
            concept_type="key_concept",
            limit=5
        )
        
        assert len(results) == 3
        assert results[0].metadata["concept_search"] == "artificial intelligence"
        assert results[0].metadata["concept_type"] == "key_concept"
        assert results[0].metadata["similarity_score"] == 0.8  # 1.0 - 0.2
        
        # Verify filters were applied
        retriever.vector_store.similarity_search_with_score.assert_called_once_with(
            query="artificial intelligence",
            k=10,  # limit * 2
            filter={"key_concept_name": "artificial intelligence"}
        )
    
    def test_semantic_search_by_concept_with_filters(self):
        """Test semantic search by concept with additional filters."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.vector_store.similarity_search_with_score.return_value = []
        
        filters = {"category": "AI", "quality_score": {"min": 0.8}}
        results = retriever.semantic_search_by_concept(
            concept="machine learning",
            concept_type="key_concept",
            filters=filters
        )
        
        # Verify combined filters
        expected_filters = {
            "key_concept_name": "machine learning",
            "category": "AI",
            "quality_score": {"min": 0.8}
        }
        retriever.vector_store.similarity_search_with_score.assert_called_once_with(
            query="machine learning",
            k=20,  # default limit * 2
            filter=expected_filters
        )
    
    def test_semantic_search_by_concept_threshold_filtering(self):
        """Test concept search with threshold filtering."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.config.similarity_threshold = 0.8
        
        # Mock results with different similarity scores
        concept_docs = [
            (Document(page_content="high_sim", metadata={"id": "1"}), 0.1),  # similarity = 0.9
            (Document(page_content="low_sim", metadata={"id": "2"}), 0.5),  # similarity = 0.5
            (Document(page_content="med_sim", metadata={"id": "3"}), 0.2),  # similarity = 0.8
        ]
        retriever.vector_store.similarity_search_with_score.return_value = concept_docs
        
        results = retriever.semantic_search_by_concept("test concept")
        
        # Should only include docs with similarity >= 0.8
        assert len(results) == 2
        assert results[0].metadata["similarity_score"] == 0.9
        assert results[1].metadata["similarity_score"] == 0.8
    
    def test_semantic_search_by_concept_error(self):
        """Test concept search with error."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.vector_store.similarity_search_with_score.side_effect = Exception("Search failed")
        
        with pytest.raises(VectorSearchError, match="Concept search failed"):
            retriever.semantic_search_by_concept("test concept")
    
    def test_get_retrieval_stats(self):
        """Test retrieval statistics."""
        retriever = SemanticRetriever()
        retriever._initialized = True
        retriever._client = Mock()
        retriever.embedding_model = Mock()
        
        stats = retriever.get_retrieval_stats()
        
        assert "retriever_type" in stats
        assert "search_method" in stats
        assert "similarity_threshold" in stats
        
        assert stats["retriever_type"] == "semantic"
        assert stats["search_method"] == "similarity"
        assert stats["similarity_threshold"] == 0.7
    
    def test_str_representation(self):
        """Test string representation."""
        retriever = SemanticRetriever()
        retriever._initialized = True
        
        retriever_str = str(retriever)
        
        assert "SemanticRetriever" in retriever_str
        assert "initialized=True" in retriever_str
    
    def test_repr_representation(self):
        """Test detailed representation."""
        retriever = SemanticRetriever()
        retriever._initialized = True
        
        retriever_repr = repr(retriever)
        
        assert "SemanticRetriever" in retriever_repr
        assert "search_method=similarity" in retriever_repr
        assert "initialized=True" in retriever_repr


class TestSemanticRetrieverEdgeCases:
    """Test edge cases for SemanticRetriever."""
    
    def test_retrieve_documents_empty_results(self):
        """Test retrieval with empty results."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.vector_store.similarity_search_with_score.return_value = []
        
        results = retriever._retrieve_documents("test query")
        
        assert len(results) == 0
    
    def test_retrieve_documents_no_metadata(self):
        """Test retrieval with documents lacking metadata."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        
        mock_docs = [
            (Document(page_content="doc1", metadata={}), 0.2),
            (Document(page_content="doc2", metadata={}), 0.4),
        ]
        retriever.vector_store.similarity_search_with_score.return_value = mock_docs
        
        results = retriever._retrieve_documents("test query")
        
        assert len(results) == 2
        # Should still add similarity scores
        assert "similarity_score" in results[0].metadata
        assert "search_score" in results[0].metadata
        assert "search_method" in results[0].metadata
    
    def test_similarity_search_with_threshold_no_results(self):
        """Test similarity search with no results above threshold."""
        retriever = SemanticRetriever()
        retriever._retrieve_documents = Mock(return_value=[
            Document(page_content="doc1", metadata={"similarity_score": 0.6}),
            Document(page_content="doc2", metadata={"similarity_score": 0.5}),
        ])
        
        results = retriever.similarity_search_with_threshold("test query", threshold=0.8)
        
        assert len(results) == 0
    
    def test_get_similar_documents_exclude_self(self):
        """Test that similar documents exclude the reference document."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        
        # Mock reference document
        ref_doc = Document(page_content="reference", metadata={"document_id": "ref_1"})
        retriever.vector_store.similarity_search.return_value = [ref_doc]
        
        # Mock similar documents including the reference itself
        similar_docs = [
            (Document(page_content="similar1", metadata={"document_id": "sim_1"}), 0.2),
            (Document(page_content="reference", metadata={"document_id": "ref_1"}), 0.1),  # Same as reference
            (Document(page_content="similar2", metadata={"document_id": "sim_2"}), 0.3),
        ]
        retriever.vector_store.similarity_search_with_score.return_value = similar_docs
        
        results = retriever.get_similar_documents("ref_1", limit=5)
        
        # Should exclude the reference document itself
        assert len(results) == 2
        assert all(doc.metadata["document_id"] != "ref_1" for doc in results)
    
    def test_semantic_search_by_concept_no_results(self):
        """Test concept search with no results."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.vector_store.similarity_search_with_score.return_value = []
        
        results = retriever.semantic_search_by_concept("nonexistent concept")
        
        assert len(results) == 0
    
    def test_semantic_search_by_concept_default_parameters(self):
        """Test concept search with default parameters."""
        retriever = SemanticRetriever()
        retriever.vector_store = Mock()
        retriever.vector_store.similarity_search_with_score.return_value = []
        
        results = retriever.semantic_search_by_concept("test concept")
        
        # Should use default limit and concept_type
        retriever.vector_store.similarity_search_with_score.assert_called_once_with(
            query="test concept",
            k=20,  # default max_results * 2
            filter={"key_concept_name": "test concept"}
        )
