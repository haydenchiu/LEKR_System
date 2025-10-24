"""
Unit tests for the RetrievalAgent class.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document

from qa_agent.retrieval_agent import RetrievalAgent
from qa_agent.config import QAConfig, DEFAULT_QA_CONFIG
from qa_agent.exceptions import RetrievalError, ValidationError


class TestRetrievalAgent:
    """Test cases for the RetrievalAgent class."""

    def test_init_default_config(self, mock_dependencies_for_qa_agent):
        """Test initialization with default configuration."""
        agent = RetrievalAgent()
        assert agent.config == DEFAULT_QA_CONFIG
        assert agent.retriever is not None
        assert agent.retriever_tool is not None
        assert agent.agent is not None

    def test_init_custom_config(self, sample_qa_config, mock_dependencies_for_qa_agent):
        """Test initialization with custom configuration."""
        agent = RetrievalAgent(config=sample_qa_config)
        assert agent.config == sample_qa_config

    def test_initialize_retriever_default(self, mock_dependencies_for_qa_agent):
        """Test retriever initialization with default config."""
        agent = RetrievalAgent()
        assert agent.retriever is not None

    def test_initialize_retriever_custom_config(self, mock_dependencies_for_qa_agent):
        """Test retriever initialization with custom config."""
        config = QAConfig(retriever_config={"search_strategy": "hybrid", "k": 10})
        agent = RetrievalAgent(config=config)
        assert agent.retriever is not None

    def test_create_agent(self, mock_dependencies_for_qa_agent):
        """Test agent creation."""
        agent = RetrievalAgent()
        assert agent.agent is not None
        assert agent.retriever_tool is not None

    def test_retrieve_documents_success(self, mock_dependencies_for_qa_agent, sample_question):
        """Test successful document retrieval."""
        agent = RetrievalAgent()
        
        # Mock the retriever's retrieve method
        expected_docs = [
            Document(page_content="Test document 1", metadata={"relevance_score": 0.9}),
            Document(page_content="Test document 2", metadata={"relevance_score": 0.8})
        ]
        agent.retriever.retrieve.return_value = expected_docs
        
        result = agent.retrieve_documents(sample_question)
        
        assert len(result) == 2
        assert result[0].page_content == "Test document 1"
        agent.retriever.retrieve.assert_called_once_with(
            sample_question,
            k=agent.config.max_retrieved_docs
        )

    def test_retrieve_documents_with_similarity_filter(self, mock_dependencies_for_qa_agent, sample_question):
        """Test document retrieval with similarity filtering."""
        config = QAConfig(similarity_threshold=0.8)
        agent = RetrievalAgent(config=config)
        
        # Mock documents with different scores
        docs = [
            Document(page_content="High relevance", metadata={"relevance_score": 0.9}),
            Document(page_content="Low relevance", metadata={"relevance_score": 0.7}),
            Document(page_content="Medium relevance", metadata={"relevance_score": 0.85})
        ]
        agent.retriever.retrieve.return_value = docs
        
        result = agent.retrieve_documents(sample_question)
        
        # Should filter out the low relevance document
        assert len(result) == 2
        assert all(doc.metadata["relevance_score"] >= 0.8 for doc in result)

    def test_retrieve_documents_invalid_question(self, mock_dependencies_for_qa_agent):
        """Test document retrieval with invalid question."""
        agent = RetrievalAgent()
        
        with pytest.raises(ValidationError):
            agent.retrieve_documents("")
        
        with pytest.raises(ValidationError):
            agent.retrieve_documents("a")

    def test_retrieve_documents_retrieval_error(self, mock_dependencies_for_qa_agent, sample_question):
        """Test document retrieval with retrieval error."""
        agent = RetrievalAgent()
        agent.retriever.retrieve.side_effect = Exception("Retrieval failed")
        
        with pytest.raises(RetrievalError, match="Failed to retrieve documents"):
            agent.retrieve_documents(sample_question)

    def test_aretrieve_documents_success(self, mock_dependencies_for_qa_agent, sample_question):
        """Test successful async document retrieval."""
        agent = RetrievalAgent()
        
        expected_docs = [
            Document(page_content="Test document", metadata={"relevance_score": 0.9})
        ]
        agent.retriever.aretrieve.return_value = expected_docs
        
        import asyncio
        result = asyncio.run(agent.aretrieve_documents(sample_question))
        
        assert len(result) == 1
        assert result[0].page_content == "Test document"
        agent.retriever.aretrieve.assert_called_once_with(
            sample_question,
            k=agent.config.max_retrieved_docs
        )

    def test_aretrieve_documents_invalid_question(self, mock_dependencies_for_qa_agent):
        """Test async document retrieval with invalid question."""
        agent = RetrievalAgent()
        
        import asyncio
        with pytest.raises(ValidationError):
            asyncio.run(agent.aretrieve_documents(""))

    def test_aretrieve_documents_retrieval_error(self, mock_dependencies_for_qa_agent, sample_question):
        """Test async document retrieval with retrieval error."""
        agent = RetrievalAgent()
        agent.retriever.aretrieve.side_effect = Exception("Async retrieval failed")
        
        import asyncio
        with pytest.raises(RetrievalError, match="Failed to retrieve documents"):
            asyncio.run(agent.aretrieve_documents(sample_question))

    def test_get_retrieval_stats(self, mock_dependencies_for_qa_agent):
        """Test retrieval statistics."""
        agent = RetrievalAgent()
        stats = agent.get_retrieval_stats()
        
        assert "retriever_type" in stats
        assert "max_docs" in stats
        assert "similarity_threshold" in stats
        assert "search_strategy" in stats
        assert "timeout" in stats
        assert stats["max_docs"] == agent.config.max_retrieved_docs

    def test_update_config(self, mock_dependencies_for_qa_agent):
        """Test configuration update."""
        agent = RetrievalAgent()
        original_max_docs = agent.config.max_retrieved_docs
        
        agent.update_config(max_retrieved_docs=10)
        
        assert agent.config.max_retrieved_docs == 10
        assert agent.config.max_retrieved_docs != original_max_docs

    def test_str_representation(self, mock_dependencies_for_qa_agent):
        """Test string representation."""
        agent = RetrievalAgent()
        str_repr = str(agent)
        
        assert "RetrievalAgent" in str_repr
        assert "max_docs" in str_repr

    def test_repr_representation(self, mock_dependencies_for_qa_agent):
        """Test detailed string representation."""
        agent = RetrievalAgent()
        repr_str = repr(agent)
        
        assert "RetrievalAgent" in repr_str
        assert "config" in repr_str
        assert "retriever" in repr_str
        assert "agent" in repr_str


class TestRetrievalAgentIntegration:
    """Integration tests for RetrievalAgent."""

    @patch('qa_agent.retrieval_agent.SemanticRetriever')
    def test_semantic_retriever_initialization(self, mock_semantic_retriever):
        """Test initialization with semantic retriever."""
        mock_retriever = Mock()
        mock_semantic_retriever.return_value = mock_retriever
        
        config = QAConfig(retriever_config={"search_strategy": "semantic"})
        agent = RetrievalAgent(config=config)
        
        assert agent.retriever == mock_retriever

    @patch('qa_agent.retrieval_agent.HybridRetriever')
    def test_hybrid_retriever_initialization(self, mock_hybrid_retriever):
        """Test initialization with hybrid retriever."""
        mock_retriever = Mock()
        mock_hybrid_retriever.return_value = mock_retriever
        
        config = QAConfig(retriever_config={"search_strategy": "hybrid"})
        agent = RetrievalAgent(config=config)
        
        assert agent.retriever == mock_retriever

    @patch('qa_agent.retrieval_agent.ContextRetriever')
    def test_context_retriever_initialization(self, mock_context_retriever):
        """Test initialization with context retriever."""
        mock_retriever = Mock()
        mock_context_retriever.return_value = mock_retriever
        
        config = QAConfig(retriever_config={"search_strategy": "context_aware"})
        agent = RetrievalAgent(config=config)
        
        assert agent.retriever == mock_retriever

    def test_retriever_initialization_error(self):
        """Test retriever initialization error handling."""
        with patch('qa_agent.retrieval_agent.SemanticRetriever') as mock_retriever:
            mock_retriever.side_effect = Exception("Initialization failed")
            
            with pytest.raises(RetrievalError, match="Failed to initialize retriever"):
                RetrievalAgent()

    def test_agent_creation_error(self):
        """Test agent creation error handling."""
        with patch('qa_agent.retrieval_agent.create_react_agent') as mock_create_agent:
            mock_create_agent.side_effect = Exception("Agent creation failed")
            
            with pytest.raises(RetrievalError, match="Failed to create retrieval agent"):
                RetrievalAgent()


class TestRetrievalAgentEdgeCases:
    """Edge case tests for RetrievalAgent."""

    def test_retrieve_documents_no_documents(self, mock_dependencies_for_qa_agent, sample_question):
        """Test retrieval when no documents are found."""
        agent = RetrievalAgent()
        agent.retriever.retrieve.return_value = []
        
        result = agent.retrieve_documents(sample_question)
        assert len(result) == 0

    def test_retrieve_documents_no_metadata(self, mock_dependencies_for_qa_agent, sample_question):
        """Test retrieval with documents that have no metadata."""
        agent = RetrievalAgent()
        docs = [
            Document(page_content="Test document", metadata={})
        ]
        agent.retriever.retrieve.return_value = docs
        
        # Should not filter out documents without relevance_score
        result = agent.retrieve_documents(sample_question)
        assert len(result) == 1

    def test_retrieve_documents_very_long_question(self, mock_dependencies_for_qa_agent):
        """Test retrieval with a very long question."""
        agent = RetrievalAgent()
        long_question = "What is machine learning? " * 100  # Very long but valid
        
        agent.retriever.retrieve.return_value = []
        
        # Should not raise an error
        result = agent.retrieve_documents(long_question)
        assert len(result) == 0

    def test_retrieve_documents_special_characters(self, mock_dependencies_for_qa_agent):
        """Test retrieval with special characters in question."""
        agent = RetrievalAgent()
        special_question = "What is machine learning? (AI/ML) - 2024"
        
        agent.retriever.retrieve.return_value = []
        
        # Should not raise an error
        result = agent.retrieve_documents(special_question)
        assert len(result) == 0
