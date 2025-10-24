"""
Pytest fixtures for retriever module tests.

This module provides common fixtures for testing retriever functionality,
including mock vector stores, embedding models, and sample documents.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
from langchain_core.documents import Document

from retriever.config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    
    # Document 1: AI/ML content
    doc1 = Document(
        page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        metadata={
            "document_id": "doc_1",
            "title": "Introduction to Machine Learning",
            "category": "AI",
            "subject": "Computer Science",
            "quality_score": 0.9,
            "similarity_score": 0.85,
            "timestamp": "2024-01-15T10:00:00Z"
        }
    )
    docs.append(doc1)
    
    # Document 2: Deep Learning content
    doc2 = Document(
        page_content="Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        metadata={
            "document_id": "doc_2", 
            "title": "Deep Learning Fundamentals",
            "category": "AI",
            "subject": "Computer Science",
            "quality_score": 0.8,
            "similarity_score": 0.75,
            "timestamp": "2024-01-16T14:30:00Z"
        }
    )
    docs.append(doc2)
    
    # Document 3: NLP content
    doc3 = Document(
        page_content="Natural language processing combines computational linguistics with machine learning to understand human language.",
        metadata={
            "document_id": "doc_3",
            "title": "Natural Language Processing Guide",
            "category": "NLP",
            "subject": "Linguistics",
            "quality_score": 0.7,
            "similarity_score": 0.65,
            "timestamp": "2024-01-17T09:15:00Z"
        }
    )
    docs.append(doc3)
    
    # Document 4: Computer Vision content
    doc4 = Document(
        page_content="Computer vision enables machines to interpret and understand visual information from the world.",
        metadata={
            "document_id": "doc_4",
            "title": "Computer Vision Applications",
            "category": "CV",
            "subject": "Computer Science",
            "quality_score": 0.85,
            "similarity_score": 0.70,
            "timestamp": "2024-01-18T16:45:00Z"
        }
    )
    docs.append(doc4)
    
    return docs


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    mock_store = Mock()
    
    # Mock similarity search
    mock_store.similarity_search_with_score = Mock()
    mock_store.similarity_search = Mock()
    
    return mock_store


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    mock_model = Mock()
    
    # Mock encode method
    mock_model.encode = Mock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    
    return mock_model


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    mock_client = Mock()
    mock_client.get_collection = Mock()
    mock_client.scroll = Mock()
    mock_client.search = Mock()
    
    return mock_client


@pytest.fixture
def mock_bm25_retriever():
    """Create a mock BM25 retriever for testing."""
    mock_retriever = Mock()
    mock_retriever.get_relevant_documents = Mock()
    mock_retriever.from_documents = Mock()
    
    return mock_retriever


@pytest.fixture
def sample_queries():
    """Create sample queries for testing."""
    return [
        "machine learning algorithms",
        "deep learning neural networks",
        "natural language processing",
        "computer vision applications",
        "artificial intelligence"
    ]


@pytest.fixture
def sample_filters():
    """Create sample metadata filters for testing."""
    return {
        "category": "AI",
        "quality_score": {"min": 0.8},
        "subject": "Computer Science"
    }


@pytest.fixture
def mock_retriever_config():
    """Create a mock retriever configuration for testing."""
    return RetrieverConfig(
        max_results=5,
        similarity_threshold=0.7,
        embedding_model="test-model",
        enable_hybrid_search=True,
        semantic_weight=0.6,
        keyword_weight=0.4
    )


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_document.txt"
    file_path.write_text("This is a test document for retrieval testing.")
    return str(file_path)


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)


@pytest.fixture
def mock_search_results(sample_documents):
    """Create mock search results with scores."""
    results = []
    for i, doc in enumerate(sample_documents):
        score = 0.9 - (i * 0.1)  # Decreasing scores
        results.append((doc, score))
    return results


@pytest.fixture
def mock_context_history():
    """Create mock context history for testing."""
    return [
        {
            "query": "What is machine learning?",
            "timestamp": "2024-01-15T10:00:00Z",
            "results_count": 3,
            "session_id": "test_session_1"
        },
        {
            "query": "How does deep learning work?",
            "timestamp": "2024-01-15T10:05:00Z", 
            "results_count": 2,
            "session_id": "test_session_1"
        },
        {
            "query": "What are neural networks?",
            "timestamp": "2024-01-15T10:10:00Z",
            "results_count": 4,
            "session_id": "test_session_1"
        }
    ]


@pytest.fixture
def mock_user_preferences():
    """Create mock user preferences for testing."""
    return {
        "categories": ["AI", "Machine Learning"],
        "subjects": ["Computer Science"],
        "preferred_document_types": ["research_paper", "tutorial"],
        "quality_threshold": 0.7
    }


@pytest.fixture
def mock_feedback():
    """Create mock user feedback for testing."""
    return {
        "liked_documents": ["doc_1", "doc_3"],
        "disliked_documents": ["doc_2"],
        "preferred_categories": ["AI", "NLP"],
        "quality_ratings": {
            "doc_1": 5,
            "doc_2": 2,
            "doc_3": 4
        }
    }
