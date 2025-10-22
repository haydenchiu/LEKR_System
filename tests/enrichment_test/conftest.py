"""
Pytest configuration and fixtures for enrichment module tests.

Provides common fixtures and test data for enrichment module testing.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import List, Dict, Any
from pathlib import Path

from enrichment.models import ChunkEnrichment
from enrichment.config import EnrichmentConfig


@pytest.fixture
def sample_enrichment_data():
    """Sample enrichment data for testing."""
    return {
        "summary": "This is a sample summary of a document chunk about machine learning and artificial intelligence.",
        "keywords": ["machine learning", "artificial intelligence", "neural networks", "deep learning", "algorithms"],
        "hypothetical_questions": [
            "What is machine learning?",
            "How do neural networks work?",
            "What are the applications of AI?"
        ],
        "table_summary": None
    }


@pytest.fixture
def sample_table_enrichment_data():
    """Sample table enrichment data for testing."""
    return {
        "summary": "This table shows performance metrics for different machine learning models.",
        "keywords": ["performance", "metrics", "models", "accuracy", "precision", "recall"],
        "hypothetical_questions": [
            "Which model has the highest accuracy?",
            "What are the performance differences?",
            "How do the models compare?"
        ],
        "table_summary": "The table compares accuracy, precision, and recall metrics across different ML models, showing Model A with 95% accuracy, Model B with 92% accuracy, and Model C with 88% accuracy."
    }


@pytest.fixture
def sample_chunk():
    """Sample text chunk for testing."""
    chunk = Mock()
    chunk.text = "This is a sample document chunk about machine learning and artificial intelligence. It contains information about neural networks and deep learning algorithms."
    chunk.metadata = Mock()
    chunk.metadata.to_dict.return_value = {
        "filetype": "text/plain",
        "page_number": 1,
        "is_table": False
    }
    return chunk


@pytest.fixture
def sample_table_chunk():
    """Sample table chunk for testing."""
    chunk = Mock()
    chunk.text = "Sample table data"
    chunk.metadata = Mock()
    chunk.metadata.to_dict.return_value = {
        "filetype": "text/html",
        "page_number": 2,
        "is_table": True,
        "text_as_html": "<table><tr><td>Model</td><td>Accuracy</td></tr><tr><td>A</td><td>95%</td></tr></table>"
    }
    chunk.metadata.text_as_html = "<table><tr><td>Model</td><td>Accuracy</td></tr><tr><td>A</td><td>95%</td></tr></table>"
    return chunk


@pytest.fixture
def sample_chunks(sample_chunk, sample_table_chunk):
    """Sample list of chunks for testing."""
    return [sample_chunk, sample_table_chunk]


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.invoke = Mock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def mock_enrichment_result(sample_enrichment_data):
    """Mock enrichment result."""
    enrichment = ChunkEnrichment(**sample_enrichment_data)
    return enrichment


@pytest.fixture
def mock_table_enrichment_result(sample_table_enrichment_data):
    """Mock table enrichment result."""
    enrichment = ChunkEnrichment(**sample_table_enrichment_data)
    return enrichment


@pytest.fixture
def default_config():
    """Default enrichment configuration."""
    return EnrichmentConfig()


@pytest.fixture
def fast_config():
    """Fast enrichment configuration."""
    return EnrichmentConfig.fast()


@pytest.fixture
def high_quality_config():
    """High quality enrichment configuration."""
    return EnrichmentConfig.high_quality()


@pytest.fixture
def custom_config():
    """Custom enrichment configuration."""
    return EnrichmentConfig(
        model_name="gpt-4o",
        temperature=0.1,
        batch_size=3,
        max_keywords=10,
        max_questions=7
    )


@pytest.fixture
def mock_enricher(mock_llm, default_config):
    """Mock document enricher for testing."""
    from enrichment.enricher import DocumentEnricher
    
    enricher = DocumentEnricher(config=default_config, llm=mock_llm)
    return enricher


@pytest.fixture
def sample_prompt_data():
    """Sample prompt data for testing."""
    return {
        "chunk_text": "This is a sample chunk about machine learning.",
        "is_table": False,
        "expected_prompt_contains": [
            "helpful assistant",
            "enrichment",
            "machine learning"
        ]
    }


@pytest.fixture
def sample_table_prompt_data():
    """Sample table prompt data for testing."""
    return {
        "chunk_text": "<table><tr><td>Data</td></tr></table>",
        "is_table": True,
        "expected_prompt_contains": [
            "TABLE chunk",
            "key insights",
            "data points"
        ]
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch processing data."""
    return {
        "chunks": [Mock() for _ in range(10)],
        "batch_size": 3,
        "expected_batches": 4  # ceil(10/3) = 4
    }


@pytest.fixture
def sample_enrichment_stats():
    """Sample enrichment statistics."""
    return {
        "total_chunks": 10,
        "enriched_chunks": 8,
        "table_chunks": 2,
        "text_chunks": 8,
        "enrichment_rate": 0.8
    }


@pytest.fixture
def sample_chunk_metadata():
    """Sample chunk metadata."""
    return {
        "filetype": "text/plain",
        "page_number": 1,
        "filename": "test_document.pdf",
        "is_table": False
    }


@pytest.fixture
def sample_table_metadata():
    """Sample table chunk metadata."""
    return {
        "filetype": "text/html",
        "page_number": 2,
        "filename": "test_document.pdf",
        "is_table": True,
        "text_as_html": "<table><tr><td>Sample</td></tr></table>"
    }


@pytest.fixture
def sample_validation_data():
    """Sample data for validation testing."""
    return {
        "valid_chunk": Mock(),
        "invalid_chunk": Mock(),
        "empty_chunk": Mock(),
        "chunk_without_metadata": Mock()
    }


@pytest.fixture
def sample_error_data():
    """Sample error data for exception testing."""
    return {
        "error_message": "Test error message",
        "chunk_id": "chunk_123",
        "chunk_type": "text",
        "model_name": "gpt-4o-mini",
        "retry_count": 2,
        "batch_size": 5,
        "failed_chunks": 2
    }


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "model_name": "gpt-4o",
        "temperature": 0.1,
        "batch_size": 5,
        "max_keywords": 8,
        "max_questions": 6,
        "enable_async_processing": True
    }


@pytest.fixture
def sample_async_data():
    """Sample data for async testing."""
    return {
        "chunks": [Mock() for _ in range(5)],
        "batch_size": 2,
        "expected_processing_time": 1.0  # seconds
    }


@pytest.fixture
def sample_integration_data():
    """Sample data for integration testing."""
    return {
        "document_path": "test_document.pdf",
        "expected_chunks": 5,
        "expected_enriched_chunks": 4,
        "expected_processing_time": 2.0
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
