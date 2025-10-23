"""
Pytest fixtures for clustering tests.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from datetime import datetime

from clustering.models import ClusterInfo, ClusteringResult, DocumentClusterAssignment
from clustering.config import ClusteringConfig, DEFAULT_CLUSTERING_CONFIG


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Data science combines statistics, programming, and domain expertise.",
        "Python is a popular programming language for data analysis.",
        "TensorFlow and PyTorch are popular frameworks for deep learning.",
        "Supervised learning uses labeled data to train machine learning models.",
        "Unsupervised learning finds patterns in data without labeled examples.",
        "Reinforcement learning learns through interaction with an environment."
    ]


@pytest.fixture
def sample_document_ids():
    """Create sample document IDs for testing."""
    return [f"doc_{i}" for i in range(10)]


@pytest.fixture
def sample_cluster_info():
    """Create a sample ClusterInfo for testing."""
    return ClusterInfo(
        cluster_id=0,
        name="Machine Learning Cluster",
        topic_words=["machine", "learning", "neural", "network", "algorithm"],
        document_count=5,
        coherence_score=0.8,
        silhouette_score=0.7,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"category": "AI/ML"}
    )


@pytest.fixture
def sample_document_assignment():
    """Create a sample DocumentClusterAssignment for testing."""
    return DocumentClusterAssignment(
        document_id="doc_1",
        cluster_id=0,
        confidence=0.85,
        distance_to_center=0.3,
        assignment_reason="initial_assignment",
        assigned_at=datetime.now(),
        metadata={"source": "test"}
    )


@pytest.fixture
def sample_clustering_result(sample_cluster_info, sample_document_assignment):
    """Create a sample ClusteringResult for testing."""
    return ClusteringResult(
        clusters=[sample_cluster_info],
        assignments=[sample_document_assignment],
        total_documents=1,
        num_clusters=1,
        overall_quality_score=0.8,
        clustering_method="BERTopic",
        parameters={"model_name": "all-MiniLM-L6-v2"},
        created_at=datetime.now(),
        metadata={"test": True}
    )


@pytest.fixture
def mock_bertopic_model():
    """Create a mock BERTopic model for testing."""
    mock_model = Mock()
    mock_model.fit_transform.return_value = ([0, 1, 0, 1, 0], [[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
    mock_model.transform.return_value = ([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    mock_model.get_topic_info.return_value = Mock()
    mock_model.get_topic_info.return_value.iterrows.return_value = [
        (0, {"Topic": 0, "Count": 3}),
        (1, {"Topic": 1, "Count": 2})
    ]
    mock_model.get_topic.return_value = [("machine", 0.5), ("learning", 0.4), ("neural", 0.3)]
    return mock_model


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return mock_model


@pytest.fixture
def clustering_config():
    """Create a clustering configuration for testing."""
    return ClusteringConfig(
        model_name="all-MiniLM-L6-v2",
        min_cluster_size=2,
        top_k_words=5,
        verbose=False,
        n_jobs=1
    )


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_document.txt"
    file_path.write_text("This is a test document for clustering.")
    return str(file_path)


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)
