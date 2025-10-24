"""
Pytest fixtures for consolidation module tests.

This module provides common fixtures and test utilities
for testing the consolidation module components.
"""

import os
import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from consolidation.models import (
    DocumentKnowledge, SubjectKnowledge, KeyConcept, KnowledgeRelation
)


@pytest.fixture
def sample_key_concept():
    """Create a sample key concept for testing."""
    return KeyConcept(
        concept_id="concept_1",
        name="Machine Learning",
        description="A subset of artificial intelligence that focuses on algorithms that can learn from data",
        category="ai",
        confidence=0.9,
        source_chunks=["chunk_1", "chunk_2"],
        keywords=["AI", "algorithms", "data"]
    )


@pytest.fixture
def sample_key_concept_2():
    """Create a second sample key concept for testing."""
    return KeyConcept(
        concept_id="concept_2",
        name="Deep Learning",
        description="A subset of machine learning that uses neural networks with multiple layers",
        category="ai",
        confidence=0.8,
        source_chunks=["chunk_3"],
        keywords=["neural networks", "layers"]
    )


@pytest.fixture
def sample_knowledge_relation():
    """Create a sample knowledge relation for testing."""
    return KnowledgeRelation(
        relation_id="rel_1",
        source_concept="concept_1",
        target_concept="taget_concept_2",
        relation_type="hierarchical",
        strength=0.8,
        description="Deep learning is a subset of machine learning",
        evidence=["Deep learning uses neural networks", "Machine learning includes various approaches"]
    )


@pytest.fixture
def sample_document_knowledge(sample_key_concept, sample_key_concept_2, sample_knowledge_relation):
    """Create a sample document knowledge for testing."""
    return DocumentKnowledge(
        document_id="doc_1",
        title="AI Fundamentals",
        summary="A comprehensive overview of artificial intelligence and machine learning concepts",
        key_concepts=[sample_key_concept, sample_key_concept_2],
        knowledge_relations=[sample_knowledge_relation],
        main_themes=["artificial intelligence", "machine learning"],
        knowledge_graph={
            "nodes": {"concept_1": {"name": "Machine Learning"}},
            "edges": [{"source": "concept_1", "target": "concept_2"}]
        },
        quality_score=0.85,
        consolidation_metadata={
            "num_chunks": 5,
            "num_concepts": 2,
            "num_relations": 1
        }
    )


@pytest.fixture
def sample_subject_knowledge(sample_key_concept, sample_key_concept_2, sample_knowledge_relation):
    """Create a sample subject knowledge for testing."""
    return SubjectKnowledge(
        subject_id="subject_ai",
        name="Artificial Intelligence",
        description="Comprehensive knowledge about artificial intelligence and related fields",
        core_concepts=[sample_key_concept, sample_key_concept_2],
        knowledge_relations=[sample_knowledge_relation],
        document_sources=["doc_1", "doc_2"],
        knowledge_hierarchy={
            "levels": {0: ["concept_1"], 1: ["concept_2"]},
            "root_concepts": ["concept_1"],
            "leaf_concepts": ["concept_2"]
        },
        expertise_level="intermediate",
        domain_tags=["AI", "machine learning", "deep learning"],
        quality_score=0.9,
        consolidation_metadata={
            "num_documents": 2,
            "num_concepts": 2,
            "num_relations": 1
        }
    )


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        Mock(text="Machine learning is a subset of artificial intelligence", metadata=Mock()),
        Mock(text="Deep learning uses neural networks with multiple layers", metadata=Mock()),
        Mock(text="Natural language processing combines linguistics with ML", metadata=Mock())
    ]


@pytest.fixture
def sample_logic_data():
    """Create sample logic extraction data for testing."""
    return [
        {
            "claims": [
                {"id": "claim_1", "statement": "ML is a subset of AI", "type": "factual", "confidence": 0.9}
            ],
            "relations": [
                {"premise": "claim_1", "conclusion": "claim_2", "type": "hierarchical", "strength": 0.8}
            ]
        },
        {
            "claims": [
                {"id": "claim_2", "statement": "Deep learning uses neural networks", "type": "factual", "confidence": 0.8}
            ],
            "relations": []
        }
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock_llm = Mock()
    mock_llm.invoke = Mock()
    mock_llm.ainvoke = AsyncMock()
    
    # Mock response for concept extraction
    concept_response = Mock()
    concept_response.content = '''[
        {
            "name": "Machine Learning",
            "description": "A subset of artificial intelligence",
            "category": "ai",
            "confidence": 0.9,
            "keywords": ["AI", "algorithms"]
        }
    ]'''
    mock_llm.invoke.return_value = concept_response
    
    # Mock response for relation extraction
    relation_response = Mock()
    relation_response.content = '''[
        {
            "source_concept": "concept_1",
            "target_concept": "concept_2",
            "relation_type": "hierarchical",
            "strength": 0.8,
            "description": "Hierarchical relationship",
            "evidence": ["Evidence 1"]
        }
    ]'''
    mock_llm.invoke.return_value = relation_response
    
    return mock_llm


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    mock_model = Mock()
    mock_model.encode = Mock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    return mock_model


@pytest.fixture
def sample_clustering_result():
    """Create a sample clustering result for testing."""
    from clustering.models import ClusteringResult, ClusterInfo, DocumentClusterAssignment
    
    cluster_info = ClusterInfo(
        cluster_id=1,
        name="AI Cluster",
        topic_words=["artificial intelligence", "machine learning"],
        document_count=3,
        coherence_score=0.8,
        silhouette_score=0.7
    )
    
    assignments = [
        DocumentClusterAssignment(
            document_id="doc_1",
            cluster_id=1,
            confidence=0.9,
            distance_to_center=0.2
        ),
        DocumentClusterAssignment(
            document_id="doc_2",
            cluster_id=1,
            confidence=0.8,
            distance_to_center=0.3
        )
    ]
    
    return ClusteringResult(
        clusters=[cluster_info],
        assignments=assignments,
        metadata={"method": "BERTopic", "num_topics": 1}
    )


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_document.txt"
    file_path.write_text("This is a test document for consolidation testing.")
    return str(file_path)


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)
