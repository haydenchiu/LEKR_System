"""
Pytest configuration and fixtures for logic_extractor tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any
from unstructured.documents.elements import Text, Table

from logic_extractor.models import Claim, LogicalRelation, LogicExtractionSchemaLiteChunk


@pytest.fixture
def sample_claim():
    """Create a sample claim for testing."""
    return Claim(
        id="claim_1",
        statement="The Transformer model uses self-attention mechanism",
        type="factual",
        confidence=0.9,
        derived_from=None
    )


@pytest.fixture
def sample_claim_with_derivation():
    """Create a sample claim with derivation for testing."""
    return Claim(
        id="claim_2",
        statement="Self-attention allows parallel processing of sequences",
        type="inferential",
        confidence=0.8,
        derived_from=["claim_1"]
    )


@pytest.fixture
def sample_logical_relation():
    """Create a sample logical relation for testing."""
    return LogicalRelation(
        premise="claim_1",
        conclusion="claim_2",
        relation_type="supportive",
        certainty=0.8
    )


@pytest.fixture
def sample_logic_extraction():
    """Create a sample logic extraction result for testing."""
    claims = [
        Claim(
            id="claim_1",
            statement="The Transformer model uses self-attention mechanism",
            type="factual",
            confidence=0.9
        ),
        Claim(
            id="claim_2",
            statement="Self-attention allows parallel processing of sequences",
            type="inferential",
            confidence=0.8,
            derived_from=["claim_1"]
        )
    ]
    
    relations = [
        LogicalRelation(
            premise="claim_1",
            conclusion="claim_2",
            relation_type="supportive",
            certainty=0.8
        )
    ]
    
    return LogicExtractionSchemaLiteChunk(
        chunk_id="test_chunk_1",
        claims=claims,
        logical_relations=relations,
        assumptions=["The model is trained on large datasets"],
        constraints=["Memory limitations affect sequence length"],
        open_questions=["How does attention scale with sequence length?"]
    )


@pytest.fixture
def sample_text_chunk():
    """Create a sample text chunk for testing."""
    chunk = Mock(spec=Text)
    chunk.text = "The Transformer model revolutionized NLP by introducing self-attention mechanism."
    chunk.metadata = Mock()
    chunk.metadata.to_dict.return_value = {
        "filetype": "text/plain",
        "page_number": 1,
        "source": "test_document.txt"
    }
    return chunk


@pytest.fixture
def sample_table_chunk():
    """Create a sample table chunk for testing."""
    chunk = Mock(spec=Table)
    chunk.text = "Performance comparison of different models"
    chunk.metadata = Mock()
    chunk.metadata.to_dict.return_value = {
        "filetype": "text/html",
        "page_number": 2,
        "source": "test_document.html",
        "text_as_html": """
        <table>
            <tr><th>Model</th><th>BLEU Score</th><th>Training Time</th></tr>
            <tr><td>RNN</td><td>25.2</td><td>48 hours</td></tr>
            <tr><td>Transformer</td><td>34.5</td><td>12 hours</td></tr>
        </table>
        """
    }
    return chunk


@pytest.fixture
def sample_chunks(sample_text_chunk, sample_table_chunk):
    """Create a list of sample chunks for testing."""
    return [sample_text_chunk, sample_table_chunk]


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock_llm = Mock()
    mock_llm.invoke = Mock()
    mock_llm.ainvoke = AsyncMock()
    return mock_llm


@pytest.fixture
def mock_logic_extraction_result(sample_logic_extraction):
    """Create a mock logic extraction result."""
    return sample_logic_extraction


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_document.txt"
    file_path.write_text("This is a test document for logic extraction.")
    return str(file_path)


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)
