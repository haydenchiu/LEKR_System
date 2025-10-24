"""
Pytest fixtures and configurations for the QA agent module tests.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
from langchain_core.documents import Document

from qa_agent.config import QAConfig, DEFAULT_QA_CONFIG
from qa_agent.exceptions import QAError, ValidationError


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="The attention mechanism is a key component of transformer models.",
            metadata={"source": "transformer_paper", "relevance_score": 0.9}
        ),
        Document(
            page_content="Transformers use self-attention to process sequences in parallel.",
            metadata={"source": "ml_textbook", "relevance_score": 0.8}
        ),
        Document(
            page_content="BERT is a bidirectional transformer model for language understanding.",
            metadata={"source": "bert_paper", "relevance_score": 0.7}
        ),
    ]


@pytest.fixture
def mock_retriever():
    """Mocks the retriever for testing."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="Test document", metadata={"relevance_score": 0.9})
    ]
    mock_retriever.aretrieve.return_value = [
        Document(page_content="Test document", metadata={"relevance_score": 0.9})
    ]
    mock_retriever.config = Mock()
    mock_retriever.config.search_strategy = "semantic"
    return mock_retriever


@pytest.fixture
def mock_llm():
    """Mocks the LLM for testing."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = Mock(content="This is a test answer.")
    mock_llm.ainvoke.return_value = Mock(content="This is a test answer.")
    return mock_llm


@pytest.fixture
def mock_supervisor():
    """Mocks the LangGraph supervisor for testing."""
    mock_supervisor = MagicMock()
    mock_supervisor.invoke.return_value = {
        "messages": [Mock(content="Test answer", type="ai")]
    }
    mock_supervisor.ainvoke.return_value = {
        "messages": [Mock(content="Test answer", type="ai")]
    }
    mock_supervisor.get_state.return_value = Mock(values={"messages": []})
    mock_supervisor.update_state.return_value = None
    return mock_supervisor


@pytest.fixture
def mock_memory():
    """Mocks the memory saver for testing."""
    mock_memory = MagicMock()
    return mock_memory


@pytest.fixture
def sample_qa_config():
    """Returns a sample QA configuration for testing."""
    return QAConfig(
        llm_model="gpt-4o-mini",
        max_retrieved_docs=3,
        similarity_threshold=0.7,
        max_answer_length=1000,
        answer_style="concise",
        session_timeout=1800,
        enable_logging=False
    )


@pytest.fixture
def sample_question():
    """Returns a sample question for testing."""
    return "What is the attention mechanism in transformers?"


@pytest.fixture
def sample_answer():
    """Returns a sample answer for testing."""
    return "The attention mechanism is a key component of transformer models that allows them to focus on relevant parts of the input sequence."


@pytest.fixture
def sample_session_id():
    """Returns a sample session ID for testing."""
    return "test_session_123"


@pytest.fixture(autouse=True)
def mock_dependencies_for_qa_agent():
    """
    Mocks external dependencies for QA agent initialization across all tests.
    """
    with patch('langchain_openai.ChatOpenAI') as mock_chat_openai, \
         patch('langgraph_supervisor.create_supervisor') as mock_create_supervisor, \
         patch('langgraph.checkpoint.memory.MemorySaver') as mock_memory_saver, \
         patch('qa_agent.retrieval_agent.RetrievalAgent') as mock_retrieval_agent_class:
        
        # Configure mocks
        mock_chat_openai.return_value = MagicMock()
        mock_create_supervisor.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        # Mock the retrieval agent
        mock_retrieval_agent = MagicMock()
        mock_retrieval_agent.agent = MagicMock()
        mock_retrieval_agent.get_retrieval_stats.return_value = {"retriever_type": "SemanticRetriever"}
        mock_retrieval_agent_class.return_value = mock_retrieval_agent
        
        yield {
            "mock_chat_openai": mock_chat_openai,
            "mock_create_supervisor": mock_create_supervisor,
            "mock_memory_saver": mock_memory_saver,
            "mock_retrieval_agent": mock_retrieval_agent
        }
