"""
QA Agent Module

This module provides functionality for question answering using the LERK System.
It integrates with the retriever module to find relevant documents and uses
LangGraph for multi-agent orchestration to provide comprehensive answers.

Main Components:
- Config: Configuration classes and presets for QA agent settings
- RetrievalAgent: Agent responsible for document retrieval
- SupervisorAgent: Orchestrates the QA workflow
- QAOrchestrator: Main interface for question answering
- Utils: Utility functions for answer processing, formatting, etc.
- Exceptions: Custom exception classes for error handling

Example Usage:
    from qa_agent import QAOrchestrator, DEFAULT_QA_CONFIG
    
    # Initialize the QA orchestrator
    qa_system = QAOrchestrator(config=DEFAULT_QA_CONFIG)
    
    # Ask a question
    answer = qa_system.ask_question("What is the attention mechanism?")
    print(answer)
"""

from .config import (
    QAConfig,
    DEFAULT_QA_CONFIG,
    FAST_QA_CONFIG,
    HIGH_QUALITY_QA_CONFIG,
    AgentType
)
from .exceptions import (
    QAError,
    RetrievalError,
    AnswerGenerationError,
    ConfigurationError,
    SessionError
)
from .retrieval_agent import RetrievalAgent
from .supervisor_agent import SupervisorAgent
from .qa_orchestrator import QAOrchestrator
from .utils import (
    format_answer,
    extract_sources,
    validate_question,
    create_session,
    cleanup_session
)

__version__ = "1.0.0"
__author__ = "LERK System Team"

# Public API
__all__ = [
    # Configuration
    "QAConfig",
    "DEFAULT_QA_CONFIG",
    "FAST_QA_CONFIG", 
    "HIGH_QUALITY_QA_CONFIG",
    "AgentType",
    
    # Core Components
    "RetrievalAgent",
    "SupervisorAgent", 
    "QAOrchestrator",
    
    # Utilities
    "format_answer",
    "extract_sources",
    "validate_question",
    "create_session",
    "cleanup_session",
    
    # Exceptions
    "QAError",
    "RetrievalError",
    "AnswerGenerationError",
    "ConfigurationError",
    "SessionError",
]
