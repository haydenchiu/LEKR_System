"""
Consolidation Module

This module provides functionality for consolidating knowledge from document chunks
into document-level and subject-level knowledge representations, and storing them
for agentic Q&A retrieval.

Main Components:
- Document Consolidation: Summarize logic chunks into key concepts and document knowledge
- Subject Consolidation: Aggregate document knowledge into general knowledge
- Storage: Save consolidated knowledge into database for retrieval
- Models: Pydantic models for knowledge representations
- Utils: Utility functions for knowledge processing
- Exceptions: Custom exception classes for error handling

Example Usage:
    from consolidation import DocumentConsolidator, SubjectConsolidator, KnowledgeStorage
    
    # Document-level consolidation
    doc_consolidator = DocumentConsolidator()
    doc_knowledge = doc_consolidator.consolidate_document(chunks)
    
    # Subject-level consolidation
    subject_consolidator = SubjectConsolidator()
    subject_knowledge = subject_consolidator.consolidate_subject(documents)
    
    # Storage
    storage = KnowledgeStorage()
    storage.save_knowledge(doc_knowledge, subject_knowledge)
"""

from .config import (
    ConsolidationConfig,
    DEFAULT_CONSOLIDATION_CONFIG,
    FAST_CONSOLIDATION_CONFIG,
    HIGH_QUALITY_CONSOLIDATION_CONFIG
)
from .exceptions import (
    ConsolidationError,
    DocumentConsolidationError,
    SubjectConsolidationError,
    StorageError,
    InvalidKnowledgeError,
    MissingAPIKeyError
)
from .models import (
    DocumentKnowledge,
    SubjectKnowledge,
    KeyConcept,
    KnowledgeRelation,
    KnowledgeStorage
)
from .document_consolidator import DocumentConsolidator
from .subject_consolidator import SubjectConsolidator
from .knowledge_storage import KnowledgeStorage
from .utils import (
    extract_key_concepts,
    build_knowledge_graph,
    validate_knowledge_consistency,
    merge_related_concepts
)

__version__ = "1.0.0"
__author__ = "LERK System Team"

# Public API
__all__ = [
    # Models
    "DocumentKnowledge",
    "SubjectKnowledge", 
    "KeyConcept",
    "KnowledgeRelation",
    "KnowledgeStorage",
    
    # Configuration
    "ConsolidationConfig",
    "DEFAULT_CONSOLIDATION_CONFIG",
    "FAST_CONSOLIDATION_CONFIG",
    "HIGH_QUALITY_CONSOLIDATION_CONFIG",
    
    # Core functionality
    "DocumentConsolidator",
    "SubjectConsolidator",
    "KnowledgeStorage",
    
    # Utilities
    "extract_key_concepts",
    "build_knowledge_graph",
    "validate_knowledge_consistency",
    "merge_related_concepts",
    
    # Exceptions
    "ConsolidationError",
    "DocumentConsolidationError",
    "SubjectConsolidationError",
    "StorageError",
    "InvalidKnowledgeError",
    "MissingAPIKeyError",
]
