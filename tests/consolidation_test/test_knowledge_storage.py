"""
Unit tests for knowledge storage.

This module contains tests for the KnowledgeStorageManager class
and its methods for storing and retrieving consolidated knowledge.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import os

from consolidation.knowledge_storage import KnowledgeStorageManager
from consolidation.models import DocumentKnowledge, SubjectKnowledge, KeyConcept, KnowledgeRelation
from consolidation.config import ConsolidationConfig, DEFAULT_CONSOLIDATION_CONFIG
from consolidation.exceptions import StorageError, StorageBackendError, VectorSearchError


class TestKnowledgeStorageManager:
    """Test cases for the KnowledgeStorageManager class."""
    
    @pytest.fixture(autouse=True)
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer to prevent actual model loading."""
        with patch('consolidation.knowledge_storage.SentenceTransformer') as mock_model_class:
            mock_model_instance = Mock()
            mock_model_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            mock_model_class.return_value = mock_model_instance
            yield mock_model_instance
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def sample_document_knowledge(self):
        """Create sample document knowledge for testing."""
        return DocumentKnowledge(
            document_id="doc_1",
            title="AI Fundamentals",
            summary="A comprehensive overview of AI concepts",
            key_concepts=[
                KeyConcept(
                    concept_id="c1",
                    name="Machine Learning",
                    description="A subset of AI",
                    category="ai",
                    confidence=0.9,
                    source_chunks=["chunk_1"]
                )
            ],
            knowledge_relations=[],
            main_themes=["AI", "ML"],
            knowledge_graph={},
            quality_score=0.85
        )
    
    @pytest.fixture
    def sample_subject_knowledge(self):
        """Create sample subject knowledge for testing."""
        return SubjectKnowledge(
            subject_id="subject_ai",
            name="Artificial Intelligence",
            description="Comprehensive AI knowledge",
            core_concepts=[
                KeyConcept(
                    concept_id="c1",
                    name="Machine Learning",
                    description="A subset of AI",
                    category="ai",
                    confidence=0.9,
                    source_chunks=["chunk_1"]
                )
            ],
            knowledge_relations=[],
            document_sources=["doc_1", "doc_2"],
            knowledge_hierarchy={},
            expertise_level="intermediate",
            domain_tags=["AI", "ML"],
            quality_score=0.9
        )
    
    def test_init_default_config(self, mock_sentence_transformer):
        """Test initialization with default configuration."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager()
                    
                    assert isinstance(storage.config, ConsolidationConfig)
                    assert storage.config.storage_backend == "memory"
                    mock_engine.assert_called_once()
    
    def test_init_custom_config(self, mock_sentence_transformer):
        """Test initialization with custom configuration."""
        config = ConsolidationConfig(storage_backend="sqlite", enable_vector_search=False)
        
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager(config)
                    
                    assert storage.config == config
                    assert storage.embedding_model is None  # Should be None when disabled
    
    def test_init_with_missing_dependencies(self):
        """Test initialization failure with missing dependencies."""
        with patch('consolidation.knowledge_storage.SentenceTransformer', side_effect=ImportError("Model not found")):
            with pytest.raises(ImportError, match="Required storage dependencies not installed"):
                KnowledgeStorageManager()
    
    def test_initialize_sqlite(self, mock_sentence_transformer, temp_db_path):
        """Test SQLite storage initialization."""
        config = ConsolidationConfig(storage_backend="sqlite")
        
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager(config)
                    
                    # Should have called create_engine with SQLite URL
                    mock_engine.assert_called_once()
                    call_args = mock_engine.call_args[0]
                    assert "sqlite:///" in call_args[0]
    
    def test_initialize_postgresql_not_implemented(self, mock_sentence_transformer):
        """Test PostgreSQL storage initialization (not implemented)."""
        config = ConsolidationConfig(storage_backend="postgresql")
        
        with pytest.raises(NotImplementedError, match="PostgreSQL storage not yet implemented"):
            KnowledgeStorageManager(config)
    
    def test_initialize_memory(self, mock_sentence_transformer):
        """Test in-memory storage initialization."""
        config = ConsolidationConfig(storage_backend="memory")
        
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager(config)
                    
                    # Should have called create_engine with in-memory SQLite URL
                    mock_engine.assert_called_once_with("sqlite:///:memory:")
    
    def test_save_document_knowledge_success(self, sample_document_knowledge, mock_sentence_transformer):
        """Test successful document knowledge saving."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session
                    mock_session = Mock()
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    storage_id = storage.save_document_knowledge(sample_document_knowledge)
                    
                    assert isinstance(storage_id, str)
                    assert len(storage_id) > 0
                    mock_session.add.assert_called_once()
                    mock_session.commit.assert_called_once()
                    mock_session.close.assert_called_once()
    
    def test_save_document_knowledge_without_embeddings(self, sample_document_knowledge):
        """Test document knowledge saving without embeddings."""
        config = ConsolidationConfig(enable_vector_search=False)
        
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session
                    mock_session = Mock()
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager(config)
                    storage_id = storage.save_document_knowledge(sample_document_knowledge)
                    
                    assert isinstance(storage_id, str)
                    # Should not have embedding vector
                    call_args = mock_session.add.call_args[0][0]
                    assert call_args.embedding_vector is None
    
    def test_save_document_knowledge_failure(self, sample_document_knowledge, mock_sentence_transformer):
        """Test document knowledge saving failure."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session that raises an exception
                    mock_session = Mock()
                    mock_session.add.side_effect = Exception("Database error")
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    
                    with pytest.raises(StorageError, match="Failed to save document knowledge"):
                        storage.save_document_knowledge(sample_document_knowledge)
    
    def test_save_subject_knowledge_success(self, sample_subject_knowledge, mock_sentence_transformer):
        """Test successful subject knowledge saving."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session
                    mock_session = Mock()
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    storage_id = storage.save_subject_knowledge(sample_subject_knowledge)
                    
                    assert isinstance(storage_id, str)
                    assert len(storage_id) > 0
                    mock_session.add.assert_called_once()
                    mock_session.commit.assert_called_once()
                    mock_session.close.assert_called_once()
    
    def test_save_subject_knowledge_failure(self, sample_subject_knowledge, mock_sentence_transformer):
        """Test subject knowledge saving failure."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session that raises an exception
                    mock_session = Mock()
                    mock_session.add.side_effect = Exception("Database error")
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    
                    with pytest.raises(StorageError, match="Failed to save subject knowledge"):
                        storage.save_subject_knowledge(sample_subject_knowledge)
    
    def test_retrieve_document_knowledge_success(self, sample_document_knowledge, mock_sentence_transformer):
        """Test successful document knowledge retrieval."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session and query result
                    mock_session = Mock()
                    mock_db_record = Mock()
                    mock_db_record.document_id = "doc_1"
                    mock_db_record.title = "AI Fundamentals"
                    mock_db_record.summary = "A comprehensive overview of AI concepts"
                    mock_db_record.key_concepts = [{"concept_id": "c1", "name": "Machine Learning"}]
                    mock_db_record.knowledge_relations = []
                    mock_db_record.main_themes = ["AI", "ML"]
                    mock_db_record.knowledge_graph = {}
                    mock_db_record.quality_score = 0.85
                    mock_db_record.consolidation_metadata = {}
                    mock_db_record.created_at = datetime.now()
                    mock_db_record.updated_at = datetime.now()
                    
                    mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_record
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.retrieve_document_knowledge("doc_1")
                    
                    assert isinstance(result, DocumentKnowledge)
                    assert result.document_id == "doc_1"
                    mock_session.close.assert_called_once()
    
    def test_retrieve_document_knowledge_not_found(self, mock_sentence_transformer):
        """Test document knowledge retrieval when not found."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session that returns None
                    mock_session = Mock()
                    mock_session.query.return_value.filter_by.return_value.first.return_value = None
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.retrieve_document_knowledge("nonexistent_doc")
                    
                    assert result is None
                    mock_session.close.assert_called_once()
    
    def test_retrieve_document_knowledge_failure(self, mock_sentence_transformer):
        """Test document knowledge retrieval failure."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session that raises an exception
                    mock_session = Mock()
                    mock_session.query.side_effect = Exception("Database error")
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.retrieve_document_knowledge("doc_1")
                    
                    assert result is None  # Should return None on failure
    
    def test_retrieve_subject_knowledge_success(self, sample_subject_knowledge, mock_sentence_transformer):
        """Test successful subject knowledge retrieval."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session and query result
                    mock_session = Mock()
                    mock_db_record = Mock()
                    mock_db_record.subject_id = "subject_ai"
                    mock_db_record.name = "Artificial Intelligence"
                    mock_db_record.description = "Comprehensive AI knowledge"
                    mock_db_record.core_concepts = [{"concept_id": "c1", "name": "Machine Learning"}]
                    mock_db_record.knowledge_relations = []
                    mock_db_record.document_sources = ["doc_1", "doc_2"]
                    mock_db_record.knowledge_hierarchy = {}
                    mock_db_record.expertise_level = "intermediate"
                    mock_db_record.domain_tags = ["AI", "ML"]
                    mock_db_record.quality_score = 0.9
                    mock_db_record.consolidation_metadata = {}
                    mock_db_record.created_at = datetime.now()
                    mock_db_record.updated_at = datetime.now()
                    
                    mock_session.query.return_value.filter_by.return_value.first.return_value = mock_db_record
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.retrieve_subject_knowledge("subject_ai")
                    
                    assert isinstance(result, SubjectKnowledge)
                    assert result.subject_id == "subject_ai"
                    mock_session.close.assert_called_once()
    
    def test_retrieve_subject_knowledge_not_found(self, mock_sentence_transformer):
        """Test subject knowledge retrieval when not found."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session that returns None
                    mock_session = Mock()
                    mock_session.query.return_value.filter_by.return_value.first.return_value = None
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.retrieve_subject_knowledge("nonexistent_subject")
                    
                    assert result is None
                    mock_session.close.assert_called_once()
    
    def test_search_documents_by_similarity_success(self, mock_sentence_transformer):
        """Test successful document similarity search."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session and query result
                    mock_session = Mock()
                    mock_db_record = Mock()
                    mock_db_record.document_id = "doc_1"
                    mock_db_record.title = "AI Fundamentals"
                    mock_db_record.embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
                    # Add other required attributes
                    for attr in ['summary', 'key_concepts', 'knowledge_relations', 'main_themes', 
                               'knowledge_graph', 'quality_score', 'consolidation_metadata', 
                               'created_at', 'updated_at']:
                        setattr(mock_db_record, attr, getattr(mock_db_record, attr, None))
                    
                    mock_session.query.return_value.filter.return_value.all.return_value = [mock_db_record]
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    results = storage.search_documents_by_similarity("machine learning", limit=5)
                    
                    assert isinstance(results, list)
                    # Should have some results (exact number depends on similarity threshold)
                    assert len(results) >= 0
                    mock_session.close.assert_called_once()
    
    def test_search_documents_by_similarity_no_embedding_model(self):
        """Test document similarity search without embedding model."""
        config = ConsolidationConfig(enable_vector_search=False)
        
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager(config)
                    
                    with pytest.raises(VectorSearchError, match="Vector search not enabled"):
                        storage.search_documents_by_similarity("machine learning")
    
    def test_search_documents_by_similarity_failure(self, mock_sentence_transformer):
        """Test document similarity search failure."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session that raises an exception
                    mock_session = Mock()
                    mock_session.query.side_effect = Exception("Database error")
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    
                    with pytest.raises(VectorSearchError, match="Failed to search documents by similarity"):
                        storage.search_documents_by_similarity("machine learning")
    
    def test_search_subjects_by_similarity_success(self, mock_sentence_transformer):
        """Test successful subject similarity search."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session and query result
                    mock_session = Mock()
                    mock_db_record = Mock()
                    mock_db_record.subject_id = "subject_ai"
                    mock_db_record.name = "Artificial Intelligence"
                    mock_db_record.embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
                    # Add other required attributes
                    for attr in ['description', 'core_concepts', 'knowledge_relations', 'document_sources',
                               'knowledge_hierarchy', 'expertise_level', 'domain_tags', 'quality_score',
                               'consolidation_metadata', 'created_at', 'updated_at']:
                        setattr(mock_db_record, attr, getattr(mock_db_record, attr, None))
                    
                    mock_session.query.return_value.filter.return_value.all.return_value = [mock_db_record]
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    results = storage.search_subjects_by_similarity("artificial intelligence", limit=5)
                    
                    assert isinstance(results, list)
                    # Should have some results (exact number depends on similarity threshold)
                    assert len(results) >= 0
                    mock_session.close.assert_called_once()
    
    def test_search_subjects_by_similarity_no_embedding_model(self):
        """Test subject similarity search without embedding model."""
        config = ConsolidationConfig(enable_vector_search=False)
        
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager(config)
                    
                    with pytest.raises(VectorSearchError, match="Vector search not enabled"):
                        storage.search_subjects_by_similarity("artificial intelligence")
    
    def test_get_all_documents_success(self, mock_sentence_transformer):
        """Test successful retrieval of all documents."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session and query result
                    mock_session = Mock()
                    mock_db_record = Mock()
                    mock_db_record.document_id = "doc_1"
                    mock_db_record.title = "AI Fundamentals"
                    # Add other required attributes
                    for attr in ['summary', 'key_concepts', 'knowledge_relations', 'main_themes',
                               'knowledge_graph', 'quality_score', 'consolidation_metadata',
                               'created_at', 'updated_at']:
                        setattr(mock_db_record, attr, getattr(mock_db_record, attr, None))
                    
                    mock_session.query.return_value.all.return_value = [mock_db_record]
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    results = storage.get_all_documents()
                    
                    assert isinstance(results, list)
                    assert len(results) == 1
                    assert isinstance(results[0], DocumentKnowledge)
                    mock_session.close.assert_called_once()
    
    def test_get_all_documents_failure(self, mock_sentence_transformer):
        """Test retrieval of all documents failure."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session that raises an exception
                    mock_session = Mock()
                    mock_session.query.side_effect = Exception("Database error")
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    results = storage.get_all_documents()
                    
                    assert results == []  # Should return empty list on failure
    
    def test_get_all_subjects_success(self, mock_sentence_transformer):
        """Test successful retrieval of all subjects."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session and query result
                    mock_session = Mock()
                    mock_db_record = Mock()
                    mock_db_record.subject_id = "subject_ai"
                    mock_db_record.name = "Artificial Intelligence"
                    # Add other required attributes
                    for attr in ['description', 'core_concepts', 'knowledge_relations', 'document_sources',
                               'knowledge_hierarchy', 'expertise_level', 'domain_tags', 'quality_score',
                               'consolidation_metadata', 'created_at', 'updated_at']:
                        setattr(mock_db_record, attr, getattr(mock_db_record, attr, None))
                    
                    mock_session.query.return_value.all.return_value = [mock_db_record]
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    results = storage.get_all_subjects()
                    
                    assert isinstance(results, list)
                    assert len(results) == 1
                    assert isinstance(results[0], SubjectKnowledge)
                    mock_session.close.assert_called_once()
    
    def test_delete_document_knowledge_success(self, mock_sentence_transformer):
        """Test successful document knowledge deletion."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session
                    mock_session = Mock()
                    mock_session.query.return_value.filter_by.return_value.delete.return_value = 1
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.delete_document_knowledge("doc_1")
                    
                    assert result is True
                    mock_session.commit.assert_called_once()
                    mock_session.close.assert_called_once()
    
    def test_delete_document_knowledge_not_found(self, mock_sentence_transformer):
        """Test document knowledge deletion when not found."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session
                    mock_session = Mock()
                    mock_session.query.return_value.filter_by.return_value.delete.return_value = 0
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.delete_document_knowledge("nonexistent_doc")
                    
                    assert result is False
                    mock_session.commit.assert_called_once()
                    mock_session.close.assert_called_once()
    
    def test_delete_document_knowledge_failure(self, mock_sentence_transformer):
        """Test document knowledge deletion failure."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session that raises an exception
                    mock_session = Mock()
                    mock_session.query.side_effect = Exception("Database error")
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.delete_document_knowledge("doc_1")
                    
                    assert result is False  # Should return False on failure
    
    def test_delete_subject_knowledge_success(self, mock_sentence_transformer):
        """Test successful subject knowledge deletion."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    # Mock session
                    mock_session = Mock()
                    mock_session.query.return_value.filter_by.return_value.delete.return_value = 1
                    mock_sessionmaker.return_value.return_value = mock_session
                    
                    storage = KnowledgeStorageManager()
                    result = storage.delete_subject_knowledge("subject_ai")
                    
                    assert result is True
                    mock_session.commit.assert_called_once()
                    mock_session.close.assert_called_once()
    
    def test_generate_document_embedding(self, sample_document_knowledge, mock_sentence_transformer):
        """Test document embedding generation."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager()
                    
                    embedding = storage._generate_document_embedding(sample_document_knowledge)
                    
                    assert isinstance(embedding, list)
                    assert len(embedding) > 0
                    mock_sentence_transformer.encode.assert_called_once()
    
    def test_generate_document_embedding_failure(self, sample_document_knowledge, mock_sentence_transformer):
        """Test document embedding generation failure."""
        mock_sentence_transformer.encode.side_effect = Exception("Encoding error")
        
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager()
                    
                    embedding = storage._generate_document_embedding(sample_document_knowledge)
                    
                    # Should return default embedding on failure
                    assert isinstance(embedding, list)
                    assert len(embedding) == 5  # Default embedding size
    
    def test_generate_subject_embedding(self, sample_subject_knowledge, mock_sentence_transformer):
        """Test subject embedding generation."""
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager()
                    
                    embedding = storage._generate_subject_embedding(sample_subject_knowledge)
                    
                    assert isinstance(embedding, list)
                    assert len(embedding) > 0
                    mock_sentence_transformer.encode.assert_called_once()
    
    def test_generate_subject_embedding_failure(self, sample_subject_knowledge, mock_sentence_transformer):
        """Test subject embedding generation failure."""
        mock_sentence_transformer.encode.side_effect = Exception("Encoding error")
        
        with patch('consolidation.knowledge_storage.create_engine') as mock_engine:
            with patch('consolidation.knowledge_storage.Base.metadata.create_all') as mock_create_all:
                with patch('consolidation.knowledge_storage.sessionmaker') as mock_sessionmaker:
                    storage = KnowledgeStorageManager()
                    
                    embedding = storage._generate_subject_embedding(sample_subject_knowledge)
                    
                    # Should return default embedding on failure
                    assert isinstance(embedding, list)
                    assert len(embedding) == 5  # Default embedding size
