"""
Unit tests for consolidation exceptions.

This module contains tests for all custom exception classes
used in the consolidation module.
"""

import pytest
from consolidation.exceptions import (
    ConsolidationError,
    DocumentConsolidationError,
    SubjectConsolidationError,
    StorageError,
    InvalidKnowledgeError,
    MissingAPIKeyError,
    ConceptExtractionError,
    RelationExtractionError,
    KnowledgeValidationError,
    StorageBackendError,
    VectorSearchError,
    KnowledgeMergeError,
    QualityValidationError
)


class TestConsolidationError:
    """Test cases for the base ConsolidationError class."""
    
    def test_consolidation_error_creation(self):
        """Test basic ConsolidationError creation."""
        error = ConsolidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_consolidation_error_with_no_message(self):
        """Test ConsolidationError with no message."""
        error = ConsolidationError()
        assert str(error) == ""
    
    def test_consolidation_error_inheritance(self):
        """Test that ConsolidationError is a proper exception."""
        error = ConsolidationError("Test error")
        assert isinstance(error, Exception)
        assert not isinstance(error, ValueError)
        assert not isinstance(error, TypeError)


class TestDocumentConsolidationError:
    """Test cases for the DocumentConsolidationError class."""
    
    def test_document_consolidation_error_creation(self):
        """Test basic DocumentConsolidationError creation."""
        error = DocumentConsolidationError("Document consolidation failed")
        assert str(error) == "Document consolidation failed"
        assert isinstance(error, ConsolidationError)
    
    def test_document_consolidation_error_with_context(self):
        """Test DocumentConsolidationError with additional context."""
        error = DocumentConsolidationError("Failed to consolidate document doc_123")
        assert "doc_123" in str(error)
    
    def test_document_consolidation_error_inheritance(self):
        """Test that DocumentConsolidationError inherits from ConsolidationError."""
        error = DocumentConsolidationError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestSubjectConsolidationError:
    """Test cases for the SubjectConsolidationError class."""
    
    def test_subject_consolidation_error_creation(self):
        """Test basic SubjectConsolidationError creation."""
        error = SubjectConsolidationError("Subject consolidation failed")
        assert str(error) == "Subject consolidation failed"
        assert isinstance(error, ConsolidationError)
    
    def test_subject_consolidation_error_with_context(self):
        """Test SubjectConsolidationError with additional context."""
        error = SubjectConsolidationError("Failed to consolidate subject subject_ai")
        assert "subject_ai" in str(error)
    
    def test_subject_consolidation_error_inheritance(self):
        """Test that SubjectConsolidationError inherits from ConsolidationError."""
        error = SubjectConsolidationError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestStorageError:
    """Test cases for the StorageError class."""
    
    def test_storage_error_creation(self):
        """Test basic StorageError creation."""
        error = StorageError("Storage operation failed")
        assert str(error) == "Storage operation failed"
        assert isinstance(error, ConsolidationError)
    
    def test_storage_error_with_operation(self):
        """Test StorageError with specific operation context."""
        error = StorageError("Failed to save document knowledge to database")
        assert "save" in str(error)
        assert "database" in str(error)
    
    def test_storage_error_inheritance(self):
        """Test that StorageError inherits from ConsolidationError."""
        error = StorageError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestInvalidKnowledgeError:
    """Test cases for the InvalidKnowledgeError class."""
    
    def test_invalid_knowledge_error_creation(self):
        """Test basic InvalidKnowledgeError creation."""
        error = InvalidKnowledgeError("Invalid knowledge data provided")
        assert str(error) == "Invalid knowledge data provided"
        assert isinstance(error, ConsolidationError)
    
    def test_invalid_knowledge_error_with_validation_context(self):
        """Test InvalidKnowledgeError with validation context."""
        error = InvalidKnowledgeError("Knowledge validation failed: missing required fields")
        assert "validation" in str(error)
        assert "missing" in str(error)
    
    def test_invalid_knowledge_error_inheritance(self):
        """Test that InvalidKnowledgeError inherits from ConsolidationError."""
        error = InvalidKnowledgeError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestMissingAPIKeyError:
    """Test cases for the MissingAPIKeyError class."""
    
    def test_missing_api_key_error_creation(self):
        """Test basic MissingAPIKeyError creation."""
        error = MissingAPIKeyError("OpenAI API key not found")
        assert str(error) == "OpenAI API key not found"
        assert isinstance(error, ConsolidationError)
    
    def test_missing_api_key_error_with_service_context(self):
        """Test MissingAPIKeyError with specific service context."""
        error = MissingAPIKeyError("Failed to initialize LLM: OPENAI_API_KEY is not set")
        assert "LLM" in str(error)
        assert "OPENAI_API_KEY" in str(error)
    
    def test_missing_api_key_error_inheritance(self):
        """Test that MissingAPIKeyError inherits from ConsolidationError."""
        error = MissingAPIKeyError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestConceptExtractionError:
    """Test cases for the ConceptExtractionError class."""
    
    def test_concept_extraction_error_creation(self):
        """Test basic ConceptExtractionError creation."""
        error = ConceptExtractionError("Failed to extract concepts from document")
        assert str(error) == "Failed to extract concepts from document"
        assert isinstance(error, ConsolidationError)
    
    def test_concept_extraction_error_with_document_context(self):
        """Test ConceptExtractionError with document context."""
        error = ConceptExtractionError("Failed to extract concepts from document doc_123")
        assert "doc_123" in str(error)
    
    def test_concept_extraction_error_inheritance(self):
        """Test that ConceptExtractionError inherits from ConsolidationError."""
        error = ConceptExtractionError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestRelationExtractionError:
    """Test cases for the RelationExtractionError class."""
    
    def test_relation_extraction_error_creation(self):
        """Test basic RelationExtractionError creation."""
        error = RelationExtractionError("Failed to extract relations between concepts")
        assert str(error) == "Failed to extract relations between concepts"
        assert isinstance(error, ConsolidationError)
    
    def test_relation_extraction_error_with_concept_context(self):
        """Test RelationExtractionError with concept context."""
        error = RelationExtractionError("Failed to extract relations for concepts: concept_1, concept_2")
        assert "concept_1" in str(error)
        assert "concept_2" in str(error)
    
    def test_relation_extraction_error_inheritance(self):
        """Test that RelationExtractionError inherits from ConsolidationError."""
        error = RelationExtractionError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestKnowledgeValidationError:
    """Test cases for the KnowledgeValidationError class."""
    
    def test_knowledge_validation_error_creation(self):
        """Test basic KnowledgeValidationError creation."""
        error = KnowledgeValidationError("Knowledge validation failed")
        assert str(error) == "Knowledge validation failed"
        assert isinstance(error, ConsolidationError)
    
    def test_knowledge_validation_error_with_validation_details(self):
        """Test KnowledgeValidationError with validation details."""
        error = KnowledgeValidationError("Knowledge validation failed: missing concept references")
        assert "missing concept references" in str(error)
    
    def test_knowledge_validation_error_inheritance(self):
        """Test that KnowledgeValidationError inherits from ConsolidationError."""
        error = KnowledgeValidationError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestStorageBackendError:
    """Test cases for the StorageBackendError class."""
    
    def test_storage_backend_error_creation(self):
        """Test basic StorageBackendError creation."""
        error = StorageBackendError("Storage backend initialization failed")
        assert str(error) == "Storage backend initialization failed"
        assert isinstance(error, StorageError)
    
    def test_storage_backend_error_with_backend_context(self):
        """Test StorageBackendError with specific backend context."""
        error = StorageBackendError("Failed to connect to PostgreSQL database")
        assert "PostgreSQL" in str(error)
        assert "database" in str(error)
    
    def test_storage_backend_error_inheritance(self):
        """Test that StorageBackendError inherits from StorageError."""
        error = StorageBackendError("Test error")
        assert isinstance(error, StorageError)
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestVectorSearchError:
    """Test cases for the VectorSearchError class."""
    
    def test_vector_search_error_creation(self):
        """Test basic VectorSearchError creation."""
        error = VectorSearchError("Vector search operation failed")
        assert str(error) == "Vector search operation failed"
        assert isinstance(error, StorageError)
    
    def test_vector_search_error_with_search_context(self):
        """Test VectorSearchError with search context."""
        error = VectorSearchError("Failed to perform similarity search for query: machine learning")
        assert "similarity search" in str(error)
        assert "machine learning" in str(error)
    
    def test_vector_search_error_inheritance(self):
        """Test that VectorSearchError inherits from StorageError."""
        error = VectorSearchError("Test error")
        assert isinstance(error, StorageError)
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestKnowledgeMergeError:
    """Test cases for the KnowledgeMergeError class."""
    
    def test_knowledge_merge_error_creation(self):
        """Test basic KnowledgeMergeError creation."""
        error = KnowledgeMergeError("Failed to merge knowledge from multiple sources")
        assert str(error) == "Failed to merge knowledge from multiple sources"
        assert isinstance(error, ConsolidationError)
    
    def test_knowledge_merge_error_with_merge_context(self):
        """Test KnowledgeMergeError with merge context."""
        error = KnowledgeMergeError("Failed to merge concepts: concept_1 and concept_2")
        assert "merge concepts" in str(error)
        assert "concept_1" in str(error)
        assert "concept_2" in str(error)
    
    def test_knowledge_merge_error_inheritance(self):
        """Test that KnowledgeMergeError inherits from ConsolidationError."""
        error = KnowledgeMergeError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestQualityValidationError:
    """Test cases for the QualityValidationError class."""
    
    def test_quality_validation_error_creation(self):
        """Test basic QualityValidationError creation."""
        error = QualityValidationError("Quality validation failed")
        assert str(error) == "Quality validation failed"
        assert isinstance(error, ConsolidationError)
    
    def test_quality_validation_error_with_quality_context(self):
        """Test QualityValidationError with quality context."""
        error = QualityValidationError("Quality score 0.3 is below minimum threshold 0.6")
        assert "Quality score" in str(error)
        assert "0.3" in str(error)
        assert "threshold" in str(error)
    
    def test_quality_validation_error_inheritance(self):
        """Test that QualityValidationError inherits from ConsolidationError."""
        error = QualityValidationError("Test error")
        assert isinstance(error, ConsolidationError)
        assert isinstance(error, Exception)


class TestExceptionHierarchy:
    """Test the exception hierarchy and relationships."""
    
    def test_exception_inheritance_chain(self):
        """Test that all exceptions properly inherit from ConsolidationError."""
        exceptions_to_test = [
            DocumentConsolidationError,
            SubjectConsolidationError,
            StorageError,
            InvalidKnowledgeError,
            MissingAPIKeyError,
            ConceptExtractionError,
            RelationExtractionError,
            KnowledgeValidationError,
            KnowledgeMergeError,
            QualityValidationError
        ]
        
        for exception_class in exceptions_to_test:
            error = exception_class("Test error")
            assert isinstance(error, ConsolidationError)
            assert isinstance(error, Exception)
    
    def test_storage_error_inheritance_chain(self):
        """Test that storage-related errors properly inherit from StorageError."""
        storage_exceptions = [
            StorageBackendError,
            VectorSearchError
        ]
        
        for exception_class in storage_exceptions:
            error = exception_class("Test error")
            assert isinstance(error, StorageError)
            assert isinstance(error, ConsolidationError)
            assert isinstance(error, Exception)
    
    def test_exception_raising_and_catching(self):
        """Test that exceptions can be raised and caught properly."""
        # Test raising and catching base exception
        with pytest.raises(ConsolidationError):
            raise ConsolidationError("Test error")
        
        # Test raising and catching specific exception
        with pytest.raises(DocumentConsolidationError):
            raise DocumentConsolidationError("Document error")
        
        # Test catching specific exception with base exception
        with pytest.raises(ConsolidationError):
            raise DocumentConsolidationError("Document error")
        
        # Test catching base exception with specific exception (should not work)
        with pytest.raises(ConsolidationError):
            raise ConsolidationError("Base error")
        
        # This should not catch the specific exception
        try:
            raise DocumentConsolidationError("Document error")
        except ConsolidationError:
            pass  # This should catch it
        except DocumentConsolidationError:
            pass  # This should also catch it
        else:
            pytest.fail("Exception should have been caught")
    
    def test_exception_message_preservation(self):
        """Test that exception messages are preserved correctly."""
        test_message = "This is a test error message"
        
        exceptions_to_test = [
            ConsolidationError,
            DocumentConsolidationError,
            SubjectConsolidationError,
            StorageError,
            InvalidKnowledgeError,
            MissingAPIKeyError,
            ConceptExtractionError,
            RelationExtractionError,
            KnowledgeValidationError,
            StorageBackendError,
            VectorSearchError,
            KnowledgeMergeError,
            QualityValidationError
        ]
        
        for exception_class in exceptions_to_test:
            error = exception_class(test_message)
            assert str(error) == test_message
