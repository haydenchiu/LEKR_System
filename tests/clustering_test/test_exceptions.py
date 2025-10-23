"""
Unit tests for clustering exceptions.
"""

import pytest
from clustering.exceptions import (
    ClusteringError,
    InvalidDocumentError,
    ClusteringInitializationError,
    ClusterAssignmentError,
    ClusteringFitError,
    ClusterNotFoundError,
    DocumentNotFoundError,
    ClusteringQualityError
)


class TestClusteringError:
    """Test cases for the base ClusteringError exception."""
    
    def test_clustering_error_creation(self):
        """Test basic ClusteringError creation."""
        error = ClusteringError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_clustering_error_inheritance(self):
        """Test that ClusteringError inherits from Exception."""
        error = ClusteringError("Test")
        assert isinstance(error, Exception)
        assert isinstance(error, ClusteringError)


class TestInvalidDocumentError:
    """Test cases for the InvalidDocumentError exception."""
    
    def test_invalid_document_error_creation(self):
        """Test basic InvalidDocumentError creation."""
        error = InvalidDocumentError("Invalid document provided")
        assert str(error) == "Invalid document provided"
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)
    
    def test_invalid_document_error_inheritance(self):
        """Test that InvalidDocumentError inherits from ClusteringError."""
        error = InvalidDocumentError("Test")
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)


class TestClusteringInitializationError:
    """Test cases for the ClusteringInitializationError exception."""
    
    def test_clustering_initialization_error_creation(self):
        """Test basic ClusteringInitializationError creation."""
        error = ClusteringInitializationError("Failed to initialize clustering model")
        assert str(error) == "Failed to initialize clustering model"
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)
    
    def test_clustering_initialization_error_inheritance(self):
        """Test that ClusteringInitializationError inherits from ClusteringError."""
        error = ClusteringInitializationError("Test")
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)


class TestClusterAssignmentError:
    """Test cases for the ClusterAssignmentError exception."""
    
    def test_cluster_assignment_error_creation(self):
        """Test basic ClusterAssignmentError creation."""
        error = ClusterAssignmentError("Failed to assign document to cluster")
        assert str(error) == "Failed to assign document to cluster"
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)
    
    def test_cluster_assignment_error_inheritance(self):
        """Test that ClusterAssignmentError inherits from ClusteringError."""
        error = ClusterAssignmentError("Test")
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)


class TestClusteringFitError:
    """Test cases for the ClusteringFitError exception."""
    
    def test_clustering_fit_error_creation(self):
        """Test basic ClusteringFitError creation."""
        error = ClusteringFitError("Failed to fit clustering model")
        assert str(error) == "Failed to fit clustering model"
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)
    
    def test_clustering_fit_error_inheritance(self):
        """Test that ClusteringFitError inherits from ClusteringError."""
        error = ClusteringFitError("Test")
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)


class TestClusterNotFoundError:
    """Test cases for the ClusterNotFoundError exception."""
    
    def test_cluster_not_found_error_creation(self):
        """Test basic ClusterNotFoundError creation."""
        error = ClusterNotFoundError("Cluster with ID 123 not found")
        assert str(error) == "Cluster with ID 123 not found"
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)
    
    def test_cluster_not_found_error_inheritance(self):
        """Test that ClusterNotFoundError inherits from ClusteringError."""
        error = ClusterNotFoundError("Test")
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)


class TestDocumentNotFoundError:
    """Test cases for the DocumentNotFoundError exception."""
    
    def test_document_not_found_error_creation(self):
        """Test basic DocumentNotFoundError creation."""
        error = DocumentNotFoundError("Document with ID doc_123 not found")
        assert str(error) == "Document with ID doc_123 not found"
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)
    
    def test_document_not_found_error_inheritance(self):
        """Test that DocumentNotFoundError inherits from ClusteringError."""
        error = DocumentNotFoundError("Test")
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)


class TestClusteringQualityError:
    """Test cases for the ClusteringQualityError exception."""
    
    def test_clustering_quality_error_creation(self):
        """Test basic ClusteringQualityError creation."""
        error = ClusteringQualityError("Clustering quality below acceptable threshold")
        assert str(error) == "Clustering quality below acceptable threshold"
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)
    
    def test_clustering_quality_error_inheritance(self):
        """Test that ClusteringQualityError inherits from ClusteringError."""
        error = ClusteringQualityError("Test")
        assert isinstance(error, ClusteringError)
        assert isinstance(error, Exception)


class TestExceptionHierarchy:
    """Test cases for exception hierarchy and relationships."""
    
    def test_exception_hierarchy(self):
        """Test that all clustering exceptions inherit from ClusteringError."""
        exceptions = [
            InvalidDocumentError,
            ClusteringInitializationError,
            ClusterAssignmentError,
            ClusteringFitError,
            ClusterNotFoundError,
            DocumentNotFoundError,
            ClusteringQualityError
        ]
        
        for exception_class in exceptions:
            error = exception_class("Test message")
            assert isinstance(error, ClusteringError)
            assert isinstance(error, Exception)
    
    def test_exception_raising(self):
        """Test that exceptions can be raised and caught properly."""
        with pytest.raises(ClusteringError):
            raise InvalidDocumentError("Test error")
        
        with pytest.raises(InvalidDocumentError):
            raise InvalidDocumentError("Test error")
        
        with pytest.raises(Exception):
            raise ClusteringInitializationError("Test error")
    
    def test_exception_chaining(self):
        """Test exception chaining with cause."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            with pytest.raises(ClusteringError):
                raise ClusteringInitializationError("Clustering failed") from e


class TestExceptionMessages:
    """Test cases for exception message handling."""
    
    def test_exception_message_types(self):
        """Test different types of exception messages."""
        # String message
        error1 = ClusteringError("String message")
        assert str(error1) == "String message"
        
        # Empty message
        error2 = ClusteringError("")
        assert str(error2) == ""
        
        # Long message
        long_message = "This is a very long error message that contains detailed information about what went wrong during the clustering process"
        error3 = ClusteringError(long_message)
        assert str(error3) == long_message
    
    def test_exception_message_formatting(self):
        """Test exception message formatting."""
        # Message with formatting
        error = ClusteringError(f"Error occurred at {__file__}:{123}")
        assert "Error occurred at" in str(error)
        
        # Message with variables
        cluster_id = 42
        error = ClusterNotFoundError(f"Cluster {cluster_id} not found")
        assert "Cluster 42 not found" in str(error)
    
    def test_exception_message_unicode(self):
        """Test exception messages with unicode characters."""
        unicode_message = "Clustering error: 聚类失败"
        error = ClusteringError(unicode_message)
        assert str(error) == unicode_message


class TestExceptionUsage:
    """Test cases for practical exception usage scenarios."""
    
    def test_validation_error_scenario(self):
        """Test exception usage in validation scenario."""
        def validate_document(document):
            if not document or not isinstance(document, str):
                raise InvalidDocumentError("Document must be a non-empty string")
            if len(document) < 10:
                raise InvalidDocumentError("Document too short for clustering")
            return True
        
        # Valid document
        assert validate_document("This is a valid document for clustering") is True
        
        # Invalid document - None
        with pytest.raises(InvalidDocumentError):
            validate_document(None)
        
        # Invalid document - too short
        with pytest.raises(InvalidDocumentError):
            validate_document("Short")
    
    def test_initialization_error_scenario(self):
        """Test exception usage in initialization scenario."""
        def initialize_clustering_model(model_name):
            if not model_name:
                raise ClusteringInitializationError("Model name cannot be empty")
            if model_name not in ["bertopic", "kmeans", "hdbscan"]:
                raise ClusteringInitializationError(f"Unsupported model: {model_name}")
            return f"Initialized {model_name}"
        
        # Valid model
        assert initialize_clustering_model("bertopic") == "Initialized bertopic"
        
        # Invalid model - empty
        with pytest.raises(ClusteringInitializationError):
            initialize_clustering_model("")
        
        # Invalid model - unsupported
        with pytest.raises(ClusteringInitializationError):
            initialize_clustering_model("unsupported")
    
    def test_assignment_error_scenario(self):
        """Test exception usage in assignment scenario."""
        def assign_document_to_cluster(document_id, cluster_id, available_clusters):
            if document_id not in ["doc_1", "doc_2", "doc_3"]:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            if cluster_id not in available_clusters:
                raise ClusterNotFoundError(f"Cluster {cluster_id} not found")
            if cluster_id == 999:  # Simulate assignment failure
                raise ClusterAssignmentError(f"Failed to assign document {document_id} to cluster {cluster_id}")
            return f"Assigned {document_id} to cluster {cluster_id}"
        
        available_clusters = [0, 1, 2]
        
        # Valid assignment
        result = assign_document_to_cluster("doc_1", 0, available_clusters)
        assert "Assigned doc_1 to cluster 0" in result
        
        # Document not found
        with pytest.raises(DocumentNotFoundError):
            assign_document_to_cluster("doc_999", 0, available_clusters)
        
        # Cluster not found
        with pytest.raises(ClusterNotFoundError):
            assign_document_to_cluster("doc_1", 999, available_clusters)
        
        # Assignment failure
        with pytest.raises(ClusterAssignmentError):
            assign_document_to_cluster("doc_1", 999, [999])  # Cluster exists but assignment fails
