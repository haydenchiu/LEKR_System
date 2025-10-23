"""
Unit tests for the clustering clusterer module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np

from clustering.clusterer import DocumentClusterer, assign_documents_to_clusters, reassign_documents_to_clusters
from clustering.models import ClusterInfo, ClusteringResult, DocumentClusterAssignment
from clustering.config import ClusteringConfig, DEFAULT_CLUSTERING_CONFIG
from clustering.exceptions import (
    ClusteringError, InvalidDocumentError, ClusteringInitializationError,
    ClusterAssignmentError, ClusteringFitError
)


class TestDocumentClusterer:
    """Test cases for the DocumentClusterer class."""
    
    @pytest.fixture
    def mock_bertopic_model(self):
        """Create a mock BERTopic model."""
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
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return mock_model
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_init_default(self, mock_sentence_transformer, mock_bertopic):
        """Test initialization with default config."""
        mock_bertopic.return_value = Mock()
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        
        assert isinstance(clusterer.config, ClusteringConfig)
        assert clusterer.is_fitted is False
        assert clusterer.clustering_result is None
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        mock_bertopic.assert_called_once()
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_init_custom_config(self, mock_sentence_transformer, mock_bertopic):
        """Test initialization with custom config."""
        mock_bertopic.return_value = Mock()
        mock_sentence_transformer.return_value = Mock()
        
        config = ClusteringConfig(model_name="custom-model", verbose=True)
        clusterer = DocumentClusterer(config)
        
        assert clusterer.config == config
        mock_sentence_transformer.assert_called_once_with("custom-model")
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_init_initialization_error(self, mock_sentence_transformer, mock_bertopic):
        """Test initialization error handling."""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        with pytest.raises(ClusteringInitializationError, match="Failed to initialize clustering models"):
            DocumentClusterer()
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_fit_clusters_success(self, mock_sentence_transformer, mock_bertopic, sample_documents, sample_document_ids):
        """Test successful cluster fitting."""
        # Setup mocks
        mock_bertopic_instance = Mock()
        mock_bertopic_instance.fit_transform.return_value = ([0, 1, 0, 1, 0], [[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        mock_bertopic_instance.get_topic_info.return_value = Mock()
        mock_bertopic_instance.get_topic_info.return_value.iterrows.return_value = [
            (0, {"Topic": 0, "Count": 3}),
            (1, {"Topic": 1, "Count": 2})
        ]
        mock_bertopic_instance.get_topic.return_value = [("machine", 0.5), ("learning", 0.4), ("neural", 0.3)]
        mock_bertopic.return_value = mock_bertopic_instance
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        result = clusterer.fit_clusters(sample_documents, sample_document_ids)
        
        assert isinstance(result, ClusteringResult)
        assert result.total_documents == len(sample_documents)
        assert clusterer.is_fitted is True
        assert clusterer.clustering_result == result
    
    def test_fit_clusters_no_documents(self):
        """Test fitting with no documents."""
        clusterer = DocumentClusterer()
        
        with pytest.raises(InvalidDocumentError, match="No documents provided for clustering"):
            clusterer.fit_clusters([])
    
    def test_fit_clusters_mismatched_ids(self, sample_documents):
        """Test fitting with mismatched document IDs."""
        clusterer = DocumentClusterer()
        document_ids = ["doc_1", "doc_2"]  # Only 2 IDs for 5 documents
        
        with pytest.raises(InvalidDocumentError, match="Number of document IDs must match number of documents"):
            clusterer.fit_clusters(sample_documents, document_ids)
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_fit_clusters_bertopic_error(self, mock_sentence_transformer, mock_bertopic, sample_documents):
        """Test fitting with BERTopic error."""
        mock_bertopic_instance = Mock()
        mock_bertopic_instance.fit_transform.side_effect = Exception("BERTopic failed")
        mock_bertopic.return_value = mock_bertopic_instance
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        
        with pytest.raises(ClusteringFitError, match="Failed to fit clusters"):
            clusterer.fit_clusters(sample_documents)
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_assign_documents_to_clusters_success(self, mock_sentence_transformer, mock_bertopic, sample_documents):
        """Test successful document assignment."""
        # Setup mocks
        mock_bertopic_instance = Mock()
        mock_bertopic_instance.fit_transform.return_value = ([0, 1, 0, 1, 0], [[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        mock_bertopic_instance.transform.return_value = ([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        mock_bertopic_instance.get_topic_info.return_value = Mock()
        mock_bertopic_instance.get_topic_info.return_value.iterrows.return_value = [
            (0, {"Topic": 0, "Count": 3}),
            (1, {"Topic": 1, "Count": 2})
        ]
        mock_bertopic_instance.get_topic.return_value = [("machine", 0.5), ("learning", 0.4), ("neural", 0.3)]
        mock_bertopic.return_value = mock_bertopic_instance
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        clusterer.fit_clusters(sample_documents)
        
        # Test assignment
        new_documents = ["New document 1", "New document 2"]
        assignments = clusterer.assign_documents_to_clusters(new_documents)
        
        assert len(assignments) == 2
        assert all(isinstance(assignment, DocumentClusterAssignment) for assignment in assignments)
    
    def test_assign_documents_not_fitted(self, sample_documents):
        """Test assignment when model is not fitted."""
        clusterer = DocumentClusterer()
        
        with pytest.raises(ClusteringError, match="Model must be fitted before assigning documents"):
            clusterer.assign_documents_to_clusters(sample_documents)
    
    def test_assign_documents_no_documents(self):
        """Test assignment with no documents."""
        clusterer = DocumentClusterer()
        clusterer.is_fitted = True  # Mock fitted state
        
        with pytest.raises(InvalidDocumentError, match="No documents provided for assignment"):
            clusterer.assign_documents_to_clusters([])
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_reassign_documents_to_clusters_success(self, mock_sentence_transformer, mock_bertopic, sample_documents):
        """Test successful document reassignment."""
        # Setup mocks
        mock_bertopic_instance = Mock()
        mock_bertopic_instance.fit_transform.return_value = ([0, 1, 0, 1, 0], [[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        mock_bertopic_instance.transform.return_value = ([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        mock_bertopic_instance.get_topic_info.return_value = Mock()
        mock_bertopic_instance.get_topic_info.return_value.iterrows.return_value = [
            (0, {"Topic": 0, "Count": 3}),
            (1, {"Topic": 1, "Count": 2})
        ]
        mock_bertopic_instance.get_topic.return_value = [("machine", 0.5), ("learning", 0.4), ("neural", 0.3)]
        mock_bertopic.return_value = mock_bertopic_instance
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        clusterer.fit_clusters(sample_documents)
        
        # Test reassignment
        new_documents = ["New document 1", "New document 2"]
        document_ids = ["new_doc_1", "new_doc_2"]
        assignments = clusterer.reassign_documents_to_clusters(new_documents, document_ids)
        
        assert len(assignments) == 2
        assert all(isinstance(assignment, DocumentClusterAssignment) for assignment in assignments)
    
    def test_reassign_documents_not_fitted(self, sample_documents):
        """Test reassignment when model is not fitted."""
        clusterer = DocumentClusterer()
        
        with pytest.raises(ClusteringError, match="Model must be fitted before reassigning documents"):
            clusterer.reassign_documents_to_clusters(sample_documents, ["doc_1", "doc_2"])
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_get_cluster_info(self, mock_sentence_transformer, mock_bertopic, sample_documents):
        """Test getting cluster information."""
        # Setup mocks
        mock_bertopic_instance = Mock()
        mock_bertopic_instance.fit_transform.return_value = ([0, 1, 0, 1, 0], [[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        mock_bertopic_instance.get_topic_info.return_value = Mock()
        mock_bertopic_instance.get_topic_info.return_value.iterrows.return_value = [
            (0, {"Topic": 0, "Count": 3}),
            (1, {"Topic": 1, "Count": 2})
        ]
        mock_bertopic_instance.get_topic.return_value = [("machine", 0.5), ("learning", 0.4), ("neural", 0.3)]
        mock_bertopic.return_value = mock_bertopic_instance
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        clusterer.fit_clusters(sample_documents)
        
        # Test getting cluster info
        cluster_info = clusterer.get_cluster_info(0)
        assert cluster_info is not None
        assert cluster_info.cluster_id == 0
        
        # Test non-existent cluster
        cluster_info = clusterer.get_cluster_info(999)
        assert cluster_info is None
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_get_document_assignment(self, mock_sentence_transformer, mock_bertopic, sample_documents, sample_document_ids):
        """Test getting document assignment."""
        # Setup mocks
        mock_bertopic_instance = Mock()
        mock_bertopic_instance.fit_transform.return_value = ([0, 1, 0, 1, 0], [[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        mock_bertopic_instance.get_topic_info.return_value = Mock()
        mock_bertopic_instance.get_topic_info.return_value.iterrows.return_value = [
            (0, {"Topic": 0, "Count": 3}),
            (1, {"Topic": 1, "Count": 2})
        ]
        mock_bertopic_instance.get_topic.return_value = [("machine", 0.5), ("learning", 0.4), ("neural", 0.3)]
        mock_bertopic.return_value = mock_bertopic_instance
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        clusterer.fit_clusters(sample_documents, sample_document_ids)
        
        # Test getting document assignment
        assignment = clusterer.get_document_assignment("doc_0")
        assert assignment is not None
        assert assignment.document_id == "doc_0"
        
        # Test non-existent document
        assignment = clusterer.get_document_assignment("non_existent")
        assert assignment is None
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_get_cluster_documents(self, mock_sentence_transformer, mock_bertopic, sample_documents, sample_document_ids):
        """Test getting cluster documents."""
        # Setup mocks
        mock_bertopic_instance = Mock()
        mock_bertopic_instance.fit_transform.return_value = ([0, 1, 0, 1, 0], [[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        mock_bertopic_instance.get_topic_info.return_value = Mock()
        mock_bertopic_instance.get_topic_info.return_value.iterrows.return_value = [
            (0, {"Topic": 0, "Count": 3}),
            (1, {"Topic": 1, "Count": 2})
        ]
        mock_bertopic_instance.get_topic.return_value = [("machine", 0.5), ("learning", 0.4), ("neural", 0.3)]
        mock_bertopic.return_value = mock_bertopic_instance
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        clusterer.fit_clusters(sample_documents, sample_document_ids)
        
        # Test getting cluster documents
        documents = clusterer.get_cluster_documents(0)
        assert isinstance(documents, list)
        
        # Test non-existent cluster
        documents = clusterer.get_cluster_documents(999)
        assert documents == []
    
    @patch('clustering.clusterer.BERTopic')
    @patch('clustering.clusterer.SentenceTransformer')
    def test_update_cluster_metadata(self, mock_sentence_transformer, mock_bertopic, sample_documents):
        """Test updating cluster metadata."""
        # Setup mocks
        mock_bertopic_instance = Mock()
        mock_bertopic_instance.fit_transform.return_value = ([0, 1, 0, 1, 0], [[0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        mock_bertopic_instance.get_topic_info.return_value = Mock()
        mock_bertopic_instance.get_topic_info.return_value.iterrows.return_value = [
            (0, {"Topic": 0, "Count": 3}),
            (1, {"Topic": 1, "Count": 2})
        ]
        mock_bertopic_instance.get_topic.return_value = [("machine", 0.5), ("learning", 0.4), ("neural", 0.3)]
        mock_bertopic.return_value = mock_bertopic_instance
        mock_sentence_transformer.return_value = Mock()
        
        clusterer = DocumentClusterer()
        clusterer.fit_clusters(sample_documents)
        
        # Test updating metadata
        success = clusterer.update_cluster_metadata(0, new_field="new_value")
        assert success is True
        
        # Test updating non-existent cluster
        success = clusterer.update_cluster_metadata(999, new_field="new_value")
        assert success is False
    
    def test_get_cluster_info_no_result(self):
        """Test getting cluster info when no result exists."""
        clusterer = DocumentClusterer()
        cluster_info = clusterer.get_cluster_info(0)
        assert cluster_info is None
    
    def test_get_document_assignment_no_result(self):
        """Test getting document assignment when no result exists."""
        clusterer = DocumentClusterer()
        assignment = clusterer.get_document_assignment("doc_1")
        assert assignment is None
    
    def test_get_cluster_documents_no_result(self):
        """Test getting cluster documents when no result exists."""
        clusterer = DocumentClusterer()
        documents = clusterer.get_cluster_documents(0)
        assert documents == []
    
    def test_update_cluster_metadata_no_result(self):
        """Test updating cluster metadata when no result exists."""
        clusterer = DocumentClusterer()
        success = clusterer.update_cluster_metadata(0, new_field="value")
        assert success is False


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @patch('clustering.clusterer.DocumentClusterer')
    def test_assign_documents_to_clusters(self, mock_clusterer_class, sample_documents, sample_document_ids):
        """Test assign_documents_to_clusters convenience function."""
        mock_clusterer = Mock()
        mock_clusterer.fit_clusters.return_value = Mock()
        mock_clusterer_class.return_value = mock_clusterer
        
        result = assign_documents_to_clusters(sample_documents, sample_document_ids)
        
        mock_clusterer_class.assert_called_once()
        mock_clusterer.fit_clusters.assert_called_once_with(sample_documents, sample_document_ids)
        assert result == mock_clusterer.fit_clusters.return_value
    
    @patch('clustering.clusterer.DocumentClusterer')
    def test_assign_documents_to_clusters_custom_config(self, mock_clusterer_class, sample_documents):
        """Test assign_documents_to_clusters with custom config."""
        mock_clusterer = Mock()
        mock_clusterer.fit_clusters.return_value = Mock()
        mock_clusterer_class.return_value = mock_clusterer
        
        config = ClusteringConfig(model_name="custom-model")
        result = assign_documents_to_clusters(sample_documents, config=config)
        
        mock_clusterer_class.assert_called_once_with(config)
        mock_clusterer.fit_clusters.assert_called_once_with(sample_documents, None)
    
    def test_reassign_documents_to_clusters(self, sample_documents):
        """Test reassign_documents_to_clusters convenience function."""
        mock_clusterer = Mock()
        mock_clusterer.reassign_documents_to_clusters.return_value = [Mock()]
        
        document_ids = ["doc_1", "doc_2"]
        result = reassign_documents_to_clusters(sample_documents, document_ids, mock_clusterer)
        
        mock_clusterer.reassign_documents_to_clusters.assert_called_once_with(sample_documents, document_ids)
        assert result == mock_clusterer.reassign_documents_to_clusters.return_value
