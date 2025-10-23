"""
Unit tests for clustering models.
"""

import pytest
from datetime import datetime
from clustering.models import ClusterInfo, ClusteringResult, DocumentClusterAssignment


class TestClusterInfo:
    """Test cases for the ClusterInfo model."""
    
    def test_cluster_info_creation(self, sample_cluster_info):
        """Test basic cluster info creation."""
        assert sample_cluster_info.cluster_id == 0
        assert sample_cluster_info.name == "Machine Learning Cluster"
        assert len(sample_cluster_info.topic_words) == 5
        assert sample_cluster_info.document_count == 5
        assert sample_cluster_info.coherence_score == 0.8
        assert sample_cluster_info.silhouette_score == 0.7
    
    def test_cluster_info_default_values(self):
        """Test cluster info with default values."""
        cluster = ClusterInfo(
            cluster_id=1,
            name="Test Cluster",
            topic_words=["test", "cluster"],
            document_count=2
        )
        
        assert cluster.cluster_id == 1
        assert cluster.coherence_score is None
        assert cluster.silhouette_score is None
        assert cluster.metadata == {}
    
    def test_get_topic_summary(self, sample_cluster_info):
        """Test topic summary generation."""
        summary = sample_cluster_info.get_topic_summary(max_words=3)
        assert summary == "machine, learning, neural"
        
        summary_default = sample_cluster_info.get_topic_summary()
        assert summary_default == "machine, learning, neural, network, algorithm"
    
    def test_is_active(self, sample_cluster_info):
        """Test cluster active status."""
        assert sample_cluster_info.is_active() is True
        
        # Test inactive cluster
        inactive_cluster = sample_cluster_info.model_copy(update={"document_count": 0})
        assert inactive_cluster.is_active() is False
    
    def test_update_metadata(self, sample_cluster_info):
        """Test metadata update."""
        original_updated_at = sample_cluster_info.updated_at
        
        updated_cluster = sample_cluster_info.update_metadata(
            new_field="new_value",
            category="updated"
        )
        
        assert updated_cluster.metadata["new_field"] == "new_value"
        assert updated_cluster.metadata["category"] == "updated"
        assert updated_cluster.updated_at > original_updated_at
    
    def test_model_dump(self, sample_cluster_info):
        """Test model serialization."""
        dumped = sample_cluster_info.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["cluster_id"] == 0
        assert dumped["name"] == "Machine Learning Cluster"


class TestDocumentClusterAssignment:
    """Test cases for the DocumentClusterAssignment model."""
    
    def test_assignment_creation(self, sample_document_assignment):
        """Test basic assignment creation."""
        assert sample_document_assignment.document_id == "doc_1"
        assert sample_document_assignment.cluster_id == 0
        assert sample_document_assignment.confidence == 0.85
        assert sample_document_assignment.distance_to_center == 0.3
    
    def test_assignment_default_values(self):
        """Test assignment with default values."""
        assignment = DocumentClusterAssignment(
            document_id="doc_2",
            cluster_id=1,
            confidence=0.7
        )
        
        assert assignment.distance_to_center is None
        assert assignment.assignment_reason is None
        assert assignment.metadata == {}
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence values
        assignment1 = DocumentClusterAssignment(
            document_id="doc_1", cluster_id=0, confidence=0.0
        )
        assignment2 = DocumentClusterAssignment(
            document_id="doc_2", cluster_id=0, confidence=1.0
        )
        assignment3 = DocumentClusterAssignment(
            document_id="doc_3", cluster_id=0, confidence=0.5
        )
        
        assert assignment1.confidence == 0.0
        assert assignment2.confidence == 1.0
        assert assignment3.confidence == 0.5
    
    def test_invalid_confidence(self):
        """Test invalid confidence values."""
        with pytest.raises(ValueError):
            DocumentClusterAssignment(
                document_id="doc_1", cluster_id=0, confidence=-0.1
            )
        
        with pytest.raises(ValueError):
            DocumentClusterAssignment(
                document_id="doc_1", cluster_id=0, confidence=1.1
            )
    
    def test_is_high_confidence(self, sample_document_assignment):
        """Test high confidence check."""
        assert sample_document_assignment.is_high_confidence(threshold=0.8) is True
        assert sample_document_assignment.is_high_confidence(threshold=0.9) is False
    
    def test_update_assignment(self, sample_document_assignment):
        """Test assignment update."""
        original_assigned_at = sample_document_assignment.assigned_at
        
        updated_assignment = sample_document_assignment.update_assignment(
            cluster_id=1,
            confidence=0.9,
            assignment_reason="reassignment"
        )
        
        assert updated_assignment.cluster_id == 1
        assert updated_assignment.confidence == 0.9
        assert updated_assignment.assignment_reason == "reassignment"
        assert updated_assignment.assigned_at > original_assigned_at
    
    def test_model_dump(self, sample_document_assignment):
        """Test model serialization."""
        dumped = sample_document_assignment.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["document_id"] == "doc_1"
        assert dumped["cluster_id"] == 0


class TestClusteringResult:
    """Test cases for the ClusteringResult model."""
    
    def test_clustering_result_creation(self, sample_clustering_result):
        """Test basic clustering result creation."""
        assert len(sample_clustering_result.clusters) == 1
        assert len(sample_clustering_result.assignments) == 1
        assert sample_clustering_result.total_documents == 1
        assert sample_clustering_result.num_clusters == 1
        assert sample_clustering_result.overall_quality_score == 0.8
    
    def test_get_cluster_by_id(self, sample_clustering_result):
        """Test cluster retrieval by ID."""
        cluster = sample_clustering_result.get_cluster_by_id(0)
        assert cluster is not None
        assert cluster.cluster_id == 0
        
        # Test non-existent cluster
        cluster = sample_clustering_result.get_cluster_by_id(999)
        assert cluster is None
    
    def test_get_assignments_for_cluster(self, sample_clustering_result):
        """Test assignment retrieval for cluster."""
        assignments = sample_clustering_result.get_assignments_for_cluster(0)
        assert len(assignments) == 1
        assert assignments[0].cluster_id == 0
        
        # Test non-existent cluster
        assignments = sample_clustering_result.get_assignments_for_cluster(999)
        assert len(assignments) == 0
    
    def test_get_document_assignment(self, sample_clustering_result):
        """Test document assignment retrieval."""
        assignment = sample_clustering_result.get_document_assignment("doc_1")
        assert assignment is not None
        assert assignment.document_id == "doc_1"
        
        # Test non-existent document
        assignment = sample_clustering_result.get_document_assignment("non_existent")
        assert assignment is None
    
    def test_get_cluster_statistics(self, sample_clustering_result):
        """Test cluster statistics generation."""
        stats = sample_clustering_result.get_cluster_statistics()
        
        assert "total_clusters" in stats
        assert "total_documents" in stats
        assert "average_cluster_size" in stats
        assert "min_cluster_size" in stats
        assert "max_cluster_size" in stats
        assert "average_confidence" in stats
        assert "high_confidence_assignments" in stats
        assert "overall_quality_score" in stats
    
    def test_filter_high_quality_clusters(self, sample_clustering_result):
        """Test filtering high-quality clusters."""
        # Create a low-quality cluster
        low_quality_cluster = ClusterInfo(
            cluster_id=1,
            name="Low Quality Cluster",
            topic_words=["test"],
            document_count=1,
            coherence_score=0.1
        )
        
        # Add to clustering result
        updated_result = sample_clustering_result.model_copy(update={
            'clusters': sample_clustering_result.clusters + [low_quality_cluster],
            'num_clusters': 2
        })
        
        # Filter high-quality clusters
        filtered_result = updated_result.filter_high_quality_clusters(
            min_documents=2,
            min_coherence=0.5
        )
        
        assert filtered_result.num_clusters == 1
        assert len(filtered_result.clusters) == 1
        assert filtered_result.clusters[0].cluster_id == 0
    
    def test_model_dump(self, sample_clustering_result):
        """Test model serialization."""
        dumped = sample_clustering_result.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["total_documents"] == 1
        assert dumped["num_clusters"] == 1
        assert len(dumped["clusters"]) == 1
        assert len(dumped["assignments"]) == 1
    
    def test_equality_comparison(self, sample_clustering_result):
        """Test equality comparison."""
        identical = sample_clustering_result.model_copy(deep=True)
        different = sample_clustering_result.model_copy(update={"total_documents": 2})
        
        # Note: Direct equality comparison may not work due to object identity
        # Instead, we can compare the model_dump results
        assert sample_clustering_result.model_dump() == identical.model_dump()
        assert sample_clustering_result.model_dump() != different.model_dump()
    
    def test_merge_similar_clusters(self, sample_clustering_result):
        """Test cluster merging functionality."""
        # This is a placeholder test since merge_similar_clusters
        # would need actual similarity calculation
        merged_result = sample_clustering_result.merge_similar_clusters()
        assert isinstance(merged_result, ClusteringResult)
        assert merged_result.num_clusters == sample_clustering_result.num_clusters
