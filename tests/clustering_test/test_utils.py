"""
Unit tests for clustering utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from clustering.utils import (
    analyze_cluster_quality,
    get_cluster_statistics,
    merge_similar_clusters,
    extract_cluster_topics,
    find_outlier_documents,
    get_cluster_summary,
    validate_clustering_result
)
from clustering.models import ClusterInfo, ClusteringResult, DocumentClusterAssignment


class TestAnalyzeClusterQuality:
    """Test cases for analyze_cluster_quality function."""
    
    def test_analyze_cluster_quality_basic(self, sample_clustering_result):
        """Test basic cluster quality analysis."""
        quality_metrics = analyze_cluster_quality(sample_clustering_result)
        
        assert isinstance(quality_metrics, dict)
        assert "num_clusters" in quality_metrics
        assert "total_documents" in quality_metrics
        assert "overall_quality_score" in quality_metrics
        assert quality_metrics["num_clusters"] == 1
        assert quality_metrics["total_documents"] == 1
    
    def test_analyze_cluster_quality_with_embeddings(self, sample_clustering_result):
        """Test cluster quality analysis with embeddings."""
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        quality_metrics = analyze_cluster_quality(sample_clustering_result, embeddings)
        
        assert isinstance(quality_metrics, dict)
        assert "num_clusters" in quality_metrics
        # Silhouette score might not be calculated if not enough clusters
        if "silhouette_score" in quality_metrics:
            assert isinstance(quality_metrics["silhouette_score"], float)
    
    def test_analyze_cluster_quality_empty_result(self):
        """Test cluster quality analysis with empty result."""
        empty_result = ClusteringResult(
            clusters=[],
            assignments=[],
            total_documents=0,
            num_clusters=0,
            overall_quality_score=None
        )
        
        quality_metrics = analyze_cluster_quality(empty_result)
        
        assert quality_metrics["num_clusters"] == 0
        assert quality_metrics["total_documents"] == 0
        assert quality_metrics["overall_quality_score"] is None
    
    def test_analyze_cluster_quality_error_handling(self):
        """Test error handling in cluster quality analysis."""
        # Create a result that might cause errors
        mock_cluster = Mock()
        mock_cluster.document_count = "invalid"  # This should cause an error
        
        mock_result = Mock()
        mock_result.clusters = [mock_cluster]
        mock_result.assignments = []
        mock_result.num_clusters = 1
        mock_result.total_documents = 1
        mock_result.overall_quality_score = 0.5
        
        quality_metrics = analyze_cluster_quality(mock_result)
        
        # Should handle errors gracefully
        assert "error" in quality_metrics


class TestGetClusterStatistics:
    """Test cases for get_cluster_statistics function."""
    
    def test_get_cluster_statistics_basic(self, sample_clustering_result):
        """Test basic cluster statistics."""
        stats = get_cluster_statistics(sample_clustering_result)
        
        assert isinstance(stats, dict)
        assert "total_clusters" in stats
        assert "total_documents" in stats
        assert "average_cluster_size" in stats
        assert "cluster_details" in stats
        assert "assignment_distribution" in stats
    
    def test_get_cluster_statistics_empty_result(self):
        """Test cluster statistics with empty result."""
        empty_result = ClusteringResult(
            clusters=[],
            assignments=[],
            total_documents=0,
            num_clusters=0
        )
        
        stats = get_cluster_statistics(empty_result)
        
        assert stats["total_clusters"] == 0
        assert stats["total_documents"] == 0
        assert stats["average_cluster_size"] == 0
    
    def test_get_cluster_statistics_error_handling(self):
        """Test error handling in cluster statistics."""
        # Create a result that might cause errors
        mock_result = Mock()
        mock_result.clusters = []
        mock_result.assignments = []
        mock_result.get_cluster_statistics.side_effect = Exception("Test error")
        
        stats = get_cluster_statistics(mock_result)
        
        # Should handle errors gracefully
        assert "error" in stats


class TestMergeSimilarClusters:
    """Test cases for merge_similar_clusters function."""
    
    def test_merge_similar_clusters_no_merge(self, sample_clustering_result):
        """Test merging when no clusters are similar."""
        # Create clusters with very different topics
        cluster1 = ClusterInfo(
            cluster_id=0,
            name="Cluster 1",
            topic_words=["machine", "learning", "ai"],
            document_count=3
        )
        cluster2 = ClusterInfo(
            cluster_id=1,
            name="Cluster 2",
            topic_words=["cooking", "recipe", "food"],
            document_count=2
        )
        
        result = ClusteringResult(
            clusters=[cluster1, cluster2],
            assignments=[
                DocumentClusterAssignment(document_id="doc_1", cluster_id=0, confidence=0.8),
                DocumentClusterAssignment(document_id="doc_2", cluster_id=1, confidence=0.7)
            ],
            total_documents=2,
            num_clusters=2
        )
        
        merged_result = merge_similar_clusters(result, similarity_threshold=0.8)
        
        # Should not merge anything
        assert merged_result.num_clusters == 2
        assert len(merged_result.clusters) == 2
    
    def test_merge_similar_clusters_single_cluster(self):
        """Test merging with single cluster."""
        cluster = ClusterInfo(
            cluster_id=0,
            name="Single Cluster",
            topic_words=["test", "cluster"],
            document_count=1
        )
        
        result = ClusteringResult(
            clusters=[cluster],
            assignments=[
                DocumentClusterAssignment(document_id="doc_1", cluster_id=0, confidence=0.8)
            ],
            total_documents=1,
            num_clusters=1
        )
        
        merged_result = merge_similar_clusters(result)
        
        # Should not change anything
        assert merged_result.num_clusters == 1
        assert len(merged_result.clusters) == 1
    
    def test_merge_similar_clusters_error_handling(self, sample_clustering_result):
        """Test error handling in cluster merging."""
        # Mock the result to cause an error
        with patch('clustering.utils._calculate_cluster_similarity', side_effect=Exception("Test error")):
            merged_result = merge_similar_clusters(sample_clustering_result)
            
            # Should return original result on error
            assert merged_result == sample_clustering_result


class TestExtractClusterTopics:
    """Test cases for extract_cluster_topics function."""
    
    def test_extract_cluster_topics_basic(self, sample_clustering_result):
        """Test basic topic extraction."""
        topics = extract_cluster_topics(sample_clustering_result, top_k=5)
        
        assert isinstance(topics, dict)
        assert 0 in topics  # Cluster 0 should be present
        assert isinstance(topics[0], list)
    
    def test_extract_cluster_topics_empty_result(self):
        """Test topic extraction with empty result."""
        empty_result = ClusteringResult(
            clusters=[],
            assignments=[],
            total_documents=0,
            num_clusters=0
        )
        
        topics = extract_cluster_topics(empty_result)
        
        assert topics == {}
    
    def test_extract_cluster_topics_error_handling(self):
        """Test error handling in topic extraction."""
        # Create a result that might cause errors
        mock_cluster = Mock()
        mock_cluster.cluster_id = 0
        mock_cluster.topic_words = None  # This might cause an error
        
        mock_result = Mock()
        mock_result.clusters = [mock_cluster]
        
        topics = extract_cluster_topics(mock_result)
        
        # Should handle errors gracefully
        assert isinstance(topics, dict)


class TestFindOutlierDocuments:
    """Test cases for find_outlier_documents function."""
    
    def test_find_outlier_documents_basic(self):
        """Test basic outlier detection."""
        result = ClusteringResult(
            clusters=[],
            assignments=[
                DocumentClusterAssignment(document_id="doc_1", cluster_id=0, confidence=0.9),
                DocumentClusterAssignment(document_id="doc_2", cluster_id=0, confidence=0.2),
                DocumentClusterAssignment(document_id="doc_3", cluster_id=0, confidence=0.8)
            ],
            total_documents=3,
            num_clusters=0
        )
        
        outliers = find_outlier_documents(result, confidence_threshold=0.3)
        
        assert "doc_2" in outliers
        assert "doc_1" not in outliers
        assert "doc_3" not in outliers
    
    def test_find_outlier_documents_no_outliers(self):
        """Test outlier detection with no outliers."""
        result = ClusteringResult(
            clusters=[],
            assignments=[
                DocumentClusterAssignment(document_id="doc_1", cluster_id=0, confidence=0.9),
                DocumentClusterAssignment(document_id="doc_2", cluster_id=0, confidence=0.8)
            ],
            total_documents=2,
            num_clusters=0
        )
        
        outliers = find_outlier_documents(result, confidence_threshold=0.3)
        
        assert len(outliers) == 0
    
    def test_find_outlier_documents_error_handling(self):
        """Test error handling in outlier detection."""
        # Create a result that might cause errors
        mock_result = Mock()
        mock_result.assignments = []
        
        outliers = find_outlier_documents(mock_result)
        
        # Should handle errors gracefully
        assert outliers == []


class TestGetClusterSummary:
    """Test cases for get_cluster_summary function."""
    
    def test_get_cluster_summary_basic(self, sample_clustering_result):
        """Test basic cluster summary generation."""
        summary = get_cluster_summary(sample_clustering_result)
        
        assert isinstance(summary, str)
        assert "Clustering Summary" in summary
        assert "Total Documents" in summary
        assert "Number of Clusters" in summary
    
    def test_get_cluster_summary_empty_result(self):
        """Test cluster summary with empty result."""
        empty_result = ClusteringResult(
            clusters=[],
            assignments=[],
            total_documents=0,
            num_clusters=0
        )
        
        summary = get_cluster_summary(empty_result)
        
        assert "Total Documents: 0" in summary
        assert "Number of Clusters: 0" in summary
    
    def test_get_cluster_summary_error_handling(self):
        """Test error handling in cluster summary generation."""
        # Create a result that might cause errors
        mock_result = Mock()
        mock_result.total_documents = "invalid"
        mock_result.num_clusters = "invalid"
        mock_result.overall_quality_score = "invalid"
        mock_result.clusters = []
        
        summary = get_cluster_summary(mock_result)
        
        # Should handle errors gracefully
        assert "Error generating summary" in summary


class TestValidateClusteringResult:
    """Test cases for validate_clustering_result function."""
    
    def test_validate_clustering_result_valid(self, sample_clustering_result):
        """Test validation of valid clustering result."""
        warnings = validate_clustering_result(sample_clustering_result)
        
        assert isinstance(warnings, list)
        # Should have no warnings for a valid result
        assert len(warnings) == 0
    
    def test_validate_clustering_result_empty_clusters(self):
        """Test validation with empty clusters."""
        empty_cluster = ClusterInfo(
            cluster_id=0,
            name="Empty Cluster",
            topic_words=[],
            document_count=0
        )
        
        result = ClusteringResult(
            clusters=[empty_cluster],
            assignments=[],
            total_documents=0,
            num_clusters=1
        )
        
        warnings = validate_clustering_result(result)
        
        assert len(warnings) > 0
        assert any("no documents" in warning.lower() for warning in warnings)
    
    def test_validate_clustering_result_orphaned_assignments(self):
        """Test validation with orphaned assignments."""
        result = ClusteringResult(
            clusters=[],  # No clusters
            assignments=[
                DocumentClusterAssignment(document_id="doc_1", cluster_id=0, confidence=0.8)
            ],
            total_documents=1,
            num_clusters=0
        )
        
        warnings = validate_clustering_result(result)
        
        assert len(warnings) > 0
        assert any("non-existent cluster" in warning.lower() for warning in warnings)
    
    def test_validate_clustering_result_low_confidence(self):
        """Test validation with low confidence assignments."""
        result = ClusteringResult(
            clusters=[],
            assignments=[
                DocumentClusterAssignment(document_id="doc_1", cluster_id=0, confidence=0.1),
                DocumentClusterAssignment(document_id="doc_2", cluster_id=0, confidence=0.2)
            ],
            total_documents=2,
            num_clusters=0
        )
        
        warnings = validate_clustering_result(result)
        
        assert len(warnings) > 0
        assert any("low confidence" in warning.lower() for warning in warnings)
    
    def test_validate_clustering_result_unbalanced_clusters(self):
        """Test validation with unbalanced clusters."""
        cluster1 = ClusterInfo(
            cluster_id=0,
            name="Large Cluster",
            topic_words=["test"],
            document_count=100
        )
        cluster2 = ClusterInfo(
            cluster_id=1,
            name="Small Cluster",
            topic_words=["test"],
            document_count=1
        )
        
        result = ClusteringResult(
            clusters=[cluster1, cluster2],
            assignments=[],
            total_documents=101,
            num_clusters=2
        )
        
        warnings = validate_clustering_result(result)
        
        assert len(warnings) > 0
        assert any("unbalanced" in warning.lower() for warning in warnings)
    
    def test_validate_clustering_result_error_handling(self):
        """Test error handling in validation."""
        # Create a result that might cause errors
        mock_result = Mock()
        mock_result.clusters = []
        mock_result.assignments = []
        
        # Mock the attributes that the validation function actually uses
        mock_result.clusters = []
        mock_result.assignments = []
        
        warnings = validate_clustering_result(mock_result)
        
        # Should handle errors gracefully and return empty warnings for empty result
        assert isinstance(warnings, list)


class TestUtilityHelperFunctions:
    """Test cases for utility helper functions."""
    
    def test_calculate_cluster_similarity_basic(self):
        """Test basic cluster similarity calculation."""
        from clustering.utils import _calculate_cluster_similarity
        
        cluster1 = ClusterInfo(
            cluster_id=0,
            name="Cluster 1",
            topic_words=["machine", "learning", "ai"],
            document_count=3
        )
        cluster2 = ClusterInfo(
            cluster_id=1,
            name="Cluster 2",
            topic_words=["machine", "learning", "neural"],
            document_count=2
        )
        
        similarity = _calculate_cluster_similarity(cluster1, cluster2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.0  # Should have some similarity
    
    def test_calculate_cluster_similarity_no_overlap(self):
        """Test cluster similarity with no word overlap."""
        from clustering.utils import _calculate_cluster_similarity
        
        cluster1 = ClusterInfo(
            cluster_id=0,
            name="Cluster 1",
            topic_words=["machine", "learning"],
            document_count=2
        )
        cluster2 = ClusterInfo(
            cluster_id=1,
            name="Cluster 2",
            topic_words=["cooking", "recipe"],
            document_count=2
        )
        
        similarity = _calculate_cluster_similarity(cluster1, cluster2)
        
        assert similarity == 0.0
    
    def test_calculate_cluster_similarity_empty_words(self):
        """Test cluster similarity with empty topic words."""
        from clustering.utils import _calculate_cluster_similarity
        
        cluster1 = ClusterInfo(
            cluster_id=0,
            name="Cluster 1",
            topic_words=[],
            document_count=1
        )
        cluster2 = ClusterInfo(
            cluster_id=1,
            name="Cluster 2",
            topic_words=["test"],
            document_count=1
        )
        
        similarity = _calculate_cluster_similarity(cluster1, cluster2)
        
        assert similarity == 0.0
    
    def test_calculate_cluster_similarity_error_handling(self):
        """Test error handling in similarity calculation."""
        from clustering.utils import _calculate_cluster_similarity
        
        # Create clusters that might cause errors
        cluster1 = Mock()
        cluster1.topic_words = None  # This should cause an error
        
        cluster2 = ClusterInfo(
            cluster_id=1,
            name="Cluster 2",
            topic_words=["test"],
            document_count=1
        )
        
        similarity = _calculate_cluster_similarity(cluster1, cluster2)
        
        # Should handle errors gracefully
        assert similarity == 0.0
