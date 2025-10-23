"""
Unit tests for the clustering example module.

This module tests the example usage of the clustering functionality,
ensuring that the example code works correctly and demonstrates
proper usage patterns.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from clustering.example import (
    create_sample_documents,
    demonstrate_basic_clustering,
    demonstrate_custom_configuration,
    demonstrate_cluster_analysis,
    demonstrate_document_assignment,
    demonstrate_cluster_merging,
    demonstrate_different_configurations
)


class TestCreateSampleDocuments:
    """Test cases for the create_sample_documents function."""
    
    def test_create_sample_documents_basic(self):
        """Test basic sample document creation."""
        documents = create_sample_documents()
        
        assert isinstance(documents, list)
        assert len(documents) > 0
        assert all(isinstance(doc, str) for doc in documents)
        assert all(len(doc) > 0 for doc in documents)
    
    def test_create_sample_documents_categories(self):
        """Test that documents cover different categories."""
        documents = create_sample_documents()
        
        # Check that we have documents from different categories
        categories = ["AI", "Machine Learning", "Data Science", "Technology"]
        found_categories = []
        
        for doc in documents:
            for category in categories:
                if category.lower() in doc.lower():
                    found_categories.append(category)
                    break
        
        # Should have documents from multiple categories
        assert len(set(found_categories)) >= 2
    
    def test_create_sample_documents_length(self):
        """Test that documents have reasonable lengths."""
        documents = create_sample_documents()
        
        for doc in documents:
            assert len(doc) >= 50  # Minimum reasonable length
            assert len(doc) <= 1000  # Maximum reasonable length


class TestDemonstrateBasicClustering:
    """Test cases for the demonstrate_basic_clustering function."""
    
    @patch('clustering.example.DocumentClusterer')
    def test_demonstrate_basic_clustering_success(self, mock_clusterer_class):
        """Test successful basic clustering demonstration."""
        # Mock the clusterer instance
        mock_clusterer = Mock()
        mock_clusterer_class.return_value = mock_clusterer
        
        # Mock the clustering result
        mock_result = Mock()
        mock_result.num_clusters = 3
        mock_result.total_documents = 10
        mock_result.overall_quality_score = 0.85
        mock_clusterer.fit_clusters.return_value = mock_result
        
        # Mock the summary function
        with patch('clustering.example.get_cluster_summary') as mock_summary:
            mock_summary.return_value = "Mock summary"
            
            result = demonstrate_basic_clustering()
            
            # Verify the clusterer was called
            mock_clusterer.fit_clusters.assert_called_once()
            mock_summary.assert_called_once_with(mock_result)
            
            # Verify the result
            assert result == mock_result
    
    @patch('clustering.example.DocumentClusterer')
    def test_demonstrate_basic_clustering_error(self, mock_clusterer_class):
        """Test basic clustering demonstration with error."""
        # Mock the clusterer to raise an exception
        mock_clusterer_class.side_effect = Exception("Clustering failed")
        
        with pytest.raises(Exception, match="Clustering failed"):
            demonstrate_basic_clustering()


class TestDemonstrateCustomConfiguration:
    """Test cases for the demonstrate_custom_configuration function."""
    
    @patch('clustering.example.DocumentClusterer')
    def test_demonstrate_custom_configuration_success(self, mock_clusterer_class):
        """Test successful custom configuration demonstration."""
        # Mock the clusterer instance
        mock_clusterer = Mock()
        mock_clusterer_class.return_value = mock_clusterer
        
        # Mock the clustering result
        mock_result = Mock()
        mock_result.num_clusters = 4
        mock_result.total_documents = 15
        mock_result.overall_quality_score = 0.88
        mock_clusterer.fit_clusters.return_value = mock_result
        
        # Mock the summary function
        with patch('clustering.example.get_cluster_summary') as mock_summary:
            mock_summary.return_value = "Mock custom config summary"
            
            result = demonstrate_custom_configuration()
            
            # Verify the clusterer was called with custom config
            mock_clusterer.fit_clusters.assert_called_once()
            mock_summary.assert_called_once_with(mock_result)
            
            # Verify the result
            assert result == mock_result
    
    @patch('clustering.example.DocumentClusterer')
    def test_demonstrate_custom_configuration_error(self, mock_clusterer_class):
        """Test custom configuration demonstration with error."""
        # Mock the clusterer to raise an exception
        mock_clusterer_class.side_effect = Exception("Custom configuration failed")
        
        with pytest.raises(Exception, match="Custom configuration failed"):
            demonstrate_custom_configuration()


class TestDemonstrateClusterAnalysis:
    """Test cases for the demonstrate_cluster_analysis function."""
    
    def test_demonstrate_cluster_analysis_success(self):
        """Test successful cluster analysis demonstration."""
        # Mock clustering result
        mock_result = Mock()
        mock_result.clusters = [Mock(), Mock()]
        mock_result.assignments = [Mock(), Mock()]
        
        # Mock the analysis function
        with patch('clustering.example.analyze_cluster_quality') as mock_analysis:
            mock_analysis.return_value = {"silhouette_score": 0.8, "calinski_harabasz_score": 150.0}
            
            result = demonstrate_cluster_analysis(mock_result)
            
            # Verify the analysis was called
            mock_analysis.assert_called_once_with(mock_result)
            
            # Verify the result
            assert result == mock_analysis.return_value
    
    def test_demonstrate_cluster_analysis_error(self):
        """Test cluster analysis demonstration with error."""
        # Mock clustering result
        mock_result = Mock()
        
        # Mock the analysis function to raise an exception
        with patch('clustering.example.analyze_cluster_quality') as mock_analysis:
            mock_analysis.side_effect = Exception("Analysis failed")
            
            with pytest.raises(Exception, match="Analysis failed"):
                demonstrate_cluster_analysis(mock_result)


class TestDemonstrateDocumentAssignment:
    """Test cases for the demonstrate_document_assignment function."""
    
    @patch('clustering.example.assign_documents_to_clusters')
    def test_demonstrate_document_assignment_success(self, mock_assign):
        """Test successful document assignment demonstration."""
        # Mock the assignment result
        mock_result = Mock()
        mock_result.num_clusters = 3
        mock_result.total_documents = 10
        mock_assign.return_value = mock_result
        
        # Mock the summary function
        with patch('clustering.example.get_cluster_summary') as mock_summary:
            mock_summary.return_value = "Mock assignment summary"
            
            result = demonstrate_document_assignment()
            
            # Verify the assignment was called
            mock_assign.assert_called_once()
            mock_summary.assert_called_once_with(mock_result)
            
            # Verify the result
            assert result == mock_result
    
    @patch('clustering.example.assign_documents_to_clusters')
    def test_demonstrate_document_assignment_error(self, mock_assign):
        """Test document assignment demonstration with error."""
        # Mock the assignment to raise an exception
        mock_assign.side_effect = Exception("Assignment failed")
        
        with pytest.raises(Exception, match="Assignment failed"):
            demonstrate_document_assignment()


class TestDemonstrateClusterMerging:
    """Test cases for the demonstrate_cluster_merging function."""
    
    def test_demonstrate_cluster_merging_success(self):
        """Test successful cluster merging demonstration."""
        # Mock clustering result
        mock_result = Mock()
        mock_result.clusters = [Mock(), Mock()]
        mock_result.assignments = [Mock(), Mock()]
        
        # Mock the merging function
        with patch('clustering.example.merge_similar_clusters') as mock_merge:
            mock_merge.return_value = mock_result
            
            result = demonstrate_cluster_merging()
            
            # Verify the merging was called
            mock_merge.assert_called_once_with(mock_result)
            
            # Verify the result
            assert result == mock_result
    
    def test_demonstrate_cluster_merging_error(self):
        """Test cluster merging demonstration with error."""
        # Mock clustering result
        mock_result = Mock()
        
        # Mock the merging function to raise an exception
        with patch('clustering.example.merge_similar_clusters') as mock_merge:
            mock_merge.side_effect = Exception("Merging failed")
            
            with pytest.raises(Exception, match="Merging failed"):
                demonstrate_cluster_merging()


class TestDemonstrateDifferentConfigurations:
    """Test cases for the demonstrate_different_configurations function."""
    
    def test_demonstrate_different_configurations_success(self):
        """Test successful different configurations demonstration."""
        # Mock the clustering results
        mock_result1 = Mock()
        mock_result1.num_clusters = 3
        mock_result1.overall_quality_score = 0.8
        
        mock_result2 = Mock()
        mock_result2.num_clusters = 5
        mock_result2.overall_quality_score = 0.9
        
        # Mock the clustering function
        with patch('clustering.example.DocumentClusterer') as mock_clusterer_class:
            mock_clusterer = Mock()
            mock_clusterer_class.return_value = mock_clusterer
            mock_clusterer.fit_clusters.side_effect = [mock_result1, mock_result2]
            
            result = demonstrate_different_configurations()
            
            # Verify the clusterer was called twice
            assert mock_clusterer.fit_clusters.call_count == 2
            
            # Verify the result
            assert result == [mock_result1, mock_result2]
    
    def test_demonstrate_different_configurations_error(self):
        """Test different configurations demonstration with error."""
        # Mock the clustering function to raise an exception
        with patch('clustering.example.DocumentClusterer') as mock_clusterer_class:
            mock_clusterer_class.side_effect = Exception("Configuration failed")
            
            with pytest.raises(Exception, match="Configuration failed"):
                demonstrate_different_configurations()


class TestExampleIntegration:
    """Integration tests for the example module."""
    
    def test_create_sample_documents_integration(self):
        """Test that sample documents work with clustering."""
        documents = create_sample_documents()
        
        # Verify documents are suitable for clustering
        assert len(documents) >= 5  # Should have enough documents
        assert all(len(doc) > 20 for doc in documents)  # Should have meaningful content
        
        # Verify documents have different content (for clustering diversity)
        unique_docs = set(documents)
        assert len(unique_docs) == len(documents)  # All documents should be unique
    
    @patch('clustering.example.DocumentClusterer')
    def test_end_to_end_demonstration(self, mock_clusterer_class):
        """Test end-to-end demonstration flow."""
        # Mock the clusterer instance
        mock_clusterer = Mock()
        mock_clusterer_class.return_value = mock_clusterer
        
        # Mock the clustering result
        mock_result = Mock()
        mock_result.num_clusters = 3
        mock_result.total_documents = 10
        mock_result.overall_quality_score = 0.85
        mock_clusterer.fit_clusters.return_value = mock_result
        
        # Mock the analysis function
        with patch('clustering.example.analyze_cluster_quality') as mock_analysis:
            mock_analysis.return_value = {"silhouette_score": 0.8}
            
            # Mock the summary function
            with patch('clustering.example.get_cluster_summary') as mock_summary:
                mock_summary.return_value = "Mock summary"
                
                # Test basic clustering
                result = demonstrate_basic_clustering()
                assert result == mock_result
                
                # Test advanced clustering
                result = demonstrate_advanced_clustering()
                assert result == mock_result
                
                # Test cluster analysis
                analysis_result = demonstrate_cluster_analysis(mock_result)
                assert analysis_result == mock_analysis.return_value
