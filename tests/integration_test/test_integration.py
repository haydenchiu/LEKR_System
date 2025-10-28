"""
Tests for the LERK System Integration Layer

This module tests the clean integration between clustering, consolidation,
and retriever modules.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import the integration components
from integration import (
    LERKIntegrationManager,
    create_integration_manager,
    process_documents_with_integration,
    ClusteringIntegrationInterface,
    ConsolidationIntegrationInterface,
    RetrieverIntegrationInterface
)


class TestLERKIntegrationManager:
    """Test the main integration manager."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        manager = LERKIntegrationManager()
        
        assert manager.clustering_manager is None
        assert manager.consolidation_manager is None
        assert manager.retriever_manager is None
        assert manager._integration_enabled is True
        assert manager._last_integration_update is None
    
    def test_init_with_managers(self):
        """Test initialization with manager instances."""
        mock_clustering = Mock()
        mock_consolidation = Mock()
        mock_retriever = Mock()
        
        manager = LERKIntegrationManager(
            clustering_manager=mock_clustering,
            consolidation_manager=mock_consolidation,
            retriever_manager=mock_retriever
        )
        
        assert manager.clustering_manager == mock_clustering
        assert manager.consolidation_manager == mock_consolidation
        assert manager.retriever_manager == mock_retriever
    
    def test_get_integration_status_default(self):
        """Test getting integration status with default state."""
        manager = LERKIntegrationManager()
        
        status = manager.get_integration_status()
        
        assert status["integration_enabled"] is True
        assert status["last_update"] is None
        assert status["managers_available"]["clustering"] is False
        assert status["managers_available"]["consolidation"] is False
        assert status["managers_available"]["retriever"] is False
        assert status["integration_health"] == "unhealthy"
    
    def test_get_integration_status_with_managers(self):
        """Test getting integration status with all managers."""
        manager = LERKIntegrationManager(
            clustering_manager=Mock(),
            consolidation_manager=Mock(),
            retriever_manager=Mock()
        )
        
        status = manager.get_integration_status()
        
        assert status["integration_enabled"] is True
        assert status["managers_available"]["clustering"] is True
        assert status["managers_available"]["consolidation"] is True
        assert status["managers_available"]["retriever"] is True
        assert status["integration_health"] == "healthy"
    
    def test_get_integration_status_degraded(self):
        """Test getting integration status with partial managers."""
        manager = LERKIntegrationManager(
            clustering_manager=Mock(),
            consolidation_manager=Mock()
        )
        
        status = manager.get_integration_status()
        
        assert status["managers_available"]["clustering"] is True
        assert status["managers_available"]["consolidation"] is True
        assert status["managers_available"]["retriever"] is False
        assert status["integration_health"] == "degraded"
    
    def test_enable_disable_integration(self):
        """Test enabling and disabling integration."""
        manager = LERKIntegrationManager()
        
        assert manager._integration_enabled is True
        
        manager.disable_integration()
        assert manager._integration_enabled is False
        
        manager.enable_integration()
        assert manager._integration_enabled is True
    
    @pytest.mark.asyncio
    async def test_process_new_documents_integrated_success(self):
        """Test successful integrated processing."""
        # Create mock managers
        mock_clustering = AsyncMock()
        mock_consolidation = AsyncMock()
        mock_retriever = AsyncMock()
        
        # Configure mock returns
        mock_clustering.process_new_documents.return_value = {
            "success": True,
            "strategy": "incremental",
            "clustering_changes": {"new_clusters": 1, "updated_clusters": 2}
        }
        
        mock_consolidation.handle_clustering_update.return_value = {
            "updated_subjects": [{"subject_id": "subject_1"}],
            "new_subjects": [{"subject_id": "subject_2"}],
            "deleted_subjects": []
        }
        
        mock_retriever._notify_retriever_of_updates.return_value = {
            "success": True,
            "cache_updated": True
        }
        
        # Create integration manager
        manager = LERKIntegrationManager(
            clustering_manager=mock_clustering,
            consolidation_manager=mock_consolidation,
            retriever_manager=mock_retriever
        )
        
        # Test data
        new_documents = [{"document_id": "doc1", "content": "Test"}]
        document_knowledge_map = {"doc1": {"concepts": ["test"]}}
        
        # Process documents
        results = await manager.process_new_documents_integrated(
            new_documents, document_knowledge_map, "auto"
        )
        
        # Verify results
        assert results["success"] is True
        assert results["clustering_results"]["success"] is True
        assert len(results["consolidation_results"]["updated_subjects"]) == 1
        assert len(results["consolidation_results"]["new_subjects"]) == 1
        assert results["retriever_notification"]["success"] is True
        
        # Verify method calls
        mock_clustering.process_new_documents.assert_called_once()
        mock_consolidation.handle_clustering_update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_new_documents_integrated_clustering_failure(self):
        """Test integrated processing with clustering failure."""
        # Create mock managers
        mock_clustering = AsyncMock()
        
        # Configure clustering failure
        mock_clustering.process_new_documents.return_value = {
            "success": False,
            "error": "Clustering failed"
        }
        
        # Create integration manager
        manager = LERKIntegrationManager(clustering_manager=mock_clustering)
        
        # Test data
        new_documents = [{"document_id": "doc1", "content": "Test"}]
        document_knowledge_map = {"doc1": {"concepts": ["test"]}}
        
        # Process documents
        results = await manager.process_new_documents_integrated(
            new_documents, document_knowledge_map, "auto"
        )
        
        # Verify results
        assert results["success"] is False
        assert "Clustering processing failed" in results["errors"]
    
    @pytest.mark.asyncio
    async def test_process_new_documents_integrated_missing_managers(self):
        """Test integrated processing with missing managers."""
        # Create integration manager without managers
        manager = LERKIntegrationManager()
        
        # Test data
        new_documents = [{"document_id": "doc1", "content": "Test"}]
        document_knowledge_map = {"doc1": {"concepts": ["test"]}}
        
        # Process documents
        results = await manager.process_new_documents_integrated(
            new_documents, document_knowledge_map, "auto"
        )
        
        # Verify results
        assert results["success"] is False
        assert "Clustering manager not available" in results["errors"]


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_integration_manager(self):
        """Test create_integration_manager function."""
        mock_clustering = Mock()
        mock_consolidation = Mock()
        mock_retriever = Mock()
        
        manager = create_integration_manager(
            clustering_manager=mock_clustering,
            consolidation_manager=mock_consolidation,
            retriever_manager=mock_retriever
        )
        
        assert isinstance(manager, LERKIntegrationManager)
        assert manager.clustering_manager == mock_clustering
        assert manager.consolidation_manager == mock_consolidation
        assert manager.retriever_manager == mock_retriever
    
    @pytest.mark.asyncio
    async def test_process_documents_with_integration(self):
        """Test process_documents_with_integration function."""
        # Mock managers
        mock_clustering = AsyncMock()
        mock_clustering.process_new_documents.return_value = {
            "success": True,
            "strategy": "incremental"
        }
        
        mock_consolidation = AsyncMock()
        mock_consolidation.handle_clustering_update.return_value = {
            "updated_subjects": [],
            "new_subjects": []
        }
        
        # Test data
        new_documents = [{"document_id": "doc1", "content": "Test"}]
        document_knowledge_map = {"doc1": {"concepts": ["test"]}}
        
        # Process documents
        results = await process_documents_with_integration(
            new_documents=new_documents,
            document_knowledge_map=document_knowledge_map,
            clustering_manager=mock_clustering,
            consolidation_manager=mock_consolidation,
            update_strategy="auto"
        )
        
        # Verify results
        assert results["success"] is True
        assert results["clustering_results"]["success"] is True


class TestIntegrationInterfaces:
    """Test module integration interfaces."""
    
    @pytest.mark.asyncio
    async def test_clustering_integration_interface_success(self):
        """Test successful clustering interface."""
        with patch('integration.DynamicClusteringManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.process_new_documents.return_value = {
                "success": True,
                "strategy": "incremental"
            }
            mock_manager_class.return_value = mock_manager
            
            results = await ClusteringIntegrationInterface.process_documents_for_clustering(
                [{"document_id": "doc1", "content": "Test"}], "auto"
            )
            
            assert results["success"] is True
            assert results["strategy"] == "incremental"
    
    @pytest.mark.asyncio
    async def test_clustering_integration_interface_import_error(self):
        """Test clustering interface with import error."""
        with patch('integration.DynamicClusteringManager', side_effect=ImportError):
            results = await ClusteringIntegrationInterface.process_documents_for_clustering(
                [{"document_id": "doc1", "content": "Test"}], "auto"
            )
            
            assert results["success"] is False
            assert "Clustering module not available" in results["error"]
    
    @pytest.mark.asyncio
    async def test_consolidation_integration_interface_success(self):
        """Test successful consolidation interface."""
        with patch('integration.ClusterBasedSubjectConsolidator') as mock_consolidator_class:
            mock_consolidator = AsyncMock()
            mock_consolidator.handle_clustering_update.return_value = {
                "updated_subjects": [{"subject_id": "subject_1"}],
                "new_subjects": []
            }
            mock_consolidator_class.return_value = mock_consolidator
            
            clustering_results = {"success": True, "updated_clusters": []}
            document_knowledge_map = {"doc1": {"concepts": ["test"]}}
            
            results = await ConsolidationIntegrationInterface.handle_clustering_updates(
                clustering_results, document_knowledge_map
            )
            
            assert not results.get("error")
            assert len(results["updated_subjects"]) == 1
    
    @pytest.mark.asyncio
    async def test_consolidation_integration_interface_import_error(self):
        """Test consolidation interface with import error."""
        with patch('integration.ClusterBasedSubjectConsolidator', side_effect=ImportError):
            clustering_results = {"success": True}
            document_knowledge_map = {"doc1": {"concepts": ["test"]}}
            
            results = await ConsolidationIntegrationInterface.handle_clustering_updates(
                clustering_results, document_knowledge_map
            )
            
            assert results["success"] is False
            assert "Consolidation module not available" in results["error"]
    
    def test_retriever_integration_interface_success(self):
        """Test successful retriever interface."""
        with patch('integration.search_subject_knowledge') as mock_search:
            mock_search.return_value = [
                {"content": "Test result", "metadata": {"similarity": 0.9}}
            ]
            
            results = RetrieverIntegrationInterface.search_subject_knowledge(
                "test query", {"filter": "value"}
            )
            
            assert len(results) == 1
            assert results[0]["content"] == "Test result"
    
    def test_retriever_integration_interface_import_error(self):
        """Test retriever interface with import error."""
        with patch('integration.search_subject_knowledge', side_effect=ImportError):
            results = RetrieverIntegrationInterface.search_subject_knowledge(
                "test query"
            )
            
            assert results == []


class TestIntegrationErrorHandling:
    """Test error handling in integration."""
    
    @pytest.mark.asyncio
    async def test_integration_manager_exception_handling(self):
        """Test exception handling in integration manager."""
        # Create mock that raises exception
        mock_clustering = AsyncMock()
        mock_clustering.process_new_documents.side_effect = Exception("Test error")
        
        manager = LERKIntegrationManager(clustering_manager=mock_clustering)
        
        # Test data
        new_documents = [{"document_id": "doc1", "content": "Test"}]
        document_knowledge_map = {"doc1": {"concepts": ["test"]}}
        
        # Process documents
        results = await manager.process_new_documents_integrated(
            new_documents, document_knowledge_map, "auto"
        )
        
        # Verify error handling
        assert results["success"] is False
        assert "Test error" in results["error"]
    
    def test_integration_status_error_handling(self):
        """Test error handling in integration status."""
        manager = LERKIntegrationManager()
        
        # Mock _assess_integration_health to raise exception
        with patch.object(manager, '_assess_integration_health', side_effect=Exception("Test error")):
            status = manager.get_integration_status()
            
            assert "error" in status
            assert "Test error" in status["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
