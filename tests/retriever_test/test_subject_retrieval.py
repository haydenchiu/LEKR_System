"""
Comprehensive tests for subject-level retrieval functionality.

This module tests the SubjectRetriever, MultiLevelSearchOrchestrator,
DiscoveryServices, and DynamicClusteringManager components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any, List

# Import the components to test
from retriever.subject_retriever import SubjectRetriever, create_subject_retriever, search_subject_knowledge
from retriever.discovery_service import ClusterDiscoveryService, DocumentDiscoveryService
# Import dynamic clustering from clustering module
from clustering.dynamic_clustering import DynamicClusteringManager, process_new_documents_async
from retriever.multi_level_orchestrator import MultiLevelSearchOrchestrator, create_multi_level_orchestrator
from retriever.config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG
from retriever.exceptions import RetrievalError, InvalidQueryError


class TestSubjectRetriever:
    """Test the SubjectRetriever class."""
    
    def test_init_default(self):
        """Test SubjectRetriever initialization with default config."""
        retriever = SubjectRetriever()
        assert retriever.config is not None
        assert retriever._subject_maturity_threshold == 0.7
        assert retriever._fallback_threshold == 3
    
    def test_init_with_config(self):
        """Test SubjectRetriever initialization with custom config."""
        config = RetrieverConfig(max_results=10)
        retriever = SubjectRetriever(config)
        assert retriever.config.max_results == 10
    
    def test_classify_query_intent_subject(self):
        """Test query intent classification for subject-level queries."""
        retriever = SubjectRetriever()
        
        # Subject-level query
        intent = retriever._classify_query_intent("What are the main concepts in machine learning?")
        assert intent["search_level"] == "subject"
        assert intent["subject_focus"] == True
        assert intent["discovery_query"] == False
    
    def test_classify_query_intent_discovery(self):
        """Test query intent classification for discovery queries."""
        retriever = SubjectRetriever()
        
        # Discovery query
        intent = retriever._classify_query_intent("What topics are available in the system?")
        assert intent["search_level"] == "discovery"
        assert intent["discovery_query"] == True
        assert intent["subject_focus"] == False
    
    def test_classify_query_intent_comparison(self):
        """Test query intent classification for comparison queries."""
        retriever = SubjectRetriever()
        
        # Comparison query
        intent = retriever._classify_query_intent("Compare machine learning and deep learning")
        assert intent["search_level"] == "comparison"
        assert intent["comparison_query"] == True
        assert intent["requires_comparison"] == True
    
    def test_extract_target_clusters(self):
        """Test target cluster extraction from queries."""
        retriever = SubjectRetriever()
        
        # Query with ML keywords
        clusters = retriever._extract_target_clusters("What are machine learning algorithms?")
        assert "ml_cluster" in clusters
        assert "ai_cluster" in clusters
    
    def test_extract_target_clusters_no_match(self):
        """Test target cluster extraction with no matches."""
        retriever = SubjectRetriever()
        
        # Query without cluster keywords
        clusters = retriever._extract_target_clusters("What is the weather like?")
        assert clusters == []
    
    @patch('retriever.subject_retriever.SubjectRetriever._initialize')
    def test_get_relevant_documents_empty_query(self, mock_init):
        """Test get_relevant_documents with empty query."""
        retriever = SubjectRetriever()
        
        with pytest.raises(InvalidQueryError):
            retriever.get_relevant_documents("")
    
    @patch('retriever.subject_retriever.SubjectRetriever._initialize')
    def test_get_relevant_documents_whitespace_query(self, mock_init):
        """Test get_relevant_documents with whitespace-only query."""
        retriever = SubjectRetriever()
        
        with pytest.raises(InvalidQueryError):
            retriever.get_relevant_documents("   ")
    
    @patch('retriever.subject_retriever.SubjectRetriever._initialize')
    def test_get_relevant_documents_success(self, mock_init):
        """Test successful document retrieval."""
        retriever = SubjectRetriever()
        
        # Mock the retrieval process
        with patch.object(retriever, '_retrieve_documents') as mock_retrieve:
            mock_doc = Mock()
            mock_doc.metadata = {"similarity_score": 0.8, "result_type": "subject_knowledge"}
            mock_retrieve.return_value = [mock_doc]
            
            results = retriever.get_relevant_documents("test query")
            assert len(results) == 1
            assert results[0].metadata["similarity_score"] == 0.8
    
    def test_filter_mature_subject_knowledge(self):
        """Test filtering of mature subject knowledge."""
        retriever = SubjectRetriever()
        
        # Create mock documents with different maturity levels
        mature_doc = Mock()
        mature_doc.metadata = {
            "quality_score": 0.8,
            "core_concepts": [1, 2, 3, 4],  # 4 concepts
            "document_sources": ["doc1", "doc2", "doc3"]  # 3 sources
        }
        
        immature_doc = Mock()
        immature_doc.metadata = {
            "quality_score": 0.5,  # Below threshold
            "core_concepts": [1],  # Only 1 concept
            "document_sources": ["doc1"]  # Only 1 source
        }
        
        results = [mature_doc, immature_doc]
        filtered_results = retriever._filter_mature_subject_knowledge(results)
        
        assert len(filtered_results) == 1
        assert filtered_results[0] == mature_doc
    
    def test_get_available_clusters(self):
        """Test getting available clusters."""
        retriever = SubjectRetriever()
        
        # Mock cluster metadata
        retriever._cluster_metadata = {
            "clusters": {"cluster1": {"label": "Test Cluster"}},
            "cluster_count": 1,
            "last_updated": "2024-01-01T00:00:00Z"
        }
        
        clusters_info = retriever.get_available_clusters()
        assert clusters_info["cluster_count"] == 1
        assert "cluster1" in clusters_info["clusters"]
    
    def test_get_available_documents(self):
        """Test getting available documents."""
        retriever = SubjectRetriever()
        
        documents_info = retriever.get_available_documents()
        assert "documents" in documents_info
        assert "document_count" in documents_info


class TestClusterDiscoveryService:
    """Test the ClusterDiscoveryService class."""
    
    def test_init(self):
        """Test ClusterDiscoveryService initialization."""
        service = ClusterDiscoveryService()
        assert service._cluster_cache == {}
        assert service._cache_ttl == 300
    
    def test_get_all_clusters_mock_data(self):
        """Test getting all clusters with mock data."""
        service = ClusterDiscoveryService()
        
        clusters_info = service.get_all_clusters()
        assert "clusters" in clusters_info
        assert "cluster_count" in clusters_info
        assert "last_updated" in clusters_info
    
    def test_get_cluster_details_existing(self):
        """Test getting details for existing cluster."""
        service = ClusterDiscoveryService()
        
        details = service.get_cluster_details("ml_cluster")
        assert details["cluster_id"] == "ml_cluster"
        assert details["exists"] == True
        assert "details" in details
    
    def test_get_cluster_details_nonexistent(self):
        """Test getting details for non-existent cluster."""
        service = ClusterDiscoveryService()
        
        details = service.get_cluster_details("nonexistent_cluster")
        assert details["cluster_id"] == "nonexistent_cluster"
        assert details["exists"] == False
    
    def test_search_clusters_by_topic(self):
        """Test searching clusters by topic."""
        service = ClusterDiscoveryService()
        
        results = service.search_clusters_by_topic("machine learning", limit=5)
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_get_cluster_statistics(self):
        """Test getting cluster statistics."""
        service = ClusterDiscoveryService()
        
        stats = service.get_cluster_statistics()
        assert "total_clusters" in stats
        assert "total_documents" in stats
        assert "last_updated" in stats


class TestDocumentDiscoveryService:
    """Test the DocumentDiscoveryService class."""
    
    def test_init(self):
        """Test DocumentDiscoveryService initialization."""
        service = DocumentDiscoveryService()
        assert service._document_cache == {}
        assert service._cache_ttl == 300
    
    def test_get_all_documents_mock_data(self):
        """Test getting all documents with mock data."""
        service = DocumentDiscoveryService()
        
        documents_info = service.get_all_documents()
        assert "documents" in documents_info
        assert "document_count" in documents_info
        assert "last_updated" in documents_info
    
    def test_get_document_details_existing(self):
        """Test getting details for existing document."""
        service = DocumentDiscoveryService()
        
        details = service.get_document_details("doc_001")
        assert details["document_id"] == "doc_001"
        assert details["exists"] == True
        assert "details" in details
    
    def test_get_document_details_nonexistent(self):
        """Test getting details for non-existent document."""
        service = DocumentDiscoveryService()
        
        details = service.get_document_details("nonexistent_doc")
        assert details["document_id"] == "nonexistent_doc"
        assert details["exists"] == False
    
    def test_search_documents_by_content(self):
        """Test searching documents by content."""
        service = DocumentDiscoveryService()
        
        results = service.search_documents_by_content("machine learning", limit=5)
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_get_documents_by_cluster(self):
        """Test getting documents by cluster."""
        service = DocumentDiscoveryService()
        
        results = service.get_documents_by_cluster("ml_cluster")
        assert isinstance(results, list)


class TestDynamicClusteringManager:
    """Test the DynamicClusteringManager class."""
    
    def test_init_default(self):
        """Test DynamicClusteringManager initialization with default config."""
        manager = DynamicClusteringManager()
        assert manager.new_document_threshold == 5
        assert manager.similarity_threshold == 0.7
        assert manager.cluster_stability_threshold == 0.8
    
    def test_init_with_config(self):
        """Test DynamicClusteringManager initialization with custom config."""
        config = {"min_cluster_size": 5, "max_cluster_size": 100}
        manager = DynamicClusteringManager(clustering_config=config)
        assert manager.clustering_config["min_cluster_size"] == 5
    
    def test_get_default_clustering_config(self):
        """Test getting default clustering configuration."""
        manager = DynamicClusteringManager()
        config = manager._get_default_clustering_config()
        assert "min_cluster_size" in config
        assert "max_cluster_size" in config
        assert "similarity_threshold" in config
    
    def test_analyze_new_documents_small_batch(self):
        """Test analyzing small batch of new documents."""
        manager = DynamicClusteringManager()
        
        new_documents = [
            {"document_id": "doc1", "file_type": "pdf"},
            {"document_id": "doc2", "file_type": "pdf"}
        ]
        
        # Run async method
        analysis = asyncio.run(manager._analyze_new_documents(new_documents))
        
        assert analysis["total_documents"] == 2
        assert analysis["clustering_impact"] == "low"
        assert analysis["recommended_strategy"] == "incremental"
    
    def test_analyze_new_documents_large_batch(self):
        """Test analyzing large batch of new documents."""
        manager = DynamicClusteringManager()
        
        new_documents = [
            {"document_id": f"doc{i}", "file_type": "pdf"} 
            for i in range(10)  # 10 documents
        ]
        
        # Run async method
        analysis = asyncio.run(manager._analyze_new_documents(new_documents))
        
        assert analysis["total_documents"] == 10
        assert analysis["clustering_impact"] == "high"
        assert analysis["recommended_strategy"] == "full_reclustering"
    
    def test_determine_update_strategy(self):
        """Test determining update strategy based on analysis."""
        manager = DynamicClusteringManager()
        
        # High impact analysis
        high_impact = {"clustering_impact": "high"}
        strategy = manager._determine_update_strategy(high_impact)
        assert strategy == "full_reclustering"
        
        # Medium impact analysis
        medium_impact = {"clustering_impact": "medium"}
        strategy = manager._determine_update_strategy(medium_impact)
        assert strategy == "hybrid"
        
        # Low impact analysis
        low_impact = {"clustering_impact": "low"}
        strategy = manager._determine_update_strategy(low_impact)
        assert strategy == "incremental"
    
    def test_calculate_document_similarity(self):
        """Test calculating document similarity."""
        manager = DynamicClusteringManager()
        
        documents = [
            {"document_id": "doc1", "content": "Machine learning is great"},
            {"document_id": "doc2", "content": "Deep learning is awesome"},
            {"document_id": "doc3", "content": "AI is the future"}
        ]
        
        similarities = manager._calculate_document_similarity(documents)
        assert isinstance(similarities, list)
        assert len(similarities) == 3  # 3 pairs from 3 documents
    
    @pytest.mark.asyncio
    async def test_process_new_documents_incremental(self):
        """Test processing new documents with incremental strategy."""
        manager = DynamicClusteringManager()
        
        new_documents = [
            {"document_id": "doc1", "content": "Test content"}
        ]
        
        # Mock the async methods
        with patch.object(manager, '_analyze_new_documents') as mock_analyze, \
             patch.object(manager, '_execute_update_strategy') as mock_execute, \
             patch.object(manager, '_update_vector_store') as mock_vector:
            
            mock_analyze.return_value = {"clustering_impact": "low"}
            mock_execute.return_value = {"success": True, "strategy": "incremental"}
            mock_vector.return_value = {"success": True}
            
            results = await manager.process_new_documents(new_documents, "incremental")
            
            assert results["success"] == True
            assert results["strategy"] == "incremental"
    
    @pytest.mark.asyncio
    async def test_process_new_documents_auto_strategy(self):
        """Test processing new documents with auto strategy detection."""
        manager = DynamicClusteringManager()
        
        new_documents = [
            {"document_id": "doc1", "content": "Test content"}
        ]
        
        # Mock the async methods
        with patch.object(manager, '_analyze_new_documents') as mock_analyze, \
             patch.object(manager, '_execute_update_strategy') as mock_execute, \
             patch.object(manager, '_update_vector_store') as mock_vector:
            
            mock_analyze.return_value = {"clustering_impact": "low"}
            mock_execute.return_value = {"success": True, "strategy": "incremental"}
            mock_vector.return_value = {"success": True}
            
            results = await manager.process_new_documents(new_documents, "auto")
            
            assert results["success"] == True
            mock_analyze.assert_called_once()
    
    def test_get_clustering_status(self):
        """Test getting clustering status."""
        manager = DynamicClusteringManager()
        
        status = manager.get_clustering_status()
        assert "last_update" in status
        assert "total_clusters" in status
        assert "clustering_config" in status
        assert "thresholds" in status
        assert "update_strategies" in status


class TestMultiLevelSearchOrchestrator:
    """Test the MultiLevelSearchOrchestrator class."""
    
    def test_init_default(self):
        """Test MultiLevelSearchOrchestrator initialization with default config."""
        orchestrator = MultiLevelSearchOrchestrator()
        assert orchestrator.config is not None
        assert orchestrator.search_levels == ["subject", "document", "chunk"]
        assert orchestrator.fallback_thresholds["subject"] == 3
    
    def test_init_with_config(self):
        """Test MultiLevelSearchOrchestrator initialization with custom config."""
        config = RetrieverConfig(max_results=20)
        orchestrator = MultiLevelSearchOrchestrator(config)
        assert orchestrator.config.max_results == 20
    
    def test_determine_search_level_subject(self):
        """Test determining search level for subject queries."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        level = orchestrator._determine_search_level("What are the main concepts in machine learning?")
        assert level == "subject"
    
    def test_determine_search_level_discovery(self):
        """Test determining search level for discovery queries."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        level = orchestrator._determine_search_level("What topics are available in the system?")
        assert level == "discovery"
    
    def test_determine_search_level_chunk(self):
        """Test determining search level for chunk queries."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        level = orchestrator._determine_search_level("Find the exact quote about neural networks")
        assert level == "chunk"
    
    def test_classify_query_intent_information_retrieval(self):
        """Test query intent classification for information retrieval."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        intent = orchestrator._classify_query_intent("What is machine learning?")
        assert intent["primary_intent"] == "information_retrieval"
        assert intent["complexity"] == "simple"
    
    def test_classify_query_intent_comparison(self):
        """Test query intent classification for comparison queries."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        intent = orchestrator._classify_query_intent("Compare machine learning and deep learning")
        assert intent["primary_intent"] == "comparison"
        assert intent["requires_comparison"] == True
        assert intent["complexity"] == "complex"
    
    def test_classify_query_intent_explanation(self):
        """Test query intent classification for explanation queries."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        intent = orchestrator._classify_query_intent("Why does gradient descent work?")
        assert intent["primary_intent"] == "explanation"
        assert intent["requires_reasoning"] == True
        assert intent["complexity"] == "complex"
    
    def test_extract_target_clusters(self):
        """Test target cluster extraction."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        clusters = orchestrator._extract_target_clusters("What are machine learning algorithms?")
        assert "ml_cluster" in clusters
        assert "ai_cluster" in clusters
    
    def test_count_total_results(self):
        """Test counting total results across levels."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        results = {
            "subject_results": [1, 2, 3],
            "document_results": [4, 5],
            "chunk_results": [6, 7, 8, 9],
            "fallback_results": [10]
        }
        
        total = orchestrator._count_total_results(results)
        assert total == 9
    
    def test_get_search_capabilities(self):
        """Test getting search capabilities."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        capabilities = orchestrator.get_search_capabilities()
        assert "available_levels" in capabilities
        assert "fallback_thresholds" in capabilities
        assert "retrievers_available" in capabilities
        assert "query_classification" in capabilities
        assert "multi_level_search" in capabilities


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_subject_retriever(self):
        """Test create_subject_retriever function."""
        retriever = create_subject_retriever()
        assert isinstance(retriever, SubjectRetriever)
    
    def test_create_subject_retriever_with_config(self):
        """Test create_subject_retriever function with config."""
        config = RetrieverConfig(max_results=15)
        retriever = create_subject_retriever(config)
        assert retriever.config.max_results == 15
    
    def test_create_cluster_discovery_service(self):
        """Test create_cluster_discovery_service function."""
        service = create_cluster_discovery_service()
        assert isinstance(service, ClusterDiscoveryService)
    
    def test_create_document_discovery_service(self):
        """Test create_document_discovery_service function."""
        service = create_document_discovery_service()
        assert isinstance(service, DocumentDiscoveryService)
    
    def test_create_dynamic_clustering_manager(self):
        """Test create_dynamic_clustering_manager function."""
        manager = create_dynamic_clustering_manager()
        assert isinstance(manager, DynamicClusteringManager)
    
    def test_create_multi_level_orchestrator(self):
        """Test create_multi_level_orchestrator function."""
        orchestrator = create_multi_level_orchestrator()
        assert isinstance(orchestrator, MultiLevelSearchOrchestrator)
    
    @pytest.mark.asyncio
    async def test_process_new_documents_async(self):
        """Test process_new_documents_async function."""
        new_documents = [{"document_id": "doc1", "content": "Test"}]
        
        # Mock the manager
        with patch('clustering.dynamic_clustering.DynamicClusteringManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.process_new_documents.return_value = {"success": True}
            mock_manager_class.return_value = mock_manager
            
            results = await process_new_documents_async(new_documents)
            assert results["success"] == True


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_subject_query_with_fallback(self):
        """Test subject query that triggers fallback."""
        # Create subject retriever
        retriever = SubjectRetriever()
        
        # Mock insufficient subject knowledge
        with patch.object(retriever, '_search_subject_knowledge') as mock_subject_search, \
             patch.object(retriever, '_apply_fallback_search') as mock_fallback:
            
            # Mock insufficient subject results
            mock_subject_search.return_value = []  # No subject results
            
            # Mock fallback results
            mock_fallback_doc = Mock()
            mock_fallback_doc.metadata = {
                "result_type": "fallback",
                "search_method": "chunk_fallback",
                "fallback_reason": "insufficient_subject_knowledge"
            }
            mock_fallback.return_value = [mock_fallback_doc]
            
            # Mock other methods
            with patch.object(retriever, '_initialize'), \
                 patch.object(retriever, '_classify_query_intent') as mock_classify, \
                 patch.object(retriever, '_filter_mature_subject_knowledge') as mock_filter, \
                 patch.object(retriever, '_enhance_with_cluster_context') as mock_enhance:
                
                mock_classify.return_value = {"search_level": "subject"}
                mock_filter.return_value = []  # No mature results
                mock_enhance.return_value = []
                
                results = retriever.get_relevant_documents("What are the main concepts in machine learning?")
                
                # Should have fallback results
                assert len(results) == 1
                assert results[0].metadata["result_type"] == "fallback"
                assert results[0].metadata["fallback_reason"] == "insufficient_subject_knowledge"
    
    @pytest.mark.asyncio
    async def test_multi_level_search_auto_detection(self):
        """Test multi-level search with auto-detection."""
        orchestrator = MultiLevelSearchOrchestrator()
        
        # Mock the search methods
        with patch.object(orchestrator, '_initialize'), \
             patch.object(orchestrator, '_search_subject_level') as mock_subject, \
             patch.object(orchestrator, '_search_chunk_level') as mock_chunk, \
             patch.object(orchestrator, '_search_document_level') as mock_document:
            
            # Mock subject search results
            mock_subject_doc = Mock()
            mock_subject_doc.metadata = {"result_type": "subject_knowledge", "similarity_score": 0.9}
            mock_subject.return_value = {
                "success": True,
                "search_level": "subject",
                "subject_results": [mock_subject_doc],
                "combined_results": [mock_subject_doc]
            }
            
            results = orchestrator.search("What are the main concepts in machine learning?")
            
            assert results["success"] == True
            assert results["search_level"] == "subject"
            assert len(results["combined_results"]) == 1
    
    @pytest.mark.asyncio
    async def test_dynamic_clustering_incremental_update(self):
        """Test dynamic clustering with incremental update."""
        manager = DynamicClusteringManager()
        
        new_documents = [
            {"document_id": "doc1", "content": "Machine learning content"},
            {"document_id": "doc2", "content": "Deep learning content"}
        ]
        
        # Mock the incremental update process
        with patch.object(manager, '_analyze_new_documents') as mock_analyze, \
             patch.object(manager, '_incremental_update') as mock_incremental, \
             patch.object(manager, '_update_vector_store') as mock_vector, \
             patch.object(manager, '_generate_update_summary') as mock_summary:
            
            mock_analyze.return_value = {"clustering_impact": "low"}
            mock_incremental.return_value = {
                "success": True,
                "strategy": "incremental",
                "new_clusters": [],
                "updated_clusters": [{"cluster_id": "ml_cluster"}],
                "assigned_documents": [{"document_id": "doc1", "cluster_id": "ml_cluster"}]
            }
            mock_vector.return_value = {"success": True}
            mock_summary.return_value = {
                "success": True,
                "strategy": "incremental",
                "clustering_changes": {"updated_clusters": 1}
            }
            
            results = await manager.process_new_documents(new_documents, "incremental")
            
            assert results["success"] == True
            assert results["strategy"] == "incremental"
            assert results["clustering_changes"]["updated_clusters"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
