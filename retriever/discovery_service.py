"""
Cluster and document discovery functionality for LERK System.

This module provides functionality for users to discover available
document clusters and documents in the system.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from langchain_core.documents import Document
except ImportError as e:
    # Create mock classes for missing dependencies
    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass
    
    class Filter:
        def __init__(self, *args, **kwargs):
            pass
    
    class FieldCondition:
        def __init__(self, *args, **kwargs):
            pass
    
    class MatchValue:
        def __init__(self, *args, **kwargs):
            pass
    
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

logger = logging.getLogger(__name__)


class ClusterDiscoveryService:
    """
    Service for discovering and managing document clusters.
    """
    
    def __init__(self, vector_store=None, knowledge_storage_manager=None):
        """
        Initialize the cluster discovery service.
        
        Args:
            vector_store: Qdrant vector store instance
            knowledge_storage_manager: Knowledge storage manager
        """
        self.vector_store = vector_store
        self.knowledge_storage_manager = knowledge_storage_manager
        self._cluster_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._last_cache_update = None
    
    def get_all_clusters(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Get all available document clusters.
        
        Args:
            include_metadata: Whether to include detailed cluster metadata
            
        Returns:
            Dictionary with cluster information
        """
        try:
            # Check cache first
            if self._is_cache_valid():
                return self._cluster_cache
            
            clusters_info = {
                "clusters": {},
                "cluster_count": 0,
                "last_updated": datetime.utcnow().isoformat(),
                "summary": {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "average_cluster_size": 0,
                    "cluster_quality_scores": []
                }
            }
            
            if self.vector_store:
                # Query Qdrant for cluster information
                cluster_data = self._query_cluster_data_from_qdrant()
                clusters_info.update(cluster_data)
            
            elif self.knowledge_storage_manager:
                # Query database for cluster information
                cluster_data = self._query_cluster_data_from_db()
                clusters_info.update(cluster_data)
            
            else:
                # Return mock data for development
                clusters_info = self._get_mock_cluster_data()
            
            # Update cache
            self._cluster_cache = clusters_info
            self._last_cache_update = datetime.utcnow()
            
            logger.info(f"Retrieved {clusters_info['cluster_count']} clusters")
            return clusters_info
            
        except Exception as e:
            logger.error(f"Failed to get clusters: {e}")
            return {
                "clusters": {},
                "cluster_count": 0,
                "error": str(e),
                "last_updated": datetime.utcnow().isoformat()
            }
    
    def get_cluster_details(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific cluster.
        
        Args:
            cluster_id: ID of the cluster to retrieve
            
        Returns:
            Detailed cluster information
        """
        try:
            cluster_info = {
                "cluster_id": cluster_id,
                "exists": False,
                "details": {},
                "documents": [],
                "subject_knowledge": {},
                "statistics": {}
            }
            
            if self.vector_store:
                # Query Qdrant for cluster details
                cluster_details = self._query_cluster_details_from_qdrant(cluster_id)
                cluster_info.update(cluster_details)
            
            elif self.knowledge_storage_manager:
                # Query database for cluster details
                cluster_details = self._query_cluster_details_from_db(cluster_id)
                cluster_info.update(cluster_details)
            
            else:
                # Return mock data
                cluster_info = self._get_mock_cluster_details(cluster_id)
            
            logger.info(f"Retrieved details for cluster {cluster_id}")
            return cluster_info
            
        except Exception as e:
            logger.error(f"Failed to get cluster details for {cluster_id}: {e}")
            return {
                "cluster_id": cluster_id,
                "exists": False,
                "error": str(e)
            }
    
    def search_clusters_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search clusters by topic or keyword.
        
        Args:
            topic: Topic or keyword to search for
            limit: Maximum number of results
            
        Returns:
            List of matching clusters
        """
        try:
            matching_clusters = []
            
            if self.vector_store:
                # Use semantic search to find relevant clusters
                matching_clusters = self._semantic_cluster_search(topic, limit)
            
            elif self.knowledge_storage_manager:
                # Use database search
                matching_clusters = self._database_cluster_search(topic, limit)
            
            else:
                # Return mock results
                matching_clusters = self._get_mock_cluster_search_results(topic, limit)
            
            logger.info(f"Found {len(matching_clusters)} clusters matching topic: {topic}")
            return matching_clusters
            
        except Exception as e:
            logger.error(f"Failed to search clusters by topic {topic}: {e}")
            return []
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about clusters.
        
        Returns:
            Cluster statistics
        """
        try:
            stats = {
                "total_clusters": 0,
                "total_documents": 0,
                "total_chunks": 0,
                "average_cluster_size": 0,
                "cluster_size_distribution": {},
                "quality_score_distribution": {},
                "domain_distribution": {},
                "last_updated": datetime.utcnow().isoformat()
            }
            
            if self.vector_store:
                # Calculate statistics from Qdrant
                stats = self._calculate_cluster_statistics_from_qdrant()
            
            elif self.knowledge_storage_manager:
                # Calculate statistics from database
                stats = self._calculate_cluster_statistics_from_db()
            
            else:
                # Return mock statistics
                stats = self._get_mock_cluster_statistics()
            
            logger.info("Retrieved cluster statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cluster statistics: {e}")
            return {
                "total_clusters": 0,
                "error": str(e),
                "last_updated": datetime.utcnow().isoformat()
            }
    
    def _is_cache_valid(self) -> bool:
        """Check if the cluster cache is still valid."""
        if not self._last_cache_update:
            return False
        
        cache_age = (datetime.utcnow() - self._last_cache_update).total_seconds()
        return cache_age < self._cache_ttl
    
    def _query_cluster_data_from_qdrant(self) -> Dict[str, Any]:
        """Query cluster data from Qdrant."""
        try:
            # This would query Qdrant for cluster information
            # For now, return mock data
            return self._get_mock_cluster_data()
        except Exception as e:
            logger.error(f"Failed to query cluster data from Qdrant: {e}")
            return {"clusters": {}, "cluster_count": 0}
    
    def _query_cluster_data_from_db(self) -> Dict[str, Any]:
        """Query cluster data from database."""
        try:
            # This would query the database for cluster information
            # For now, return mock data
            return self._get_mock_cluster_data()
        except Exception as e:
            logger.error(f"Failed to query cluster data from database: {e}")
            return {"clusters": {}, "cluster_count": 0}
    
    def _query_cluster_details_from_qdrant(self, cluster_id: str) -> Dict[str, Any]:
        """Query cluster details from Qdrant."""
        try:
            # This would query Qdrant for specific cluster details
            # For now, return mock data
            return self._get_mock_cluster_details(cluster_id)
        except Exception as e:
            logger.error(f"Failed to query cluster details from Qdrant: {e}")
            return {"cluster_id": cluster_id, "exists": False}
    
    def _query_cluster_details_from_db(self, cluster_id: str) -> Dict[str, Any]:
        """Query cluster details from database."""
        try:
            # This would query the database for specific cluster details
            # For now, return mock data
            return self._get_mock_cluster_details(cluster_id)
        except Exception as e:
            logger.error(f"Failed to query cluster details from database: {e}")
            return {"cluster_id": cluster_id, "exists": False}
    
    def _semantic_cluster_search(self, topic: str, limit: int) -> List[Dict[str, Any]]:
        """Perform semantic search for clusters."""
        try:
            # This would use semantic search to find relevant clusters
            # For now, return mock results
            return self._get_mock_cluster_search_results(topic, limit)
        except Exception as e:
            logger.error(f"Failed to perform semantic cluster search: {e}")
            return []
    
    def _database_cluster_search(self, topic: str, limit: int) -> List[Dict[str, Any]]:
        """Perform database search for clusters."""
        try:
            # This would search the database for relevant clusters
            # For now, return mock results
            return self._get_mock_cluster_search_results(topic, limit)
        except Exception as e:
            logger.error(f"Failed to perform database cluster search: {e}")
            return []
    
    def _calculate_cluster_statistics_from_qdrant(self) -> Dict[str, Any]:
        """Calculate cluster statistics from Qdrant."""
        try:
            # This would calculate statistics from Qdrant data
            # For now, return mock statistics
            return self._get_mock_cluster_statistics()
        except Exception as e:
            logger.error(f"Failed to calculate cluster statistics from Qdrant: {e}")
            return {"total_clusters": 0}
    
    def _calculate_cluster_statistics_from_db(self) -> Dict[str, Any]:
        """Calculate cluster statistics from database."""
        try:
            # This would calculate statistics from database data
            # For now, return mock statistics
            return self._get_mock_cluster_statistics()
        except Exception as e:
            logger.error(f"Failed to calculate cluster statistics from database: {e}")
            return {"total_clusters": 0}
    
    def _get_mock_cluster_data(self) -> Dict[str, Any]:
        """Get mock cluster data for development."""
        return {
            "clusters": {
                "ml_cluster": {
                    "cluster_id": "ml_cluster",
                    "label": "Machine Learning",
                    "description": "Documents about machine learning algorithms, techniques, and applications",
                    "document_count": 15,
                    "chunk_count": 450,
                    "main_topics": ["algorithms", "neural networks", "supervised learning", "unsupervised learning"],
                    "quality_score": 0.85,
                    "last_updated": "2024-01-15T10:30:00Z"
                },
                "ai_cluster": {
                    "cluster_id": "ai_cluster",
                    "label": "Artificial Intelligence",
                    "description": "Documents about artificial intelligence concepts, history, and future",
                    "document_count": 12,
                    "chunk_count": 380,
                    "main_topics": ["AI history", "machine learning", "deep learning", "neural networks"],
                    "quality_score": 0.82,
                    "last_updated": "2024-01-14T15:20:00Z"
                },
                "dl_cluster": {
                    "cluster_id": "dl_cluster",
                    "label": "Deep Learning",
                    "description": "Documents specifically about deep learning architectures and applications",
                    "document_count": 8,
                    "chunk_count": 250,
                    "main_topics": ["neural networks", "CNN", "RNN", "transformer", "backpropagation"],
                    "quality_score": 0.88,
                    "last_updated": "2024-01-16T09:15:00Z"
                }
            },
            "cluster_count": 3,
            "summary": {
                "total_documents": 35,
                "total_chunks": 1080,
                "average_cluster_size": 11.7,
                "cluster_quality_scores": [0.85, 0.82, 0.88]
            }
        }
    
    def _get_mock_cluster_details(self, cluster_id: str) -> Dict[str, Any]:
        """Get mock cluster details for development."""
        mock_clusters = {
            "ml_cluster": {
                "cluster_id": "ml_cluster",
                "exists": True,
                "details": {
                    "label": "Machine Learning",
                    "description": "Documents about machine learning algorithms, techniques, and applications",
                    "created_at": "2024-01-10T08:00:00Z",
                    "last_updated": "2024-01-15T10:30:00Z"
                },
                "documents": [
                    {"doc_id": "doc_001", "title": "Introduction to Machine Learning", "chunk_count": 25},
                    {"doc_id": "doc_002", "title": "Supervised Learning Algorithms", "chunk_count": 30},
                    {"doc_id": "doc_003", "title": "Unsupervised Learning Techniques", "chunk_count": 28}
                ],
                "subject_knowledge": {
                    "subject_id": "subject_ml_001",
                    "name": "Machine Learning Fundamentals",
                    "quality_score": 0.85,
                    "core_concepts": ["algorithms", "training", "prediction", "optimization"]
                },
                "statistics": {
                    "document_count": 15,
                    "chunk_count": 450,
                    "average_chunks_per_doc": 30,
                    "quality_score": 0.85
                }
            }
        }
        
        return mock_clusters.get(cluster_id, {
            "cluster_id": cluster_id,
            "exists": False,
            "details": {},
            "documents": [],
            "subject_knowledge": {},
            "statistics": {}
        })
    
    def _get_mock_cluster_search_results(self, topic: str, limit: int) -> List[Dict[str, Any]]:
        """Get mock cluster search results for development."""
        mock_results = [
            {
                "cluster_id": "ml_cluster",
                "label": "Machine Learning",
                "relevance_score": 0.95,
                "match_reason": f"Contains documents about {topic}",
                "document_count": 15,
                "quality_score": 0.85
            },
            {
                "cluster_id": "ai_cluster",
                "label": "Artificial Intelligence",
                "relevance_score": 0.88,
                "match_reason": f"Related to {topic} concepts",
                "document_count": 12,
                "quality_score": 0.82
            }
        ]
        
        return mock_results[:limit]
    
    def _get_mock_cluster_statistics(self) -> Dict[str, Any]:
        """Get mock cluster statistics for development."""
        return {
            "total_clusters": 3,
            "total_documents": 35,
            "total_chunks": 1080,
            "average_cluster_size": 11.7,
            "cluster_size_distribution": {
                "small (1-5 docs)": 0,
                "medium (6-15 docs)": 2,
                "large (16+ docs)": 1
            },
            "quality_score_distribution": {
                "high (0.8+)": 3,
                "medium (0.6-0.8)": 0,
                "low (<0.6)": 0
            },
            "domain_distribution": {
                "Machine Learning": 1,
                "Artificial Intelligence": 1,
                "Deep Learning": 1
            },
            "last_updated": datetime.utcnow().isoformat()
        }


class DocumentDiscoveryService:
    """
    Service for discovering and managing documents.
    """
    
    def __init__(self, vector_store=None, knowledge_storage_manager=None):
        """
        Initialize the document discovery service.
        
        Args:
            vector_store: Qdrant vector store instance
            knowledge_storage_manager: Knowledge storage manager
        """
        self.vector_store = vector_store
        self.knowledge_storage_manager = knowledge_storage_manager
        self._document_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._last_cache_update = None
    
    def get_all_documents(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Get all available documents.
        
        Args:
            include_metadata: Whether to include detailed document metadata
            
        Returns:
            Dictionary with document information
        """
        try:
            # Check cache first
            if self._is_cache_valid():
                return self._document_cache
            
            documents_info = {
                "documents": {},
                "document_count": 0,
                "last_updated": datetime.utcnow().isoformat(),
                "summary": {
                    "total_chunks": 0,
                    "average_chunks_per_doc": 0,
                    "document_types": {},
                    "processing_status": {}
                }
            }
            
            if self.vector_store:
                # Query Qdrant for document information
                document_data = self._query_document_data_from_qdrant()
                documents_info.update(document_data)
            
            elif self.knowledge_storage_manager:
                # Query database for document information
                document_data = self._query_document_data_from_db()
                documents_info.update(document_data)
            
            else:
                # Return mock data for development
                documents_info = self._get_mock_document_data()
            
            # Update cache
            self._document_cache = documents_info
            self._last_cache_update = datetime.utcnow()
            
            logger.info(f"Retrieved {documents_info['document_count']} documents")
            return documents_info
            
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return {
                "documents": {},
                "document_count": 0,
                "error": str(e),
                "last_updated": datetime.utcnow().isoformat()
            }
    
    def get_document_details(self, document_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific document.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Detailed document information
        """
        try:
            document_info = {
                "document_id": document_id,
                "exists": False,
                "details": {},
                "chunks": [],
                "enrichment_status": {},
                "logic_extraction_status": {},
                "cluster_assignment": {},
                "statistics": {}
            }
            
            if self.vector_store:
                # Query Qdrant for document details
                document_details = self._query_document_details_from_qdrant(document_id)
                document_info.update(document_details)
            
            elif self.knowledge_storage_manager:
                # Query database for document details
                document_details = self._query_document_details_from_db(document_id)
                document_info.update(document_details)
            
            else:
                # Return mock data
                document_info = self._get_mock_document_details(document_id)
            
            logger.info(f"Retrieved details for document {document_id}")
            return document_info
            
        except Exception as e:
            logger.error(f"Failed to get document details for {document_id}: {e}")
            return {
                "document_id": document_id,
                "exists": False,
                "error": str(e)
            }
    
    def search_documents_by_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            matching_documents = []
            
            if self.vector_store:
                # Use semantic search to find relevant documents
                matching_documents = self._semantic_document_search(query, limit)
            
            elif self.knowledge_storage_manager:
                # Use database search
                matching_documents = self._database_document_search(query, limit)
            
            else:
                # Return mock results
                matching_documents = self._get_mock_document_search_results(query, limit)
            
            logger.info(f"Found {len(matching_documents)} documents matching query: {query}")
            return matching_documents
            
        except Exception as e:
            logger.error(f"Failed to search documents by content {query}: {e}")
            return []
    
    def get_documents_by_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        """
        Get all documents assigned to a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            List of documents in the cluster
        """
        try:
            cluster_documents = []
            
            if self.vector_store:
                # Query Qdrant for documents in cluster
                cluster_documents = self._query_documents_in_cluster_from_qdrant(cluster_id)
            
            elif self.knowledge_storage_manager:
                # Query database for documents in cluster
                cluster_documents = self._query_documents_in_cluster_from_db(cluster_id)
            
            else:
                # Return mock results
                cluster_documents = self._get_mock_documents_in_cluster(cluster_id)
            
            logger.info(f"Found {len(cluster_documents)} documents in cluster {cluster_id}")
            return cluster_documents
            
        except Exception as e:
            logger.error(f"Failed to get documents for cluster {cluster_id}: {e}")
            return []
    
    def _is_cache_valid(self) -> bool:
        """Check if the document cache is still valid."""
        if not self._last_cache_update:
            return False
        
        cache_age = (datetime.utcnow() - self._last_cache_update).total_seconds()
        return cache_age < self._cache_ttl
    
    def _query_document_data_from_qdrant(self) -> Dict[str, Any]:
        """Query document data from Qdrant."""
        try:
            # This would query Qdrant for document information
            # For now, return mock data
            return self._get_mock_document_data()
        except Exception as e:
            logger.error(f"Failed to query document data from Qdrant: {e}")
            return {"documents": {}, "document_count": 0}
    
    def _query_document_data_from_db(self) -> Dict[str, Any]:
        """Query document data from database."""
        try:
            # This would query the database for document information
            # For now, return mock data
            return self._get_mock_document_data()
        except Exception as e:
            logger.error(f"Failed to query document data from database: {e}")
            return {"documents": {}, "document_count": 0}
    
    def _query_document_details_from_qdrant(self, document_id: str) -> Dict[str, Any]:
        """Query document details from Qdrant."""
        try:
            # This would query Qdrant for specific document details
            # For now, return mock data
            return self._get_mock_document_details(document_id)
        except Exception as e:
            logger.error(f"Failed to query document details from Qdrant: {e}")
            return {"document_id": document_id, "exists": False}
    
    def _query_document_details_from_db(self, document_id: str) -> Dict[str, Any]:
        """Query document details from database."""
        try:
            # This would query the database for specific document details
            # For now, return mock data
            return self._get_mock_document_details(document_id)
        except Exception as e:
            logger.error(f"Failed to query document details from database: {e}")
            return {"document_id": document_id, "exists": False}
    
    def _semantic_document_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform semantic search for documents."""
        try:
            # This would use semantic search to find relevant documents
            # For now, return mock results
            return self._get_mock_document_search_results(query, limit)
        except Exception as e:
            logger.error(f"Failed to perform semantic document search: {e}")
            return []
    
    def _database_document_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform database search for documents."""
        try:
            # This would search the database for relevant documents
            # For now, return mock results
            return self._get_mock_document_search_results(query, limit)
        except Exception as e:
            logger.error(f"Failed to perform database document search: {e}")
            return []
    
    def _query_documents_in_cluster_from_qdrant(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Query documents in cluster from Qdrant."""
        try:
            # This would query Qdrant for documents in cluster
            # For now, return mock results
            return self._get_mock_documents_in_cluster(cluster_id)
        except Exception as e:
            logger.error(f"Failed to query documents in cluster from Qdrant: {e}")
            return []
    
    def _query_documents_in_cluster_from_db(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Query documents in cluster from database."""
        try:
            # This would query the database for documents in cluster
            # For now, return mock results
            return self._get_mock_documents_in_cluster(cluster_id)
        except Exception as e:
            logger.error(f"Failed to query documents in cluster from database: {e}")
            return []
    
    def _get_mock_document_data(self) -> Dict[str, Any]:
        """Get mock document data for development."""
        return {
            "documents": {
                "doc_001": {
                    "document_id": "doc_001",
                    "title": "Introduction to Machine Learning",
                    "file_path": "/documents/ml_intro.pdf",
                    "file_type": "pdf",
                    "chunk_count": 25,
                    "cluster_id": "ml_cluster",
                    "processing_status": "completed",
                    "enrichment_status": "completed",
                    "logic_extraction_status": "completed",
                    "created_at": "2024-01-10T08:00:00Z",
                    "last_updated": "2024-01-15T10:30:00Z"
                },
                "doc_002": {
                    "document_id": "doc_002",
                    "title": "Deep Learning Fundamentals",
                    "file_path": "/documents/dl_fundamentals.pdf",
                    "file_type": "pdf",
                    "chunk_count": 30,
                    "cluster_id": "dl_cluster",
                    "processing_status": "completed",
                    "enrichment_status": "completed",
                    "logic_extraction_status": "completed",
                    "created_at": "2024-01-12T14:20:00Z",
                    "last_updated": "2024-01-16T09:15:00Z"
                }
            },
            "document_count": 2,
            "summary": {
                "total_chunks": 55,
                "average_chunks_per_doc": 27.5,
                "document_types": {"pdf": 2},
                "processing_status": {"completed": 2, "processing": 0, "failed": 0}
            }
        }
    
    def _get_mock_document_details(self, document_id: str) -> Dict[str, Any]:
        """Get mock document details for development."""
        mock_documents = {
            "doc_001": {
                "document_id": "doc_001",
                "exists": True,
                "details": {
                    "title": "Introduction to Machine Learning",
                    "file_path": "/documents/ml_intro.pdf",
                    "file_type": "pdf",
                    "file_size": 2048576,
                    "created_at": "2024-01-10T08:00:00Z",
                    "last_updated": "2024-01-15T10:30:00Z"
                },
                "chunks": [
                    {"chunk_id": "chunk_001", "content_preview": "Machine learning is...", "chunk_index": 0},
                    {"chunk_id": "chunk_002", "content_preview": "There are three main types...", "chunk_index": 1}
                ],
                "enrichment_status": {
                    "enriched_chunks": 25,
                    "total_chunks": 25,
                    "enrichment_rate": 1.0
                },
                "logic_extraction_status": {
                    "extracted_chunks": 25,
                    "total_chunks": 25,
                    "extraction_rate": 1.0
                },
                "cluster_assignment": {
                    "cluster_id": "ml_cluster",
                    "cluster_label": "Machine Learning",
                    "assignment_confidence": 0.92
                },
                "statistics": {
                    "chunk_count": 25,
                    "average_chunk_length": 150,
                    "quality_score": 0.85
                }
            }
        }
        
        return mock_documents.get(document_id, {
            "document_id": document_id,
            "exists": False,
            "details": {},
            "chunks": [],
            "enrichment_status": {},
            "logic_extraction_status": {},
            "cluster_assignment": {},
            "statistics": {}
        })
    
    def _get_mock_document_search_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Get mock document search results for development."""
        mock_results = [
            {
                "document_id": "doc_001",
                "title": "Introduction to Machine Learning",
                "relevance_score": 0.95,
                "match_reason": f"Contains content about {query}",
                "chunk_count": 25,
                "cluster_id": "ml_cluster"
            },
            {
                "document_id": "doc_002",
                "title": "Deep Learning Fundamentals",
                "relevance_score": 0.88,
                "match_reason": f"Related to {query} concepts",
                "chunk_count": 30,
                "cluster_id": "dl_cluster"
            }
        ]
        
        return mock_results[:limit]
    
    def _get_mock_documents_in_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Get mock documents in cluster for development."""
        cluster_documents = {
            "ml_cluster": [
                {
                    "document_id": "doc_001",
                    "title": "Introduction to Machine Learning",
                    "chunk_count": 25,
                    "quality_score": 0.85
                }
            ],
            "dl_cluster": [
                {
                    "document_id": "doc_002",
                    "title": "Deep Learning Fundamentals",
                    "chunk_count": 30,
                    "quality_score": 0.88
                }
            ]
        }
        
        return cluster_documents.get(cluster_id, [])


# Convenience functions
def create_cluster_discovery_service(vector_store=None, knowledge_storage_manager=None) -> ClusterDiscoveryService:
    """Create a cluster discovery service instance."""
    return ClusterDiscoveryService(vector_store, knowledge_storage_manager)


def create_document_discovery_service(vector_store=None, knowledge_storage_manager=None) -> DocumentDiscoveryService:
    """Create a document discovery service instance."""
    return DocumentDiscoveryService(vector_store, knowledge_storage_manager)
