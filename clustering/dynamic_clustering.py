"""
Dynamic clustering and subject knowledge update functionality for LERK System.

This module handles the behavior when new documents are added to the system,
including reclustering, new cluster creation, and subject knowledge updates.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    import numpy as np
except ImportError as e:
    # Create mock classes for missing dependencies
    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass
    
    class Distance:
        COSINE = "Cosine"
    
    class VectorParams:
        def __init__(self, *args, **kwargs):
            pass
    
    class PointStruct:
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
    
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
    
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class np:
        @staticmethod
        def array(data):
            return data

logger = logging.getLogger(__name__)


class DynamicClusteringManager:
    """
    Manages dynamic clustering and subject knowledge updates.
    """
    
    def __init__(
        self,
        vector_store=None,
        knowledge_storage_manager=None,
        clustering_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the dynamic clustering manager.
        
        Args:
            vector_store: Qdrant vector store instance
            knowledge_storage_manager: Knowledge storage manager
            clustering_config: Clustering configuration
        """
        self.vector_store = vector_store
        self.knowledge_storage_manager = knowledge_storage_manager
        self.clustering_config = clustering_config or self._get_default_clustering_config()
        
        # Clustering thresholds
        self.new_document_threshold = 5  # Minimum new documents to trigger reclustering
        self.similarity_threshold = 0.7  # Similarity threshold for cluster assignment
        self.cluster_stability_threshold = 0.8  # Threshold for cluster stability
        
        # Update strategies
        self.update_strategies = {
            "incremental": self._incremental_update,
            "full_reclustering": self._full_reclustering,
            "hybrid": self._hybrid_update
        }
        
        self._cluster_cache: Dict[str, Any] = {}
        self._last_clustering_update = None
    
    def _get_default_clustering_config(self) -> Dict[str, Any]:
        """Get default clustering configuration."""
        return {
            "min_cluster_size": 3,
            "max_cluster_size": 50,
            "similarity_threshold": 0.7,
            "reclustering_threshold": 0.8,
            "new_cluster_threshold": 0.6,
            "update_strategy": "hybrid"
        }
    
    async def process_new_documents(
        self,
        new_documents: List[Dict[str, Any]],
        update_strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Process new documents and update clustering/subject knowledge.
        
        Args:
            new_documents: List of new document data
            update_strategy: Strategy for updating ("auto", "incremental", "full_reclustering", "hybrid")
            
        Returns:
            Update results including new clusters, updated clusters, and subject knowledge changes
        """
        try:
            logger.info(f"Processing {len(new_documents)} new documents with strategy: {update_strategy}")
            
            # Step 1: Analyze new documents
            document_analysis = await self._analyze_new_documents(new_documents)
            
            # Step 2: Determine update strategy
            if update_strategy == "auto":
                update_strategy = self._determine_update_strategy(document_analysis)
            
            # Step 3: Execute update strategy
            update_results = await self._execute_update_strategy(
                new_documents, document_analysis, update_strategy
            )
            
            # Step 4: Update vector store
            await self._update_vector_store(update_results)
            
            # Step 5: Generate update summary
            update_summary = self._generate_update_summary(update_results)
            
            logger.info(f"Successfully processed {len(new_documents)} new documents")
            return update_summary
            
        except Exception as e:
            logger.error(f"Failed to process new documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "processed_documents": 0,
                "new_clusters": [],
                "updated_clusters": [],
                "subject_updates": []
            }
    
    async def _analyze_new_documents(self, new_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze new documents to determine clustering impact.
        
        Args:
            new_documents: List of new document data
            
        Returns:
            Analysis results
        """
        try:
            analysis = {
                "total_documents": len(new_documents),
                "document_types": {},
                "content_similarity": {},
                "topic_distribution": {},
                "clustering_impact": "low",
                "recommended_strategy": "incremental"
            }
            
            # Analyze document types
            for doc in new_documents:
                doc_type = doc.get("file_type", "unknown")
                analysis["document_types"][doc_type] = analysis["document_types"].get(doc_type, 0) + 1
            
            # Analyze content similarity (simplified)
            if len(new_documents) > 1:
                similarity_scores = self._calculate_document_similarity(new_documents)
                analysis["content_similarity"] = {
                    "average_similarity": np.mean(similarity_scores) if similarity_scores else 0,
                    "max_similarity": np.max(similarity_scores) if similarity_scores else 0,
                    "min_similarity": np.min(similarity_scores) if similarity_scores else 0
                }
            
            # Determine clustering impact
            if len(new_documents) >= self.new_document_threshold:
                analysis["clustering_impact"] = "high"
                analysis["recommended_strategy"] = "full_reclustering"
            elif len(new_documents) >= 3:
                analysis["clustering_impact"] = "medium"
                analysis["recommended_strategy"] = "hybrid"
            else:
                analysis["clustering_impact"] = "low"
                analysis["recommended_strategy"] = "incremental"
            
            logger.info(f"Document analysis completed: {analysis['clustering_impact']} impact")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze new documents: {e}")
            return {
                "total_documents": len(new_documents),
                "clustering_impact": "unknown",
                "recommended_strategy": "incremental",
                "error": str(e)
            }
    
    def _calculate_document_similarity(self, documents: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate similarity between documents.
        
        Args:
            documents: List of document data
            
        Returns:
            List of similarity scores
        """
        try:
            # This would use embeddings to calculate similarity
            # For now, return mock similarity scores
            similarity_scores = []
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    # Mock similarity calculation
                    similarity = 0.5 + (i + j) * 0.1  # Mock increasing similarity
                    similarity_scores.append(min(similarity, 1.0))
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"Failed to calculate document similarity: {e}")
            return []
    
    def _determine_update_strategy(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the best update strategy based on analysis.
        
        Args:
            analysis: Document analysis results
            
        Returns:
            Recommended update strategy
        """
        clustering_impact = analysis.get("clustering_impact", "low")
        
        if clustering_impact == "high":
            return "full_reclustering"
        elif clustering_impact == "medium":
            return "hybrid"
        else:
            return "incremental"
    
    async def _execute_update_strategy(
        self,
        new_documents: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        strategy: str
    ) -> Dict[str, Any]:
        """
        Execute the chosen update strategy.
        
        Args:
            new_documents: List of new document data
            analysis: Document analysis results
            strategy: Update strategy to execute
            
        Returns:
            Update results
        """
        try:
            if strategy in self.update_strategies:
                update_function = self.update_strategies[strategy]
                return await update_function(new_documents, analysis)
            else:
                logger.warning(f"Unknown update strategy: {strategy}, using incremental")
                return await self.update_strategies["incremental"](new_documents, analysis)
                
        except Exception as e:
            logger.error(f"Failed to execute update strategy {strategy}: {e}")
            return {
                "success": False,
                "error": str(e),
                "new_clusters": [],
                "updated_clusters": [],
                "assigned_documents": []
            }
    
    async def _incremental_update(
        self,
        new_documents: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform incremental update (assign documents to existing clusters).
        
        Args:
            new_documents: List of new document data
            analysis: Document analysis results
            
        Returns:
            Incremental update results
        """
        try:
            logger.info("Performing incremental update")
            
            # Get existing clusters
            existing_clusters = await self._get_existing_clusters()
            
            # Assign new documents to existing clusters
            assignment_results = await self._assign_documents_to_clusters(
                new_documents, existing_clusters
            )
            
            # Update cluster metadata
            updated_clusters = await self._update_cluster_metadata(assignment_results)
            
            return {
                "success": True,
                "strategy": "incremental",
                "new_clusters": [],
                "updated_clusters": updated_clusters,
                "assigned_documents": assignment_results["assignments"],
                "unassigned_documents": assignment_results["unassigned"]
            }
            
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "strategy": "incremental",
                "new_clusters": [],
                "updated_clusters": [],
                "assigned_documents": []
            }
    
    async def _full_reclustering(
        self,
        new_documents: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform full reclustering of all documents.
        
        Args:
            new_documents: List of new document data
            analysis: Document analysis results
            
        Returns:
            Full reclustering results
        """
        try:
            logger.info("Performing full reclustering")
            
            # Get all documents (existing + new)
            all_documents = await self._get_all_documents_for_clustering(new_documents)
            
            # Perform clustering
            clustering_result = await self._perform_clustering(all_documents)
            
            # Create new cluster assignments
            cluster_assignments = await self._create_cluster_assignments(clustering_result)
            
            # Update cluster metadata
            updated_clusters = await self._update_all_cluster_metadata(cluster_assignments)
            
            return {
                "success": True,
                "strategy": "full_reclustering",
                "new_clusters": clustering_result.get("new_clusters", []),
                "updated_clusters": updated_clusters,
                "assigned_documents": cluster_assignments["assignments"],
                "clustering_quality": clustering_result.get("quality_metrics", {})
            }
            
        except Exception as e:
            logger.error(f"Full reclustering failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "strategy": "full_reclustering",
                "new_clusters": [],
                "updated_clusters": [],
                "assigned_documents": []
            }
    
    async def _hybrid_update(
        self,
        new_documents: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform hybrid update (incremental + selective reclustering).
        
        Args:
            new_documents: List of new document data
            analysis: Document analysis results
            
        Returns:
            Hybrid update results
        """
        try:
            logger.info("Performing hybrid update")
            
            # Step 1: Try incremental assignment
            incremental_results = await self._incremental_update(new_documents, analysis)
            
            # Step 2: Check if selective reclustering is needed
            if self._needs_selective_reclustering(incremental_results):
                logger.info("Performing selective reclustering")
                selective_results = await self._selective_reclustering(new_documents, analysis)
                
                # Merge results
                merged_results = self._merge_update_results(incremental_results, selective_results)
                return merged_results
            
            return incremental_results
            
        except Exception as e:
            logger.error(f"Hybrid update failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "strategy": "hybrid",
                "new_clusters": [],
                "updated_clusters": [],
                "assigned_documents": []
            }
    
    def _needs_selective_reclustering(self, incremental_results: Dict[str, Any]) -> bool:
        """
        Determine if selective reclustering is needed.
        
        Args:
            incremental_results: Results from incremental update
            
        Returns:
            True if selective reclustering is needed
        """
        unassigned_count = len(incremental_results.get("unassigned_documents", []))
        total_documents = len(incremental_results.get("assigned_documents", [])) + unassigned_count
        
        # If more than 30% of documents are unassigned, do selective reclustering
        return unassigned_count > 0 and (unassigned_count / total_documents) > 0.3
    
    async def _selective_reclustering(
        self,
        new_documents: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform selective reclustering for unassigned documents.
        
        Args:
            new_documents: List of new document data
            analysis: Document analysis results
            
        Returns:
            Selective reclustering results
        """
        try:
            logger.info("Performing selective reclustering")
            
            # Get unassigned documents
            unassigned_documents = await self._get_unassigned_documents(new_documents)
            
            if not unassigned_documents:
                return {
                    "success": True,
                    "strategy": "selective_reclustering",
                    "new_clusters": [],
                    "updated_clusters": [],
                    "assigned_documents": []
                }
            
            # Perform clustering on unassigned documents
            clustering_result = await self._perform_clustering(unassigned_documents)
            
            # Create new clusters if needed
            new_clusters = await self._create_new_clusters(clustering_result)
            
            return {
                "success": True,
                "strategy": "selective_reclustering",
                "new_clusters": new_clusters,
                "updated_clusters": [],
                "assigned_documents": clustering_result.get("assignments", [])
            }
            
        except Exception as e:
            logger.error(f"Selective reclustering failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "strategy": "selective_reclustering",
                "new_clusters": [],
                "updated_clusters": [],
                "assigned_documents": []
            }
    
    async def _get_existing_clusters(self) -> List[Dict[str, Any]]:
        """Get existing clusters from the system."""
        try:
            # This would query the system for existing clusters
            # For now, return mock data
            return [
                {
                    "cluster_id": "ml_cluster",
                    "label": "Machine Learning",
                    "centroid": [0.1, 0.2, 0.3],  # Mock centroid
                    "document_count": 15,
                    "similarity_threshold": 0.7
                },
                {
                    "cluster_id": "ai_cluster",
                    "label": "Artificial Intelligence",
                    "centroid": [0.4, 0.5, 0.6],  # Mock centroid
                    "document_count": 12,
                    "similarity_threshold": 0.7
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get existing clusters: {e}")
            return []
    
    async def _assign_documents_to_clusters(
        self,
        documents: List[Dict[str, Any]],
        clusters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assign documents to existing clusters.
        
        Args:
            documents: List of documents to assign
            clusters: List of existing clusters
            
        Returns:
            Assignment results
        """
        try:
            assignments = []
            unassigned = []
            
            for doc in documents:
                best_cluster = None
                best_similarity = 0
                
                # Calculate similarity to each cluster
                for cluster in clusters:
                    similarity = self._calculate_document_cluster_similarity(doc, cluster)
                    if similarity > best_similarity and similarity >= cluster["similarity_threshold"]:
                        best_similarity = similarity
                        best_cluster = cluster
                
                if best_cluster:
                    assignments.append({
                        "document_id": doc.get("document_id"),
                        "cluster_id": best_cluster["cluster_id"],
                        "similarity_score": best_similarity,
                        "assignment_confidence": best_similarity
                    })
                else:
                    unassigned.append(doc)
            
            return {
                "assignments": assignments,
                "unassigned": unassigned
            }
            
        except Exception as e:
            logger.error(f"Failed to assign documents to clusters: {e}")
            return {"assignments": [], "unassigned": documents}
    
    def _calculate_document_cluster_similarity(
        self,
        document: Dict[str, Any],
        cluster: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between document and cluster.
        
        Args:
            document: Document data
            cluster: Cluster data
            
        Returns:
            Similarity score
        """
        try:
            # This would use embeddings to calculate similarity
            # For now, return mock similarity
            return 0.75  # Mock similarity score
            
        except Exception as e:
            logger.error(f"Failed to calculate document-cluster similarity: {e}")
            return 0.0
    
    async def _update_cluster_metadata(self, assignment_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update metadata for clusters that received new documents."""
        try:
            updated_clusters = []
            
            # Group assignments by cluster
            cluster_assignments = {}
            for assignment in assignment_results["assignments"]:
                cluster_id = assignment["cluster_id"]
                if cluster_id not in cluster_assignments:
                    cluster_assignments[cluster_id] = []
                cluster_assignments[cluster_id].append(assignment)
            
            # Update each cluster
            for cluster_id, assignments in cluster_assignments.items():
                updated_cluster = await self._update_single_cluster_metadata(cluster_id, assignments)
                updated_clusters.append(updated_cluster)
            
            return updated_clusters
            
        except Exception as e:
            logger.error(f"Failed to update cluster metadata: {e}")
            return []
    
    async def _update_single_cluster_metadata(
        self,
        cluster_id: str,
        assignments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update metadata for a single cluster."""
        try:
            # This would update the cluster metadata in the database
            # For now, return mock updated cluster
            return {
                "cluster_id": cluster_id,
                "updated_at": datetime.utcnow().isoformat(),
                "new_documents": len(assignments),
                "total_documents": 15 + len(assignments),  # Mock total
                "average_similarity": np.mean([a["similarity_score"] for a in assignments]) if assignments else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to update cluster {cluster_id} metadata: {e}")
            return {"cluster_id": cluster_id, "error": str(e)}
    
    async def _get_all_documents_for_clustering(self, new_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get all documents (existing + new) for clustering."""
        try:
            # This would get all documents from the system
            # For now, return mock data
            existing_documents = [
                {"document_id": "doc_001", "content": "Existing document 1"},
                {"document_id": "doc_002", "content": "Existing document 2"}
            ]
            
            return existing_documents + new_documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents for clustering: {e}")
            return new_documents
    
    async def _perform_clustering(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform clustering on documents."""
        try:
            # This would use BERTopic or similar clustering algorithm
            # For now, return mock clustering results
            return {
                "clusters": [
                    {
                        "cluster_id": "new_cluster_001",
                        "label": "New Topic Cluster",
                        "documents": documents[:len(documents)//2],
                        "centroid": [0.7, 0.8, 0.9]
                    },
                    {
                        "cluster_id": "new_cluster_002",
                        "label": "Another New Cluster",
                        "documents": documents[len(documents)//2:],
                        "centroid": [0.2, 0.3, 0.4]
                    }
                ],
                "quality_metrics": {
                    "silhouette_score": 0.75,
                    "calinski_harabasz_score": 150.5
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to perform clustering: {e}")
            return {"clusters": [], "quality_metrics": {}}
    
    async def _create_cluster_assignments(self, clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create cluster assignments from clustering results."""
        try:
            assignments = []
            
            for cluster in clustering_result.get("clusters", []):
                for doc in cluster.get("documents", []):
                    assignments.append({
                        "document_id": doc.get("document_id"),
                        "cluster_id": cluster["cluster_id"],
                        "similarity_score": 0.8,  # Mock score
                        "assignment_confidence": 0.8
                    })
            
            return {"assignments": assignments}
            
        except Exception as e:
            logger.error(f"Failed to create cluster assignments: {e}")
            return {"assignments": []}
    
    async def _update_all_cluster_metadata(self, cluster_assignments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update metadata for all clusters."""
        try:
            # This would update all cluster metadata
            # For now, return mock data
            return [
                {
                    "cluster_id": "new_cluster_001",
                    "updated_at": datetime.utcnow().isoformat(),
                    "document_count": 5,
                    "quality_score": 0.85
                },
                {
                    "cluster_id": "new_cluster_002",
                    "updated_at": datetime.utcnow().isoformat(),
                    "document_count": 3,
                    "quality_score": 0.82
                }
            ]
            
        except Exception as e:
            logger.error(f"Failed to update all cluster metadata: {e}")
            return []
    
    async def _create_new_clusters(self, clustering_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create new clusters from clustering results."""
        try:
            new_clusters = []
            
            for cluster in clustering_result.get("clusters", []):
                new_cluster = {
                    "cluster_id": cluster["cluster_id"],
                    "label": cluster["label"],
                    "description": f"New cluster created from clustering",
                    "document_count": len(cluster.get("documents", [])),
                    "centroid": cluster.get("centroid", []),
                    "created_at": datetime.utcnow().isoformat(),
                    "quality_score": 0.8  # Mock quality score
                }
                new_clusters.append(new_cluster)
            
            return new_clusters
            
        except Exception as e:
            logger.error(f"Failed to create new clusters: {e}")
            return []
    
    async def _get_unassigned_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get documents that couldn't be assigned to existing clusters."""
        try:
            # This would identify unassigned documents
            # For now, return mock data
            return documents[:len(documents)//2]  # Mock unassigned documents
            
        except Exception as e:
            logger.error(f"Failed to get unassigned documents: {e}")
            return []
    
    def _merge_update_results(
        self,
        incremental_results: Dict[str, Any],
        selective_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge incremental and selective update results."""
        try:
            merged_results = {
                "success": incremental_results.get("success", False) and selective_results.get("success", False),
                "strategy": "hybrid",
                "new_clusters": selective_results.get("new_clusters", []),
                "updated_clusters": incremental_results.get("updated_clusters", []),
                "assigned_documents": (
                    incremental_results.get("assigned_documents", []) +
                    selective_results.get("assigned_documents", [])
                )
            }
            
            return merged_results
            
        except Exception as e:
            logger.error(f"Failed to merge update results: {e}")
            return {
                "success": False,
                "error": str(e),
                "strategy": "hybrid",
                "new_clusters": [],
                "updated_clusters": [],
                "assigned_documents": []
            }
    
    async def _update_vector_store(self, update_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update the vector store with new clustering information."""
        try:
            logger.info("Updating vector store")
            
            # This would update Qdrant with new cluster information
            # For now, return mock success
            return {
                "success": True,
                "updated_points": len(update_results.get("assigned_documents", [])),
                "new_collections": len(update_results.get("new_clusters", []))
            }
            
        except Exception as e:
            logger.error(f"Failed to update vector store: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_update_summary(self, update_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the update process."""
        try:
            summary = {
                "success": update_results.get("success", False),
                "strategy": update_results.get("strategy", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
                "clustering_changes": {
                    "new_clusters": len(update_results.get("new_clusters", [])),
                    "updated_clusters": len(update_results.get("updated_clusters", [])),
                    "assigned_documents": len(update_results.get("assigned_documents", []))
                },
                "quality_metrics": update_results.get("clustering_quality", {}),
                "errors": []
            }
            
            if update_results.get("error"):
                summary["errors"].append(update_results["error"])
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate update summary: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_clustering_status(self) -> Dict[str, Any]:
        """Get current clustering status and statistics."""
        try:
            return {
                "last_update": self._last_clustering_update,
                "total_clusters": len(self._cluster_cache.get("clusters", {})),
                "clustering_config": self.clustering_config,
                "thresholds": {
                    "new_document_threshold": self.new_document_threshold,
                    "similarity_threshold": self.similarity_threshold,
                    "cluster_stability_threshold": self.cluster_stability_threshold
                },
                "update_strategies": list(self.update_strategies.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to get clustering status: {e}")
            return {"error": str(e)}


# Convenience functions
def create_dynamic_clustering_manager(
    vector_store=None,
    knowledge_storage_manager=None,
    clustering_config: Optional[Dict[str, Any]] = None
) -> DynamicClusteringManager:
    """Create a dynamic clustering manager instance."""
    return DynamicClusteringManager(vector_store, knowledge_storage_manager, clustering_config)


async def process_new_documents_async(
    new_documents: List[Dict[str, Any]],
    vector_store=None,
    knowledge_storage_manager=None,
    update_strategy: str = "auto"
) -> Dict[str, Any]:
    """
    Process new documents asynchronously.
    
    Args:
        new_documents: List of new document data
        vector_store: Qdrant vector store instance
        knowledge_storage_manager: Knowledge storage manager
        update_strategy: Update strategy to use
        
    Returns:
        Update results
    """
    manager = DynamicClusteringManager(vector_store, knowledge_storage_manager)
    return await manager.process_new_documents(new_documents, update_strategy)
