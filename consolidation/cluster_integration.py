"""
Integration with clustering module for subject consolidation.

This module provides functionality to integrate the consolidation module
with the clustering module to create a proper flow from document clusters
to subject knowledge.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from clustering.models import ClusteringResult, ClusterInfo, DocumentClusterAssignment
    from clustering.clusterer import DocumentClusterer
except ImportError as e:
    raise ImportError(
        "Clustering module not available. "
        "Please ensure the clustering module is properly installed."
    ) from e

from .models import DocumentKnowledge, SubjectKnowledge
from .subject_consolidator import SubjectConsolidator
from .config import ConsolidationConfig, DEFAULT_CONSOLIDATION_CONFIG
from .exceptions import ConsolidationError, SubjectConsolidationError

logger = logging.getLogger(__name__)


class ClusterBasedSubjectConsolidator:
    """
    Enhanced subject consolidator that works with document clusters from the clustering module.
    
    This class provides functionality to consolidate knowledge from document clusters
    (subjects) identified by the clustering module into comprehensive subject knowledge.
    """
    
    def __init__(self, config: ConsolidationConfig = DEFAULT_CONSOLIDATION_CONFIG):
        """
        Initialize the ClusterBasedSubjectConsolidator.
        
        Args:
            config: Configuration for consolidation parameters
        """
        self.config = config
        self.subject_consolidator = SubjectConsolidator(config)
    
    def consolidate_subjects_from_clusters(
        self,
        clustering_result: ClusteringResult,
        document_knowledge_map: Dict[str, DocumentKnowledge]
    ) -> List[SubjectKnowledge]:
        """
        Consolidate subjects from document clusters.
        
        Args:
            clustering_result: Result from the clustering module
            document_knowledge_map: Mapping of document_id to DocumentKnowledge
            
        Returns:
            List of SubjectKnowledge objects, one for each cluster/subject
            
        Raises:
            SubjectConsolidationError: If consolidation fails
        """
        try:
            logger.info(f"Starting subject consolidation from {len(clustering_result.clusters)} clusters")
            
            subject_knowledge_list = []
            
            for cluster in clustering_result.clusters:
                # Get documents assigned to this cluster
                cluster_documents = self._get_documents_for_cluster(
                    cluster, clustering_result.assignments, document_knowledge_map
                )
                
                if not cluster_documents:
                    logger.warning(f"No documents found for cluster {cluster.cluster_id}")
                    continue
                
                # Consolidate subject knowledge for this cluster
                subject_knowledge = self._consolidate_single_subject(
                    cluster, cluster_documents
                )
                
                if subject_knowledge:
                    subject_knowledge_list.append(subject_knowledge)
                    logger.info(f"Consolidated subject {subject_knowledge.subject_id} with {len(subject_knowledge.core_concepts)} concepts")
            
            logger.info(f"Successfully consolidated {len(subject_knowledge_list)} subjects")
            return subject_knowledge_list
            
        except Exception as e:
            raise SubjectConsolidationError(f"Failed to consolidate subjects from clusters: {e}") from e
    
    def _get_documents_for_cluster(
        self,
        cluster: ClusterInfo,
        assignments: List[DocumentClusterAssignment],
        document_knowledge_map: Dict[str, DocumentKnowledge]
    ) -> List[DocumentKnowledge]:
        """Get document knowledge objects assigned to a specific cluster."""
        try:
            # Find assignments for this cluster
            cluster_assignments = [
                assignment for assignment in assignments
                if assignment.cluster_id == cluster.cluster_id
            ]
            
            # Get document knowledge for assigned documents
            cluster_documents = []
            for assignment in cluster_assignments:
                if assignment.document_id in document_knowledge_map:
                    cluster_documents.append(document_knowledge_map[assignment.document_id])
                else:
                    logger.warning(f"Document knowledge not found for document {assignment.document_id}")
            
            return cluster_documents
            
        except Exception as e:
            logger.error(f"Failed to get documents for cluster {cluster.cluster_id}: {e}")
            return []
    
    def _consolidate_single_subject(
        self,
        cluster: ClusterInfo,
        cluster_documents: List[DocumentKnowledge]
    ) -> Optional[SubjectKnowledge]:
        """Consolidate a single subject from cluster documents."""
        try:
            # Generate subject ID and name from cluster
            subject_id = f"subject_cluster_{cluster.cluster_id}"
            subject_name = self._generate_subject_name(cluster, cluster_documents)
            subject_description = self._generate_subject_description(cluster, cluster_documents)
            
            # Use the base subject consolidator
            subject_knowledge = self.subject_consolidator.consolidate_subject(
                subject_id=subject_id,
                subject_name=subject_name,
                document_knowledge=cluster_documents,
                subject_description=subject_description
            )
            
            # Add cluster-specific metadata
            subject_knowledge.consolidation_metadata.update({
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.name,
                "cluster_topic_words": cluster.topic_words,
                "cluster_document_count": cluster.document_count,
                "cluster_coherence_score": cluster.coherence_score,
                "consolidation_method": "cluster_based"
            })
            
            return subject_knowledge
            
        except Exception as e:
            logger.error(f"Failed to consolidate subject for cluster {cluster.cluster_id}: {e}")
            return None
    
    def _generate_subject_name(
        self,
        cluster: ClusterInfo,
        cluster_documents: List[DocumentKnowledge]
    ) -> str:
        """Generate a subject name from cluster and document information."""
        try:
            # Use cluster name if available
            if cluster.name and cluster.name != f"Cluster {cluster.cluster_id}":
                return cluster.name
            
            # Generate from topic words
            if cluster.topic_words:
                # Take top 3 topic words
                top_words = cluster.topic_words[:3]
                return " ".join(top_words).title()
            
            # Fallback to document themes
            all_themes = []
            for doc in cluster_documents:
                all_themes.extend(doc.main_themes)
            
            if all_themes:
                # Get most common themes
                from collections import Counter
                theme_counts = Counter(all_themes)
                top_themes = [theme for theme, count in theme_counts.most_common(3)]
                return " ".join(top_themes).title()
            
            # Final fallback
            return f"Subject {cluster.cluster_id}"
            
        except Exception as e:
            logger.warning(f"Failed to generate subject name: {e}")
            return f"Subject {cluster.cluster_id}"
    
    def _generate_subject_description(
        self,
        cluster: ClusterInfo,
        cluster_documents: List[DocumentKnowledge]
    ) -> str:
        """Generate a subject description from cluster and document information."""
        try:
            # Start with cluster information
            description_parts = []
            
            if cluster.topic_words:
                description_parts.append(f"Topics: {', '.join(cluster.topic_words[:5])}")
            
            if cluster.document_count:
                description_parts.append(f"Based on {cluster.document_count} documents")
            
            # Add document themes
            all_themes = []
            for doc in cluster_documents:
                all_themes.extend(doc.main_themes)
            
            if all_themes:
                from collections import Counter
                theme_counts = Counter(all_themes)
                top_themes = [theme for theme, count in theme_counts.most_common(5)]
                description_parts.append(f"Main themes: {', '.join(top_themes)}")
            
            # Combine into description
            if description_parts:
                return " | ".join(description_parts)
            else:
                return f"Consolidated knowledge from cluster {cluster.cluster_id}"
                
        except Exception as e:
            logger.warning(f"Failed to generate subject description: {e}")
            return f"Consolidated knowledge from cluster {cluster.cluster_id}"
    
    async def handle_clustering_update(
        self,
        clustering_update_results: Dict[str, Any],
        document_knowledge_map: Dict[str, DocumentKnowledge]
    ) -> Dict[str, Any]:
        """
        Handle dynamic clustering updates and update subject knowledge accordingly.
        
        This method is called when the clustering module processes new documents
        and updates clusters. It ensures subject knowledge stays in sync.
        
        Args:
            clustering_update_results: Results from DynamicClusteringManager.process_new_documents()
            document_knowledge_map: Mapping of document_id to DocumentKnowledge
            
        Returns:
            Subject knowledge update results
        """
        try:
            logger.info("Handling clustering update for subject knowledge")
            
            subject_updates = {
                "updated_subjects": [],
                "new_subjects": [],
                "deleted_subjects": [],
                "quality_changes": []
            }
            
            # Handle updated clusters
            for cluster_update in clustering_update_results.get("updated_clusters", []):
                subject_update = await self._update_subject_for_cluster(
                    cluster_update, document_knowledge_map
                )
                if subject_update:
                    subject_updates["updated_subjects"].append(subject_update)
            
            # Handle new clusters
            for new_cluster in clustering_update_results.get("new_clusters", []):
                new_subject = await self._create_subject_for_cluster(
                    new_cluster, document_knowledge_map
                )
                if new_subject:
                    subject_updates["new_subjects"].append(new_subject)
            
            # Handle cluster deletions (if any)
            deleted_clusters = clustering_update_results.get("deleted_clusters", [])
            for deleted_cluster in deleted_clusters:
                deleted_subject = await self._delete_subject_for_cluster(deleted_cluster)
                if deleted_subject:
                    subject_updates["deleted_subjects"].append(deleted_subject)
            
            logger.info(f"Subject knowledge update completed: "
                       f"{len(subject_updates['updated_subjects'])} updated, "
                       f"{len(subject_updates['new_subjects'])} new, "
                       f"{len(subject_updates['deleted_subjects'])} deleted")
            
            return subject_updates
            
        except Exception as e:
            logger.error(f"Failed to handle clustering update: {e}")
            return {
                "updated_subjects": [],
                "new_subjects": [],
                "deleted_subjects": [],
                "quality_changes": [],
                "error": str(e)
            }
    
    async def _update_subject_for_cluster(
        self,
        cluster_update: Dict[str, Any],
        document_knowledge_map: Dict[str, DocumentKnowledge]
    ) -> Optional[Dict[str, Any]]:
        """
        Update subject knowledge for an updated cluster.
        
        Args:
            cluster_update: Cluster update information
            document_knowledge_map: Mapping of document_id to DocumentKnowledge
            
        Returns:
            Subject update information or None if failed
        """
        try:
            cluster_id = cluster_update.get("cluster_id")
            if not cluster_id:
                logger.warning("Cluster update missing cluster_id")
                return None
            
            # Get documents for this cluster
            cluster_documents = self._get_documents_for_cluster_from_update(
                cluster_update, document_knowledge_map
            )
            
            if not cluster_documents:
                logger.warning(f"No documents found for updated cluster {cluster_id}")
                return None
            
            # Update subject knowledge
            subject_id = f"subject_{cluster_id}"
            
            # This would update the subject knowledge in the database
            # For now, return mock update information
            return {
                "subject_id": subject_id,
                "cluster_id": cluster_id,
                "updated_at": datetime.utcnow().isoformat(),
                "quality_score": cluster_update.get("quality_score", 0.8),
                "document_count": len(cluster_documents),
                "update_type": "cluster_update"
            }
            
        except Exception as e:
            logger.error(f"Failed to update subject for cluster: {e}")
            return None
    
    async def _create_subject_for_cluster(
        self,
        new_cluster: Dict[str, Any],
        document_knowledge_map: Dict[str, DocumentKnowledge]
    ) -> Optional[Dict[str, Any]]:
        """
        Create subject knowledge for a new cluster.
        
        Args:
            new_cluster: New cluster information
            document_knowledge_map: Mapping of document_id to DocumentKnowledge
            
        Returns:
            New subject information or None if failed
        """
        try:
            cluster_id = new_cluster.get("cluster_id")
            if not cluster_id:
                logger.warning("New cluster missing cluster_id")
                return None
            
            # Get documents for this cluster
            cluster_documents = self._get_documents_for_cluster_from_update(
                new_cluster, document_knowledge_map
            )
            
            if not cluster_documents:
                logger.warning(f"No documents found for new cluster {cluster_id}")
                return None
            
            # Create subject knowledge
            subject_id = f"subject_{cluster_id}"
            
            # This would create new subject knowledge in the database
            # For now, return mock creation information
            return {
                "subject_id": subject_id,
                "cluster_id": cluster_id,
                "name": new_cluster.get("label", "New Subject"),
                "description": new_cluster.get("description", "New subject knowledge"),
                "created_at": datetime.utcnow().isoformat(),
                "quality_score": new_cluster.get("quality_score", 0.8),
                "document_count": len(cluster_documents),
                "update_type": "cluster_creation"
            }
            
        except Exception as e:
            logger.error(f"Failed to create subject for cluster: {e}")
            return None
    
    async def _delete_subject_for_cluster(self, deleted_cluster: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Delete subject knowledge for a deleted cluster.
        
        Args:
            deleted_cluster: Deleted cluster information
            
        Returns:
            Deletion information or None if failed
        """
        try:
            cluster_id = deleted_cluster.get("cluster_id")
            if not cluster_id:
                logger.warning("Deleted cluster missing cluster_id")
                return None
            
            subject_id = f"subject_{cluster_id}"
            
            # This would delete the subject knowledge from the database
            # For now, return mock deletion information
            return {
                "subject_id": subject_id,
                "cluster_id": cluster_id,
                "deleted_at": datetime.utcnow().isoformat(),
                "update_type": "cluster_deletion"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete subject for cluster: {e}")
            return None
    
    def _get_documents_for_cluster_from_update(
        self,
        cluster_info: Dict[str, Any],
        document_knowledge_map: Dict[str, DocumentKnowledge]
    ) -> List[DocumentKnowledge]:
        """
        Get documents for a cluster from update information.
        
        Args:
            cluster_info: Cluster information from update
            document_knowledge_map: Mapping of document_id to DocumentKnowledge
            
        Returns:
            List of DocumentKnowledge objects for the cluster
        """
        try:
            cluster_documents = []
            
            # Get document IDs from cluster assignments
            assigned_documents = cluster_info.get("assigned_documents", [])
            for assignment in assigned_documents:
                doc_id = assignment.get("document_id")
                if doc_id and doc_id in document_knowledge_map:
                    cluster_documents.append(document_knowledge_map[doc_id])
            
            return cluster_documents
            
        except Exception as e:
            logger.error(f"Failed to get documents for cluster from update: {e}")
            return []


class IntegratedConsolidationPipeline:
    """
    Integrated pipeline that combines clustering and consolidation.
    
    This class provides a complete pipeline from raw documents to consolidated
    subject knowledge, integrating both clustering and consolidation modules.
    """
    
    def __init__(self, consolidation_config: ConsolidationConfig = DEFAULT_CONSOLIDATION_CONFIG):
        """
        Initialize the integrated pipeline.
        
        Args:
            consolidation_config: Configuration for consolidation
        """
        self.consolidation_config = consolidation_config
        self.cluster_based_consolidator = ClusterBasedSubjectConsolidator(consolidation_config)
    
    def process_documents_to_subjects(
        self,
        documents: List[str],
        document_ids: List[str],
        document_chunks_map: Dict[str, List[Any]],
        chunk_logic_data_map: Optional[Dict[str, List[Any]]] = None
    ) -> Tuple[List[DocumentKnowledge], List[SubjectKnowledge]]:
        """
        Complete pipeline from documents to subject knowledge.
        
        Args:
            documents: List of document texts
            document_ids: List of document IDs
            document_chunks_map: Mapping of document_id to document chunks
            chunk_logic_data_map: Optional mapping of document_id to logic data
            
        Returns:
            Tuple of (document_knowledge_list, subject_knowledge_list)
        """
        try:
            from .document_consolidator import DocumentConsolidator
            
            logger.info("Starting integrated consolidation pipeline")
            
            # Step 1: Document-level consolidation
            logger.info("Step 1: Document-level consolidation")
            document_consolidator = DocumentConsolidator(self.consolidation_config)
            document_knowledge_list = []
            
            for i, (doc_id, doc_text) in enumerate(zip(document_ids, documents)):
                try:
                    # Get chunks and logic data for this document
                    chunks = document_chunks_map.get(doc_id, [])
                    logic_data = chunk_logic_data_map.get(doc_id, []) if chunk_logic_data_map else None
                    
                    # Consolidate document
                    doc_knowledge = document_consolidator.consolidate_document(
                        document_id=doc_id,
                        document_title=f"Document {doc_id}",
                        chunks=chunks,
                        chunk_logic_data=logic_data
                    )
                    
                    document_knowledge_list.append(doc_knowledge)
                    logger.info(f"Consolidated document {doc_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to consolidate document {doc_id}: {e}")
                    continue
            
            # Step 2: Clustering
            logger.info("Step 2: Document clustering")
            from clustering.clusterer import DocumentClusterer
            clusterer = DocumentClusterer()
            clustering_result = clusterer.fit_clusters(documents, document_ids)
            
            # Step 3: Subject-level consolidation
            logger.info("Step 3: Subject-level consolidation")
            document_knowledge_map = {doc.document_id: doc for doc in document_knowledge_list}
            subject_knowledge_list = self.cluster_based_consolidator.consolidate_subjects_from_clusters(
                clustering_result, document_knowledge_map
            )
            
            logger.info(f"Pipeline completed: {len(document_knowledge_list)} documents, {len(subject_knowledge_list)} subjects")
            return document_knowledge_list, subject_knowledge_list
            
        except Exception as e:
            raise ConsolidationError(f"Integrated pipeline failed: {e}") from e
