"""
Core clustering functionality using BERTopic.

This module implements the main clustering functionality for the LERK System,
providing document clustering, reassignment, and cluster management capabilities.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import silhouette_score
except ImportError as e:
    raise ImportError(
        "Required clustering dependencies not installed. "
        "Please install: pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn"
    ) from e

from .models import ClusterInfo, ClusteringResult, DocumentClusterAssignment
from .config import ClusteringConfig, DEFAULT_CLUSTERING_CONFIG
from .exceptions import (
    ClusteringError, InvalidDocumentError, ClusteringInitializationError,
    ClusterAssignmentError, ClusteringFitError, ClusterNotFoundError
)

logger = logging.getLogger(__name__)


class DocumentClusterer:
    """
    Main clustering class using BERTopic for document clustering.
    
    This class provides functionality to cluster documents, reassign documents,
    and manage clusters in the LERK System.
    """
    
    def __init__(self, config: ClusteringConfig = DEFAULT_CLUSTERING_CONFIG):
        """
        Initialize the DocumentClusterer.
        
        Args:
            config: Configuration for clustering parameters
        """
        self.config = config
        self.bertopic_model: Optional[BERTopic] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.is_fitted = False
        self.clustering_result: Optional[ClusteringResult] = None
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize BERTopic and embedding models."""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.config.model_name)
            
            # Get BERTopic parameters
            bertopic_params = self.config.get_bertopic_params()
            
            # Initialize BERTopic model
            self.bertopic_model = BERTopic(**bertopic_params)
            
            logger.info(f"Initialized DocumentClusterer with model: {self.config.model_name}")
            
        except Exception as e:
            raise ClusteringInitializationError(f"Failed to initialize clustering models: {e}") from e
    
    def fit_clusters(self, documents: List[str], document_ids: Optional[List[str]] = None) -> ClusteringResult:
        """
        Fit clusters to documents using BERTopic.
        
        Args:
            documents: List of document texts to cluster
            document_ids: Optional list of document IDs
            
        Returns:
            ClusteringResult containing clusters and assignments
            
        Raises:
            InvalidDocumentError: If documents are invalid
            ClusteringFitError: If clustering fails
        """
        if not documents:
            raise InvalidDocumentError("No documents provided for clustering")
        
        if document_ids and len(document_ids) != len(documents):
            raise InvalidDocumentError("Number of document IDs must match number of documents")
        
        if not document_ids:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
        
        try:
            logger.info(f"Fitting clusters for {len(documents)} documents")
            
            # Fit BERTopic model
            topics, probabilities = self.bertopic_model.fit_transform(documents)
            
            # Get topic information
            topic_info = self.bertopic_model.get_topic_info()
            
            # Create cluster information
            clusters = []
            for _, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Skip outlier topic
                    cluster = ClusterInfo(
                        cluster_id=int(row['Topic']),
                        name=f"Cluster {row['Topic']}",
                        topic_words=self._get_topic_words(int(row['Topic'])),
                        document_count=int(row['Count']),
                        coherence_score=self._calculate_coherence_score(int(row['Topic'])),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    clusters.append(cluster)
            
            # Create document assignments
            assignments = []
            for i, (doc_id, topic, prob) in enumerate(zip(document_ids, topics, probabilities)):
                if topic != -1:  # Skip outliers
                    assignment = DocumentClusterAssignment(
                        document_id=doc_id,
                        cluster_id=int(topic),
                        confidence=float(prob[topic]) if prob is not None else 0.5,
                        distance_to_center=self._calculate_distance_to_center(i, int(topic)),
                        assigned_at=datetime.now()
                    )
                    assignments.append(assignment)
            
            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(clusters, assignments)
            
            # Create clustering result
            self.clustering_result = ClusteringResult(
                clusters=clusters,
                assignments=assignments,
                total_documents=len(documents),
                num_clusters=len(clusters),
                overall_quality_score=quality_score,
                clustering_method=self.config.clustering_method,
                parameters=self.config.get_bertopic_params(),
                created_at=datetime.now()
            )
            
            self.is_fitted = True
            logger.info(f"Successfully fitted {len(clusters)} clusters")
            
            return self.clustering_result
            
        except Exception as e:
            raise ClusteringFitError(f"Failed to fit clusters: {e}") from e
    
    def assign_documents_to_clusters(self, documents: List[str], document_ids: Optional[List[str]] = None) -> List[DocumentClusterAssignment]:
        """
        Assign new documents to existing clusters.
        
        Args:
            documents: List of document texts to assign
            document_ids: Optional list of document IDs
            
        Returns:
            List of document assignments
            
        Raises:
            ClusteringError: If model is not fitted
            InvalidDocumentError: If documents are invalid
        """
        if not self.is_fitted or self.bertopic_model is None:
            raise ClusteringError("Model must be fitted before assigning documents")
        
        if not documents:
            raise InvalidDocumentError("No documents provided for assignment")
        
        if document_ids and len(document_ids) != len(documents):
            raise InvalidDocumentError("Number of document IDs must match number of documents")
        
        if not document_ids:
            document_ids = [f"new_doc_{i}" for i in range(len(documents))]
        
        try:
            # Transform documents to get topics and probabilities
            topics, probabilities = self.bertopic_model.transform(documents)
            
            # Create assignments
            assignments = []
            for doc_id, topic, prob in zip(document_ids, topics, probabilities):
                if topic != -1:  # Skip outliers
                    assignment = DocumentClusterAssignment(
                        document_id=doc_id,
                        cluster_id=int(topic),
                        confidence=float(prob[topic]) if prob is not None else 0.5,
                        distance_to_center=None,  # Would need to calculate
                        assigned_at=datetime.now()
                    )
                    assignments.append(assignment)
            
            logger.info(f"Assigned {len(assignments)} documents to clusters")
            return assignments
            
        except Exception as e:
            raise ClusterAssignmentError(f"Failed to assign documents: {e}") from e
    
    def reassign_documents_to_clusters(self, documents: List[str], document_ids: List[str]) -> List[DocumentClusterAssignment]:
        """
        Reassign documents to clusters, potentially creating new clusters.
        
        Args:
            documents: List of document texts to reassign
            document_ids: List of document IDs
            
        Returns:
            List of updated document assignments
        """
        if not self.is_fitted or self.bertopic_model is None:
            raise ClusteringError("Model must be fitted before reassigning documents")
        
        try:
            # Use partial_fit to update the model with new documents
            topics, probabilities = self.bertopic_model.transform(documents)
            
            # Create updated assignments
            assignments = []
            for doc_id, topic, prob in zip(document_ids, topics, probabilities):
                if topic != -1:
                    assignment = DocumentClusterAssignment(
                        document_id=doc_id,
                        cluster_id=int(topic),
                        confidence=float(prob[topic]) if prob is not None else 0.5,
                        distance_to_center=None,
                        assigned_at=datetime.now(),
                        assignment_reason="reassignment"
                    )
                    assignments.append(assignment)
            
            logger.info(f"Reassigned {len(assignments)} documents")
            return assignments
            
        except Exception as e:
            raise ClusterAssignmentError(f"Failed to reassign documents: {e}") from e
    
    def get_cluster_info(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Get information about a specific cluster."""
        if not self.clustering_result:
            return None
        
        return self.clustering_result.get_cluster_by_id(cluster_id)
    
    def get_document_assignment(self, document_id: str) -> Optional[DocumentClusterAssignment]:
        """Get assignment for a specific document."""
        if not self.clustering_result:
            return None
        
        return self.clustering_result.get_document_assignment(document_id)
    
    def get_cluster_documents(self, cluster_id: int) -> List[str]:
        """Get all document IDs in a specific cluster."""
        if not self.clustering_result:
            return []
        
        assignments = self.clustering_result.get_assignments_for_cluster(cluster_id)
        return [assignment.document_id for assignment in assignments]
    
    def update_cluster_metadata(self, cluster_id: int, **metadata) -> bool:
        """Update metadata for a specific cluster."""
        if not self.clustering_result:
            return False
        
        cluster = self.clustering_result.get_cluster_by_id(cluster_id)
        if not cluster:
            return False
        
        # Update cluster metadata
        updated_cluster = cluster.update_metadata(**metadata)
        
        # Update in clustering result
        for i, c in enumerate(self.clustering_result.clusters):
            if c.cluster_id == cluster_id:
                self.clustering_result.clusters[i] = updated_cluster
                break
        
        return True
    
    def _get_topic_words(self, topic_id: int) -> List[str]:
        """Get top words for a topic."""
        try:
            if self.bertopic_model:
                topic_words = self.bertopic_model.get_topic(topic_id)
                return [word for word, _ in topic_words[:self.config.top_k_words]]
        except Exception as e:
            logger.warning(f"Failed to get topic words for topic {topic_id}: {e}")
        
        return []
    
    def _calculate_coherence_score(self, topic_id: int) -> Optional[float]:
        """Calculate coherence score for a topic."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated coherence metrics
        try:
            topic_words = self._get_topic_words(topic_id)
            if len(topic_words) < 2:
                return None
            
            # Simple coherence based on word frequency
            return min(1.0, len(topic_words) / 10.0)
        except Exception:
            return None
    
    def _calculate_distance_to_center(self, doc_index: int, cluster_id: int) -> Optional[float]:
        """Calculate distance from document to cluster center."""
        # This would require access to embeddings and cluster centers
        # For now, return None
        return None
    
    def _calculate_overall_quality_score(self, clusters: List[ClusterInfo], assignments: List[DocumentClusterAssignment]) -> Optional[float]:
        """Calculate overall quality score for the clustering."""
        try:
            if not assignments:
                return None
            
            # Calculate average confidence
            confidences = [assignment.confidence for assignment in assignments]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Calculate cluster balance (lower is better)
            cluster_sizes = [cluster.document_count for cluster in clusters]
            if cluster_sizes:
                balance_score = 1.0 - (max(cluster_sizes) - min(cluster_sizes)) / max(cluster_sizes)
            else:
                balance_score = 1.0
            
            # Combine scores
            quality_score = (avg_confidence + balance_score) / 2.0
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate quality score: {e}")
            return None


# Convenience functions
def assign_documents_to_clusters(
    documents: List[str], 
    document_ids: Optional[List[str]] = None,
    config: ClusteringConfig = DEFAULT_CLUSTERING_CONFIG
) -> ClusteringResult:
    """
    Convenience function to cluster documents.
    
    Args:
        documents: List of document texts
        document_ids: Optional list of document IDs
        config: Clustering configuration
        
    Returns:
        ClusteringResult
    """
    clusterer = DocumentClusterer(config)
    return clusterer.fit_clusters(documents, document_ids)


def reassign_documents_to_clusters(
    documents: List[str],
    document_ids: List[str],
    existing_clusterer: DocumentClusterer
) -> List[DocumentClusterAssignment]:
    """
    Convenience function to reassign documents to existing clusters.
    
    Args:
        documents: List of document texts
        document_ids: List of document IDs
        existing_clusterer: Pre-fitted DocumentClusterer
        
    Returns:
        List of document assignments
    """
    return existing_clusterer.reassign_documents_to_clusters(documents, document_ids)
