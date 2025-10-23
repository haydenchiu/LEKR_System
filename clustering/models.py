"""
Pydantic models for clustering functionality.

This module defines the data structures used for representing clustering results,
cluster information, and document assignments in the LERK System.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class ClusterInfo(BaseModel):
    """Information about a single cluster."""
    
    cluster_id: int = Field(description="Unique identifier for the cluster")
    name: str = Field(description="Human-readable name for the cluster")
    topic_words: List[str] = Field(description="Top topic words for this cluster")
    document_count: int = Field(description="Number of documents in this cluster")
    coherence_score: Optional[float] = Field(
        default=None, 
        description="Coherence score for the cluster topic"
    )
    silhouette_score: Optional[float] = Field(
        default=None,
        description="Silhouette score for cluster quality"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when cluster was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when cluster was last updated"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the cluster"
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)

    def get_topic_summary(self, max_words: int = 10) -> str:
        """Get a summary of the cluster topic."""
        return ", ".join(self.topic_words[:max_words])

    def is_active(self) -> bool:
        """Check if cluster is active (has documents)."""
        return self.document_count > 0

    def update_metadata(self, **kwargs) -> 'ClusterInfo':
        """Update cluster metadata and return new instance."""
        new_metadata = self.metadata.copy()
        new_metadata.update(kwargs)
        return self.model_copy(update={
            'metadata': new_metadata,
            'updated_at': datetime.now()
        })


class DocumentClusterAssignment(BaseModel):
    """Assignment of a document to a cluster."""
    
    document_id: str = Field(description="Unique identifier for the document")
    cluster_id: int = Field(description="ID of the assigned cluster")
    confidence: float = Field(
        ge=0.0, 
        le=1.0,
        description="Confidence score for the assignment"
    )
    distance_to_center: Optional[float] = Field(
        default=None,
        description="Distance to cluster center"
    )
    assignment_reason: Optional[str] = Field(
        default=None,
        description="Reason for the assignment"
    )
    assigned_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when assignment was made"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the assignment"
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if assignment is high confidence."""
        return self.confidence >= threshold

    def update_assignment(self, cluster_id: int, confidence: float, **kwargs) -> 'DocumentClusterAssignment':
        """Update assignment and return new instance."""
        return self.model_copy(update={
            'cluster_id': cluster_id,
            'confidence': confidence,
            'assigned_at': datetime.now(),
            **kwargs
        })


class ClusteringResult(BaseModel):
    """Complete clustering result containing all clusters and assignments."""
    
    clusters: List[ClusterInfo] = Field(description="List of all clusters")
    assignments: List[DocumentClusterAssignment] = Field(
        description="List of document assignments"
    )
    total_documents: int = Field(description="Total number of documents clustered")
    num_clusters: int = Field(description="Number of clusters created")
    overall_quality_score: Optional[float] = Field(
        default=None,
        description="Overall quality score for the clustering"
    )
    clustering_method: str = Field(
        default="BERTopic",
        description="Method used for clustering"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for clustering"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when clustering was performed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the clustering result"
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)

    def get_cluster_by_id(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Get cluster information by ID."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def get_assignments_for_cluster(self, cluster_id: int) -> List[DocumentClusterAssignment]:
        """Get all assignments for a specific cluster."""
        return [assignment for assignment in self.assignments if assignment.cluster_id == cluster_id]

    def get_document_assignment(self, document_id: str) -> Optional[DocumentClusterAssignment]:
        """Get assignment for a specific document."""
        for assignment in self.assignments:
            if assignment.document_id == document_id:
                return assignment
        return None

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get statistics about the clustering result."""
        cluster_sizes = [cluster.document_count for cluster in self.clusters]
        assignment_confidences = [assignment.confidence for assignment in self.assignments]
        
        return {
            "total_clusters": self.num_clusters,
            "total_documents": self.total_documents,
            "average_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "average_confidence": sum(assignment_confidences) / len(assignment_confidences) if assignment_confidences else 0,
            "high_confidence_assignments": sum(1 for conf in assignment_confidences if conf >= 0.7),
            "overall_quality_score": self.overall_quality_score
        }

    def filter_high_quality_clusters(self, min_documents: int = 3, min_coherence: float = 0.3) -> 'ClusteringResult':
        """Filter to only high-quality clusters."""
        high_quality_clusters = []
        for cluster in self.clusters:
            if (cluster.document_count >= min_documents and 
                (cluster.coherence_score is None or cluster.coherence_score >= min_coherence)):
                high_quality_clusters.append(cluster)
        
        high_quality_cluster_ids = {cluster.cluster_id for cluster in high_quality_clusters}
        high_quality_assignments = [
            assignment for assignment in self.assignments 
            if assignment.cluster_id in high_quality_cluster_ids
        ]
        
        return self.model_copy(update={
            'clusters': high_quality_clusters,
            'assignments': high_quality_assignments,
            'num_clusters': len(high_quality_clusters)
        })

    def merge_similar_clusters(self, similarity_threshold: float = 0.8) -> 'ClusteringResult':
        """Merge clusters that are too similar."""
        # This would be implemented with actual similarity calculation
        # For now, return the original result
        return self
