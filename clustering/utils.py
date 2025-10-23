"""
Utility functions for clustering operations and analysis.

This module provides utility functions for analyzing cluster quality,
extracting cluster topics, and performing various clustering operations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "Required utility dependencies not installed. "
        "Please install: pip install scikit-learn pandas"
    ) from e

from .models import ClusteringResult, ClusterInfo, DocumentClusterAssignment
from .exceptions import ClusteringError, ClusterNotFoundError

logger = logging.getLogger(__name__)


def analyze_cluster_quality(
    clustering_result: ClusteringResult,
    embeddings: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Analyze the quality of clustering results.
    
    Args:
        clustering_result: The clustering result to analyze
        embeddings: Optional embeddings for quality metrics
        
    Returns:
        Dictionary containing quality metrics
    """
    try:
        quality_metrics = {
            "num_clusters": clustering_result.num_clusters,
            "total_documents": clustering_result.total_documents,
            "overall_quality_score": clustering_result.overall_quality_score
        }
        
        # Calculate cluster size statistics
        cluster_sizes = [cluster.document_count for cluster in clustering_result.clusters]
        if cluster_sizes:
            quality_metrics.update({
                "avg_cluster_size": np.mean(cluster_sizes),
                "min_cluster_size": min(cluster_sizes),
                "max_cluster_size": max(cluster_sizes),
                "cluster_size_std": np.std(cluster_sizes)
            })
        
        # Calculate assignment confidence statistics
        confidences = [assignment.confidence for assignment in clustering_result.assignments]
        if confidences:
            quality_metrics.update({
                "avg_confidence": np.mean(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences),
                "confidence_std": np.std(confidences),
                "high_confidence_ratio": sum(1 for c in confidences if c >= 0.7) / len(confidences)
            })
        
        # Calculate coherence scores if available
        coherence_scores = [cluster.coherence_score for cluster in clustering_result.clusters 
                          if cluster.coherence_score is not None]
        if coherence_scores:
            quality_metrics.update({
                "avg_coherence": np.mean(coherence_scores),
                "min_coherence": min(coherence_scores),
                "max_coherence": max(coherence_scores)
            })
        
        # Calculate silhouette score if embeddings are provided
        if embeddings is not None and len(embeddings) > 1:
            try:
                cluster_labels = [assignment.cluster_id for assignment in clustering_result.assignments]
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                    silhouette = silhouette_score(embeddings, cluster_labels)
                    quality_metrics["silhouette_score"] = silhouette
            except Exception as e:
                logger.warning(f"Failed to calculate silhouette score: {e}")
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Failed to analyze cluster quality: {e}")
        return {"error": str(e)}


def get_cluster_statistics(clustering_result: ClusteringResult) -> Dict[str, Any]:
    """
    Get comprehensive statistics about clustering results.
    
    Args:
        clustering_result: The clustering result to analyze
        
    Returns:
        Dictionary containing cluster statistics
    """
    try:
        stats = clustering_result.get_cluster_statistics()
        
        # Add additional statistics
        cluster_info = []
        for cluster in clustering_result.clusters:
            cluster_info.append({
                "cluster_id": cluster.cluster_id,
                "name": cluster.name,
                "document_count": cluster.document_count,
                "coherence_score": cluster.coherence_score,
                "topic_words": cluster.topic_words[:5],  # Top 5 words
                "created_at": cluster.created_at.isoformat() if cluster.created_at else None
            })
        
        stats["cluster_details"] = cluster_info
        
        # Calculate cluster balance
        cluster_sizes = [cluster.document_count for cluster in clustering_result.clusters]
        if cluster_sizes:
            stats["cluster_balance"] = 1.0 - (max(cluster_sizes) - min(cluster_sizes)) / max(cluster_sizes)
        
        # Calculate assignment distribution
        assignment_counts = {}
        for assignment in clustering_result.assignments:
            cluster_id = assignment.cluster_id
            assignment_counts[cluster_id] = assignment_counts.get(cluster_id, 0) + 1
        
        stats["assignment_distribution"] = assignment_counts
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get cluster statistics: {e}")
        return {"error": str(e)}


def merge_similar_clusters(
    clustering_result: ClusteringResult,
    similarity_threshold: float = 0.8
) -> ClusteringResult:
    """
    Merge clusters that are too similar based on topic words.
    
    Args:
        clustering_result: The clustering result to process
        similarity_threshold: Threshold for cluster similarity
        
    Returns:
        Updated clustering result with merged clusters
    """
    try:
        if len(clustering_result.clusters) <= 1:
            return clustering_result
        
        # Calculate similarity between clusters based on topic words
        clusters_to_merge = []
        processed_clusters = set()
        
        for i, cluster1 in enumerate(clustering_result.clusters):
            if cluster1.cluster_id in processed_clusters:
                continue
                
            for j, cluster2 in enumerate(clustering_result.clusters[i+1:], i+1):
                if cluster2.cluster_id in processed_clusters:
                    continue
                
                # Calculate similarity based on topic words
                similarity = _calculate_cluster_similarity(cluster1, cluster2)
                
                if similarity >= similarity_threshold:
                    clusters_to_merge.append((cluster1.cluster_id, cluster2.cluster_id))
                    processed_clusters.add(cluster2.cluster_id)
                    break
        
        if not clusters_to_merge:
            return clustering_result
        
        # Merge clusters
        merged_clusters = []
        cluster_mapping = {}
        
        for cluster in clustering_result.clusters:
            if cluster.cluster_id not in processed_clusters:
                merged_clusters.append(cluster)
            else:
                # Find the cluster to merge into
                target_cluster_id = None
                for merge_pair in clusters_to_merge:
                    if cluster.cluster_id == merge_pair[1]:
                        target_cluster_id = merge_pair[0]
                        break
                
                if target_cluster_id:
                    cluster_mapping[cluster.cluster_id] = target_cluster_id
        
        # Update assignments
        updated_assignments = []
        for assignment in clustering_result.assignments:
            if assignment.cluster_id in cluster_mapping:
                new_assignment = assignment.model_copy(update={
                    'cluster_id': cluster_mapping[assignment.cluster_id]
                })
                updated_assignments.append(new_assignment)
            else:
                updated_assignments.append(assignment)
        
        # Update cluster document counts
        for cluster in merged_clusters:
            cluster_assignments = [a for a in updated_assignments if a.cluster_id == cluster.cluster_id]
            cluster.document_count = len(cluster_assignments)
        
        # Create updated clustering result
        updated_result = clustering_result.model_copy(update={
            'clusters': merged_clusters,
            'assignments': updated_assignments,
            'num_clusters': len(merged_clusters)
        })
        
        logger.info(f"Merged {len(clusters_to_merge)} cluster pairs")
        return updated_result
        
    except Exception as e:
        logger.error(f"Failed to merge similar clusters: {e}")
        return clustering_result


def extract_cluster_topics(
    clustering_result: ClusteringResult,
    top_k: int = 10
) -> Dict[int, List[str]]:
    """
    Extract top topics for each cluster.
    
    Args:
        clustering_result: The clustering result
        top_k: Number of top words to extract per cluster
        
    Returns:
        Dictionary mapping cluster_id to list of topic words
    """
    try:
        cluster_topics = {}
        
        for cluster in clustering_result.clusters:
            cluster_topics[cluster.cluster_id] = cluster.topic_words[:top_k]
        
        return cluster_topics
        
    except Exception as e:
        logger.error(f"Failed to extract cluster topics: {e}")
        return {}


def find_outlier_documents(
    clustering_result: ClusteringResult,
    confidence_threshold: float = 0.3
) -> List[str]:
    """
    Find documents that are outliers (low confidence assignments).
    
    Args:
        clustering_result: The clustering result
        confidence_threshold: Confidence threshold for outliers
        
    Returns:
        List of document IDs that are outliers
    """
    try:
        outliers = []
        
        for assignment in clustering_result.assignments:
            if assignment.confidence < confidence_threshold:
                outliers.append(assignment.document_id)
        
        return outliers
        
    except Exception as e:
        logger.error(f"Failed to find outlier documents: {e}")
        return []


def get_cluster_summary(clustering_result: ClusteringResult) -> str:
    """
    Generate a human-readable summary of clustering results.
    
    Args:
        clustering_result: The clustering result
        
    Returns:
        String summary of the clustering
    """
    try:
        stats = get_cluster_statistics(clustering_result)
        
        quality_score_str = f"{clustering_result.overall_quality_score:.3f}" if clustering_result.overall_quality_score is not None else "N/A"
        
        summary = f"""
Clustering Summary:
==================
Total Documents: {clustering_result.total_documents}
Number of Clusters: {clustering_result.num_clusters}
Overall Quality Score: {quality_score_str}

Cluster Distribution:
"""
        
        for cluster in clustering_result.clusters:
            topic_summary = ", ".join(cluster.topic_words[:3])
            summary += f"  Cluster {cluster.cluster_id}: {cluster.document_count} documents\n"
            summary += f"    Topics: {topic_summary}\n"
            if cluster.coherence_score is not None:
                summary += f"    Coherence: {cluster.coherence_score:.3f}\n"
        
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate cluster summary: {e}")
        return f"Error generating summary: {e}"


def _calculate_cluster_similarity(cluster1: ClusterInfo, cluster2: ClusterInfo) -> float:
    """
    Calculate similarity between two clusters based on topic words.
    
    Args:
        cluster1: First cluster
        cluster2: Second cluster
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        words1 = set(cluster1.topic_words)
        words2 = set(cluster2.topic_words)
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Failed to calculate cluster similarity: {e}")
        return 0.0


def validate_clustering_result(clustering_result: ClusteringResult) -> List[str]:
    """
    Validate a clustering result for consistency and quality.
    
    Args:
        clustering_result: The clustering result to validate
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    try:
        # Check for empty clusters
        for cluster in clustering_result.clusters:
            if cluster.document_count == 0:
                warnings.append(f"Cluster {cluster.cluster_id} has no documents")
        
        # Check for orphaned assignments
        cluster_ids = {cluster.cluster_id for cluster in clustering_result.clusters}
        for assignment in clustering_result.assignments:
            if assignment.cluster_id not in cluster_ids:
                warnings.append(f"Assignment for document {assignment.document_id} references non-existent cluster {assignment.cluster_id}")
        
        # Check for low confidence assignments
        low_confidence_count = sum(1 for assignment in clustering_result.assignments if assignment.confidence < 0.3)
        if low_confidence_count > 0:
            warnings.append(f"{low_confidence_count} assignments have low confidence (< 0.3)")
        
        # Check for unbalanced clusters
        cluster_sizes = [cluster.document_count for cluster in clustering_result.clusters]
        if cluster_sizes:
            size_ratio = max(cluster_sizes) / min(cluster_sizes) if min(cluster_sizes) > 0 else float('inf')
            if size_ratio > 10:
                warnings.append(f"Clusters are highly unbalanced (size ratio: {size_ratio:.1f})")
        
        return warnings
        
    except Exception as e:
        logger.error(f"Failed to validate clustering result: {e}")
        return [f"Validation error: {e}"]
