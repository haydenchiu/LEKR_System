"""
Clustering Module

This module provides functionality for clustering documents into subject clusters using BERTopic.
It supports dynamic clustering, reassignment, and cluster management for the LERK System.

Main Components:
- Models: Pydantic models for clustering results and cluster information
- Clusterer: Core clustering functionality using BERTopic
- Config: Configuration classes and presets for clustering parameters
- Utils: Utility functions for cluster operations and analysis
- Exceptions: Custom exception classes for error handling

Example Usage:
    from clustering import DocumentClusterer, DEFAULT_CLUSTERING_CONFIG
    
    clusterer = DocumentClusterer()
    clusters = clusterer.fit_clusters(documents)
"""

from .config import (
    ClusteringConfig,
    DEFAULT_CLUSTERING_CONFIG,
    FAST_CLUSTERING_CONFIG,
    HIGH_QUALITY_CLUSTERING_CONFIG
)
from .exceptions import (
    ClusteringError,
    InvalidDocumentError,
    ClusteringInitializationError,
    ClusterAssignmentError
)
from .models import (
    ClusterInfo,
    ClusteringResult,
    DocumentClusterAssignment
)
from .clusterer import (
    DocumentClusterer,
    assign_documents_to_clusters,
    reassign_documents_to_clusters
)
from .dynamic_clustering import (
    DynamicClusteringManager,
    create_dynamic_clustering_manager,
    process_new_documents_async
)
from .utils import (
    analyze_cluster_quality,
    get_cluster_statistics,
    merge_similar_clusters,
    extract_cluster_topics
)

__version__ = "1.0.0"
__author__ = "LERK System Team"

# Public API
__all__ = [
    # Models
    "ClusterInfo",
    "ClusteringResult", 
    "DocumentClusterAssignment",

    # Configuration
    "ClusteringConfig",
    "DEFAULT_CLUSTERING_CONFIG",
    "FAST_CLUSTERING_CONFIG",
    "HIGH_QUALITY_CLUSTERING_CONFIG",

    # Core functionality
    "DocumentClusterer",
    "assign_documents_to_clusters",
    "reassign_documents_to_clusters",

    # Dynamic clustering
    "DynamicClusteringManager",
    "create_dynamic_clustering_manager",
    "process_new_documents_async",

    # Utilities
    "analyze_cluster_quality",
    "get_cluster_statistics",
    "merge_similar_clusters",
    "extract_cluster_topics",

    # Exceptions
    "ClusteringError",
    "InvalidDocumentError",
    "ClusteringInitializationError",
    "ClusterAssignmentError",
]
