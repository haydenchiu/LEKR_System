"""
Configuration classes and presets for clustering.

This module provides configuration classes and predefined configuration presets
for the clustering functionality, allowing users to customize the behavior
of the BERTopic-based clustering process.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict


class ClusteringConfig(BaseModel):
    """Configuration for the clustering module."""

    # BERTopic parameters
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    umap_model: Optional[Dict[str, Any]] = Field(
        default=None,
        description="UMAP parameters for dimensionality reduction"
    )
    hdbscan_model: Optional[Dict[str, Any]] = Field(
        default=None,
        description="HDBSCAN parameters for clustering"
    )
    vectorizer_model: Optional[Dict[str, Any]] = Field(
        default=None,
        description="CountVectorizer parameters for topic modeling"
    )
    ctfidf_model: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Class-based TF-IDF parameters"
    )
    
    # Clustering behavior
    min_cluster_size: int = Field(
        default=3,
        ge=1,
        description="Minimum number of documents per cluster"
    )
    min_samples: int = Field(
        default=1,
        ge=1,
        description="Minimum number of samples in a neighborhood for a core point"
    )
    cluster_selection_epsilon: float = Field(
        default=0.0,
        ge=0.0,
        description="Distance threshold for cluster selection"
    )
    
    # Topic modeling
    nr_topics: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of topics to reduce to"
    )
    top_k_words: int = Field(
        default=10,
        ge=1,
        description="Number of top words to extract per topic"
    )
    
    # Quality metrics
    calculate_probabilities: bool = Field(
        default=True,
        description="Whether to calculate assignment probabilities"
    )
    calculate_distances: bool = Field(
        default=True,
        description="Whether to calculate distances to cluster centers"
    )
    
    # Performance
    n_jobs: int = Field(
        default=1,
        ge=1,
        description="Number of parallel jobs for computation"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to show progress information"
    )
    
    # Advanced parameters
    embedding_model: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom embedding model parameters"
    )
    representation_model: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom representation model parameters"
    )
    
    # Clustering thresholds
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for cluster assignment"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for merging similar clusters"
    )
    
    # Metadata
    clustering_method: str = Field(
        default="BERTopic",
        description="Method used for clustering"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the configuration"
    )

    model_config = ConfigDict(
        json_encoders={
            # Add any custom encoders if needed
        }
    )

    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def update(self, **kwargs) -> 'ClusteringConfig':
        """Create a new configuration with updated values."""
        current_dict = self.model_dump()
        current_dict.update(kwargs)
        return ClusteringConfig(**current_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"ClusteringConfig(model={self.model_name}, "
                f"min_cluster_size={self.min_cluster_size}, "
                f"nr_topics={self.nr_topics})")

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, ClusteringConfig):
            return False
        return self.model_dump() == other.model_dump()

    def __hash__(self) -> int:
        """Hash for configuration."""
        # Convert to hashable tuple for hashing, handling unhashable dict types
        config_dict = self.model_dump()
        hashable_items = []
        for key, value in sorted(config_dict.items()):
            if isinstance(value, dict):
                # Convert dict to sorted tuple of items
                hashable_items.append((key, tuple(sorted(value.items()))))
            elif isinstance(value, list):
                # Convert list to tuple
                hashable_items.append((key, tuple(value)))
            else:
                hashable_items.append((key, value))
        config_tuple = tuple(hashable_items)
        return hash(config_tuple)

    def get_bertopic_params(self) -> Dict[str, Any]:
        """Get parameters for BERTopic initialization."""
        params = {
            "embedding_model": self.model_name,
            "verbose": self.verbose,
            "calculate_probabilities": self.calculate_probabilities
        }
        
        if self.umap_model:
            params["umap_model"] = self.umap_model
        if self.hdbscan_model:
            params["hdbscan_model"] = self.hdbscan_model
        if self.vectorizer_model:
            params["vectorizer_model"] = self.vectorizer_model
        if self.ctfidf_model:
            params["ctfidf_model"] = self.ctfidf_model
        if self.embedding_model:
            params["embedding_model"] = self.embedding_model
        if self.representation_model:
            params["representation_model"] = self.representation_model
            
        return params

    def get_hdbscan_params(self) -> Dict[str, Any]:
        """Get HDBSCAN parameters."""
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_epsilon": self.cluster_selection_epsilon
        }

    def get_umap_params(self) -> Dict[str, Any]:
        """Get UMAP parameters."""
        default_params = {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine"
        }
        if self.umap_model:
            default_params.update(self.umap_model)
        return default_params


# Predefined configurations
DEFAULT_CLUSTERING_CONFIG = ClusteringConfig()

FAST_CLUSTERING_CONFIG = ClusteringConfig(
    model_name="all-MiniLM-L6-v2",
    min_cluster_size=5,
    min_samples=2,
    top_k_words=5,
    n_jobs=2,
    verbose=True,
    umap_model={
        "n_neighbors": 10,
        "n_components": 3,
        "min_dist": 0.1
    },
    hdbscan_model={
        "min_cluster_size": 5,
        "min_samples": 2
    }
)

HIGH_QUALITY_CLUSTERING_CONFIG = ClusteringConfig(
    model_name="all-mpnet-base-v2",
    min_cluster_size=3,
    min_samples=1,
    top_k_words=15,
    calculate_probabilities=True,
    calculate_distances=True,
    n_jobs=4,
    verbose=True,
    umap_model={
        "n_neighbors": 20,
        "n_components": 10,
        "min_dist": 0.0,
        "metric": "cosine"
    },
    hdbscan_model={
        "min_cluster_size": 3,
        "min_samples": 1,
        "cluster_selection_epsilon": 0.1
    }
)

BALANCED_CLUSTERING_CONFIG = ClusteringConfig(
    model_name="all-MiniLM-L12-v2",
    min_cluster_size=4,
    min_samples=2,
    top_k_words=10,
    calculate_probabilities=True,
    n_jobs=2,
    verbose=False,
    umap_model={
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.05
    },
    hdbscan_model={
        "min_cluster_size": 4,
        "min_samples": 2
    }
)

# Configuration registry for easy access
CONFIG_REGISTRY = {
    "default": DEFAULT_CLUSTERING_CONFIG,
    "fast": FAST_CLUSTERING_CONFIG,
    "high_quality": HIGH_QUALITY_CLUSTERING_CONFIG,
    "balanced": BALANCED_CLUSTERING_CONFIG
}


def get_config(config_name: str = "default") -> ClusteringConfig:
    """
    Get a predefined configuration by name.
    
    Args:
        config_name: Name of the configuration preset
        
    Returns:
        ClusteringConfig instance
        
    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in CONFIG_REGISTRY:
        available_configs = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config name '{config_name}'. Available configs: {available_configs}")
    
    return CONFIG_REGISTRY[config_name]


def list_available_configs() -> List[str]:
    """
    List all available configuration presets.
    
    Returns:
        List of available configuration names
    """
    return list(CONFIG_REGISTRY.keys())


def create_custom_config(**kwargs) -> ClusteringConfig:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        ClusteringConfig instance with custom settings
    """
    return DEFAULT_CLUSTERING_CONFIG.update(**kwargs)
