"""
Configuration classes and presets for the retriever module.

This module provides configuration classes and predefined configuration presets
for the retriever functionality, allowing users to customize retrieval behavior
including search strategies, ranking algorithms, and performance settings.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class SearchStrategy(str, Enum):
    """Available search strategies for retrieval."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    CONTEXT_AWARE = "context_aware"


class RankingMethod(str, Enum):
    """Available ranking methods for retrieval results."""
    RELEVANCE = "relevance"
    QUALITY = "quality"
    COMBINED = "combined"
    RECENCY = "recency"


class RetrieverConfig(BaseModel):
    """Configuration for the retriever module."""
    
    # Search configuration
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.SEMANTIC,
        description="Primary search strategy to use"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for results"
    )
    
    # Embedding configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        description="Dimension of embedding vectors"
    )
    
    # Vector database configuration
    vector_store_type: str = Field(
        default="qdrant",
        description="Type of vector database to use"
    )
    collection_name: str = Field(
        default="lerk_knowledge",
        description="Name of the vector collection"
    )
    host: str = Field(
        default="localhost",
        description="Vector database host"
    )
    port: int = Field(
        default=6333,
        ge=1,
        le=65535,
        description="Vector database port"
    )
    
    # Hybrid search configuration
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid semantic + keyword search"
    )
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in hybrid mode"
    )
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in hybrid mode"
    )
    
    # Context-aware configuration
    enable_context_awareness: bool = Field(
        default=True,
        description="Enable context-aware retrieval"
    )
    context_window_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of previous queries to consider for context"
    )
    context_decay_factor: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Decay factor for older context"
    )
    
    # Ranking configuration
    ranking_method: RankingMethod = Field(
        default=RankingMethod.COMBINED,
        description="Method for ranking retrieval results"
    )
    quality_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for quality score in ranking"
    )
    relevance_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for relevance score in ranking"
    )
    
    # Performance configuration
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for batch retrieval operations"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout for retrieval operations in seconds"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of retrieval results"
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="Cache time-to-live in seconds"
    )
    
    # Metadata filtering
    enable_metadata_filtering: bool = Field(
        default=True,
        description="Enable filtering by metadata"
    )
    default_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default metadata filters to apply"
    )
    
    # Advanced configuration
    enable_reranking: bool = Field(
        default=False,
        description="Enable result reranking using advanced algorithms"
    )
    reranking_model: Optional[str] = Field(
        default=None,
        description="Model to use for reranking (if enabled)"
    )
    enable_diversity: bool = Field(
        default=True,
        description="Enable diversity in retrieval results"
    )
    diversity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for diversity filtering"
    )
    
    # Custom parameters
    custom_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom parameters for specific retrieval strategies"
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
    
    def update(self, **kwargs) -> 'RetrieverConfig':
        """Create a new configuration with updated values."""
        current_dict = self.model_dump()
        current_dict.update(kwargs)
        return RetrieverConfig(**current_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RetrieverConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"RetrieverConfig(strategy={self.search_strategy}, "
                f"max_results={self.max_results}, threshold={self.similarity_threshold})")
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, RetrieverConfig):
            return False
        return self.model_dump() == other.model_dump()
    
    def __hash__(self) -> int:
        """Hash for configuration."""
        # Convert to hashable tuple for hashing, handling nested dicts
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)
            else:
                return obj
        
        config_dict = self.model_dump()
        config_tuple = tuple(sorted((k, make_hashable(v)) for k, v in config_dict.items()))
        return hash(config_tuple)


# Predefined configurations
DEFAULT_RETRIEVER_CONFIG = RetrieverConfig()

FAST_RETRIEVER_CONFIG = RetrieverConfig(
    search_strategy=SearchStrategy.SEMANTIC,
    max_results=5,
    similarity_threshold=0.6,
    embedding_model="all-MiniLM-L6-v2",
    enable_hybrid_search=False,
    enable_context_awareness=False,
    enable_metadata_filtering=False,
    enable_reranking=False,
    enable_diversity=False,
    batch_size=20,
    timeout=10,
    enable_caching=True,
    cache_ttl=1800
)

HIGH_QUALITY_RETRIEVER_CONFIG = RetrieverConfig(
    search_strategy=SearchStrategy.HYBRID,
    max_results=20,
    similarity_threshold=0.8,
    embedding_model="all-mpnet-base-v2",
    embedding_dimension=768,
    enable_hybrid_search=True,
    semantic_weight=0.8,
    keyword_weight=0.2,
    enable_context_awareness=True,
    context_window_size=10,
    ranking_method=RankingMethod.COMBINED,
    quality_weight=0.5,
    relevance_weight=0.5,
    enable_reranking=True,
    reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    enable_diversity=True,
    diversity_threshold=0.4,
    batch_size=5,
    timeout=60,
    enable_caching=True,
    cache_ttl=7200
)

SEMANTIC_RETRIEVER_CONFIG = RetrieverConfig(
    search_strategy=SearchStrategy.SEMANTIC,
    max_results=15,
    similarity_threshold=0.75,
    embedding_model="all-mpnet-base-v2",
    embedding_dimension=768,
    enable_hybrid_search=False,
    enable_context_awareness=True,
    context_window_size=3,
    ranking_method=RankingMethod.RELEVANCE,
    relevance_weight=1.0,
    quality_weight=0.0,
    enable_reranking=False,
    enable_diversity=True,
    diversity_threshold=0.3,
    batch_size=10,
    timeout=30
)

HYBRID_RETRIEVER_CONFIG = RetrieverConfig(
    search_strategy=SearchStrategy.HYBRID,
    max_results=12,
    similarity_threshold=0.7,
    embedding_model="all-mpnet-base-v2",
    embedding_dimension=768,
    enable_hybrid_search=True,
    semantic_weight=0.6,
    keyword_weight=0.4,
    enable_context_awareness=True,
    context_window_size=5,
    ranking_method=RankingMethod.COMBINED,
    relevance_weight=0.6,
    quality_weight=0.4,
    enable_reranking=True,
    reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    enable_diversity=True,
    diversity_threshold=0.35,
    batch_size=8,
    timeout=45
)

CONTEXT_AWARE_RETRIEVER_CONFIG = RetrieverConfig(
    search_strategy=SearchStrategy.CONTEXT_AWARE,
    max_results=10,
    similarity_threshold=0.75,
    embedding_model="all-mpnet-base-v2",
    embedding_dimension=768,
    enable_hybrid_search=True,
    semantic_weight=0.7,
    keyword_weight=0.3,
    enable_context_awareness=True,
    context_window_size=8,
    context_decay_factor=0.7,
    ranking_method=RankingMethod.COMBINED,
    relevance_weight=0.7,
    quality_weight=0.3,
    enable_reranking=True,
    reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    enable_diversity=True,
    diversity_threshold=0.4,
    batch_size=6,
    timeout=50
)

# Configuration registry for easy access
CONFIG_REGISTRY = {
    "default": DEFAULT_RETRIEVER_CONFIG,
    "fast": FAST_RETRIEVER_CONFIG,
    "high_quality": HIGH_QUALITY_RETRIEVER_CONFIG,
    "semantic": SEMANTIC_RETRIEVER_CONFIG,
    "hybrid": HYBRID_RETRIEVER_CONFIG,
    "context_aware": CONTEXT_AWARE_RETRIEVER_CONFIG
}


def get_config(config_name: str = "default") -> RetrieverConfig:
    """
    Get a predefined configuration by name.
    
    Args:
        config_name: Name of the configuration preset
        
    Returns:
        RetrieverConfig instance
        
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


def create_custom_config(**kwargs) -> RetrieverConfig:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        RetrieverConfig instance with custom settings
    """
    return DEFAULT_RETRIEVER_CONFIG.update(**kwargs)
