"""
Configuration classes and presets for consolidation.

This module provides configuration classes and predefined configuration presets
for the consolidation functionality, allowing users to customize the behavior
of the knowledge consolidation process.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict


class ConsolidationConfig(BaseModel):
    """Configuration for the consolidation module."""

    # LLM Configuration
    model_name: str = Field(
        default="gpt-4o-mini",
        description="The LLM model to use for consolidation"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Temperature for LLM generation"
    )
    max_tokens: int = Field(
        default=4000,
        ge=100,
        description="Maximum tokens for LLM responses"
    )
    
    # Document Consolidation
    max_concepts_per_document: int = Field(
        default=20,
        ge=1,
        description="Maximum number of concepts to extract per document"
    )
    min_concept_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for concept extraction"
    )
    max_relations_per_document: int = Field(
        default=30,
        ge=1,
        description="Maximum number of relations to extract per document"
    )
    
    # Subject Consolidation
    max_concepts_per_subject: int = Field(
        default=50,
        ge=1,
        description="Maximum number of concepts to consolidate per subject"
    )
    concept_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for merging similar concepts"
    )
    relation_strength_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum strength for keeping relations"
    )
    
    # Storage Configuration
    storage_backend: str = Field(
        default="memory",
        description="Storage backend (memory, sqlite, postgresql)"
    )
    enable_vector_search: bool = Field(
        default=True,
        description="Enable vector embeddings for similarity search"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model for generating embeddings"
    )
    
    # Processing Configuration
    batch_size: int = Field(
        default=5,
        ge=1,
        description="Number of documents to process in each batch"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed operations"
    )
    timeout: int = Field(
        default=120,
        ge=1,
        description="Timeout for operations in seconds"
    )
    
    # Quality Control
    enable_quality_validation: bool = Field(
        default=True,
        description="Enable quality validation for consolidated knowledge"
    )
    min_quality_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for accepting consolidated knowledge"
    )
    
    # Advanced Options
    custom_prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom prompts for different consolidation tasks"
    )
    llm_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters to pass to the LLM"
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

    def update(self, **kwargs) -> 'ConsolidationConfig':
        """Create a new configuration with updated values."""
        current_dict = self.model_dump()
        current_dict.update(kwargs)
        return ConsolidationConfig(**current_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConsolidationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"ConsolidationConfig(model={self.model_name}, "
                f"temp={self.temperature}, batch_size={self.batch_size})")

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, ConsolidationConfig):
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


# Predefined configurations
DEFAULT_CONSOLIDATION_CONFIG = ConsolidationConfig()

FAST_CONSOLIDATION_CONFIG = ConsolidationConfig(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=2000,
    max_concepts_per_document=10,
    max_relations_per_document=15,
    max_concepts_per_subject=25,
    batch_size=10,
    timeout=60,
    min_quality_score=0.5
)

HIGH_QUALITY_CONSOLIDATION_CONFIG = ConsolidationConfig(
    model_name="gpt-4o",
    temperature=0.0,
    max_tokens=8000,
    max_concepts_per_document=30,
    max_relations_per_document=50,
    max_concepts_per_subject=100,
    concept_similarity_threshold=0.8,
    relation_strength_threshold=0.6,
    batch_size=3,
    timeout=300,
    min_quality_score=0.8,
    enable_quality_validation=True
)

BALANCED_CONSOLIDATION_CONFIG = ConsolidationConfig(
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=4000,
    max_concepts_per_document=20,
    max_relations_per_document=30,
    max_concepts_per_subject=50,
    concept_similarity_threshold=0.7,
    relation_strength_threshold=0.5,
    batch_size=5,
    timeout=120,
    min_quality_score=0.6
)

# Configuration registry for easy access
CONFIG_REGISTRY = {
    "default": DEFAULT_CONSOLIDATION_CONFIG,
    "fast": FAST_CONSOLIDATION_CONFIG,
    "high_quality": HIGH_QUALITY_CONSOLIDATION_CONFIG,
    "balanced": BALANCED_CONSOLIDATION_CONFIG
}


def get_config(config_name: str = "default") -> ConsolidationConfig:
    """
    Get a predefined configuration by name.
    
    Args:
        config_name: Name of the configuration preset
        
    Returns:
        ConsolidationConfig instance
        
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


def create_custom_config(**kwargs) -> ConsolidationConfig:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        ConsolidationConfig instance with custom settings
    """
    return DEFAULT_CONSOLIDATION_CONFIG.update(**kwargs)
