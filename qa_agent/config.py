"""
Configuration classes and presets for the QA agent module.

This module provides configuration classes and predefined configuration presets
for the QA functionality, allowing users to customize the behavior
of different agent implementations.
"""

from typing import Dict, Any, Optional, Literal, List
from pydantic import BaseModel, Field, ConfigDict


AgentType = Literal["retrieval", "supervisor", "qa_orchestrator"]


class QAConfig(BaseModel):
    """Configuration for the QA agent module."""

    # LLM Configuration
    llm_model: str = Field(
        default="gpt-4o",
        description="The LLM model to use for answer generation."
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation (0.0 for deterministic, 2.0 for creative)."
    )
    
    # Retriever Configuration
    retriever_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for the retriever module."
    )
    max_retrieved_docs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of documents to retrieve for answering."
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for retrieved documents."
    )
    
    # Answer Generation Configuration
    max_answer_length: int = Field(
        default=2000,
        ge=100,
        le=5000,
        description="Maximum length of generated answers."
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source citations in answers."
    )
    answer_style: Literal["concise", "detailed", "academic"] = Field(
        default="detailed",
        description="Style of answer generation."
    )
    
    # Session Configuration
    session_timeout: int = Field(
        default=3600,
        ge=60,
        description="Session timeout in seconds."
    )
    max_conversation_turns: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of conversation turns to remember."
    )
    enable_context_memory: bool = Field(
        default=True,
        description="Whether to maintain conversation context."
    )
    
    # Advanced Configuration
    enable_reranking: bool = Field(
        default=False,
        description="Whether to rerank retrieved documents."
    )
    reranking_model: Optional[str] = Field(
        default=None,
        description="Model to use for reranking documents."
    )
    enable_fact_checking: bool = Field(
        default=False,
        description="Whether to perform fact checking on answers."
    )
    fact_checking_model: Optional[str] = Field(
        default=None,
        description="Model to use for fact checking."
    )
    
    # System Configuration
    timeout: int = Field(
        default=30,
        ge=5,
        description="Timeout for individual operations in seconds."
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for failed operations."
    )
    enable_logging: bool = Field(
        default=True,
        description="Whether to enable detailed logging."
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level."
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

    def update(self, **kwargs) -> 'QAConfig':
        """Create a new configuration with updated values."""
        current_dict = self.model_dump()
        current_dict.update(kwargs)
        return QAConfig(**current_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QAConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"QAConfig(model={self.llm_model}, style={self.answer_style}, "
                f"max_docs={self.max_retrieved_docs})")

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, QAConfig):
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
DEFAULT_QA_CONFIG = QAConfig()

FAST_QA_CONFIG = QAConfig(
    llm_model="gpt-4o-mini",
    llm_temperature=0.1,
    max_retrieved_docs=3,
    similarity_threshold=0.8,
    max_answer_length=1000,
    answer_style="concise",
    session_timeout=1800,
    max_conversation_turns=5,
    enable_context_memory=False,
    timeout=15,
    max_retries=2
)

HIGH_QUALITY_QA_CONFIG = QAConfig(
    llm_model="gpt-4o",
    llm_temperature=0.0,
    max_retrieved_docs=10,
    similarity_threshold=0.6,
    max_answer_length=3000,
    answer_style="academic",
    session_timeout=7200,
    max_conversation_turns=20,
    enable_context_memory=True,
    enable_reranking=True,
    reranking_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    enable_fact_checking=True,
    fact_checking_model="gpt-4o",
    timeout=60,
    max_retries=5,
    log_level="DEBUG"
)

ACADEMIC_QA_CONFIG = QAConfig(
    llm_model="gpt-4o",
    llm_temperature=0.0,
    max_retrieved_docs=8,
    similarity_threshold=0.7,
    max_answer_length=2500,
    answer_style="academic",
    include_sources=True,
    session_timeout=5400,
    max_conversation_turns=15,
    enable_context_memory=True,
    enable_reranking=True,
    enable_fact_checking=True,
    timeout=45,
    max_retries=3
)

CONVERSATIONAL_QA_CONFIG = QAConfig(
    llm_model="gpt-4o",
    llm_temperature=0.3,
    max_retrieved_docs=5,
    similarity_threshold=0.7,
    max_answer_length=1500,
    answer_style="detailed",
    include_sources=True,
    session_timeout=3600,
    max_conversation_turns=25,
    enable_context_memory=True,
    timeout=30,
    max_retries=3
)

# Configuration registry for easy access
CONFIG_REGISTRY = {
    "default": DEFAULT_QA_CONFIG,
    "fast": FAST_QA_CONFIG,
    "high_quality": HIGH_QUALITY_QA_CONFIG,
    "academic": ACADEMIC_QA_CONFIG,
    "conversational": CONVERSATIONAL_QA_CONFIG
}


def get_config(config_name: str = "default") -> QAConfig:
    """
    Get a predefined configuration by name.
    
    Args:
        config_name: Name of the configuration preset
        
    Returns:
        QAConfig instance
        
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


def create_custom_config(**kwargs) -> QAConfig:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        QAConfig instance with custom settings
    """
    return DEFAULT_QA_CONFIG.update(**kwargs)
