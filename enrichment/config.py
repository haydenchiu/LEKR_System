"""
Configuration classes for document enrichment.

Defines configuration options for enrichment processing including
LLM settings, batch processing, and enrichment parameters.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class EnrichmentConfig(BaseModel):
    """Configuration for document enrichment processing."""
    
    # LLM Configuration
    model_name: str = Field(
        default="gpt-4o-mini",
        description="LLM model name for enrichment"
    )
    
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens for LLM response"
    )
    
    # Batch Processing
    batch_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to process concurrently"
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed enrichments"
    )
    
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay between retry attempts in seconds"
    )
    
    # Enrichment Parameters
    max_keywords: int = Field(
        default=7,
        ge=1,
        le=20,
        description="Maximum number of keywords to extract"
    )
    
    max_questions: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum number of hypothetical questions"
    )
    
    summary_max_length: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Maximum length of summary in characters"
    )
    
    # Processing Options
    enable_table_summary: bool = Field(
        default=True,
        description="Whether to generate table summaries"
    )
    
    enable_async_processing: bool = Field(
        default=True,
        description="Whether to use async processing"
    )
    
    # Additional LLM Parameters
    llm_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters to pass to the LLM"
    )
    
    model_config = ConfigDict(
        json_encoders={
            # Add any custom encoders if needed
        }
    )
    
    @classmethod
    def default(cls) -> "EnrichmentConfig":
        """Create default enrichment configuration."""
        return cls()
    
    @classmethod
    def fast(cls) -> "EnrichmentConfig":
        """Create fast processing configuration."""
        return cls(
            batch_size=10,
            max_retries=1,
            retry_delay=0.5,
            max_keywords=5,
            max_questions=3,
            summary_max_length=150
        )
    
    @classmethod
    def high_quality(cls) -> "EnrichmentConfig":
        """Create high-quality processing configuration."""
        return cls(
            model_name="gpt-4o",
            temperature=0.1,
            batch_size=3,
            max_retries=5,
            retry_delay=2.0,
            max_keywords=10,
            max_questions=7,
            summary_max_length=300
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnrichmentConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def update(self, **kwargs) -> "EnrichmentConfig":
        """Update configuration with new values."""
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return self.__class__(**current_dict)


# Predefined configurations
DEFAULT_ENRICHMENT_CONFIG = EnrichmentConfig.default()
FAST_ENRICHMENT_CONFIG = EnrichmentConfig.fast()
HIGH_QUALITY_ENRICHMENT_CONFIG = EnrichmentConfig.high_quality()
