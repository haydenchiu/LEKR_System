"""
Configuration classes and presets for logic extraction.

This module provides configuration classes and predefined configuration presets
for the logic extraction functionality, allowing users to customize the behavior
of the LLM-based logic extraction process.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class LogicExtractionConfig(BaseModel):
    """Configuration for the logic extraction module."""

    model_name: str = Field(
        default="gpt-4o-mini",
        description="The LLM model to use for logic extraction."
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The temperature for LLM generation."
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for LLM calls."
    )
    timeout: int = Field(
        default=60,
        ge=1,
        description="Timeout for LLM calls in seconds."
    )
    batch_size: int = Field(
        default=5,
        ge=1,
        description="Number of chunks to process concurrently in async mode."
    )

    model_config = ConfigDict(
        json_encoders={
            # Add any custom encoders if needed
        }
    )

    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)


# Predefined configurations
DEFAULT_LOGIC_EXTRACTION_CONFIG = LogicExtractionConfig()

FAST_LOGIC_EXTRACTION_CONFIG = LogicExtractionConfig(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    batch_size=10
)

HIGH_QUALITY_LOGIC_EXTRACTION_CONFIG = LogicExtractionConfig(
    model_name="gpt-4o",
    temperature=0.0,
    max_retries=5
)