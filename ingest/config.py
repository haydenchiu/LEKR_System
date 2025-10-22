"""
Configuration management for the ingestion module.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsingConfig:
    """Configuration for document parsing."""
    strategy: str = "hi_res"
    skip_infer_table_types: Optional[List[str]] = None
    max_partition: Optional[int] = None
    
    def __post_init__(self):
        if self.skip_infer_table_types is None:
            self.skip_infer_table_types = []


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    max_characters: int = 2048
    combine_text_under_n_chars: int = 256
    new_after_n_chars: int = 1800


@dataclass
class IngestionConfig:
    """Complete configuration for document ingestion."""
    parsing: ParsingConfig
    chunking: ChunkingConfig
    
    @classmethod
    def default(cls) -> "IngestionConfig":
        """Create default configuration."""
        return cls(
            parsing=ParsingConfig(),
            chunking=ChunkingConfig()
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IngestionConfig":
        """Create configuration from dictionary."""
        parsing_config = ParsingConfig(**config_dict.get("parsing", {}))
        chunking_config = ChunkingConfig(**config_dict.get("chunking", {}))
        return cls(parsing=parsing_config, chunking=chunking_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "parsing": {
                "strategy": self.parsing.strategy,
                "skip_infer_table_types": self.parsing.skip_infer_table_types,
                "max_partition": self.parsing.max_partition
            },
            "chunking": {
                "max_characters": self.chunking.max_characters,
                "combine_text_under_n_chars": self.chunking.combine_text_under_n_chars,
                "new_after_n_chars": self.chunking.new_after_n_chars
            }
        }


# Default configurations for different use cases
DEFAULT_CONFIG = IngestionConfig.default()

# Configuration for large documents
LARGE_DOCUMENT_CONFIG = IngestionConfig(
    parsing=ParsingConfig(
        strategy="hi_res",
        max_partition=1000
    ),
    chunking=ChunkingConfig(
        max_characters=4096,
        combine_text_under_n_chars=512,
        new_after_n_chars=3600
    )
)

# Configuration for fast processing
FAST_CONFIG = IngestionConfig(
    parsing=ParsingConfig(
        strategy="fast",
        max_partition=100
    ),
    chunking=ChunkingConfig(
        max_characters=1024,
        combine_text_under_n_chars=128,
        new_after_n_chars=900
    )
)

# Configuration for high-quality processing
HIGH_QUALITY_CONFIG = IngestionConfig(
    parsing=ParsingConfig(
        strategy="hi_res",
        max_partition=None
    ),
    chunking=ChunkingConfig(
        max_characters=2048,
        combine_text_under_n_chars=256,
        new_after_n_chars=1800
    )
)
