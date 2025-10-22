"""
Unit tests for the config module.
"""

import pytest
from ingest.config import (
    ParsingConfig,
    ChunkingConfig,
    IngestionConfig,
    DEFAULT_CONFIG,
    LARGE_DOCUMENT_CONFIG,
    FAST_CONFIG,
    HIGH_QUALITY_CONFIG
)


class TestParsingConfig:
    """Test cases for ParsingConfig class."""
    
    def test_init_default(self):
        """Test ParsingConfig initialization with default values."""
        config = ParsingConfig()
        
        assert config.strategy == "hi_res"
        assert config.skip_infer_table_types == []
        assert config.max_partition is None
    
    def test_init_custom(self):
        """Test ParsingConfig initialization with custom values."""
        config = ParsingConfig(
            strategy="fast",
            skip_infer_table_types=["pdf", "html"],
            max_partition=100
        )
        
        assert config.strategy == "fast"
        assert config.skip_infer_table_types == ["pdf", "html"]
        assert config.max_partition == 100
    
    def test_post_init_empty_list(self):
        """Test that skip_infer_table_types is initialized as empty list if None."""
        config = ParsingConfig(skip_infer_table_types=None)
        assert config.skip_infer_table_types == []


class TestChunkingConfig:
    """Test cases for ChunkingConfig class."""
    
    def test_init_default(self):
        """Test ChunkingConfig initialization with default values."""
        config = ChunkingConfig()
        
        assert config.max_characters == 2048
        assert config.combine_text_under_n_chars == 256
        assert config.new_after_n_chars == 1800
    
    def test_init_custom(self):
        """Test ChunkingConfig initialization with custom values."""
        config = ChunkingConfig(
            max_characters=4096,
            combine_text_under_n_chars=512,
            new_after_n_chars=3600
        )
        
        assert config.max_characters == 4096
        assert config.combine_text_under_n_chars == 512
        assert config.new_after_n_chars == 3600


class TestIngestionConfig:
    """Test cases for IngestionConfig class."""
    
    def test_init(self):
        """Test IngestionConfig initialization."""
        parsing_config = ParsingConfig(strategy="fast")
        chunking_config = ChunkingConfig(max_characters=1024)
        
        config = IngestionConfig(
            parsing=parsing_config,
            chunking=chunking_config
        )
        
        assert config.parsing.strategy == "fast"
        assert config.chunking.max_characters == 1024
    
    def test_default_classmethod(self):
        """Test IngestionConfig.default() class method."""
        config = IngestionConfig.default()
        
        assert isinstance(config, IngestionConfig)
        assert isinstance(config.parsing, ParsingConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert config.parsing.strategy == "hi_res"
        assert config.chunking.max_characters == 2048
    
    def test_from_dict(self):
        """Test IngestionConfig.from_dict() class method."""
        config_dict = {
            "parsing": {
                "strategy": "fast",
                "skip_infer_table_types": ["pdf"],
                "max_partition": 100
            },
            "chunking": {
                "max_characters": 1024,
                "combine_text_under_n_chars": 128,
                "new_after_n_chars": 900
            }
        }
        
        config = IngestionConfig.from_dict(config_dict)
        
        assert config.parsing.strategy == "fast"
        assert config.parsing.skip_infer_table_types == ["pdf"]
        assert config.parsing.max_partition == 100
        assert config.chunking.max_characters == 1024
        assert config.chunking.combine_text_under_n_chars == 128
        assert config.chunking.new_after_n_chars == 900
    
    def test_from_dict_empty(self):
        """Test IngestionConfig.from_dict() with empty dictionary."""
        config = IngestionConfig.from_dict({})
        
        # Should use default values
        assert config.parsing.strategy == "hi_res"
        assert config.parsing.skip_infer_table_types == []
        assert config.parsing.max_partition is None
        assert config.chunking.max_characters == 2048
        assert config.chunking.combine_text_under_n_chars == 256
        assert config.chunking.new_after_n_chars == 1800
    
    def test_from_dict_partial(self):
        """Test IngestionConfig.from_dict() with partial dictionary."""
        config_dict = {
            "parsing": {
                "strategy": "fast"
            }
        }
        
        config = IngestionConfig.from_dict(config_dict)
        
        # Parsing should have custom values
        assert config.parsing.strategy == "fast"
        assert config.parsing.skip_infer_table_types == []
        assert config.parsing.max_partition is None
        
        # Chunking should have default values
        assert config.chunking.max_characters == 2048
        assert config.chunking.combine_text_under_n_chars == 256
        assert config.chunking.new_after_n_chars == 1800
    
    def test_to_dict(self):
        """Test IngestionConfig.to_dict() method."""
        config = IngestionConfig.default()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "parsing" in config_dict
        assert "chunking" in config_dict
        
        # Check parsing values
        assert config_dict["parsing"]["strategy"] == "hi_res"
        assert config_dict["parsing"]["skip_infer_table_types"] == []
        assert config_dict["parsing"]["max_partition"] is None
        
        # Check chunking values
        assert config_dict["chunking"]["max_characters"] == 2048
        assert config_dict["chunking"]["combine_text_under_n_chars"] == 256
        assert config_dict["chunking"]["new_after_n_chars"] == 1800
    
    def test_roundtrip_dict(self):
        """Test that to_dict() and from_dict() are inverse operations."""
        original_config = IngestionConfig.default()
        config_dict = original_config.to_dict()
        restored_config = IngestionConfig.from_dict(config_dict)
        
        assert restored_config.parsing.strategy == original_config.parsing.strategy
        assert restored_config.parsing.skip_infer_table_types == original_config.parsing.skip_infer_table_types
        assert restored_config.parsing.max_partition == original_config.parsing.max_partition
        assert restored_config.chunking.max_characters == original_config.chunking.max_characters
        assert restored_config.chunking.combine_text_under_n_chars == original_config.chunking.combine_text_under_n_chars
        assert restored_config.chunking.new_after_n_chars == original_config.chunking.new_after_n_chars


class TestPresetConfigurations:
    """Test cases for preset configurations."""
    
    def test_default_config(self):
        """Test DEFAULT_CONFIG preset."""
        config = DEFAULT_CONFIG
        
        assert isinstance(config, IngestionConfig)
        assert config.parsing.strategy == "hi_res"
        assert config.parsing.max_partition is None
        assert config.chunking.max_characters == 2048
        assert config.chunking.combine_text_under_n_chars == 256
        assert config.chunking.new_after_n_chars == 1800
    
    def test_large_document_config(self):
        """Test LARGE_DOCUMENT_CONFIG preset."""
        config = LARGE_DOCUMENT_CONFIG
        
        assert isinstance(config, IngestionConfig)
        assert config.parsing.strategy == "hi_res"
        assert config.parsing.max_partition == 1000
        assert config.chunking.max_characters == 4096
        assert config.chunking.combine_text_under_n_chars == 512
        assert config.chunking.new_after_n_chars == 3600
    
    def test_fast_config(self):
        """Test FAST_CONFIG preset."""
        config = FAST_CONFIG
        
        assert isinstance(config, IngestionConfig)
        assert config.parsing.strategy == "fast"
        assert config.parsing.max_partition == 100
        assert config.chunking.max_characters == 1024
        assert config.chunking.combine_text_under_n_chars == 128
        assert config.chunking.new_after_n_chars == 900
    
    def test_high_quality_config(self):
        """Test HIGH_QUALITY_CONFIG preset."""
        config = HIGH_QUALITY_CONFIG
        
        assert isinstance(config, IngestionConfig)
        assert config.parsing.strategy == "hi_res"
        assert config.parsing.max_partition is None
        assert config.chunking.max_characters == 2048
        assert config.chunking.combine_text_under_n_chars == 256
        assert config.chunking.new_after_n_chars == 1800
    
    def test_config_differences(self):
        """Test that different configs have different values."""
        configs = [
            DEFAULT_CONFIG,
            LARGE_DOCUMENT_CONFIG,
            FAST_CONFIG,
            HIGH_QUALITY_CONFIG
        ]
        
        # Check that at least some configs have different values
        max_chars_values = [config.chunking.max_characters for config in configs]
        assert len(set(max_chars_values)) > 1  # At least two different values
        
        strategies = [config.parsing.strategy for config in configs]
        assert len(set(strategies)) > 1  # At least two different strategies


class TestConfigValidation:
    """Test cases for configuration validation."""
    
    def test_parsing_config_validation(self):
        """Test that ParsingConfig accepts valid values."""
        # Valid strategy
        config = ParsingConfig(strategy="hi_res")
        assert config.strategy == "hi_res"
        
        config = ParsingConfig(strategy="fast")
        assert config.strategy == "fast"
        
        # Valid skip_infer_table_types
        config = ParsingConfig(skip_infer_table_types=["pdf", "html"])
        assert config.skip_infer_table_types == ["pdf", "html"]
        
        # Valid max_partition
        config = ParsingConfig(max_partition=100)
        assert config.max_partition == 100
        
        config = ParsingConfig(max_partition=None)
        assert config.max_partition is None
    
    def test_chunking_config_validation(self):
        """Test that ChunkingConfig accepts valid values."""
        # Valid max_characters
        config = ChunkingConfig(max_characters=1024)
        assert config.max_characters == 1024
        
        config = ChunkingConfig(max_characters=4096)
        assert config.max_characters == 4096
        
        # Valid combine_text_under_n_chars
        config = ChunkingConfig(combine_text_under_n_chars=128)
        assert config.combine_text_under_n_chars == 128
        
        # Valid new_after_n_chars
        config = ChunkingConfig(new_after_n_chars=900)
        assert config.new_after_n_chars == 900
    
    def test_ingestion_config_immutability(self):
        """Test that IngestionConfig components are properly set."""
        config = IngestionConfig.default()
        
        # Components should be properly initialized
        assert hasattr(config, 'parsing')
        assert hasattr(config, 'chunking')
        assert config.parsing is not None
        assert config.chunking is not None
        
        # Components should be the correct types
        assert isinstance(config.parsing, ParsingConfig)
        assert isinstance(config.chunking, ChunkingConfig)
