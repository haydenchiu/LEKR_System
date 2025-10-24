"""
Simple unit tests for the retriever configuration module.

This module tests the RetrieverConfig class and configuration presets
without requiring external dependencies.
"""

import pytest
from pydantic import ValidationError

from retriever.config import (
    RetrieverConfig, DEFAULT_RETRIEVER_CONFIG, FAST_RETRIEVER_CONFIG,
    HIGH_QUALITY_RETRIEVER_CONFIG, SEMANTIC_RETRIEVER_CONFIG,
    HYBRID_RETRIEVER_CONFIG, CONTEXT_AWARE_RETRIEVER_CONFIG,
    SearchStrategy, RankingMethod, get_config, list_available_configs,
    create_custom_config
)


class TestRetrieverConfig:
    """Test cases for the RetrieverConfig class."""
    
    def test_config_creation_default(self):
        """Test default configuration creation."""
        config = RetrieverConfig()
        
        assert config.search_strategy == SearchStrategy.SEMANTIC
        assert config.max_results == 10
        assert config.similarity_threshold == 0.7
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_dimension == 384
        assert config.vector_store_type == "qdrant"
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.enable_hybrid_search == True
        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.enable_context_awareness == True
        assert config.context_window_size == 5
        assert config.ranking_method == RankingMethod.COMBINED
        assert config.enable_diversity == True
    
    def test_config_creation_custom(self):
        """Test custom configuration creation."""
        config = RetrieverConfig(
            search_strategy=SearchStrategy.HYBRID,
            max_results=20,
            similarity_threshold=0.8,
            embedding_model="custom-model",
            enable_hybrid_search=False
        )
        
        assert config.search_strategy == SearchStrategy.HYBRID
        assert config.max_results == 20
        assert config.similarity_threshold == 0.8
        assert config.embedding_model == "custom-model"
        assert config.enable_hybrid_search == False
    
    def test_config_validation_similarity_threshold(self):
        """Test similarity threshold validation."""
        # Valid thresholds
        config1 = RetrieverConfig(similarity_threshold=0.0)
        config2 = RetrieverConfig(similarity_threshold=1.0)
        config3 = RetrieverConfig(similarity_threshold=0.5)
        
        assert config1.similarity_threshold == 0.0
        assert config2.similarity_threshold == 1.0
        assert config3.similarity_threshold == 0.5
        
        # Invalid thresholds
        with pytest.raises(ValidationError):
            RetrieverConfig(similarity_threshold=-0.1)
        
        with pytest.raises(ValidationError):
            RetrieverConfig(similarity_threshold=1.1)
    
    def test_config_validation_max_results(self):
        """Test max_results validation."""
        # Valid values
        config1 = RetrieverConfig(max_results=1)
        config2 = RetrieverConfig(max_results=100)
        config3 = RetrieverConfig(max_results=50)
        
        assert config1.max_results == 1
        assert config2.max_results == 100
        assert config3.max_results == 50
        
        # Invalid values
        with pytest.raises(ValidationError):
            RetrieverConfig(max_results=0)
        
        with pytest.raises(ValidationError):
            RetrieverConfig(max_results=101)
    
    def test_config_model_dump(self):
        """Test model_dump method."""
        config = RetrieverConfig(max_results=15, similarity_threshold=0.8)
        dumped = config.model_dump()
        
        assert isinstance(dumped, dict)
        assert dumped["max_results"] == 15
        assert dumped["similarity_threshold"] == 0.8
        assert "search_strategy" in dumped
        assert "embedding_model" in dumped
    
    def test_config_to_dict(self):
        """Test to_dict method."""
        config = RetrieverConfig(max_results=20)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["max_results"] == 20
    
    def test_config_update(self):
        """Test update method."""
        config = RetrieverConfig(max_results=10, similarity_threshold=0.7)
        updated_config = config.update(max_results=20, similarity_threshold=0.8)
        
        assert updated_config.max_results == 20
        assert updated_config.similarity_threshold == 0.8
        assert config.max_results == 10  # Original unchanged
        assert config.similarity_threshold == 0.7
    
    def test_config_from_dict(self):
        """Test from_dict class method."""
        config_dict = {
            "max_results": 25,
            "similarity_threshold": 0.9,
            "embedding_model": "test-model"
        }
        config = RetrieverConfig.from_dict(config_dict)
        
        assert config.max_results == 25
        assert config.similarity_threshold == 0.9
        assert config.embedding_model == "test-model"
    
    def test_config_str_representation(self):
        """Test string representation."""
        config = RetrieverConfig(max_results=15, similarity_threshold=0.8)
        config_str = str(config)
        
        assert "RetrieverConfig" in config_str
        assert "strategy=SearchStrategy.SEMANTIC" in config_str
        assert "max_results=15" in config_str
        assert "threshold=0.8" in config_str
    
    def test_config_equality(self):
        """Test equality comparison."""
        config1 = RetrieverConfig(max_results=10, similarity_threshold=0.7)
        config2 = RetrieverConfig(max_results=10, similarity_threshold=0.7)
        config3 = RetrieverConfig(max_results=15, similarity_threshold=0.7)
        
        assert config1 == config2
        assert config1 != config3
    
    def test_config_hash(self):
        """Test hash functionality."""
        config1 = RetrieverConfig(max_results=10)
        config2 = RetrieverConfig(max_results=10)
        config3 = RetrieverConfig(max_results=15)
        
        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)


class TestConfigurationPresets:
    """Test cases for configuration presets."""
    
    def test_default_config(self):
        """Test default configuration preset."""
        config = DEFAULT_RETRIEVER_CONFIG
        
        assert config.search_strategy == SearchStrategy.SEMANTIC
        assert config.max_results == 10
        assert config.similarity_threshold == 0.7
    
    def test_fast_config(self):
        """Test fast configuration preset."""
        config = FAST_RETRIEVER_CONFIG
        
        assert config.max_results == 5
        assert config.similarity_threshold == 0.6
        assert config.enable_hybrid_search == False
        assert config.enable_context_awareness == False
        assert config.enable_metadata_filtering == False
        assert config.enable_reranking == False
        assert config.enable_diversity == False
    
    def test_high_quality_config(self):
        """Test high quality configuration preset."""
        config = HIGH_QUALITY_RETRIEVER_CONFIG
        
        assert config.search_strategy == SearchStrategy.HYBRID
        assert config.max_results == 20
        assert config.similarity_threshold == 0.8
        assert config.embedding_model == "all-mpnet-base-v2"
        assert config.enable_hybrid_search == True
        assert config.semantic_weight == 0.8
        assert config.keyword_weight == 0.2
        assert config.enable_context_awareness == True
        assert config.enable_reranking == True
        assert config.enable_diversity == True
    
    def test_semantic_config(self):
        """Test semantic configuration preset."""
        config = SEMANTIC_RETRIEVER_CONFIG
        
        assert config.search_strategy == SearchStrategy.SEMANTIC
        assert config.max_results == 15
        assert config.similarity_threshold == 0.75
        assert config.enable_hybrid_search == False
        assert config.ranking_method == RankingMethod.RELEVANCE
        assert config.relevance_weight == 1.0
        assert config.quality_weight == 0.0
    
    def test_hybrid_config(self):
        """Test hybrid configuration preset."""
        config = HYBRID_RETRIEVER_CONFIG
        
        assert config.search_strategy == SearchStrategy.HYBRID
        assert config.max_results == 12
        assert config.similarity_threshold == 0.7
        assert config.enable_hybrid_search == True
        assert config.semantic_weight == 0.6
        assert config.keyword_weight == 0.4
        assert config.enable_reranking == True
    
    def test_context_aware_config(self):
        """Test context-aware configuration preset."""
        config = CONTEXT_AWARE_RETRIEVER_CONFIG
        
        assert config.search_strategy == SearchStrategy.CONTEXT_AWARE
        assert config.max_results == 10
        assert config.similarity_threshold == 0.75
        assert config.enable_context_awareness == True
        assert config.context_window_size == 8
        assert config.context_decay_factor == 0.7


class TestConfigurationRegistry:
    """Test cases for configuration registry functions."""
    
    def test_get_config_default(self):
        """Test getting default configuration."""
        config = get_config("default")
        assert config == DEFAULT_RETRIEVER_CONFIG
    
    def test_get_config_fast(self):
        """Test getting fast configuration."""
        config = get_config("fast")
        assert config == FAST_RETRIEVER_CONFIG
    
    def test_get_config_high_quality(self):
        """Test getting high quality configuration."""
        config = get_config("high_quality")
        assert config == HIGH_QUALITY_RETRIEVER_CONFIG
    
    def test_get_config_semantic(self):
        """Test getting semantic configuration."""
        config = get_config("semantic")
        assert config == SEMANTIC_RETRIEVER_CONFIG
    
    def test_get_config_hybrid(self):
        """Test getting hybrid configuration."""
        config = get_config("hybrid")
        assert config == HYBRID_RETRIEVER_CONFIG
    
    def test_get_config_context_aware(self):
        """Test getting context-aware configuration."""
        config = get_config("context_aware")
        assert config == CONTEXT_AWARE_RETRIEVER_CONFIG
    
    def test_get_config_invalid(self):
        """Test getting invalid configuration name."""
        with pytest.raises(ValueError, match="Unknown config name"):
            get_config("invalid_config")
    
    def test_list_available_configs(self):
        """Test listing available configurations."""
        configs = list_available_configs()
        
        expected_configs = [
            "default", "fast", "high_quality", "semantic", 
            "hybrid", "context_aware"
        ]
        
        assert set(configs) == set(expected_configs)
        assert len(configs) == len(expected_configs)
    
    def test_create_custom_config(self):
        """Test creating custom configuration."""
        custom_config = create_custom_config(
            max_results=30,
            similarity_threshold=0.9,
            embedding_model="custom-model"
        )
        
        assert custom_config.max_results == 30
        assert custom_config.similarity_threshold == 0.9
        assert custom_config.embedding_model == "custom-model"
        
        # Other values should be from default
        assert custom_config.search_strategy == SearchStrategy.SEMANTIC
        assert custom_config.host == "localhost"
        assert custom_config.port == 6333


class TestSearchStrategy:
    """Test cases for SearchStrategy enum."""
    
    def test_search_strategy_values(self):
        """Test search strategy enum values."""
        assert SearchStrategy.SEMANTIC == "semantic"
        assert SearchStrategy.KEYWORD == "keyword"
        assert SearchStrategy.HYBRID == "hybrid"
        assert SearchStrategy.CONTEXT_AWARE == "context_aware"
    
    def test_search_strategy_usage(self):
        """Test search strategy usage in configuration."""
        config = RetrieverConfig(search_strategy=SearchStrategy.HYBRID)
        assert config.search_strategy == SearchStrategy.HYBRID


class TestRankingMethod:
    """Test cases for RankingMethod enum."""
    
    def test_ranking_method_values(self):
        """Test ranking method enum values."""
        assert RankingMethod.RELEVANCE == "relevance"
        assert RankingMethod.QUALITY == "quality"
        assert RankingMethod.COMBINED == "combined"
        assert RankingMethod.RECENCY == "recency"
    
    def test_ranking_method_usage(self):
        """Test ranking method usage in configuration."""
        config = RetrieverConfig(ranking_method=RankingMethod.QUALITY)
        assert config.ranking_method == RankingMethod.QUALITY
