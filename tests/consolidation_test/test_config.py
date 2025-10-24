"""
Unit tests for consolidation configuration.

This module contains tests for the ConsolidationConfig class and
various configuration presets.
"""

import pytest
from consolidation.config import (
    ConsolidationConfig,
    DEFAULT_CONSOLIDATION_CONFIG,
    FAST_CONSOLIDATION_CONFIG,
    HIGH_QUALITY_CONSOLIDATION_CONFIG,
    BALANCED_CONSOLIDATION_CONFIG,
    get_config,
    list_available_configs,
    create_custom_config
)


class TestConsolidationConfig:
    """Test cases for the ConsolidationConfig class."""
    
    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = ConsolidationConfig()
        
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.1
        assert config.max_tokens == 4000
        assert config.max_concepts_per_document == 20
        assert config.min_concept_confidence == 0.3
        assert config.max_relations_per_document == 30
        assert config.max_concepts_per_subject == 50
        assert config.concept_similarity_threshold == 0.7
        assert config.relation_strength_threshold == 0.5
        assert config.storage_backend == "memory"
        assert config.enable_vector_search is True
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.batch_size == 5
        assert config.max_retries == 3
        assert config.timeout == 120
        assert config.enable_quality_validation is True
        assert config.min_quality_score == 0.6
    
    def test_custom_config_creation(self):
        """Test creation of custom configuration."""
        config = ConsolidationConfig(
            model_name="gpt-4o",
            temperature=0.2,
            max_concepts_per_document=30,
            min_concept_confidence=0.5,
            batch_size=10
        )
        
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.2
        assert config.max_concepts_per_document == 30
        assert config.min_concept_confidence == 0.5
        assert config.batch_size == 10
    
    def test_config_validation_temperature(self):
        """Test temperature validation."""
        # Valid temperature values
        config1 = ConsolidationConfig(temperature=0.0)
        config2 = ConsolidationConfig(temperature=1.0)
        config3 = ConsolidationConfig(temperature=0.5)
        
        assert config1.temperature == 0.0
        assert config2.temperature == 1.0
        assert config3.temperature == 0.5
        
        # Invalid temperature values
        with pytest.raises(ValueError):
            ConsolidationConfig(temperature=-0.1)
        
        with pytest.raises(ValueError):
            ConsolidationConfig(temperature=1.1)
    
    def test_config_validation_max_tokens(self):
        """Test max_tokens validation."""
        # Valid max_tokens values
        config1 = ConsolidationConfig(max_tokens=100)
        config2 = ConsolidationConfig(max_tokens=8000)
        
        assert config1.max_tokens == 100
        assert config2.max_tokens == 8000
        
        # Invalid max_tokens values
        with pytest.raises(ValueError):
            ConsolidationConfig(max_tokens=50)  # Below minimum
    
    def test_config_validation_confidence_thresholds(self):
        """Test confidence threshold validation."""
        # Valid confidence values
        config = ConsolidationConfig(
            min_concept_confidence=0.5,
            min_quality_score=0.7
        )
        
        assert config.min_concept_confidence == 0.5
        assert config.min_quality_score == 0.7
        
        # Invalid confidence values
        with pytest.raises(ValueError):
            ConsolidationConfig(min_concept_confidence=-0.1)
        
        with pytest.raises(ValueError):
            ConsolidationConfig(min_concept_confidence=1.1)
        
        with pytest.raises(ValueError):
            ConsolidationConfig(min_quality_score=-0.1)
        
        with pytest.raises(ValueError):
            ConsolidationConfig(min_quality_score=1.1)
    
    def test_config_validation_positive_integers(self):
        """Test validation of positive integer fields."""
        # Valid positive integer values
        config = ConsolidationConfig(
            max_concepts_per_document=10,
            max_relations_per_document=20,
            max_concepts_per_subject=30,
            batch_size=5,
            max_retries=3,
            timeout=60
        )
        
        assert config.max_concepts_per_document == 10
        assert config.max_relations_per_document == 20
        assert config.max_concepts_per_subject == 30
        assert config.batch_size == 5
        assert config.max_retries == 3
        assert config.timeout == 60
        
        # Invalid values (should raise ValueError)
        with pytest.raises(ValueError):
            ConsolidationConfig(max_concepts_per_document=0)
        
        with pytest.raises(ValueError):
            ConsolidationConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            ConsolidationConfig(timeout=0)
    
    def test_config_model_dump(self):
        """Test model_dump method."""
        config = ConsolidationConfig(
            model_name="test-model",
            temperature=0.3,
            max_concepts_per_document=15
        )
        
        dumped_data = config.model_dump()
        assert dumped_data["model_name"] == "test-model"
        assert dumped_data["temperature"] == 0.3
        assert dumped_data["max_concepts_per_document"] == 15
        assert "llm_parameters" in dumped_data
        assert "custom_prompts" in dumped_data
    
    def test_config_to_dict(self):
        """Test to_dict method."""
        config = ConsolidationConfig(model_name="test-model")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test-model"
    
    def test_config_update(self):
        """Test update method."""
        config = ConsolidationConfig(model_name="original")
        updated_config = config.update(model_name="updated", temperature=0.5)
        
        assert updated_config.model_name == "updated"
        assert updated_config.temperature == 0.5
        assert config.model_name == "original"  # Original should be unchanged
    
    def test_config_from_dict(self):
        """Test from_dict class method."""
        config_dict = {
            "model_name": "test-model",
            "temperature": 0.4,
            "max_concepts_per_document": 25
        }
        
        config = ConsolidationConfig.from_dict(config_dict)
        assert config.model_name == "test-model"
        assert config.temperature == 0.4
        assert config.max_concepts_per_document == 25
    
    def test_config_str_representation(self):
        """Test string representation."""
        config = ConsolidationConfig(model_name="test-model", temperature=0.3, batch_size=8)
        config_str = str(config)
        
        assert "test-model" in config_str
        assert "0.3" in config_str
        assert "8" in config_str
    
    def test_config_equality(self):
        """Test equality comparison."""
        config1 = ConsolidationConfig(model_name="test", temperature=0.3)
        config2 = ConsolidationConfig(model_name="test", temperature=0.3)
        config3 = ConsolidationConfig(model_name="different", temperature=0.3)
        
        assert config1 == config2
        assert config1 != config3
    
    def test_config_hash(self):
        """Test hash functionality."""
        config1 = ConsolidationConfig(model_name="test", temperature=0.3)
        config2 = ConsolidationConfig(model_name="test", temperature=0.3)
        config3 = ConsolidationConfig(model_name="different", temperature=0.3)
        
        # Same configs should have same hash
        assert hash(config1) == hash(config2)
        
        # Different configs should have different hashes
        assert hash(config1) != hash(config3)
        
        # Hash should be consistent
        assert hash(config1) == hash(config1)


class TestConfigurationPresets:
    """Test cases for configuration presets."""
    
    def test_default_config_preset(self):
        """Test default configuration preset."""
        config = DEFAULT_CONSOLIDATION_CONFIG
        
        assert isinstance(config, ConsolidationConfig)
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.1
        assert config.max_concepts_per_document == 20
    
    def test_fast_config_preset(self):
        """Test fast configuration preset."""
        config = FAST_CONSOLIDATION_CONFIG
        
        assert isinstance(config, ConsolidationConfig)
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.2
        assert config.max_tokens == 2000
        assert config.max_concepts_per_document == 10
        assert config.batch_size == 10
        assert config.timeout == 60
        assert config.min_quality_score == 0.5
    
    def test_high_quality_config_preset(self):
        """Test high quality configuration preset."""
        config = HIGH_QUALITY_CONSOLIDATION_CONFIG
        
        assert isinstance(config, ConsolidationConfig)
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 8000
        assert config.max_concepts_per_document == 30
        assert config.max_relations_per_document == 50
        assert config.max_concepts_per_subject == 100
        assert config.concept_similarity_threshold == 0.8
        assert config.relation_strength_threshold == 0.6
        assert config.batch_size == 3
        assert config.timeout == 300
        assert config.min_quality_score == 0.8
        assert config.enable_quality_validation is True
    
    def test_balanced_config_preset(self):
        """Test balanced configuration preset."""
        config = BALANCED_CONSOLIDATION_CONFIG
        
        assert isinstance(config, ConsolidationConfig)
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.1
        assert config.max_tokens == 4000
        assert config.max_concepts_per_document == 20
        assert config.max_relations_per_document == 30
        assert config.max_concepts_per_subject == 50
        assert config.concept_similarity_threshold == 0.7
        assert config.relation_strength_threshold == 0.5
        assert config.batch_size == 5
        assert config.timeout == 120
        assert config.min_quality_score == 0.6


class TestConfigurationUtilities:
    """Test cases for configuration utility functions."""
    
    def test_get_config_valid_names(self):
        """Test get_config with valid configuration names."""
        configs_to_test = [
            ("default", DEFAULT_CONSOLIDATION_CONFIG),
            ("fast", FAST_CONSOLIDATION_CONFIG),
            ("high_quality", HIGH_QUALITY_CONSOLIDATION_CONFIG),
            ("balanced", BALANCED_CONSOLIDATION_CONFIG)
        ]
        
        for config_name, expected_config in configs_to_test:
            config = get_config(config_name)
            assert isinstance(config, ConsolidationConfig)
            assert config.model_name == expected_config.model_name
    
    def test_get_config_invalid_name(self):
        """Test get_config with invalid configuration name."""
        with pytest.raises(ValueError, match="Unknown config name"):
            get_config("invalid_config")
    
    def test_list_available_configs(self):
        """Test list_available_configs function."""
        available_configs = list_available_configs()
        
        assert isinstance(available_configs, list)
        assert "default" in available_configs
        assert "fast" in available_configs
        assert "high_quality" in available_configs
        assert "balanced" in available_configs
    
    def test_create_custom_config(self):
        """Test create_custom_config function."""
        custom_config = create_custom_config(
            model_name="custom-model",
            temperature=0.3,
            max_concepts_per_document=25,
            batch_size=8
        )
        
        assert isinstance(custom_config, ConsolidationConfig)
        assert custom_config.model_name == "custom-model"
        assert custom_config.temperature == 0.3
        assert custom_config.max_concepts_per_document == 25
        assert custom_config.batch_size == 8
        
        # Should inherit default values for unspecified parameters
        assert custom_config.max_tokens == 4000  # Default value
        assert custom_config.storage_backend == "memory"  # Default value


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_config_with_none_values(self):
        """Test configuration with None values where appropriate."""
        config = ConsolidationConfig(
            custom_prompts=None,  # Should be handled gracefully
            llm_parameters=None   # Should be handled gracefully
        )
        
        assert config.custom_prompts == {}
        assert config.llm_parameters == {}
    
    def test_config_with_empty_dicts(self):
        """Test configuration with empty dictionaries."""
        config = ConsolidationConfig(
            custom_prompts={},
            llm_parameters={}
        )
        
        assert config.custom_prompts == {}
        assert config.llm_parameters == {}
    
    def test_config_with_custom_llm_parameters(self):
        """Test configuration with custom LLM parameters."""
        custom_params = {
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
        
        config = ConsolidationConfig(llm_parameters=custom_params)
        assert config.llm_parameters == custom_params
    
    def test_config_with_custom_prompts(self):
        """Test configuration with custom prompts."""
        custom_prompts = {
            "concept_extraction": "Custom concept extraction prompt",
            "relation_extraction": "Custom relation extraction prompt"
        }
        
        config = ConsolidationConfig(custom_prompts=custom_prompts)
        assert config.custom_prompts == custom_prompts
    
    def test_config_boundary_values(self):
        """Test configuration with boundary values."""
        config = ConsolidationConfig(
            temperature=0.0,  # Minimum valid value
            max_tokens=100,   # Minimum valid value
            min_concept_confidence=0.0,  # Minimum valid value
            min_quality_score=0.0,  # Minimum valid value
            max_concepts_per_document=1,  # Minimum valid value
            batch_size=1,  # Minimum valid value
            timeout=1  # Minimum valid value
        )
        
        assert config.temperature == 0.0
        assert config.max_tokens == 100
        assert config.min_concept_confidence == 0.0
        assert config.min_quality_score == 0.0
        assert config.max_concepts_per_document == 1
        assert config.batch_size == 1
        assert config.timeout == 1
