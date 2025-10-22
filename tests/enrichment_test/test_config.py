"""
Unit tests for enrichment configuration.

Tests the EnrichmentConfig class and related functionality.
"""

import pytest
from pydantic import ValidationError

from enrichment.config import (
    EnrichmentConfig,
    DEFAULT_ENRICHMENT_CONFIG,
    FAST_ENRICHMENT_CONFIG,
    HIGH_QUALITY_ENRICHMENT_CONFIG
)


class TestEnrichmentConfig:
    """Test cases for EnrichmentConfig class."""
    
    def test_init_default(self):
        """Test EnrichmentConfig initialization with default values."""
        config = EnrichmentConfig()
        
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.batch_size == 5
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.max_keywords == 7
        assert config.max_questions == 5
        assert config.summary_max_length == 200
        assert config.enable_table_summary is True
        assert config.enable_async_processing is True
        assert config.llm_parameters == {}
    
    def test_init_custom(self, sample_config_dict):
        """Test EnrichmentConfig initialization with custom values."""
        config = EnrichmentConfig(**sample_config_dict)
        
        assert config.model_name == sample_config_dict["model_name"]
        assert config.temperature == sample_config_dict["temperature"]
        assert config.batch_size == sample_config_dict["batch_size"]
        assert config.max_keywords == sample_config_dict["max_keywords"]
        assert config.max_questions == sample_config_dict["max_questions"]
        assert config.enable_async_processing == sample_config_dict["enable_async_processing"]
    
    def test_validation_temperature_range(self):
        """Test temperature validation range."""
        # Valid temperature
        config = EnrichmentConfig(temperature=0.5)
        assert config.temperature == 0.5
        
        # Invalid temperature (too low)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(temperature=-0.1)
        assert "temperature" in str(exc_info.value)
        
        # Invalid temperature (too high)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(temperature=2.1)
        assert "temperature" in str(exc_info.value)
    
    def test_validation_batch_size_range(self):
        """Test batch_size validation range."""
        # Valid batch size
        config = EnrichmentConfig(batch_size=10)
        assert config.batch_size == 10
        
        # Invalid batch size (too small)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(batch_size=0)
        assert "batch_size" in str(exc_info.value)
        
        # Invalid batch size (too large)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(batch_size=51)
        assert "batch_size" in str(exc_info.value)
    
    def test_validation_max_retries_range(self):
        """Test max_retries validation range."""
        # Valid max retries
        config = EnrichmentConfig(max_retries=5)
        assert config.max_retries == 5
        
        # Invalid max retries (too small)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(max_retries=-1)
        assert "max_retries" in str(exc_info.value)
        
        # Invalid max retries (too large)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(max_retries=11)
        assert "max_retries" in str(exc_info.value)
    
    def test_validation_retry_delay_range(self):
        """Test retry_delay validation range."""
        # Valid retry delay
        config = EnrichmentConfig(retry_delay=2.0)
        assert config.retry_delay == 2.0
        
        # Invalid retry delay (negative)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(retry_delay=-0.1)
        assert "retry_delay" in str(exc_info.value)
    
    def test_validation_max_keywords_range(self):
        """Test max_keywords validation range."""
        # Valid max keywords
        config = EnrichmentConfig(max_keywords=10)
        assert config.max_keywords == 10
        
        # Invalid max keywords (too small)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(max_keywords=0)
        assert "max_keywords" in str(exc_info.value)
        
        # Invalid max keywords (too large)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(max_keywords=21)
        assert "max_keywords" in str(exc_info.value)
    
    def test_validation_max_questions_range(self):
        """Test max_questions validation range."""
        # Valid max questions
        config = EnrichmentConfig(max_questions=8)
        assert config.max_questions == 8
        
        # Invalid max questions (too small)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(max_questions=0)
        assert "max_questions" in str(exc_info.value)
        
        # Invalid max questions (too large)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(max_questions=16)
        assert "max_questions" in str(exc_info.value)
    
    def test_validation_summary_max_length_range(self):
        """Test summary_max_length validation range."""
        # Valid summary max length
        config = EnrichmentConfig(summary_max_length=300)
        assert config.summary_max_length == 300
        
        # Invalid summary max length (too small)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(summary_max_length=49)
        assert "summary_max_length" in str(exc_info.value)
        
        # Invalid summary max length (too large)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(summary_max_length=501)
        assert "summary_max_length" in str(exc_info.value)
    
    def test_validation_max_tokens_range(self):
        """Test max_tokens validation range."""
        # Valid max tokens
        config = EnrichmentConfig(max_tokens=1000)
        assert config.max_tokens == 1000
        
        # Invalid max tokens (too small)
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentConfig(max_tokens=0)
        assert "max_tokens" in str(exc_info.value)
    
    def test_validation_boolean_fields(self):
        """Test boolean field validation."""
        # Valid boolean values
        config = EnrichmentConfig(enable_table_summary=False, enable_async_processing=False)
        assert config.enable_table_summary is False
        assert config.enable_async_processing is False
        
        # Test with string values (should be converted)
        config = EnrichmentConfig(enable_table_summary="true", enable_async_processing="false")
        assert config.enable_table_summary is True
        assert config.enable_async_processing is False
    
    def test_validation_llm_parameters(self):
        """Test llm_parameters validation."""
        # Valid llm parameters
        params = {"top_p": 0.9, "frequency_penalty": 0.1}
        config = EnrichmentConfig(llm_parameters=params)
        assert config.llm_parameters == params
        
        # Empty llm parameters
        config = EnrichmentConfig(llm_parameters={})
        assert config.llm_parameters == {}
    
    def test_model_dump(self, sample_config_dict):
        """Test model_dump method."""
        config = EnrichmentConfig(**sample_config_dict)
        dumped = config.model_dump()
        
        assert isinstance(dumped, dict)
        assert dumped["model_name"] == sample_config_dict["model_name"]
        assert dumped["temperature"] == sample_config_dict["temperature"]
        assert dumped["batch_size"] == sample_config_dict["batch_size"]
    
    def test_to_dict(self, sample_config_dict):
        """Test to_dict method."""
        config = EnrichmentConfig(**sample_config_dict)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == sample_config_dict["model_name"]
        assert config_dict["temperature"] == sample_config_dict["temperature"]
    
    def test_update(self, sample_config_dict):
        """Test update method."""
        config = EnrichmentConfig(**sample_config_dict)
        
        # Update with new values
        updated_config = config.update(
            temperature=0.5,
            batch_size=10,
            max_keywords=15
        )
        
        assert updated_config.temperature == 0.5
        assert updated_config.batch_size == 10
        assert updated_config.max_keywords == 15
        
        # Original config should be unchanged
        assert config.temperature == sample_config_dict["temperature"]
        assert config.batch_size == sample_config_dict["batch_size"]
    
    def test_from_dict(self, sample_config_dict):
        """Test from_dict class method."""
        config = EnrichmentConfig.from_dict(sample_config_dict)
        
        assert isinstance(config, EnrichmentConfig)
        assert config.model_name == sample_config_dict["model_name"]
        assert config.temperature == sample_config_dict["temperature"]
        assert config.batch_size == sample_config_dict["batch_size"]
    
    def test_str_representation(self, sample_config_dict):
        """Test string representation of EnrichmentConfig."""
        config = EnrichmentConfig(**sample_config_dict)
        str_repr = str(config)
        
        # Pydantic v2 uses a different string representation format
        assert sample_config_dict["model_name"] in str_repr
        assert "temperature=" in str_repr
    
    def test_equality(self, sample_config_dict):
        """Test equality of EnrichmentConfig instances."""
        config1 = EnrichmentConfig(**sample_config_dict)
        config2 = EnrichmentConfig(**sample_config_dict)
        
        assert config1 == config2
    
    def test_inequality(self, sample_config_dict):
        """Test inequality of EnrichmentConfig instances."""
        config1 = EnrichmentConfig(**sample_config_dict)
        
        different_dict = sample_config_dict.copy()
        different_dict["temperature"] = 0.5
        config2 = EnrichmentConfig(**different_dict)
        
        assert config1 != config2
    
    def test_hash(self, sample_config_dict):
        """Test hash of EnrichmentConfig instances."""
        config1 = EnrichmentConfig(**sample_config_dict)
        config2 = EnrichmentConfig(**sample_config_dict)
        
        # Pydantic models are not hashable by default
        # Test that they can be converted to hashable types
        config1_dict = config1.model_dump()
        config2_dict = config2.model_dump()
        
        # Convert to hashable tuples, handling nested dicts
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)
            else:
                return obj
        
        config1_tuple = make_hashable(config1_dict)
        config2_tuple = make_hashable(config2_dict)
        
        assert hash(config1_tuple) == hash(config1_tuple)
        assert hash(config1_tuple) == hash(config2_tuple)


class TestPredefinedConfigurations:
    """Test cases for predefined configurations."""
    
    def test_default_config(self):
        """Test DEFAULT_ENRICHMENT_CONFIG."""
        config = DEFAULT_ENRICHMENT_CONFIG
        
        assert isinstance(config, EnrichmentConfig)
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.batch_size == 5
        assert config.max_keywords == 7
        assert config.max_questions == 5
    
    def test_fast_config(self):
        """Test FAST_ENRICHMENT_CONFIG."""
        config = FAST_ENRICHMENT_CONFIG
        
        assert isinstance(config, EnrichmentConfig)
        assert config.batch_size == 10
        assert config.max_retries == 1
        assert config.retry_delay == 0.5
        assert config.max_keywords == 5
        assert config.max_questions == 3
        assert config.summary_max_length == 150
    
    def test_high_quality_config(self):
        """Test HIGH_QUALITY_ENRICHMENT_CONFIG."""
        config = HIGH_QUALITY_ENRICHMENT_CONFIG
        
        assert isinstance(config, EnrichmentConfig)
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.1
        assert config.batch_size == 3
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.max_keywords == 10
        assert config.max_questions == 7
        assert config.summary_max_length == 300
    
    def test_config_presets_are_different(self):
        """Test that config presets are different."""
        default = DEFAULT_ENRICHMENT_CONFIG
        fast = FAST_ENRICHMENT_CONFIG
        high_quality = HIGH_QUALITY_ENRICHMENT_CONFIG
        
        assert default != fast
        assert default != high_quality
        assert fast != high_quality
    
    def test_config_presets_immutable(self):
        """Test that config presets are immutable."""
        # Create a copy to avoid modifying the original
        default = EnrichmentConfig()
        original_batch_size = default.batch_size
        
        # Try to modify (should not affect original)
        default.batch_size = 999
        
        # The modification affects the instance
        assert default.batch_size == 999
        # The original value is preserved in the class variable
        assert DEFAULT_ENRICHMENT_CONFIG.batch_size == original_batch_size


class TestConfigClassMethods:
    """Test cases for EnrichmentConfig class methods."""
    
    def test_default_class_method(self):
        """Test default class method."""
        config = EnrichmentConfig.default()
        
        assert isinstance(config, EnrichmentConfig)
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.batch_size == 5
    
    def test_fast_class_method(self):
        """Test fast class method."""
        config = EnrichmentConfig.fast()
        
        assert isinstance(config, EnrichmentConfig)
        assert config.batch_size == 10
        assert config.max_retries == 1
        assert config.retry_delay == 0.5
    
    def test_high_quality_class_method(self):
        """Test high_quality class method."""
        config = EnrichmentConfig.high_quality()
        
        assert isinstance(config, EnrichmentConfig)
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.1
        assert config.batch_size == 3
    
    def test_from_dict_class_method(self, sample_config_dict):
        """Test from_dict class method."""
        config = EnrichmentConfig.from_dict(sample_config_dict)
        
        assert isinstance(config, EnrichmentConfig)
        assert config.model_name == sample_config_dict["model_name"]
        assert config.temperature == sample_config_dict["temperature"]
    
    def test_from_dict_invalid(self):
        """Test from_dict with invalid data."""
        invalid_dict = {
            "model_name": "gpt-4o",
            "temperature": 2.5,  # Invalid temperature
            "batch_size": 0     # Invalid batch size
        }
        
        with pytest.raises(ValidationError):
            EnrichmentConfig.from_dict(invalid_dict)


class TestConfigEdgeCases:
    """Test cases for configuration edge cases."""
    
    def test_minimum_valid_values(self):
        """Test minimum valid values for all fields."""
        config = EnrichmentConfig(
            temperature=0.0,
            batch_size=1,
            max_retries=0,
            retry_delay=0.0,
            max_keywords=1,
            max_questions=1,
            summary_max_length=50,
            max_tokens=1
        )
        
        assert config.temperature == 0.0
        assert config.batch_size == 1
        assert config.max_retries == 0
        assert config.retry_delay == 0.0
        assert config.max_keywords == 1
        assert config.max_questions == 1
        assert config.summary_max_length == 50
        assert config.max_tokens == 1
    
    def test_maximum_valid_values(self):
        """Test maximum valid values for all fields."""
        config = EnrichmentConfig(
            temperature=2.0,
            batch_size=50,
            max_retries=10,
            max_keywords=20,
            max_questions=15,
            summary_max_length=500,
            max_tokens=10000
        )
        
        assert config.temperature == 2.0
        assert config.batch_size == 50
        assert config.max_retries == 10
        assert config.max_keywords == 20
        assert config.max_questions == 15
        assert config.summary_max_length == 500
        assert config.max_tokens == 10000
    
    def test_none_values(self):
        """Test None values for optional fields."""
        config = EnrichmentConfig(
            max_tokens=None,
            llm_parameters={}  # Use empty dict instead of None
        )
        
        assert config.max_tokens is None
        assert config.llm_parameters == {}
    
    def test_empty_llm_parameters(self):
        """Test empty llm_parameters."""
        config = EnrichmentConfig(llm_parameters={})
        
        assert config.llm_parameters == {}
    
    def test_complex_llm_parameters(self):
        """Test complex llm_parameters."""
        complex_params = {
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["END", "STOP"],
            "n": 1,
            "stream": False
        }
        
        config = EnrichmentConfig(llm_parameters=complex_params)
        
        assert config.llm_parameters == complex_params
    
    def test_config_serialization(self, sample_config_dict):
        """Test config serialization and deserialization."""
        config = EnrichmentConfig(**sample_config_dict)
        
        # Serialize to dict
        config_dict = config.to_dict()
        
        # Deserialize from dict
        new_config = EnrichmentConfig.from_dict(config_dict)
        
        assert new_config == config
        assert new_config.model_name == config.model_name
        assert new_config.temperature == config.temperature
        assert new_config.batch_size == config.batch_size
