"""
Tests for the logic_extractor config module.
"""

import pytest
from logic_extractor.config import (
    LogicExtractionConfig,
    DEFAULT_LOGIC_EXTRACTION_CONFIG,
    FAST_LOGIC_EXTRACTION_CONFIG,
    HIGH_QUALITY_LOGIC_EXTRACTION_CONFIG
)


class TestLogicExtractionConfig:
    """Test cases for the LogicExtractionConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogicExtractionConfig()
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_retries == 3
        assert config.timeout == 60
        assert config.batch_size == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LogicExtractionConfig(
            model_name="gpt-4o",
            temperature=0.5,
            max_retries=5,
            timeout=120,
            batch_size=10
        )
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_retries == 5
        assert config.timeout == 120
        assert config.batch_size == 10

    def test_invalid_temperature(self):
        """Test invalid temperature values raise ValidationError."""
        with pytest.raises(ValueError):
            LogicExtractionConfig(temperature=1.1)
        with pytest.raises(ValueError):
            LogicExtractionConfig(temperature=-0.1)

    def test_invalid_max_retries(self):
        """Test invalid max_retries values raise ValidationError."""
        with pytest.raises(ValueError):
            LogicExtractionConfig(max_retries=-1)

    def test_invalid_timeout(self):
        """Test invalid timeout values raise ValidationError."""
        with pytest.raises(ValueError):
            LogicExtractionConfig(timeout=0)

    def test_invalid_batch_size(self):
        """Test invalid batch_size values raise ValidationError."""
        with pytest.raises(ValueError):
            LogicExtractionConfig(batch_size=0)

    def test_config_presets(self):
        """Test predefined configuration presets."""
        default = DEFAULT_LOGIC_EXTRACTION_CONFIG
        fast = FAST_LOGIC_EXTRACTION_CONFIG
        high_quality = HIGH_QUALITY_LOGIC_EXTRACTION_CONFIG

        assert default.model_name == "gpt-4o-mini"
        assert fast.model_name == "gpt-3.5-turbo"
        assert high_quality.model_name == "gpt-4o"

        assert fast.temperature == 0.1
        assert high_quality.temperature == 0.0

        assert fast.batch_size == 10
        assert high_quality.max_retries == 5

    def test_config_presets_immutable(self):
        """Test that modifying a preset does not affect the original."""
        original_temp = DEFAULT_LOGIC_EXTRACTION_CONFIG.temperature

        # Create a copy and modify it
        modified_config = DEFAULT_LOGIC_EXTRACTION_CONFIG.model_copy(deep=True)
        modified_config.temperature = 0.9

        assert DEFAULT_LOGIC_EXTRACTION_CONFIG.temperature == original_temp
        assert modified_config.temperature == 0.9

    def test_str_representation(self):
        """Test string representation of the config."""
        config = LogicExtractionConfig(model_name="test-model")
        assert "model_name='test-model'" in str(config)

    def test_hash(self):
        """Test that config objects can be compared for equality."""
        config1 = LogicExtractionConfig(model_name="model1", temperature=0.1)
        config2 = LogicExtractionConfig(model_name="model1", temperature=0.1)
        config3 = LogicExtractionConfig(model_name="model2", temperature=0.2)

        assert config1 == config2
        assert config1 != config3

        # Test equality in a list
        configs = [config1, config2]
        assert configs.count(config1) == 2
        configs.append(config3)
        assert configs.count(config1) == 2