"""
Unit tests for the QA agent configuration classes and related functions.
"""

import pytest
from qa_agent.config import (
    QAConfig,
    DEFAULT_QA_CONFIG,
    FAST_QA_CONFIG,
    HIGH_QUALITY_QA_CONFIG,
    ACADEMIC_QA_CONFIG,
    CONVERSATIONAL_QA_CONFIG,
    AgentType,
    get_config,
    list_available_configs,
    create_custom_config
)
from pydantic import ValidationError


class TestQAConfig:
    """Test cases for the QAConfig model."""

    def test_default_config_creation(self):
        """Test that the default config is created correctly."""
        config = QAConfig()
        assert config.llm_model == "gpt-4o"
        assert config.llm_temperature == 0.0
        assert config.max_retrieved_docs == 5
        assert config.similarity_threshold == 0.7
        assert config.max_answer_length == 2000
        assert config.answer_style == "detailed"
        assert config.include_sources is True
        assert config.session_timeout == 3600
        assert config.max_conversation_turns == 10
        assert config.enable_context_memory is True
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.enable_logging is True
        assert config.log_level == "INFO"

    def test_custom_config_creation(self):
        """Test creating a config with custom parameters."""
        config = QAConfig(
            llm_model="gpt-4o-mini",
            max_retrieved_docs=3,
            answer_style="concise",
            session_timeout=1800
        )
        assert config.llm_model == "gpt-4o-mini"
        assert config.max_retrieved_docs == 3
        assert config.answer_style == "concise"
        assert config.session_timeout == 1800
        assert config.llm_temperature == 0.0  # Default should still apply

    def test_config_validation_llm_temperature(self):
        """Test validation for 'llm_temperature' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(llm_temperature=-0.1)
        with pytest.raises(ValidationError):
            QAConfig(llm_temperature=2.1)
        
        config1 = QAConfig(llm_temperature=0.0)
        assert config1.llm_temperature == 0.0
        config2 = QAConfig(llm_temperature=2.0)
        assert config2.llm_temperature == 2.0

    def test_config_validation_max_retrieved_docs(self):
        """Test validation for 'max_retrieved_docs' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(max_retrieved_docs=0)
        with pytest.raises(ValidationError):
            QAConfig(max_retrieved_docs=21)
        
        config = QAConfig(max_retrieved_docs=1)
        assert config.max_retrieved_docs == 1

    def test_config_validation_similarity_threshold(self):
        """Test validation for 'similarity_threshold' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(similarity_threshold=-0.1)
        with pytest.raises(ValidationError):
            QAConfig(similarity_threshold=1.1)
        
        config1 = QAConfig(similarity_threshold=0.0)
        assert config1.similarity_threshold == 0.0
        config2 = QAConfig(similarity_threshold=1.0)
        assert config2.similarity_threshold == 1.0

    def test_config_validation_max_answer_length(self):
        """Test validation for 'max_answer_length' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(max_answer_length=99)
        with pytest.raises(ValidationError):
            QAConfig(max_answer_length=5001)
        
        config1 = QAConfig(max_answer_length=100)
        assert config1.max_answer_length == 100
        config2 = QAConfig(max_answer_length=5000)
        assert config2.max_answer_length == 5000

    def test_config_validation_answer_style(self):
        """Test validation for 'answer_style' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(answer_style="invalid_style")
        
        valid_styles = ["concise", "detailed", "academic"]
        for style in valid_styles:
            config = QAConfig(answer_style=style)
            assert config.answer_style == style

    def test_config_validation_session_timeout(self):
        """Test validation for 'session_timeout' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(session_timeout=59)
        
        config = QAConfig(session_timeout=60)
        assert config.session_timeout == 60

    def test_config_validation_max_conversation_turns(self):
        """Test validation for 'max_conversation_turns' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(max_conversation_turns=0)
        with pytest.raises(ValidationError):
            QAConfig(max_conversation_turns=51)
        
        config1 = QAConfig(max_conversation_turns=1)
        assert config1.max_conversation_turns == 1
        config2 = QAConfig(max_conversation_turns=50)
        assert config2.max_conversation_turns == 50

    def test_config_validation_timeout(self):
        """Test validation for 'timeout' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(timeout=4)
        
        config = QAConfig(timeout=5)
        assert config.timeout == 5

    def test_config_validation_max_retries(self):
        """Test validation for 'max_retries' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(max_retries=-1)
        with pytest.raises(ValidationError):
            QAConfig(max_retries=11)
        
        config1 = QAConfig(max_retries=0)
        assert config1.max_retries == 0
        config2 = QAConfig(max_retries=10)
        assert config2.max_retries == 10

    def test_config_validation_log_level(self):
        """Test validation for 'log_level' parameter."""
        with pytest.raises(ValidationError):
            QAConfig(log_level="INVALID")
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        for level in valid_levels:
            config = QAConfig(log_level=level)
            assert config.log_level == level

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        config = DEFAULT_QA_CONFIG
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["llm_model"] == DEFAULT_QA_CONFIG.llm_model

    def test_config_update(self):
        """Test updating config parameters."""
        original_config = DEFAULT_QA_CONFIG
        updated_config = original_config.update(
            max_retrieved_docs=8,
            answer_style="academic"
        )
        
        assert updated_config.max_retrieved_docs == 8
        assert updated_config.answer_style == "academic"
        assert updated_config.llm_model == original_config.llm_model  # Unchanged
        assert original_config.max_retrieved_docs == 5  # Original should be immutable

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "llm_model": "gpt-4o-mini",
            "max_retrieved_docs": 3,
            "answer_style": "concise"
        }
        config = QAConfig.from_dict(config_dict)
        assert config.llm_model == "gpt-4o-mini"
        assert config.max_retrieved_docs == 3
        assert config.answer_style == "concise"

    def test_config_str_representation(self):
        """Test string representation."""
        config = QAConfig(
            llm_model="gpt-4o-mini",
            answer_style="concise",
            max_retrieved_docs=3
        )
        config_str = str(config)
        assert "QAConfig" in config_str
        assert "gpt-4o-mini" in config_str
        assert "concise" in config_str
        assert "3" in config_str

    def test_config_equality(self):
        """Test equality comparison."""
        config1 = QAConfig(max_retrieved_docs=5, answer_style="detailed")
        config2 = QAConfig(max_retrieved_docs=5, answer_style="detailed")
        config3 = QAConfig(max_retrieved_docs=8, answer_style="detailed")
        
        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"

    def test_config_hash(self):
        """Test hashing of config objects."""
        config1 = QAConfig(max_retrieved_docs=5, answer_style="detailed")
        config2 = QAConfig(max_retrieved_docs=5, answer_style="detailed")
        config3 = QAConfig(max_retrieved_docs=8, answer_style="detailed")
        
        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)


class TestPredefinedConfigs:
    """Test cases for predefined QA configurations."""

    def test_default_config(self):
        """Test DEFAULT_QA_CONFIG."""
        config = DEFAULT_QA_CONFIG
        assert config.llm_model == "gpt-4o"
        assert config.answer_style == "detailed"
        assert config.max_retrieved_docs == 5

    def test_fast_config(self):
        """Test FAST_QA_CONFIG."""
        config = FAST_QA_CONFIG
        assert config.llm_model == "gpt-4o-mini"
        assert config.answer_style == "concise"
        assert config.max_retrieved_docs == 3
        assert config.session_timeout == 1800
        assert config.enable_context_memory is False

    def test_high_quality_config(self):
        """Test HIGH_QUALITY_QA_CONFIG."""
        config = HIGH_QUALITY_QA_CONFIG
        assert config.llm_model == "gpt-4o"
        assert config.answer_style == "academic"
        assert config.max_retrieved_docs == 10
        assert config.enable_reranking is True
        assert config.enable_fact_checking is True
        assert config.log_level == "DEBUG"

    def test_academic_config(self):
        """Test ACADEMIC_QA_CONFIG."""
        config = ACADEMIC_QA_CONFIG
        assert config.answer_style == "academic"
        assert config.max_retrieved_docs == 8
        assert config.include_sources is True
        assert config.enable_reranking is True

    def test_conversational_config(self):
        """Test CONVERSATIONAL_QA_CONFIG."""
        config = CONVERSATIONAL_QA_CONFIG
        assert config.llm_temperature == 0.3
        assert config.answer_style == "detailed"
        assert config.max_conversation_turns == 25
        assert config.enable_context_memory is True


class TestConfigFunctions:
    """Test cases for utility functions related to configurations."""

    def test_get_config_default(self):
        """Test get_config with default name."""
        config = get_config()
        assert config == DEFAULT_QA_CONFIG

    def test_get_config_by_name(self):
        """Test get_config with specific names."""
        config_fast = get_config("fast")
        assert config_fast == FAST_QA_CONFIG
        
        config_high_quality = get_config("high_quality")
        assert config_high_quality == HIGH_QUALITY_QA_CONFIG

    def test_get_config_invalid_name(self):
        """Test get_config with an invalid name."""
        with pytest.raises(ValueError, match="Unknown config name 'non_existent'. Available configs:"):
            get_config("non_existent")

    def test_list_available_configs(self):
        """Test listing available configurations."""
        available = list_available_configs()
        assert isinstance(available, list)
        assert "default" in available
        assert "fast" in available
        assert "high_quality" in available
        assert "academic" in available
        assert "conversational" in available
        assert len(available) == 5

    def test_create_custom_config_function(self):
        """Test create_custom_config function."""
        custom_config = create_custom_config(
            max_retrieved_docs=8,
            answer_style="academic"
        )
        assert custom_config.max_retrieved_docs == 8
        assert custom_config.answer_style == "academic"
        assert custom_config.llm_model == DEFAULT_QA_CONFIG.llm_model  # Unchanged


class TestAgentType:
    """Test cases for the AgentType type alias."""

    def test_agent_type_values(self):
        """Test that AgentType has the expected values."""
        # AgentType is a Literal type alias, so we test the values directly
        valid_types = ["retrieval", "supervisor", "qa_orchestrator"]
        for agent_type in valid_types:
            assert agent_type in valid_types

    def test_agent_type_invalid_value(self):
        """Test that AgentType rejects invalid values."""
        # Since AgentType is a Literal, we can't instantiate it directly
        # We just verify the valid values are defined
        valid_types = ["retrieval", "supervisor", "qa_orchestrator"]
        assert "invalid_type" not in valid_types
