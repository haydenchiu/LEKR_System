"""
Unit tests for clustering configuration.
"""

import pytest
from clustering.config import (
    ClusteringConfig,
    DEFAULT_CLUSTERING_CONFIG,
    FAST_CLUSTERING_CONFIG,
    HIGH_QUALITY_CLUSTERING_CONFIG,
    get_config,
    list_available_configs,
    create_custom_config
)


class TestClusteringConfig:
    """Test cases for the ClusteringConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClusteringConfig()
        
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.min_cluster_size == 3
        assert config.min_samples == 1
        assert config.top_k_words == 10
        assert config.n_jobs == 1
        assert config.verbose is False
        assert config.calculate_probabilities is True
        assert config.calculate_distances is True
        assert config.confidence_threshold == 0.5
        assert config.similarity_threshold == 0.7
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClusteringConfig(
            model_name="all-mpnet-base-v2",
            min_cluster_size=5,
            top_k_words=15,
            verbose=True,
            n_jobs=4
        )
        
        assert config.model_name == "all-mpnet-base-v2"
        assert config.min_cluster_size == 5
        assert config.top_k_words == 15
        assert config.verbose is True
        assert config.n_jobs == 4
    
    def test_validation_min_cluster_size(self):
        """Test min_cluster_size validation."""
        # Valid values
        config1 = ClusteringConfig(min_cluster_size=1)
        config2 = ClusteringConfig(min_cluster_size=10)
        
        assert config1.min_cluster_size == 1
        assert config2.min_cluster_size == 10
        
        # Invalid values
        with pytest.raises(ValueError):
            ClusteringConfig(min_cluster_size=0)
        
        with pytest.raises(ValueError):
            ClusteringConfig(min_cluster_size=-1)
    
    def test_validation_min_samples(self):
        """Test min_samples validation."""
        # Valid values
        config1 = ClusteringConfig(min_samples=1)
        config2 = ClusteringConfig(min_samples=5)
        
        assert config1.min_samples == 1
        assert config2.min_samples == 5
        
        # Invalid values
        with pytest.raises(ValueError):
            ClusteringConfig(min_samples=0)
    
    def test_validation_confidence_threshold(self):
        """Test confidence_threshold validation."""
        # Valid values
        config1 = ClusteringConfig(confidence_threshold=0.0)
        config2 = ClusteringConfig(confidence_threshold=1.0)
        config3 = ClusteringConfig(confidence_threshold=0.5)
        
        assert config1.confidence_threshold == 0.0
        assert config2.confidence_threshold == 1.0
        assert config3.confidence_threshold == 0.5
        
        # Invalid values
        with pytest.raises(ValueError):
            ClusteringConfig(confidence_threshold=-0.1)
        
        with pytest.raises(ValueError):
            ClusteringConfig(confidence_threshold=1.1)
    
    def test_validation_similarity_threshold(self):
        """Test similarity_threshold validation."""
        # Valid values
        config1 = ClusteringConfig(similarity_threshold=0.0)
        config2 = ClusteringConfig(similarity_threshold=1.0)
        config3 = ClusteringConfig(similarity_threshold=0.8)
        
        assert config1.similarity_threshold == 0.0
        assert config2.similarity_threshold == 1.0
        assert config3.similarity_threshold == 0.8
        
        # Invalid values
        with pytest.raises(ValueError):
            ClusteringConfig(similarity_threshold=-0.1)
        
        with pytest.raises(ValueError):
            ClusteringConfig(similarity_threshold=1.1)
    
    def test_umap_model_config(self):
        """Test UMAP model configuration."""
        umap_params = {
            "n_neighbors": 20,
            "n_components": 10,
            "min_dist": 0.1,
            "metric": "euclidean"
        }
        
        config = ClusteringConfig(umap_model=umap_params)
        assert config.umap_model == umap_params
    
    def test_hdbscan_model_config(self):
        """Test HDBSCAN model configuration."""
        hdbscan_params = {
            "min_cluster_size": 5,
            "min_samples": 3,
            "cluster_selection_epsilon": 0.2
        }
        
        config = ClusteringConfig(hdbscan_model=hdbscan_params)
        assert config.hdbscan_model == hdbscan_params
    
    def test_model_dump(self):
        """Test model serialization."""
        config = ClusteringConfig(model_name="test-model")
        dumped = config.model_dump()
        
        assert isinstance(dumped, dict)
        assert dumped["model_name"] == "test-model"
        assert dumped["min_cluster_size"] == 3
    
    def test_to_dict(self):
        """Test to_dict method."""
        config = ClusteringConfig(verbose=True)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["verbose"] is True
    
    def test_update(self):
        """Test configuration update."""
        config = ClusteringConfig()
        updated_config = config.update(
            model_name="updated-model",
            verbose=True
        )
        
        assert updated_config.model_name == "updated-model"
        assert updated_config.verbose is True
        assert updated_config.min_cluster_size == 3  # Unchanged
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model_name": "dict-model",
            "min_cluster_size": 5,
            "verbose": True
        }
        
        config = ClusteringConfig.from_dict(config_dict)
        
        assert config.model_name == "dict-model"
        assert config.min_cluster_size == 5
        assert config.verbose is True
    
    def test_str_representation(self):
        """Test string representation."""
        config = ClusteringConfig(model_name="test-model", min_cluster_size=5)
        config_str = str(config)
        
        assert "test-model" in config_str
        assert "5" in config_str
        assert "ClusteringConfig" in config_str
    
    def test_equality(self):
        """Test equality comparison."""
        config1 = ClusteringConfig(model_name="test")
        config2 = ClusteringConfig(model_name="test")
        config3 = ClusteringConfig(model_name="different")
        
        assert config1 == config2
        assert config1 != config3
    
    def test_hash(self):
        """Test hash functionality."""
        config1 = ClusteringConfig(model_name="test")
        config2 = ClusteringConfig(model_name="test")
        config3 = ClusteringConfig(model_name="different")
        
        # Hash should be consistent for identical configs
        assert hash(config1) == hash(config2)
        # Hash should be different for different configs
        assert hash(config1) != hash(config3)
        
        # Test that hash is actually a number
        assert isinstance(hash(config1), int)
    
    def test_get_bertopic_params(self):
        """Test BERTopic parameters extraction."""
        config = ClusteringConfig(
            model_name="test-model",
            verbose=True,
            calculate_probabilities=True,
            n_jobs=2
        )
        
        params = config.get_bertopic_params()
        
        assert params["embedding_model"] == "test-model"
        assert params["verbose"] is True
        assert params["calculate_probabilities"] is True
        # n_jobs is not included in BERTopic parameters
    
    def test_get_hdbscan_params(self):
        """Test HDBSCAN parameters extraction."""
        config = ClusteringConfig(
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.2
        )
        
        params = config.get_hdbscan_params()
        
        assert params["min_cluster_size"] == 5
        assert params["min_samples"] == 3
        assert params["cluster_selection_epsilon"] == 0.2
    
    def test_get_umap_params(self):
        """Test UMAP parameters extraction."""
        config = ClusteringConfig()
        params = config.get_umap_params()
        
        assert "n_neighbors" in params
        assert "n_components" in params
        assert "min_dist" in params
        assert "metric" in params
        
        # Test with custom UMAP model
        custom_umap = {
            "n_neighbors": 25,
            "n_components": 8,
            "min_dist": 0.2,
            "metric": "manhattan"
        }
        
        config_custom = ClusteringConfig(umap_model=custom_umap)
        params_custom = config_custom.get_umap_params()
        
        assert params_custom["n_neighbors"] == 25
        assert params_custom["n_components"] == 8
        assert params_custom["min_dist"] == 0.2
        assert params_custom["metric"] == "manhattan"


class TestConfigPresets:
    """Test cases for configuration presets."""
    
    def test_default_config_preset(self):
        """Test default configuration preset."""
        assert DEFAULT_CLUSTERING_CONFIG.model_name == "all-MiniLM-L6-v2"
        assert DEFAULT_CLUSTERING_CONFIG.min_cluster_size == 3
        assert DEFAULT_CLUSTERING_CONFIG.verbose is False
    
    def test_fast_config_preset(self):
        """Test fast configuration preset."""
        assert FAST_CLUSTERING_CONFIG.model_name == "all-MiniLM-L6-v2"
        assert FAST_CLUSTERING_CONFIG.min_cluster_size == 5
        assert FAST_CLUSTERING_CONFIG.verbose is True
        assert FAST_CLUSTERING_CONFIG.n_jobs == 2
    
    def test_high_quality_config_preset(self):
        """Test high quality configuration preset."""
        assert HIGH_QUALITY_CLUSTERING_CONFIG.model_name == "all-mpnet-base-v2"
        assert HIGH_QUALITY_CLUSTERING_CONFIG.min_cluster_size == 3
        assert HIGH_QUALITY_CLUSTERING_CONFIG.verbose is True
        assert HIGH_QUALITY_CLUSTERING_CONFIG.n_jobs == 4
    
    def test_config_presets_immutable(self):
        """Test that config presets are immutable."""
        original_model = DEFAULT_CLUSTERING_CONFIG.model_name
        
        # Create a copy to test immutability
        new_config = DEFAULT_CLUSTERING_CONFIG.update(model_name="modified")
        assert new_config.model_name == "modified"
        # Original should remain unchanged
        assert DEFAULT_CLUSTERING_CONFIG.model_name == original_model
        
        # Test that we can create multiple independent configs
        config1 = DEFAULT_CLUSTERING_CONFIG.update(model_name="config1")
        config2 = DEFAULT_CLUSTERING_CONFIG.update(model_name="config2")
        assert config1.model_name == "config1"
        assert config2.model_name == "config2"
        assert config1 != config2


class TestConfigRegistry:
    """Test cases for configuration registry."""
    
    def test_get_config_default(self):
        """Test getting default configuration."""
        config = get_config("default")
        assert isinstance(config, ClusteringConfig)
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.min_cluster_size == 3
        assert config.verbose is False
    
    def test_get_config_fast(self):
        """Test getting fast configuration."""
        config = get_config("fast")
        assert isinstance(config, ClusteringConfig)
        assert config.verbose is True
    
    def test_get_config_high_quality(self):
        """Test getting high quality configuration."""
        config = get_config("high_quality")
        assert isinstance(config, ClusteringConfig)
        assert config.model_name == "all-mpnet-base-v2"
    
    def test_get_config_invalid(self):
        """Test getting invalid configuration."""
        with pytest.raises(ValueError, match="Unknown config name"):
            get_config("invalid_config")
    
    def test_list_available_configs(self):
        """Test listing available configurations."""
        configs = list_available_configs()
        
        assert isinstance(configs, list)
        assert "default" in configs
        assert "fast" in configs
        assert "high_quality" in configs
        assert "balanced" in configs
    
    def test_create_custom_config(self):
        """Test creating custom configuration."""
        custom_config = create_custom_config(
            model_name="custom-model",
            min_cluster_size=10,
            verbose=True
        )
        
        assert isinstance(custom_config, ClusteringConfig)
        assert custom_config.model_name == "custom-model"
        assert custom_config.min_cluster_size == 10
        assert custom_config.verbose is True
        assert custom_config.min_samples == 1  # Default value
