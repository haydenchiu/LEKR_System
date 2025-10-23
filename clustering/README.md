# Clustering Module

The clustering module provides functionality for clustering documents into subject clusters using BERTopic. It supports dynamic clustering, reassignment, and cluster management for the LERK System.

## Features

- **Document Clustering**: Cluster documents into subject-based groups using BERTopic
- **Dynamic Reassignment**: Reassign documents to clusters as new documents are added
- **Cluster Management**: Manage clusters, extract topics, and analyze cluster quality
- **Configurable Parameters**: Customize clustering behavior with various configuration options
- **Quality Metrics**: Analyze cluster quality with coherence scores and silhouette analysis
- **Topic Extraction**: Extract meaningful topics and keywords from clusters

## Installation

The clustering module requires several dependencies:

```bash
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas
```

## Quick Start

### Basic Clustering

```python
from clustering import DocumentClusterer, DEFAULT_CLUSTERING_CONFIG

# Create sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information."
]

# Initialize clusterer
clusterer = DocumentClusterer()

# Fit clusters
result = clusterer.fit_clusters(documents)

print(f"Created {result.num_clusters} clusters")
print(f"Quality score: {result.overall_quality_score:.3f}")
```

### Custom Configuration

```python
from clustering import ClusteringConfig, DocumentClusterer

# Create custom configuration
config = ClusteringConfig(
    model_name="all-MiniLM-L6-v2",
    min_cluster_size=3,
    top_k_words=8,
    verbose=True
)

# Use custom configuration
clusterer = DocumentClusterer(config)
result = clusterer.fit_clusters(documents)
```

### Document Assignment

```python
# Assign new documents to existing clusters
new_documents = [
    "Artificial intelligence is transforming industries.",
    "Neural networks are inspired by biological structures."
]

assignments = clusterer.assign_documents_to_clusters(new_documents)
```

## Core Components

### Models

- **`ClusterInfo`**: Information about a single cluster
- **`ClusteringResult`**: Complete clustering result with all clusters and assignments
- **`DocumentClusterAssignment`**: Assignment of a document to a cluster

### Configuration

- **`ClusteringConfig`**: Main configuration class
- **`DEFAULT_CLUSTERING_CONFIG`**: Default configuration
- **`FAST_CLUSTERING_CONFIG`**: Fast clustering configuration
- **`HIGH_QUALITY_CLUSTERING_CONFIG`**: High-quality clustering configuration

### Core Functionality

- **`DocumentClusterer`**: Main clustering class
- **`assign_documents_to_clusters()`**: Convenience function for clustering
- **`reassign_documents_to_clusters()`**: Convenience function for reassignment

### Utilities

- **`analyze_cluster_quality()`**: Analyze cluster quality metrics
- **`get_cluster_statistics()`**: Get comprehensive cluster statistics
- **`merge_similar_clusters()`**: Merge clusters that are too similar
- **`extract_cluster_topics()`**: Extract topics from clusters

## Configuration Options

### BERTopic Parameters

```python
config = ClusteringConfig(
    model_name="all-MiniLM-L6-v2",  # Sentence transformer model
    min_cluster_size=3,             # Minimum documents per cluster
    top_k_words=10,                 # Number of top words per topic
    verbose=True                    # Show progress information
)
```

### UMAP Parameters

```python
config = ClusteringConfig(
    umap_model={
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": "cosine"
    }
)
```

### HDBSCAN Parameters

```python
config = ClusteringConfig(
    hdbscan_model={
        "min_cluster_size": 3,
        "min_samples": 1,
        "cluster_selection_epsilon": 0.1
    }
)
```

## Advanced Usage

### Cluster Analysis

```python
from clustering.utils import analyze_cluster_quality, get_cluster_statistics

# Analyze cluster quality
quality_metrics = analyze_cluster_quality(result)
print(f"Average confidence: {quality_metrics['avg_confidence']:.3f}")
print(f"High confidence ratio: {quality_metrics['high_confidence_ratio']:.3f}")

# Get comprehensive statistics
stats = get_cluster_statistics(result)
print(f"Cluster balance: {stats['cluster_balance']:.3f}")
```

### Cluster Merging

```python
from clustering.utils import merge_similar_clusters

# Merge similar clusters
merged_result = merge_similar_clusters(result, similarity_threshold=0.8)
print(f"Clusters after merging: {merged_result.num_clusters}")
```

### Topic Extraction

```python
from clustering.utils import extract_cluster_topics

# Extract cluster topics
topics = extract_cluster_topics(result, top_k=5)
for cluster_id, topic_words in topics.items():
    print(f"Cluster {cluster_id}: {', '.join(topic_words)}")
```

## Quality Metrics

The clustering module provides several quality metrics:

- **Confidence Scores**: Assignment confidence for each document
- **Coherence Scores**: Topic coherence for each cluster
- **Silhouette Score**: Overall clustering quality (if embeddings provided)
- **Cluster Balance**: Distribution of documents across clusters
- **High Confidence Ratio**: Percentage of high-confidence assignments

## Error Handling

The module includes comprehensive error handling:

- **`ClusteringError`**: Base exception for clustering errors
- **`InvalidDocumentError`**: Invalid document input
- **`ClusteringInitializationError`**: Model initialization failures
- **`ClusterAssignmentError`**: Document assignment failures

## Performance Considerations

### Fast Clustering

For faster clustering with lower quality:

```python
from clustering import FAST_CLUSTERING_CONFIG

clusterer = DocumentClusterer(FAST_CLUSTERING_CONFIG)
```

### High Quality Clustering

For higher quality clustering with longer processing time:

```python
from clustering import HIGH_QUALITY_CLUSTERING_CONFIG

clusterer = DocumentClusterer(HIGH_QUALITY_CLUSTERING_CONFIG)
```

### Parallel Processing

```python
config = ClusteringConfig(
    n_jobs=4,  # Use 4 parallel jobs
    verbose=True
)
```

## Integration with LERK System

The clustering module integrates with the LERK System by:

1. **Document Processing**: Clustering enriched and logic-extracted documents
2. **Subject Organization**: Organizing documents by subject clusters
3. **Dynamic Updates**: Reassigning documents as new content is added
4. **Quality Monitoring**: Tracking clustering quality over time

## Examples

See `clustering/example.py` for comprehensive examples of:

- Basic clustering operations
- Custom configuration usage
- Cluster analysis and statistics
- Document assignment and reassignment
- Cluster merging and topic extraction
- Different configuration presets

## Dependencies

- **bertopic**: Main clustering library
- **sentence-transformers**: Embedding models
- **umap-learn**: Dimensionality reduction
- **hdbscan**: Density-based clustering
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **numpy**: Numerical operations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Use smaller models or reduce batch sizes
3. **Low Quality Clusters**: Adjust `min_cluster_size` and `min_samples`
4. **Slow Performance**: Use `FAST_CLUSTERING_CONFIG` or reduce `n_jobs`

### Performance Tips

1. Use appropriate model sizes for your data
2. Tune UMAP and HDBSCAN parameters
3. Consider preprocessing documents
4. Monitor memory usage with large datasets

## Contributing

When contributing to the clustering module:

1. Follow the existing code structure
2. Add comprehensive tests for new functionality
3. Update documentation for new features
4. Ensure backward compatibility
5. Add type hints and docstrings

## License

This module is part of the LERK System and follows the same licensing terms.
