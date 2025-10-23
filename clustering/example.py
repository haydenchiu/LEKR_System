"""
Example usage of the clustering module.

This module demonstrates how to use the clustering functionality
to cluster documents into subject clusters using BERTopic.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Import clustering module components
from clustering.clusterer import DocumentClusterer, assign_documents_to_clusters
from clustering.models import ClusterInfo, ClusteringResult, DocumentClusterAssignment
from clustering.config import (
    ClusteringConfig,
    DEFAULT_CLUSTERING_CONFIG,
    FAST_CLUSTERING_CONFIG,
    HIGH_QUALITY_CLUSTERING_CONFIG
)
from clustering.utils import (
    analyze_cluster_quality,
    get_cluster_statistics,
    merge_similar_clusters,
    extract_cluster_topics,
    get_cluster_summary
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_documents() -> List[str]:
    """Create sample documents for clustering demonstration."""
    return [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret and analyze visual information.",
        "Data science combines statistics, programming, and domain expertise.",
        "Python is a popular programming language for data analysis and machine learning.",
        "TensorFlow and PyTorch are popular frameworks for deep learning.",
        "Supervised learning uses labeled data to train machine learning models.",
        "Unsupervised learning finds patterns in data without labeled examples.",
        "Reinforcement learning learns through interaction with an environment.",
        "The stock market experienced significant volatility this week.",
        "Economic indicators suggest a potential recession in the coming months.",
        "Federal Reserve announced new interest rate policies.",
        "Cryptocurrency prices fluctuated dramatically in recent trading sessions.",
        "Global supply chain disruptions continue to affect manufacturing.",
        "Climate change is causing more frequent extreme weather events.",
        "Renewable energy sources are becoming more cost-effective.",
        "Carbon emissions reached record levels this year.",
        "Sustainable development goals require international cooperation.",
        "Green technology investments are increasing worldwide."
    ]


def demonstrate_basic_clustering():
    """Demonstrate basic document clustering."""
    logger.info("Demonstrating basic document clustering...")
    
    # Create sample documents
    documents = create_sample_documents()
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Use default configuration
    clusterer = DocumentClusterer()
    
    # Fit clusters
    result = clusterer.fit_clusters(documents, document_ids)
    
    # Print results
    logger.info(f"Created {result.num_clusters} clusters from {result.total_documents} documents")
    logger.info(f"Overall quality score: {result.overall_quality_score:.3f}")
    
    # Print cluster information
    for cluster in result.clusters:
        logger.info(f"Cluster {cluster.cluster_id}: {cluster.document_count} documents")
        logger.info(f"  Topics: {', '.join(cluster.topic_words[:5])}")
        if cluster.coherence_score:
            logger.info(f"  Coherence: {cluster.coherence_score:.3f}")


def demonstrate_custom_configuration():
    """Demonstrate clustering with custom configuration."""
    logger.info("Demonstrating clustering with custom configuration...")
    
    # Create custom configuration
    custom_config = ClusteringConfig(
        model_name="all-MiniLM-L6-v2",
        min_cluster_size=3,
        top_k_words=8,
        verbose=True,
        umap_model={
            "n_neighbors": 10,
            "n_components": 5,
            "min_dist": 0.1
        }
    )
    
    documents = create_sample_documents()
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Use custom configuration
    clusterer = DocumentClusterer(custom_config)
    result = clusterer.fit_clusters(documents, document_ids)
    
    logger.info(f"Custom clustering created {result.num_clusters} clusters")
    logger.info(f"Quality score: {result.overall_quality_score:.3f}")


def demonstrate_cluster_analysis():
    """Demonstrate cluster analysis and statistics."""
    logger.info("Demonstrating cluster analysis...")
    
    documents = create_sample_documents()
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Cluster documents
    result = assign_documents_to_clusters(documents, document_ids)
    
    # Analyze cluster quality
    quality_metrics = analyze_cluster_quality(result)
    logger.info("Quality Metrics:")
    for metric, value in quality_metrics.items():
        logger.info(f"  {metric}: {value}")
    
    # Get cluster statistics
    stats = get_cluster_statistics(result)
    logger.info("Cluster Statistics:")
    logger.info(f"  Average cluster size: {stats.get('average_cluster_size', 'N/A')}")
    logger.info(f"  High confidence ratio: {stats.get('high_confidence_ratio', 'N/A'):.3f}")
    
    # Extract cluster topics
    topics = extract_cluster_topics(result, top_k=5)
    logger.info("Cluster Topics:")
    for cluster_id, topic_words in topics.items():
        logger.info(f"  Cluster {cluster_id}: {', '.join(topic_words)}")


def demonstrate_document_assignment():
    """Demonstrate assigning new documents to existing clusters."""
    logger.info("Demonstrating document assignment to existing clusters...")
    
    # Create initial documents and cluster them
    initial_docs = create_sample_documents()[:10]
    initial_ids = [f"initial_{i}" for i in range(len(initial_docs))]
    
    clusterer = DocumentClusterer()
    clusterer.fit_clusters(initial_docs, initial_ids)
    
    # Create new documents
    new_docs = [
        "Artificial intelligence is transforming various industries.",
        "Neural networks are inspired by biological brain structures.",
        "Data preprocessing is crucial for machine learning success."
    ]
    new_ids = [f"new_{i}" for i in range(len(new_docs))]
    
    # Assign new documents to existing clusters
    assignments = clusterer.assign_documents_to_clusters(new_docs, new_ids)
    
    logger.info(f"Assigned {len(assignments)} new documents to clusters:")
    for assignment in assignments:
        logger.info(f"  Document {assignment.document_id} -> Cluster {assignment.cluster_id} (confidence: {assignment.confidence:.3f})")


def demonstrate_cluster_merging():
    """Demonstrate merging similar clusters."""
    logger.info("Demonstrating cluster merging...")
    
    documents = create_sample_documents()
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Cluster documents
    result = assign_documents_to_clusters(documents, document_ids)
    
    logger.info(f"Original clusters: {result.num_clusters}")
    
    # Merge similar clusters
    merged_result = merge_similar_clusters(result, similarity_threshold=0.6)
    
    logger.info(f"After merging: {merged_result.num_clusters} clusters")
    
    # Show cluster summary
    summary = get_cluster_summary(merged_result)
    logger.info("Cluster Summary:")
    logger.info(summary)


def demonstrate_different_configurations():
    """Demonstrate different clustering configurations."""
    logger.info("Demonstrating different clustering configurations...")
    
    documents = create_sample_documents()
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    configs = {
        "Fast": FAST_CLUSTERING_CONFIG,
        "High Quality": HIGH_QUALITY_CLUSTERING_CONFIG,
        "Default": DEFAULT_CLUSTERING_CONFIG
    }
    
    for config_name, config in configs.items():
        logger.info(f"\n{config_name} Configuration:")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Min cluster size: {config.min_cluster_size}")
        logger.info(f"  Top K words: {config.top_k_words}")
        
        clusterer = DocumentClusterer(config)
        result = clusterer.fit_clusters(documents, document_ids)
        
        logger.info(f"  Results: {result.num_clusters} clusters, quality: {result.overall_quality_score:.3f}")


async def demonstrate_async_clustering():
    """Demonstrate asynchronous clustering operations."""
    logger.info("Demonstrating asynchronous clustering...")
    
    # This would be implemented with actual async operations
    # For now, just show the structure
    documents = create_sample_documents()
    document_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Simulate async operation
    await asyncio.sleep(0.1)
    
    result = assign_documents_to_clusters(documents, document_ids)
    logger.info(f"Async clustering completed: {result.num_clusters} clusters")


async def main():
    """Main demonstration function."""
    logger.info("Starting clustering module demonstration...")
    
    try:
        # Basic clustering
        demonstrate_basic_clustering()
        
        # Custom configuration
        demonstrate_custom_configuration()
        
        # Cluster analysis
        demonstrate_cluster_analysis()
        
        # Document assignment
        demonstrate_document_assignment()
        
        # Cluster merging
        demonstrate_cluster_merging()
        
        # Different configurations
        demonstrate_different_configurations()
        
        # Async operations
        await demonstrate_async_clustering()
        
        logger.info("Clustering demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
