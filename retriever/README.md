# LERK Retriever Module

The LERK Retriever Module provides comprehensive retrieval functionality for the LERK System, supporting semantic search, hybrid search, and context-aware retrieval using LangChain integration with Qdrant vector database.

## Features

- **Semantic Retrieval**: Vector similarity search using embeddings
- **Hybrid Search**: Combines semantic and keyword-based search (BM25)
- **Context-Aware Retrieval**: Considers conversation history and user preferences
- **Metadata Filtering**: Filter results by document properties
- **Batch Processing**: Efficient batch retrieval for multiple queries
- **Result Ranking**: Multiple ranking strategies (relevance, quality, combined)
- **Diversity Filtering**: Avoid similar results for better coverage
- **Caching**: Optional result caching for improved performance

## Installation

The retriever module requires the following dependencies:

```bash
pip install langchain qdrant-client sentence-transformers scikit-learn
```

## Quick Start

### Basic Semantic Retrieval

```python
from retriever import SemanticRetriever, DEFAULT_RETRIEVER_CONFIG

# Initialize retriever
retriever = SemanticRetriever(config=DEFAULT_RETRIEVER_CONFIG)

# Retrieve documents
results = retriever.get_relevant_documents("machine learning algorithms", limit=5)

# Format results
from retriever.utils import format_retrieval_results
formatted_results = format_retrieval_results(results, "standard")
```

### Hybrid Search

```python
from retriever import HybridRetriever, HYBRID_RETRIEVER_CONFIG

# Initialize hybrid retriever
retriever = HybridRetriever(config=HYBRID_RETRIEVER_CONFIG)

# Search with custom weights
results = retriever.search_with_weights(
    query="deep learning for computer vision",
    semantic_weight=0.7,
    keyword_weight=0.3,
    limit=5
)
```

### Context-Aware Retrieval

```python
from retriever import ContextRetriever, HIGH_QUALITY_RETRIEVER_CONFIG

# Initialize context-aware retriever
retriever = ContextRetriever(config=HIGH_QUALITY_RETRIEVER_CONFIG)

# Start a session
session_id = retriever.start_session()

# Add to context
retriever.add_to_context("What is machine learning?", results)

# Retrieve with context
results = retriever.get_relevant_documents("How does it work?", limit=5)
```

## Configuration

### RetrieverConfig

The `RetrieverConfig` class provides comprehensive configuration options:

```python
from retriever import RetrieverConfig, SearchStrategy, RankingMethod

config = RetrieverConfig(
    search_strategy=SearchStrategy.HYBRID,
    max_results=10,
    similarity_threshold=0.7,
    embedding_model="all-mpnet-base-v2",
    enable_hybrid_search=True,
    semantic_weight=0.6,
    keyword_weight=0.4,
    enable_context_awareness=True,
    context_window_size=5,
    ranking_method=RankingMethod.COMBINED,
    enable_diversity=True
)
```

### Predefined Configurations

```python
from retriever import (
    DEFAULT_RETRIEVER_CONFIG,
    FAST_RETRIEVER_CONFIG,
    HIGH_QUALITY_RETRIEVER_CONFIG,
    SEMANTIC_RETRIEVER_CONFIG,
    HYBRID_RETRIEVER_CONFIG,
    CONTEXT_AWARE_RETRIEVER_CONFIG
)

# Use predefined configurations
retriever = SemanticRetriever(config=FAST_RETRIEVER_CONFIG)
```

## Retrieval Strategies

### 1. Semantic Retrieval

Uses vector similarity search to find semantically similar documents:

```python
from retriever import SemanticRetriever

retriever = SemanticRetriever()

# Basic semantic search
results = retriever.get_relevant_documents("neural networks")

# Search with threshold
results = retriever.similarity_search_with_threshold(
    query="machine learning",
    threshold=0.8
)

# Find similar documents
similar_docs = retriever.get_similar_documents("doc_123", limit=5)

# Search by concept
concept_results = retriever.semantic_search_by_concept(
    concept="artificial intelligence",
    concept_type="key_concept"
)
```

### 2. Hybrid Retrieval

Combines semantic and keyword search for better accuracy:

```python
from retriever import HybridRetriever

retriever = HybridRetriever()

# Standard hybrid search
results = retriever.get_relevant_documents("deep learning")

# Search with custom weights
results = retriever.search_with_weights(
    query="computer vision",
    semantic_weight=0.8,
    keyword_weight=0.2
)
```

### 3. Context-Aware Retrieval

Considers conversation history and user preferences:

```python
from retriever import ContextRetriever

retriever = ContextRetriever()

# Start session
session_id = retriever.start_session()

# Update user preferences
retriever.update_user_preferences({
    "categories": ["AI", "Machine Learning"],
    "subjects": ["Computer Science"]
})

# Retrieve with feedback
results = retriever.retrieve_with_feedback(
    query="latest AI research",
    feedback={
        "liked_documents": ["doc_1", "doc_3"],
        "disliked_documents": ["doc_2"]
    }
)
```

## Advanced Features

### Metadata Filtering

Filter results by document metadata:

```python
# Filter by category
filters = {"category": "AI"}
results = retriever.get_relevant_documents("machine learning", filters=filters)

# Filter by quality score
filters = {"quality_score": {"min": 0.8}}
results = retriever.get_relevant_documents("deep learning", filters=filters)

# Combined filters
filters = {
    "category": "AI",
    "quality_score": {"min": 0.7},
    "subject": "Computer Science"
}
results = retriever.get_relevant_documents("neural networks", filters=filters)
```

### Batch Processing

Process multiple queries efficiently:

```python
queries = [
    "machine learning",
    "deep learning", 
    "neural networks",
    "natural language processing"
]

# Batch retrieval
batch_results = retriever.batch_retrieve(queries, limit=3)

# Process results
for query, results in zip(queries, batch_results):
    print(f"Query: {query}")
    for result in results:
        print(f"  - {result.metadata.get('title')}")
```

### Result Formatting

Format results for different use cases:

```python
from retriever.utils import format_retrieval_results

# Minimal format
minimal_results = format_retrieval_results(results, "minimal")

# Standard format
standard_results = format_retrieval_results(results, "standard")

# Detailed format
detailed_results = format_retrieval_results(results, "detailed")
```

### Result Merging

Merge results from multiple retrievers:

```python
from retriever.utils import merge_retrieval_results

# Merge with different strategies
union_results = merge_retrieval_results([results1, results2], "union")
intersection_results = merge_retrieval_results([results1, results2], "intersection")
weighted_results = merge_retrieval_results([results1, results2], "weighted")
```

## Utility Functions

### Relevance Scoring

```python
from retriever.utils import calculate_relevance_score

# Calculate relevance score
for doc in results:
    relevance = calculate_relevance_score(doc, "machine learning")
    doc.metadata['relevance_score'] = relevance
```

### Metadata Filtering

```python
from retriever.utils import filter_by_metadata

# Filter documents
filters = {"category": "AI", "quality_score": {"min": 0.8}}
filtered_docs = filter_by_metadata(results, filters)
```

### Quality Ranking

```python
from retriever.utils import rank_by_quality

# Rank by quality score
ranked_docs = rank_by_quality(results)
```

## Error Handling

The retriever module provides comprehensive error handling:

```python
from retriever.exceptions import (
    RetrievalError, VectorSearchError, InvalidQueryError,
    MissingEmbeddingModelError, DatabaseConnectionError
)

try:
    results = retriever.get_relevant_documents("machine learning")
except VectorSearchError as e:
    print(f"Vector search failed: {e}")
except InvalidQueryError as e:
    print(f"Invalid query: {e}")
except DatabaseConnectionError as e:
    print(f"Database connection failed: {e}")
```

## Performance Optimization

### Caching

Enable caching for improved performance:

```python
config = RetrieverConfig(
    enable_caching=True,
    cache_ttl=3600  # 1 hour
)
retriever = SemanticRetriever(config=config)
```

### Batch Processing

Use batch processing for multiple queries:

```python
# Process queries in batches
batch_results = retriever.batch_retrieve(
    queries, 
    batch_size=10,
    limit=5
)
```

### Diversity Filtering

Enable diversity filtering to avoid similar results:

```python
config = RetrieverConfig(
    enable_diversity=True,
    diversity_threshold=0.3
)
retriever = SemanticRetriever(config=config)
```

## Monitoring and Statistics

Get detailed statistics about retriever performance:

```python
# Get retriever statistics
stats = retriever.get_retrieval_stats()
print(f"Retriever type: {stats['retriever_type']}")
print(f"Initialized: {stats['initialized']}")
print(f"Vector store connected: {stats['vector_store_connected']}")

# Get context summary (for context-aware retriever)
if hasattr(retriever, 'get_context_summary'):
    context_summary = retriever.get_context_summary()
    print(f"Context entries: {context_summary['context_entries']}")
```

## Integration with LERK System

The retriever module integrates seamlessly with other LERK System components:

```python
# Integration with consolidation module
from consolidation import DocumentKnowledge, SubjectKnowledge
from retriever import SemanticRetriever

# Retrieve document knowledge
retriever = SemanticRetriever()
results = retriever.get_relevant_documents("machine learning concepts")

# Filter by document knowledge
doc_knowledge_results = [
    doc for doc in results 
    if doc.metadata.get('knowledge_type') == 'document'
]

# Filter by subject knowledge  
subject_knowledge_results = [
    doc for doc in results
    if doc.metadata.get('knowledge_type') == 'subject'
]
```

## Examples

See `example.py` for comprehensive usage examples demonstrating all retriever functionality.

## API Reference

### Classes

- `RetrieverConfig`: Configuration class for retrievers
- `LERKBaseRetriever`: Base retriever class
- `SemanticRetriever`: Semantic search retriever
- `HybridRetriever`: Hybrid search retriever
- `ContextRetriever`: Context-aware retriever

### Functions

- `format_retrieval_results()`: Format results for different outputs
- `calculate_relevance_score()`: Calculate relevance scores
- `filter_by_metadata()`: Filter by metadata criteria
- `rank_by_quality()`: Rank by quality scores
- `batch_retrieve()`: Batch processing
- `merge_retrieval_results()`: Merge multiple result sets

### Exceptions

- `RetrievalError`: Base exception for retrieval errors
- `VectorSearchError`: Vector search operation errors
- `InvalidQueryError`: Invalid query errors
- `MissingEmbeddingModelError`: Missing embedding model errors
- `DatabaseConnectionError`: Database connection errors
- `ConfigurationError`: Configuration errors

## Contributing

Contributions to the retriever module are welcome! Please ensure all new functionality includes appropriate tests and documentation.

## License

This module is part of the LERK System and follows the same licensing terms.
