# Consolidation Module

The Consolidation Module provides functionality for consolidating knowledge from document chunks into document-level and subject-level knowledge representations, and storing them for agentic Q&A retrieval.

## Overview

This module implements three main functions as requested:

1. **Document-level consolidation**: Summarize logic chunks → key concepts → document knowledge
2. **Subject-level consolidation**: Aggregate document knowledge → general knowledge  
3. **Storage**: Save into database for agentic Q&A retrieval

## Features

- **Document Consolidation**: Extract key concepts and relations from document chunks
- **Subject Consolidation**: Aggregate knowledge from multiple documents into subject knowledge
- **Knowledge Storage**: Store and retrieve consolidated knowledge with vector search capabilities
- **Quality Validation**: Ensure consistency and quality of consolidated knowledge
- **Flexible Configuration**: Multiple presets for different use cases
- **Vector Search**: Semantic similarity search for knowledge retrieval

## Architecture

### Basic Flow
```
Document Chunks → Document Consolidation → Document Knowledge
                                        ↓
Subject Knowledge ← Subject Consolidation ← Multiple Documents
                                        ↓
Knowledge Storage → Database → Agentic Q&A Retrieval
```

### Integrated Flow with Clustering
```
Document Chunks → Document Consolidation → Document Knowledge
                                        ↓
Document Clustering → Document Clusters (Subjects)
                                        ↓
Subject Knowledge ← Cluster-Based Consolidation ← Document Clusters
                                        ↓
Knowledge Storage → Database → Agentic Q&A Retrieval
```

## Components

### Core Classes

- **`DocumentConsolidator`**: Consolidates knowledge from document chunks
- **`SubjectConsolidator`**: Aggregates document knowledge into subject knowledge
- **`KnowledgeStorageManager`**: Manages storage and retrieval of knowledge
- **`ClusterBasedSubjectConsolidator`**: Consolidates subjects from document clusters
- **`IntegratedConsolidationPipeline`**: Complete pipeline with clustering integration

### Models

- **`DocumentKnowledge`**: Represents consolidated knowledge at document level
- **`SubjectKnowledge`**: Represents consolidated knowledge at subject level
- **`KeyConcept`**: Represents a key concept with metadata
- **`KnowledgeRelation`**: Represents relationships between concepts
- **`KnowledgeStorage`**: Represents storage information

### Configuration

- **`ConsolidationConfig`**: Main configuration class
- **Presets**: `DEFAULT_CONSOLIDATION_CONFIG`, `FAST_CONSOLIDATION_CONFIG`, `HIGH_QUALITY_CONSOLIDATION_CONFIG`

## Installation

```bash
pip install langchain-openai sentence-transformers scikit-learn sqlalchemy
```

## Usage

### Basic Document Consolidation

```python
from consolidation import DocumentConsolidator, DEFAULT_CONSOLIDATION_CONFIG

# Initialize consolidator
consolidator = DocumentConsolidator(DEFAULT_CONSOLIDATION_CONFIG)

# Consolidate document chunks
document_knowledge = consolidator.consolidate_document(
    document_id="doc_001",
    document_title="AI Fundamentals",
    chunks=document_chunks,
    chunk_logic_data=logic_extraction_data
)

print(f"Extracted {len(document_knowledge.key_concepts)} concepts")
print(f"Quality score: {document_knowledge.quality_score}")
```

### Subject-Level Consolidation

#### Basic Subject Consolidation
```python
from consolidation import SubjectConsolidator

# Initialize subject consolidator
subject_consolidator = SubjectConsolidator()

# Consolidate multiple documents into subject knowledge
subject_knowledge = subject_consolidator.consolidate_subject(
    subject_id="subject_ai",
    subject_name="Artificial Intelligence",
    document_knowledge=[doc1_knowledge, doc2_knowledge],
    subject_description="Comprehensive AI knowledge"
)

print(f"Subject expertise level: {subject_knowledge.expertise_level}")
print(f"Core concepts: {len(subject_knowledge.core_concepts)}")
```

#### Cluster-Based Subject Consolidation
```python
from consolidation import ClusterBasedSubjectConsolidator
from clustering import DocumentClusterer

# Step 1: Cluster documents
clusterer = DocumentClusterer()
clustering_result = clusterer.fit_clusters(documents, document_ids)

# Step 2: Consolidate subjects from clusters
cluster_consolidator = ClusterBasedSubjectConsolidator()
subject_knowledge_list = cluster_consolidator.consolidate_subjects_from_clusters(
    clustering_result, document_knowledge_map
)

print(f"Created {len(subject_knowledge_list)} subjects from clusters")
```

#### Integrated Pipeline
```python
from consolidation import IntegratedConsolidationPipeline

# Complete pipeline from documents to subjects
pipeline = IntegratedConsolidationPipeline()

doc_knowledge_list, subject_knowledge_list = pipeline.process_documents_to_subjects(
    documents=document_texts,
    document_ids=document_ids,
    document_chunks_map=chunks_map,
    chunk_logic_data_map=logic_data_map
)

print(f"Processed {len(doc_knowledge_list)} documents into {len(subject_knowledge_list)} subjects")
```

### Knowledge Storage and Retrieval

```python
from consolidation import KnowledgeStorageManager

# Initialize storage manager
storage = KnowledgeStorageManager()

# Save knowledge
doc_storage_id = storage.save_document_knowledge(document_knowledge)
subject_storage_id = storage.save_subject_knowledge(subject_knowledge)

# Retrieve knowledge
retrieved_doc = storage.retrieve_document_knowledge("doc_001")
retrieved_subject = storage.retrieve_subject_knowledge("subject_ai")

# Semantic search
similar_docs = storage.search_documents_by_similarity(
    "machine learning algorithms",
    limit=10,
    min_similarity=0.7
)
```

### Configuration Presets

```python
from consolidation import (
    FAST_CONSOLIDATION_CONFIG,
    HIGH_QUALITY_CONSOLIDATION_CONFIG,
    create_custom_config
)

# Use fast configuration for quick processing
fast_consolidator = DocumentConsolidator(FAST_CONSOLIDATION_CONFIG)

# Use high-quality configuration for best results
quality_consolidator = DocumentConsolidator(HIGH_QUALITY_CONSOLIDATION_CONFIG)

# Create custom configuration
custom_config = create_custom_config(
    model_name="gpt-4o",
    max_concepts_per_document=30,
    min_concept_confidence=0.7
)
```

## Configuration Options

### LLM Configuration
- `model_name`: LLM model to use (default: "gpt-4o-mini")
- `temperature`: Temperature for generation (default: 0.1)
- `max_tokens`: Maximum tokens for responses (default: 4000)

### Document Consolidation
- `max_concepts_per_document`: Maximum concepts per document (default: 20)
- `min_concept_confidence`: Minimum confidence threshold (default: 0.3)
- `max_relations_per_document`: Maximum relations per document (default: 30)

### Subject Consolidation
- `max_concepts_per_subject`: Maximum concepts per subject (default: 50)
- `concept_similarity_threshold`: Threshold for merging concepts (default: 0.7)
- `relation_strength_threshold`: Minimum relation strength (default: 0.5)

### Storage Configuration
- `storage_backend`: Storage backend ("memory", "sqlite", "postgresql")
- `enable_vector_search`: Enable vector embeddings (default: True)
- `embedding_model`: Model for embeddings (default: "all-MiniLM-L6-v2")

## Quality Metrics

The module provides several quality metrics:

- **Concept Confidence**: Confidence scores for extracted concepts
- **Relation Strength**: Strength of relationships between concepts
- **Coverage Score**: How many concepts have relations
- **Diversity Score**: Category diversity of concepts
- **Overall Quality**: Combined quality score

## Error Handling

The module includes comprehensive error handling:

- **`ConsolidationError`**: Base exception for consolidation errors
- **`DocumentConsolidationError`**: Document-level consolidation failures
- **`SubjectConsolidationError`**: Subject-level consolidation failures
- **`StorageError`**: Storage operation failures
- **`KnowledgeValidationError`**: Knowledge validation failures

## Examples

See `example.py` for comprehensive examples including:

- Document consolidation with different configurations
- Subject consolidation from multiple documents
- Knowledge storage and retrieval
- Semantic similarity search
- Complete consolidation pipeline

## Integration with LERK System

The consolidation module integrates with other LERK System components:

- **Input**: Document chunks from the `ingest` module
- **Input**: Logic extraction data from the `logic_extractor` module
- **Input**: Clustering results from the `clustering` module
- **Output**: Consolidated knowledge for agentic Q&A systems

## Performance Considerations

- **Batch Processing**: Process multiple documents in batches
- **Vector Search**: Use embeddings for similarity matching
- **Caching**: Cache embeddings and processed results
- **Configuration**: Choose appropriate configuration for your use case

## Future Enhancements

- **Multi-language Support**: Support for non-English documents
- **Advanced Analytics**: More sophisticated quality metrics
- **Real-time Processing**: Stream processing capabilities
- **Distributed Storage**: Support for distributed storage backends
- **Knowledge Graphs**: Advanced graph-based knowledge representation

## Contributing

Contributions are welcome! Please see the main LERK System documentation for contribution guidelines.

## License

This module is part of the LERK System and follows the same licensing terms.
