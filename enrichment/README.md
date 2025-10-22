# LERK System - Enrichment Module

The enrichment module handles the enrichment of document chunks with summaries, keywords, questions, and table data using LLM-based processing.

## Overview

The enrichment module takes parsed document chunks and enhances them with:
- **Summaries**: Concise 1-2 sentence summaries of chunk content
- **Keywords**: 5-7 key topics or entities mentioned
- **Hypothetical Questions**: 3-5 questions the chunk could answer
- **Table Summaries**: Natural language summaries of table data (for table chunks)

## Module Structure

```
enrichment/
├── __init__.py              # Module initialization and public API
├── models.py                # Pydantic models for enrichment data
├── enricher.py              # Main enrichment processing classes
├── prompts.py                # Prompt templates for LLM processing
├── config.py                 # Configuration classes and presets
├── utils.py                  # Utility functions for chunk processing
├── exceptions.py             # Custom exception classes
├── example.py                # Usage examples and demonstrations
└── README.md                 # This documentation
```

## Key Components

### 1. ChunkEnrichment Model (`models.py`)

The core data structure for enrichment results:

```python
from enrichment import ChunkEnrichment

# Enrichment data structure
enrichment = ChunkEnrichment(
    summary="Brief summary of the chunk",
    keywords=["keyword1", "keyword2", "keyword3"],
    hypothetical_questions=["Question 1?", "Question 2?"],
    table_summary="Table summary (if applicable)"
)
```

### 2. DocumentEnricher Class (`enricher.py`)

Main class for processing document chunks:

```python
from enrichment import DocumentEnricher, EnrichmentConfig

# Create enricher with default configuration
enricher = DocumentEnricher()

# Enrich a single chunk
enriched_chunk = enricher.enrich_chunk(chunk)

# Enrich multiple chunks
enriched_chunks = enricher.enrich_chunks(chunks)

# Async enrichment
enriched_chunks = await enricher.enrich_chunks_async(chunks)
```

### 3. Configuration (`config.py`)

Flexible configuration system with presets:

```python
from enrichment import (
    EnrichmentConfig,
    DEFAULT_ENRICHMENT_CONFIG,
    FAST_ENRICHMENT_CONFIG,
    HIGH_QUALITY_ENRICHMENT_CONFIG
)

# Use predefined configurations
enricher = DocumentEnricher(config=FAST_ENRICHMENT_CONFIG)

# Create custom configuration
config = EnrichmentConfig(
    model_name="gpt-4o",
    temperature=0.1,
    batch_size=3,
    max_keywords=10
)
enricher = DocumentEnricher(config=config)
```

### 4. Utility Functions (`utils.py`)

Helper functions for chunk processing:

```python
from enrichment import (
    is_table_chunk,
    get_chunk_content,
    process_chunks_concurrently,
    get_chunk_statistics
)

# Check if chunk is a table
is_table = is_table_chunk(chunk)

# Get chunk content
content = get_chunk_content(chunk, is_table)

# Get processing statistics
stats = get_chunk_statistics(chunks)
```

## Usage Examples

### Basic Usage

```python
from enrichment import DocumentEnricher, add_enrichment_to_chunk

# Simple enrichment
enriched_chunk = add_enrichment_to_chunk(chunk)

# Using the enricher class
enricher = DocumentEnricher()
enriched_chunk = enricher.enrich_chunk(chunk)
```

### Batch Processing

```python
from enrichment import DocumentEnricher

enricher = DocumentEnricher()

# Process multiple chunks
chunks = [chunk1, chunk2, chunk3]
enriched_chunks = enricher.enrich_chunks(chunks)

# Get statistics
stats = enricher.get_enrichment_stats(enriched_chunks)
print(f"Enriched {stats['enriched_chunks']} out of {stats['total_chunks']} chunks")
```

### Async Processing

```python
import asyncio
from enrichment import DocumentEnricher

async def process_documents():
    enricher = DocumentEnricher()
    chunks = get_document_chunks()
    
    # Process chunks asynchronously
    enriched_chunks = await enricher.enrich_chunks_async(chunks)
    return enriched_chunks

# Run async processing
enriched_chunks = asyncio.run(process_documents())
```

### Custom Configuration

```python
from enrichment import EnrichmentConfig, DocumentEnricher

# Create custom configuration
config = EnrichmentConfig(
    model_name="gpt-4o",
    temperature=0.1,
    batch_size=5,
    max_keywords=8,
    max_questions=6,
    enable_async_processing=True
)

# Create enricher with custom config
enricher = DocumentEnricher(config=config)
```

## Configuration Options

### EnrichmentConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "gpt-4o-mini" | LLM model for enrichment |
| `temperature` | float | 0.0 | LLM temperature (0.0-2.0) |
| `max_tokens` | int | None | Maximum tokens for LLM response |
| `batch_size` | int | 5 | Concurrent processing batch size |
| `max_retries` | int | 3 | Maximum retry attempts |
| `retry_delay` | float | 1.0 | Delay between retries (seconds) |
| `max_keywords` | int | 7 | Maximum keywords to extract |
| `max_questions` | int | 5 | Maximum hypothetical questions |
| `summary_max_length` | int | 200 | Maximum summary length |
| `enable_table_summary` | bool | True | Enable table summaries |
| `enable_async_processing` | bool | True | Enable async processing |

### Predefined Configurations

- **DEFAULT_ENRICHMENT_CONFIG**: Balanced settings for general use
- **FAST_ENRICHMENT_CONFIG**: Optimized for speed
- **HIGH_QUALITY_ENRICHMENT_CONFIG**: Optimized for quality

## Error Handling

The module provides comprehensive error handling with custom exceptions:

```python
from enrichment import (
    EnrichmentError,
    ChunkProcessingError,
    LLMInvocationError,
    BatchProcessingError
)

try:
    enriched_chunk = enricher.enrich_chunk(chunk)
except ChunkProcessingError as e:
    print(f"Failed to process chunk: {e}")
except LLMInvocationError as e:
    print(f"LLM invocation failed: {e}")
```

## Integration with LERK System

The enrichment module integrates seamlessly with the broader LERK system:

```python
from ingest import DocumentIngestionOrchestrator
from enrichment import DocumentEnricher

# Process documents through ingestion
orchestrator = DocumentIngestionOrchestrator()
result = orchestrator.ingest_file("document.pdf")

# Enrich the processed chunks
enricher = DocumentEnricher()
enriched_chunks = enricher.enrich_chunks(result["chunks"])
```

## Performance Considerations

### Batch Processing
- Use appropriate batch sizes based on available memory
- Consider LLM rate limits when setting batch sizes
- Monitor processing time and adjust batch size accordingly

### Async Processing
- Enable async processing for better performance
- Use concurrent batch processing for large document sets
- Consider memory usage with large batch sizes

### Configuration Tuning
- Use faster models (gpt-4o-mini) for speed
- Use higher-quality models (gpt-4o) for accuracy
- Adjust temperature for creativity vs consistency

## Examples

See `example.py` for comprehensive usage examples including:
- Synchronous and asynchronous processing
- Different configuration options
- Batch processing with concurrent execution
- Error handling and statistics
- Integration with other LERK modules

## Dependencies

- `langchain-openai`: For LLM integration
- `pydantic`: For data validation and models
- `asyncio`: For asynchronous processing
- `logging`: For processing logs

## Future Enhancements

- Support for additional LLM providers
- Custom prompt templates
- Advanced batch processing strategies
- Integration with vector databases
- Real-time processing capabilities
