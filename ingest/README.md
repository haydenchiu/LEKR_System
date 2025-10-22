# LERK System - Ingestion Module

The ingestion module handles the parsing and chunking of documents in the LERK System. It provides a clean interface for processing various document types and preparing them for enrichment and logic extraction.

## Features

- **Multi-format Support**: PDF, HTML, DOCX, TXT, CSV, JSON, XML, and more
- **Intelligent Chunking**: Title-based chunking with configurable parameters
- **URL Support**: Direct ingestion from web URLs
- **Text Processing**: Direct text content ingestion
- **Batch Processing**: Process multiple files efficiently
- **Configurable**: Multiple preset configurations for different use cases
- **Error Handling**: Comprehensive error handling and logging

## Quick Start

```python
from ingest import DocumentIngestionOrchestrator

# Create orchestrator with default settings
orchestrator = DocumentIngestionOrchestrator()

# Ingest a single file
result = orchestrator.ingest_file("document.pdf")

if result["success"]:
    print(f"Processed {result['statistics']['total_chunks']} chunks")
    chunks = result["chunks"]
else:
    print(f"Failed: {result['error']}")
```

## Configuration

The module provides several preset configurations:

```python
from ingest import DEFAULT_CONFIG, LARGE_DOCUMENT_CONFIG, FAST_CONFIG

# Default configuration (balanced)
orchestrator = DocumentIngestionOrchestrator()

# For large documents
orchestrator = DocumentIngestionOrchestrator(
    max_characters=4096,
    new_after_n_chars=3600
)

# For fast processing
orchestrator = DocumentIngestionOrchestrator(
    parsing_strategy="fast",
    max_characters=1024
)
```

## Supported File Types

- **PDF**: `application/pdf`
- **HTML**: `text/html`
- **Word**: `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- **Text**: `text/plain`
- **CSV**: `text/csv`
- **JSON**: `application/json`
- **XML**: `application/xml`
- **Markdown**: `text/markdown`

## API Reference

### DocumentIngestionOrchestrator

Main orchestrator class for document ingestion.

#### Methods

- `ingest_file(file_path)`: Ingest a single file
- `ingest_url(url)`: Ingest content from URL
- `ingest_text(text, filetype)`: Ingest text content directly
- `ingest_multiple_files(file_paths)`: Ingest multiple files
- `get_supported_file_types()`: Get list of supported file types

### DocumentParser

Handles parsing of various document types.

#### Methods

- `parse_file(file_path)`: Parse a file
- `parse_url(url)`: Parse content from URL
- `parse_text(text, filetype)`: Parse text content
- `get_file_type(file_path)`: Get MIME type of file

### DocumentChunker

Handles chunking of parsed elements.

#### Methods

- `chunk_elements(elements)`: Chunk a list of elements
- `get_chunk_statistics(chunks)`: Get statistics about chunks

## Examples

### Single File Ingestion

```python
from ingest import DocumentIngestionOrchestrator

orchestrator = DocumentIngestionOrchestrator()
result = orchestrator.ingest_file("research_paper.pdf")

print(f"Chunks: {result['statistics']['total_chunks']}")
print(f"Text chunks: {result['statistics']['text_chunks']}")
print(f"Table chunks: {result['statistics']['table_chunks']}")
```

### Multiple Files

```python
from ingest import DocumentIngestionOrchestrator, get_ingestion_summary

orchestrator = DocumentIngestionOrchestrator()
file_paths = ["doc1.pdf", "doc2.html", "doc3.txt"]

results = orchestrator.ingest_multiple_files(file_paths)
summary = get_ingestion_summary(results)

print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Total chunks: {summary['total_chunks']}")
```

### URL Ingestion

```python
orchestrator = DocumentIngestionOrchestrator()
result = orchestrator.ingest_url("https://example.com/article")

if result["success"]:
    print(f"Processed {len(result['chunks'])} chunks")
```

### Text Content

```python
text_content = """
This is a sample document.
It has multiple paragraphs.
"""

result = orchestrator.ingest_text(text_content, "text/plain")
chunks = result["chunks"]
```

## Configuration Options

### Parsing Configuration

- `strategy`: Parsing strategy ("hi_res", "fast", "ocr_only")
- `skip_infer_table_types`: List of table types to skip
- `max_partition`: Maximum number of partitions to process

### Chunking Configuration

- `max_characters`: Maximum size of a chunk
- `combine_text_under_n_chars`: Combine small text elements
- `new_after_n_chars`: Start new chunk if current exceeds this size

## Error Handling

The module provides comprehensive error handling:

```python
from ingest import (
    IngestionError,
    ParsingError,
    ChunkingError,
    UnsupportedFileTypeError
)

try:
    result = orchestrator.ingest_file("document.pdf")
except ParsingError as e:
    print(f"Parsing failed: {e}")
except UnsupportedFileTypeError as e:
    print(f"Unsupported file type: {e}")
```

## Utilities

### File Type Detection

```python
from ingest import get_file_type, is_supported_file_type

file_type = get_file_type("document.pdf")
print(f"File type: {file_type}")

if is_supported_file_type("document.pdf"):
    print("File type is supported")
```

### Chunk Utilities

```python
from ingest import get_chunk_content, is_table_chunk, get_chunk_metadata

for chunk in chunks:
    content = get_chunk_content(chunk)
    is_table = is_table_chunk(chunk)
    metadata = get_chunk_metadata(chunk)
    
    print(f"Content: {content[:100]}...")
    print(f"Is table: {is_table}")
```

## Integration with LERK System

The ingestion module is designed to work seamlessly with other LERK System components:

1. **Enrichment Module**: Chunks are passed to enrichment for adding summaries, keywords, and questions
2. **Logic Extractor**: Chunks are processed for logic extraction
3. **Vector Store**: Chunks are vectorized and stored for retrieval
4. **QA Agent**: Processed chunks are used for question answering

## Performance Considerations

- Use `FAST_CONFIG` for development and testing
- Use `LARGE_DOCUMENT_CONFIG` for large documents
- Use `HIGH_QUALITY_CONFIG` for production when quality is critical
- Consider batch processing for multiple files
- Monitor memory usage with large documents

## Troubleshooting

### Common Issues

1. **File not found**: Ensure file path is correct and file exists
2. **Unsupported file type**: Check if file type is in supported list
3. **Memory issues**: Use smaller chunk sizes or process files individually
4. **Parsing errors**: Try different parsing strategies

### Debug Mode

Enable debug logging to see detailed processing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
