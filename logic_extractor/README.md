# Logic Extractor Module

The Logic Extractor module is a powerful component of the LERK (Logic Extraction and Reasoning Knowledge) System that extracts logical and causal structure from document chunks using Large Language Models (LLMs). It identifies claims, relationships, assumptions, constraints, and open questions from text content.

## Features

- **Structured Logic Extraction**: Extract atomic claims, logical relations, assumptions, constraints, and open questions
- **Multiple LLM Support**: Compatible with various OpenAI models (GPT-3.5, GPT-4, GPT-4o)
- **Async Processing**: Full async/await support for concurrent processing
- **Configurable**: Multiple configuration presets and custom configuration options
- **Validation**: Built-in validation for extracted logic structures
- **Error Handling**: Comprehensive error handling with custom exception classes
- **Batch Processing**: Efficient batch processing of multiple chunks
- **Export Options**: Export results to JSON format

## Installation

The Logic Extractor module is part of the LERK System. Ensure you have the required dependencies installed:

```bash
pip install langchain-openai pydantic unstructured
```

## Quick Start

### Basic Usage

```python
from logic_extractor import LogicExtractor, DEFAULT_LOGIC_EXTRACTION_CONFIG

# Initialize the extractor
extractor = LogicExtractor()

# Extract logic from a chunk
processed_chunk = extractor.extract_logic(chunk)

# Access extracted logic
if hasattr(processed_chunk, 'logic') and processed_chunk.logic:
    logic = processed_chunk.logic
    print(f"Extracted {len(logic.claims)} claims")
    print(f"Extracted {len(logic.logical_relations)} relations")
```

### Async Usage

```python
import asyncio
from logic_extractor import add_logic_extraction_to_chunk_async

async def extract_logic_async(chunk):
    processed_chunk = await add_logic_extraction_to_chunk_async(chunk)
    return processed_chunk

# Usage
result = asyncio.run(extract_logic_async(chunk))
```

### Batch Processing

```python
from logic_extractor import LogicExtractor

# Initialize extractor
extractor = LogicExtractor()

# Process multiple chunks
processed_chunks = await extractor.extract_logic_batch_async(chunks)

# Get statistics
stats = extractor.get_extraction_stats(processed_chunks)
print(f"Extraction rate: {stats['extraction_rate']:.2%}")
```

## Configuration

### Predefined Configurations

The module provides several predefined configurations:

```python
from logic_extractor import (
    DEFAULT_LOGIC_EXTRACTION_CONFIG,
    FAST_LOGIC_EXTRACTION_CONFIG,
    HIGH_QUALITY_LOGIC_EXTRACTION_CONFIG
)

# Use different presets
fast_extractor = LogicExtractor(config=FAST_LOGIC_EXTRACTION_CONFIG)
quality_extractor = LogicExtractor(config=HIGH_QUALITY_LOGIC_EXTRACTION_CONFIG)
```

### Custom Configuration

```python
from logic_extractor import LogicExtractor, LogicExtractionConfig

# Create custom configuration
custom_config = LogicExtractionConfig(
    model_name="gpt-4o",
    temperature=0.1,
    batch_size=3,
    timeout=120,
    max_claims_per_chunk=15,
    max_relations_per_chunk=20,
    min_confidence_threshold=0.5
)

# Use custom configuration
extractor = LogicExtractor(config=custom_config)
```

## Data Models

### Claim

Represents an atomic logical or causal statement:

```python
from logic_extractor import Claim

claim = Claim(
    id="claim_1",
    statement="The Transformer model uses self-attention mechanism",
    type="factual",
    confidence=0.9,
    derived_from=None
)
```

### LogicalRelation

Represents a causal or inferential connection between claims:

```python
from logic_extractor import LogicalRelation

relation = LogicalRelation(
    premise="claim_1",
    conclusion="claim_2",
    relation_type="supportive",
    certainty=0.8
)
```

### LogicExtractionSchemaLiteChunk

Complete logic extraction result for a chunk:

```python
from logic_extractor import LogicExtractionSchemaLiteChunk

logic_result = LogicExtractionSchemaLiteChunk(
    chunk_id="chunk_1",
    claims=[claim1, claim2],
    logical_relations=[relation1],
    assumptions=["Assumption 1"],
    constraints=["Constraint 1"],
    open_questions=["Question 1"]
)
```

## Utility Functions

### Validation

```python
from logic_extractor.utils import validate_logic_extraction, validate_chunk

# Validate a chunk before processing
if validate_chunk(chunk):
    result = extractor.extract_logic(chunk)

# Validate extraction results
errors = validate_logic_extraction(logic_result)
if errors:
    print(f"Validation errors: {errors}")
```

### Statistics

```python
from logic_extractor.utils import get_logic_extraction_statistics

stats = get_logic_extraction_statistics(processed_chunks)
print(f"Total claims: {stats['total_claims']}")
print(f"Average claims per chunk: {stats['avg_claims_per_chunk']}")
```

### Export

```python
from logic_extractor.utils import export_logic_extraction_to_json

# Export to JSON
json_output = export_logic_extraction_to_json(logic_result)
print(json_output)
```

## Error Handling

The module provides comprehensive error handling:

```python
from logic_extractor.exceptions import (
    LogicExtractionError,
    LLMInvocationError,
    InvalidChunkError,
    MissingAPIKeyError
)

try:
    result = extractor.extract_logic(chunk)
except InvalidChunkError as e:
    print(f"Invalid chunk: {e}")
except LLMInvocationError as e:
    print(f"LLM error: {e}")
except MissingAPIKeyError as e:
    print(f"Missing API key: {e}")
```

## Examples

### Complete Example

```python
import asyncio
from logic_extractor import LogicExtractor, DEFAULT_LOGIC_EXTRACTION_CONFIG

async def main():
    # Initialize extractor
    extractor = LogicExtractor(config=DEFAULT_LOGIC_EXTRACTION_CONFIG)
    
    # Process chunk
    processed_chunk = await extractor.extract_logic_async(chunk)
    
    # Display results
    if hasattr(processed_chunk, 'logic') and processed_chunk.logic:
        logic = processed_chunk.logic
        
        print("Claims:")
        for claim in logic.claims:
            print(f"  - {claim.statement} (confidence: {claim.confidence})")
        
        print("Relations:")
        for relation in logic.logical_relations:
            print(f"  - {relation.premise} -> {relation.conclusion}")
        
        if logic.assumptions:
            print(f"Assumptions: {logic.assumptions}")
        
        if logic.constraints:
            print(f"Constraints: {logic.constraints}")
        
        if logic.open_questions:
            print(f"Open questions: {logic.open_questions}")

# Run the example
asyncio.run(main())
```

### Batch Processing Example

```python
import asyncio
from logic_extractor import LogicExtractor, FAST_LOGIC_EXTRACTION_CONFIG

async def process_document(chunks):
    # Initialize extractor with fast config
    extractor = LogicExtractor(config=FAST_LOGIC_EXTRACTION_CONFIG)
    
    # Process all chunks
    processed_chunks = await extractor.extract_logic_batch_async(chunks)
    
    # Get statistics
    stats = extractor.get_extraction_stats(processed_chunks)
    
    print(f"Processed {stats['total_chunks']} chunks")
    print(f"Extraction rate: {stats['extraction_rate']:.2%}")
    print(f"Total claims: {stats['total_claims']}")
    print(f"Total relations: {stats['total_relations']}")
    
    return processed_chunks

# Usage
processed_chunks = asyncio.run(process_document(chunks))
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_name` | str | "gpt-4o-mini" | LLM model to use |
| `temperature` | float | 0.0 | Generation temperature |
| `max_retries` | int | 3 | Maximum retry attempts |
| `timeout` | int | 60 | Request timeout in seconds |
| `batch_size` | int | 5 | Batch size for async processing |
| `enable_async_processing` | bool | True | Enable async processing |
| `max_claims_per_chunk` | int | 10 | Maximum claims per chunk |
| `max_relations_per_chunk` | int | 15 | Maximum relations per chunk |
| `min_confidence_threshold` | float | 0.3 | Minimum confidence threshold |

## API Reference

### Classes

- `LogicExtractor`: Main extraction class
- `Claim`: Atomic claim representation
- `LogicalRelation`: Logical relationship between claims
- `LogicExtractionSchemaLiteChunk`: Complete extraction result
- `LogicExtractionConfig`: Configuration class

### Functions

- `add_logic_extraction_to_chunk()`: Synchronous single chunk extraction
- `add_logic_extraction_to_chunk_async()`: Asynchronous single chunk extraction
- `generate_logic_extraction_prompt()`: Generate LLM prompts
- `validate_logic_extraction()`: Validate extraction results
- `get_logic_extraction_statistics()`: Get processing statistics

### Exceptions

- `LogicExtractionError`: Base exception class
- `LLMInvocationError`: LLM-related errors
- `InvalidChunkError`: Invalid chunk errors
- `MissingAPIKeyError`: Missing API key errors
- `ChunkProcessingError`: Chunk processing errors

## Contributing

Contributions are welcome! Please ensure that:

1. All tests pass
2. Code coverage is maintained above 80%
3. Documentation is updated
4. Type hints are provided

## License

This module is part of the LERK System and follows the same licensing terms.
