# LERK System - Ingestion Module Tests

This directory contains comprehensive unit tests for the LERK System ingestion module.

## Test Structure

```
tests/ingest_test/
├── __init__.py              # Test package initialization
├── conftest.py             # Pytest configuration and fixtures
├── test_parsing.py         # Tests for parsing module
├── test_chunking.py        # Tests for chunking module
├── test_orchestrator.py    # Tests for orchestrator module
├── test_config.py          # Tests for config module
├── test_utils.py           # Tests for utils module
├── test_exceptions.py      # Tests for exceptions module
├── run_tests.py            # Test runner script
└── README.md               # This file
```

## Running Tests

### Using the Test Runner

The easiest way to run tests is using the provided test runner:

```bash
# Run all tests
python tests/ingest_test/run_tests.py

# Run with verbose output
python tests/ingest_test/run_tests.py -v

# Run with coverage reporting
python tests/ingest_test/run_tests.py -c

# Run all tests with verbose output and coverage
python tests/ingest_test/run_tests.py --all

# Run specific test file
python tests/ingest_test/run_tests.py -t parsing
```

### Using pytest directly

```bash
# Run all tests
pytest tests/ingest_test/

# Run with verbose output
pytest tests/ingest_test/ -v

# Run with coverage
pytest tests/ingest_test/ --cov=ingest --cov-report=html

# Run specific test file
pytest tests/ingest_test/test_parsing.py

# Run specific test function
pytest tests/ingest_test/test_parsing.py::TestDocumentParser::test_parse_file_success
```

## Test Coverage

The tests provide comprehensive coverage for:

### Parsing Module (`test_parsing.py`)
- ✅ DocumentParser class initialization
- ✅ File parsing with various parameters
- ✅ URL parsing
- ✅ Text content parsing
- ✅ Error handling for parsing failures
- ✅ File type detection
- ✅ Convenience function testing

### Chunking Module (`test_chunking.py`)
- ✅ DocumentChunker class initialization
- ✅ Element chunking with various parameters
- ✅ Chunk statistics calculation
- ✅ Error handling for chunking failures
- ✅ Convenience function testing
- ✅ Integration testing

### Orchestrator Module (`test_orchestrator.py`)
- ✅ DocumentIngestionOrchestrator initialization
- ✅ Single file ingestion
- ✅ Multiple file ingestion
- ✅ URL ingestion
- ✅ Text content ingestion
- ✅ Error handling and recovery
- ✅ Integration testing

### Configuration Module (`test_config.py`)
- ✅ ParsingConfig and ChunkingConfig classes
- ✅ IngestionConfig class
- ✅ Configuration serialization/deserialization
- ✅ Preset configurations
- ✅ Configuration validation

### Utilities Module (`test_utils.py`)
- ✅ File type detection
- ✅ Supported file type checking
- ✅ Chunk content extraction
- ✅ Table chunk detection
- ✅ Metadata handling
- ✅ File validation
- ✅ Ingestion summary generation
- ✅ Logging utilities

### Exceptions Module (`test_exceptions.py`)
- ✅ Exception hierarchy
- ✅ Exception creation and raising
- ✅ Exception catching and handling
- ✅ Error message formatting
- ✅ Exception chaining

## Test Fixtures

The `conftest.py` file provides comprehensive fixtures:

### File Fixtures
- `temp_file`: Temporary text file for testing
- `temp_pdf_file`: Temporary PDF file for testing
- `sample_text_content`: Sample text content
- `sample_html_content`: Sample HTML content

### Element Fixtures
- `sample_elements`: Sample unstructured elements
- `sample_chunks`: Sample chunked elements

### Mock Fixtures
- `mock_parser`: Mock parser for testing
- `mock_chunker`: Mock chunker for testing
- `mock_partition`: Mock partition function
- `mock_chunk_by_title`: Mock chunking function

### Result Fixtures
- `sample_ingestion_result`: Sample ingestion result
- `sample_ingestion_results`: Multiple ingestion results

## Test Categories

### Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Test error conditions
- Test edge cases

### Integration Tests
- Test component interactions
- Test complete pipelines
- Test error propagation

### Configuration Tests
- Test configuration creation
- Test configuration validation
- Test preset configurations

## Mocking Strategy

The tests use comprehensive mocking to:

1. **Isolate Components**: Each module is tested in isolation
2. **Control Dependencies**: External libraries are mocked
3. **Test Error Conditions**: Simulate various failure scenarios
4. **Speed Up Tests**: Avoid actual file I/O and network calls

### Key Mocks

- `unstructured.partition`: Mocked to return predictable elements
- `unstructured.chunking.chunk_by_title`: Mocked to return predictable chunks
- File system operations: Mocked to avoid actual file creation
- Network operations: Mocked to avoid actual HTTP requests

## Test Data

### Sample Content
- Text documents with various structures
- HTML documents with tables and formatting
- Mixed content types

### Sample Elements
- Text elements with metadata
- Table elements with HTML content
- Title elements with hierarchical structure

### Sample Chunks
- Text chunks with different sizes
- Table chunks with HTML content
- Mixed chunk types

## Error Testing

The tests extensively cover error conditions:

### Parsing Errors
- File not found
- Invalid file formats
- Corrupted files
- Network errors for URLs

### Chunking Errors
- Invalid element types
- Memory constraints
- Configuration errors

### Orchestrator Errors
- Pipeline failures
- Resource constraints
- Configuration issues

## Performance Testing

### Test Execution Time
- Individual test timing
- Suite execution time
- Memory usage monitoring

### Scalability Testing
- Large document handling
- Batch processing
- Memory management

## Continuous Integration

The tests are designed to run in CI environments:

### Requirements
- Python 3.8+
- pytest
- pytest-cov (for coverage)
- unittest.mock (built-in)

### Environment Variables
- `TEST_DATA_DIR`: Directory for test data files
- `LOG_LEVEL`: Logging level for tests
- `COVERAGE_THRESHOLD`: Minimum coverage threshold

## Debugging Tests

### Verbose Output
```bash
pytest tests/ingest_test/ -v -s
```

### Debug Mode
```bash
pytest tests/ingest_test/ --pdb
```

### Specific Test Debugging
```bash
pytest tests/ingest_test/test_parsing.py::TestDocumentParser::test_parse_file_success -v -s --pdb
```

## Test Maintenance

### Adding New Tests
1. Follow the existing naming conventions
2. Use appropriate fixtures
3. Mock external dependencies
4. Test both success and failure cases
5. Add docstrings to test methods

### Updating Tests
1. Update tests when changing module interfaces
2. Maintain backward compatibility
3. Update fixtures when adding new test data
4. Keep tests focused and atomic

### Test Documentation
1. Update this README when adding new test categories
2. Document new fixtures and their purposes
3. Explain complex test scenarios
4. Keep test names descriptive

## Best Practices

### Test Organization
- One test file per module
- Group related tests in classes
- Use descriptive test names
- Keep tests independent

### Test Data
- Use fixtures for reusable test data
- Create realistic test scenarios
- Test edge cases and error conditions
- Keep test data minimal but comprehensive

### Mocking
- Mock at the right level
- Don't over-mock
- Use realistic mock data
- Test both mocked and real scenarios

### Assertions
- Use specific assertions
- Test both positive and negative cases
- Verify side effects
- Check error messages and types
