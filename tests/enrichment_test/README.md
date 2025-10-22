# LERK System - Enrichment Module Tests

This directory contains comprehensive unit tests for the enrichment module of the LERK System.

## Test Structure

```
tests/enrichment_test/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_models.py           # Tests for ChunkEnrichment model
├── test_enricher.py         # Tests for DocumentEnricher class
├── test_prompts.py          # Tests for prompt generation functions
├── test_config.py           # Tests for configuration classes
├── test_utils.py            # Tests for utility functions
├── test_exceptions.py       # Tests for custom exceptions
├── run_tests.py             # Test runner script
└── README.md                # This documentation
```

## Test Coverage

The test suite provides comprehensive coverage for all enrichment module components:

### 1. Models (`test_models.py`)
- **ChunkEnrichment Model**: Tests for Pydantic model validation, serialization, and data handling
- **Field Validation**: Tests for required fields, data types, and constraints
- **Serialization**: Tests for `model_dump()` and `model_dump_json()` methods
- **Edge Cases**: Tests for empty values, special characters, and unicode content

### 2. Enricher (`test_enricher.py`)
- **DocumentEnricher Class**: Tests for main enrichment processing class
- **Synchronous Processing**: Tests for `enrich_chunk()` and `enrich_chunks()` methods
- **Asynchronous Processing**: Tests for `enrich_chunk_async()` and `enrich_chunks_async()` methods
- **Error Handling**: Tests for various error conditions and retry logic
- **Statistics**: Tests for enrichment statistics calculation
- **Integration**: Tests for complete enrichment pipelines

### 3. Prompts (`test_prompts.py`)
- **Prompt Generation**: Tests for `generate_enrichment_prompt()` function
- **Table Prompts**: Tests for table-specific prompt generation
- **Text Prompts**: Tests for text-specific prompt generation
- **Custom Prompts**: Tests for custom prompt creation with additional instructions
- **Edge Cases**: Tests for empty content, special characters, and unicode

### 4. Configuration (`test_config.py`)
- **EnrichmentConfig Class**: Tests for configuration validation and constraints
- **Field Validation**: Tests for all configuration parameters and their ranges
- **Predefined Configs**: Tests for DEFAULT, FAST, and HIGH_QUALITY configurations
- **Serialization**: Tests for configuration serialization and deserialization
- **Edge Cases**: Tests for minimum/maximum values and edge conditions

### 5. Utilities (`test_utils.py`)
- **Chunk Processing**: Tests for chunk content extraction and metadata handling
- **Table Detection**: Tests for table chunk identification
- **Enrichment Detection**: Tests for enriched chunk identification
- **Batch Processing**: Tests for concurrent chunk processing
- **Validation**: Tests for chunk validation and filtering
- **Statistics**: Tests for chunk statistics calculation

### 6. Exceptions (`test_exceptions.py`)
- **Exception Hierarchy**: Tests for custom exception inheritance
- **Error Context**: Tests for exception context information
- **Error Handling**: Tests for exception catching and handling
- **Error Chaining**: Tests for exception chaining with causes

## Running Tests

### Basic Test Execution

```bash
# Run all tests with coverage
python tests/enrichment_test/run_tests.py

# Run tests without coverage
python tests/enrichment_test/run_tests.py --no-cov

# Run tests in parallel
python tests/enrichment_test/run_tests.py --parallel
```

### Specific Test Execution

```bash
# Run specific test file
python tests/enrichment_test/run_tests.py test_models.py

# Run tests with pytest directly
pytest tests/enrichment_test/test_models.py -v

# Run tests with coverage
pytest tests/enrichment_test/ --cov=enrichment --cov-report=html
```

### Test Options

```bash
# Run with verbose output
pytest tests/enrichment_test/ -v

# Run with short traceback
pytest tests/enrichment_test/ --tb=short

# Run specific test method
pytest tests/enrichment_test/test_models.py::TestChunkEnrichment::test_init_default

# Run tests matching pattern
pytest tests/enrichment_test/ -k "test_enrich"
```

## Test Fixtures

The test suite includes comprehensive fixtures in `conftest.py`:

### Sample Data Fixtures
- `sample_enrichment_data`: Sample enrichment data for testing
- `sample_table_enrichment_data`: Sample table enrichment data
- `sample_chunk`: Sample text chunk for testing
- `sample_table_chunk`: Sample table chunk for testing
- `sample_chunks`: List of sample chunks

### Mock Fixtures
- `mock_llm`: Mock LLM for testing
- `mock_enrichment_result`: Mock enrichment result
- `mock_table_enrichment_result`: Mock table enrichment result
- `mock_enricher`: Mock document enricher

### Configuration Fixtures
- `default_config`: Default enrichment configuration
- `fast_config`: Fast processing configuration
- `high_quality_config`: High quality configuration
- `custom_config`: Custom configuration

### Test Data Fixtures
- `sample_prompt_data`: Sample prompt data for testing
- `sample_batch_data`: Sample batch processing data
- `sample_enrichment_stats`: Sample enrichment statistics
- `sample_async_data`: Sample async processing data

## Test Categories

### Unit Tests
- **Model Tests**: Test individual model components in isolation
- **Function Tests**: Test individual functions with mocked dependencies
- **Class Tests**: Test class methods and properties
- **Validation Tests**: Test data validation and constraints

### Integration Tests
- **Pipeline Tests**: Test complete enrichment pipelines
- **Configuration Tests**: Test configuration integration
- **Error Handling Tests**: Test error handling across components
- **Async Tests**: Test asynchronous processing workflows

### Edge Case Tests
- **Empty Data**: Tests with empty or None values
- **Invalid Data**: Tests with invalid or malformed data
- **Boundary Values**: Tests with minimum/maximum valid values
- **Error Conditions**: Tests with various error scenarios

## Test Coverage

The test suite aims for comprehensive coverage:

- **Line Coverage**: > 90% of code lines covered
- **Branch Coverage**: > 85% of code branches covered
- **Function Coverage**: 100% of functions tested
- **Class Coverage**: 100% of classes tested

### Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/enrichment_test/ --cov=enrichment --cov-report=html

# Generate terminal coverage report
pytest tests/enrichment_test/ --cov=enrichment --cov-report=term-missing

# Generate XML coverage report
pytest tests/enrichment_test/ --cov=enrichment --cov-report=xml
```

## Test Dependencies

The test suite requires the following dependencies:

```bash
# Core testing dependencies
pytest>=8.4.2
pytest-cov>=7.0.0
pytest-asyncio>=0.21.0

# Mocking dependencies
unittest.mock (built-in)

# Async testing
asyncio (built-in)
```

## Test Data

### Sample Chunks
- **Text Chunks**: Regular document text with metadata
- **Table Chunks**: HTML table data with table-specific metadata
- **Mixed Chunks**: Combination of text and table chunks

### Sample Enrichment Data
- **Basic Enrichment**: Standard enrichment with summary, keywords, questions
- **Table Enrichment**: Table-specific enrichment with table summaries
- **Empty Enrichment**: Edge cases with empty or missing data

### Sample Configurations
- **Default Config**: Balanced configuration for general use
- **Fast Config**: Optimized for speed and performance
- **High Quality Config**: Optimized for accuracy and quality
- **Custom Config**: User-defined configuration parameters

## Test Patterns

### Test Structure
```python
class TestComponentName:
    """Test cases for ComponentName class."""
    
    def test_method_success(self):
        """Test successful method execution."""
        # Arrange
        # Act
        # Assert
    
    def test_method_error(self):
        """Test method error handling."""
        # Arrange
        # Act & Assert
        with pytest.raises(ExpectedError):
            # Error condition
```

### Async Test Structure
```python
@pytest.mark.asyncio
async def test_async_method(self):
    """Test async method execution."""
    # Arrange
    # Act
    result = await async_method()
    # Assert
    assert result == expected
```

### Mock Test Structure
```python
def test_method_with_mock(self, mock_dependency):
    """Test method with mocked dependency."""
    # Arrange
    mock_dependency.return_value = expected_value
    
    # Act
    result = method_under_test()
    
    # Assert
    assert result == expected_result
    mock_dependency.assert_called_once()
```

## Best Practices

### Test Naming
- Use descriptive test names that explain the scenario
- Include the expected outcome in the test name
- Use consistent naming patterns across test files

### Test Organization
- Group related tests in the same class
- Use fixtures for common test data
- Keep tests focused on single functionality

### Test Data
- Use realistic test data that reflects real-world usage
- Include edge cases and error conditions
- Use fixtures for reusable test data

### Test Assertions
- Use specific assertions that test the exact behavior
- Include error message testing for exception cases
- Test both positive and negative scenarios

## Continuous Integration

The test suite is designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Enrichment Tests
  run: |
    python tests/enrichment_test/run_tests.py
    pytest tests/enrichment_test/ --cov=enrichment --cov-report=xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in Python path
2. **Mock Issues**: Check that mocks are properly configured
3. **Async Issues**: Use `@pytest.mark.asyncio` for async tests
4. **Coverage Issues**: Ensure all code paths are tested

### Debug Tips

```bash
# Run tests with debug output
pytest tests/enrichment_test/ -v -s

# Run specific test with debug
pytest tests/enrichment_test/test_models.py::TestChunkEnrichment::test_init_default -v -s

# Run tests with pdb debugger
pytest tests/enrichment_test/ --pdb
```

## Contributing

When adding new tests:

1. Follow the existing test structure and patterns
2. Add appropriate fixtures for new test data
3. Include both positive and negative test cases
4. Ensure tests are deterministic and don't depend on external state
5. Update this README if adding new test categories

## Test Maintenance

- **Regular Updates**: Update tests when adding new features
- **Refactoring**: Update tests when refactoring code
- **Dependencies**: Keep test dependencies up to date
- **Coverage**: Monitor test coverage and add tests for uncovered code
