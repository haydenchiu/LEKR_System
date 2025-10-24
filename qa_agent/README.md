# QA Agent Module

The QA Agent module provides intelligent question answering capabilities for the LERK System, integrating with the retriever module to find relevant documents and using LangGraph for multi-agent orchestration to provide comprehensive answers.

## Features

- **Intelligent Question Answering**: Ask questions and get comprehensive, well-sourced answers
- **Conversation Context**: Maintain conversation history and context across multiple questions
- **Multiple Answer Styles**: Support for concise, detailed, academic, and conversational answer styles
- **Session Management**: Track and manage user sessions with automatic cleanup
- **Source Citation**: Automatically extract and cite sources from retrieved documents
- **Configurable Quality**: Multiple predefined configurations for different use cases
- **Async Support**: Full asynchronous support for high-performance applications
- **Error Handling**: Comprehensive error handling and validation

## Architecture

The QA Agent module consists of several key components:

```
QAOrchestrator
├── SupervisorAgent
│   ├── RetrievalAgent
│   │   ├── SemanticRetriever
│   │   ├── HybridRetriever
│   │   └── ContextRetriever
│   └── Answer Generation (LLM)
├── Session Management
├── Context Memory
└── Response Formatting
```

## Quick Start

### Basic Usage

```python
from qa_agent import QAOrchestrator, DEFAULT_QA_CONFIG

# Initialize the QA system
qa_system = QAOrchestrator(config=DEFAULT_QA_CONFIG)

# Ask a question
response = qa_system.ask_question("What is the attention mechanism?")
print(response['answer'])
```

### Async Usage

```python
import asyncio
from qa_agent import QAOrchestrator

async def main():
    qa_system = QAOrchestrator()
    
    response = await qa_system.aask_question("How do transformers work?")
    print(response['answer'])

asyncio.run(main())
```

### Conversational QA

```python
# Create a session for conversation context
session_id = "user_123"

# First question
response1 = qa_system.ask_question("What is machine learning?", session_id=session_id)

# Follow-up question (uses context)
response2 = qa_system.ask_question("How does it relate to deep learning?", session_id=session_id)

# Get conversation history
history = qa_system.get_conversation_history(session_id)
```

## Configuration

### Predefined Configurations

The module provides several predefined configurations for different use cases:

```python
from qa_agent import (
    DEFAULT_QA_CONFIG,      # Balanced settings
    FAST_QA_CONFIG,         # Optimized for speed
    HIGH_QUALITY_QA_CONFIG, # Maximum accuracy
    ACADEMIC_QA_CONFIG,     # Academic writing style
    CONVERSATIONAL_QA_CONFIG # Conversational style
)

# Use a specific configuration
qa_system = QAOrchestrator(config=HIGH_QUALITY_QA_CONFIG)
```

### Custom Configuration

```python
from qa_agent import QAConfig

# Create custom configuration
custom_config = QAConfig(
    llm_model="gpt-4o",
    max_retrieved_docs=8,
    answer_style="academic",
    include_sources=True,
    session_timeout=7200,
    enable_context_memory=True
)

qa_system = QAOrchestrator(config=custom_config)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_model` | str | "gpt-4o" | LLM model for answer generation |
| `llm_temperature` | float | 0.0 | Temperature for LLM generation |
| `max_retrieved_docs` | int | 5 | Maximum documents to retrieve |
| `similarity_threshold` | float | 0.7 | Minimum similarity threshold |
| `max_answer_length` | int | 2000 | Maximum answer length |
| `answer_style` | str | "detailed" | Answer style (concise/detailed/academic) |
| `include_sources` | bool | True | Whether to include source citations |
| `session_timeout` | int | 3600 | Session timeout in seconds |
| `enable_context_memory` | bool | True | Enable conversation context |

## API Reference

### QAOrchestrator

Main interface for question answering.

#### Methods

- `ask_question(question, session_id=None, context=None)`: Ask a question synchronously
- `aask_question(question, session_id=None, context=None)`: Ask a question asynchronously
- `get_conversation_history(session_id)`: Get conversation history for a session
- `clear_session(session_id)`: Clear conversation history for a session
- `get_session_stats(session_id)`: Get statistics for a session
- `get_system_stats()`: Get overall system statistics
- `update_config(**kwargs)`: Update configuration

#### Response Format

```python
{
    "answer": "The generated answer text",
    "sources": [
        {
            "type": "citation",
            "text": "Source reference",
            "pattern": "citation_pattern"
        }
    ],
    "session_id": "session_uuid",
    "timestamp": "2024-01-01T12:00:00",
    "question": "Original question",
    "context_used": True,
    "message_count": 5
}
```

### RetrievalAgent

Agent responsible for document retrieval.

#### Methods

- `retrieve_documents(question, **kwargs)`: Retrieve documents synchronously
- `aretrieve_documents(question, **kwargs)`: Retrieve documents asynchronously
- `get_retrieval_stats()`: Get retrieval statistics
- `update_config(**kwargs)`: Update configuration

### SupervisorAgent

Agent that orchestrates the QA workflow.

#### Methods

- `ask_question(question, session_id=None)`: Ask a question synchronously
- `aask_question(question, session_id=None)`: Ask a question asynchronously
- `get_conversation_history(session_id)`: Get conversation history
- `clear_session(session_id)`: Clear session
- `get_agent_stats()`: Get agent statistics

## Advanced Usage

### Custom Retriever Configuration

```python
from qa_agent import QAConfig
from retriever import RetrieverConfig

# Configure the retriever
retriever_config = RetrieverConfig(
    search_strategy="hybrid",
    k=10,
    similarity_threshold=0.6
)

# Use with QA system
qa_config = QAConfig(
    retriever_config=retriever_config.to_dict(),
    max_retrieved_docs=8
)

qa_system = QAOrchestrator(config=qa_config)
```

### Session Management

```python
# Create a new session
session_id = qa_system.create_session()

# Use the session for multiple questions
response1 = qa_system.ask_question("Question 1", session_id=session_id)
response2 = qa_system.ask_question("Question 2", session_id=session_id)

# Get session statistics
stats = qa_system.get_session_stats(session_id)
print(f"Messages in session: {stats['message_count']}")

# Clear the session when done
qa_system.clear_session(session_id)
```

### Error Handling

```python
from qa_agent import QAOrchestrator, ValidationError, QAError

qa_system = QAOrchestrator()

try:
    response = qa_system.ask_question("What is AI?")
    print(response['answer'])
except ValidationError as e:
    print(f"Invalid question: {e}")
except QAError as e:
    print(f"QA system error: {e}")
```

### Monitoring and Statistics

```python
# Get system statistics
stats = qa_system.get_system_stats()
print(f"Active sessions: {stats['active_sessions']}")
print(f"Total sessions: {stats['total_sessions_created']}")

# Get agent statistics
agent_stats = qa_system.supervisor_agent.get_agent_stats()
print(f"LLM model: {agent_stats['llm_model']}")
print(f"Answer style: {agent_stats['answer_style']}")
```

## Integration with LERK System

The QA Agent module integrates seamlessly with other LERK System components:

- **Retriever Module**: Uses semantic, hybrid, and context-aware retrieval
- **Consolidation Module**: Can work with consolidated knowledge
- **Clustering Module**: Can leverage document clusters for better retrieval

## Performance Considerations

- **Async Support**: Use async methods for better performance in concurrent applications
- **Session Management**: Regular cleanup of expired sessions to manage memory
- **Configuration Tuning**: Adjust similarity thresholds and document limits based on your use case
- **Caching**: Consider implementing caching for frequently asked questions

## Troubleshooting

### Common Issues

1. **No relevant documents found**: Lower the similarity threshold or increase max_retrieved_docs
2. **Answers too short**: Adjust answer_style to "detailed" or increase max_answer_length
3. **Session timeout**: Increase session_timeout or implement session renewal
4. **Memory issues**: Enable automatic session cleanup and monitor active sessions

### Debugging

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or configure the QA system with debug logging
config = QAConfig(log_level="DEBUG")
qa_system = QAOrchestrator(config=config)
```

## Examples

See `example.py` for comprehensive examples of:
- Basic question answering
- Conversational QA with session management
- Different configuration presets
- Batch question processing
- Error handling
- System monitoring

## Dependencies

- `langchain`: Core LangChain functionality
- `langchain-openai`: OpenAI integration
- `langgraph-supervisor`: Multi-agent orchestration
- `langgraph`: Graph-based agent workflows
- `pydantic`: Data validation and configuration
- `retriever`: LERK System retriever module

## License

This module is part of the LERK System and follows the same licensing terms.
