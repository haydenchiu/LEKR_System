# AgentChat Integration for LERK System

This module provides seamless integration between [AgentChat](https://agentchat.vercel.app/) and the LERK System's QA agent capabilities, enabling conversational interfaces for sophisticated knowledge retrieval and question answering.

## Features

- **🤖 Conversational Interface**: Natural language interaction with the LERK System
- **🔍 Intelligent Query Classification**: Automatically detects query types (subject-level, document-level, chunk-level, discovery, system)
- **📚 Multi-Level Knowledge Retrieval**: Leverages the full power of LERK's subject-level retrieval with fallback mechanisms
- **💬 Session Management**: Maintains conversation context and history
- **🎯 Confidence Scoring**: Provides confidence scores for responses
- **📖 Source Attribution**: Includes sources and metadata with responses
- **🔄 Fallback Handling**: Gracefully handles cases where subject knowledge is insufficient

## Architecture

The AgentChat integration sits on top of the LERK System's clean module architecture:

```
AgentChat Interface
        ↓
AgentChatLERKIntegration
        ↓
LERK Integration Manager
        ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   QA Agent      │   Retriever     │  Consolidation  │
│   Module        │   Module        │   Module        │
└─────────────────┴─────────────────┴─────────────────┘
```

## Quick Start

### Basic Integration

```python
from integration import AgentChatLERKIntegration, create_agentchat_integration
from qa_agent import QAOrchestrator
from integration import LERKIntegrationManager

# Initialize LERK System components
qa_orchestrator = QAOrchestrator()
integration_manager = LERKIntegrationManager()

# Create AgentChat integration
agentchat_integration = create_agentchat_integration(
    qa_orchestrator=qa_orchestrator,
    integration_manager=integration_manager
)

# Process a chat message
response = await agentchat_integration.process_chat_message(
    message="What are the main concepts in machine learning?",
    session_id="user_session_001",
    user_context={"user_type": "researcher"}
)

print(f"Assistant: {response['message']}")
print(f"Confidence: {response['metadata']['confidence']}")
print(f"Sources: {response['metadata']['sources_count']}")
```

### Convenience Function

```python
from integration import process_chat_message_with_lerk

# Direct processing without creating integration instance
response = await process_chat_message_with_lerk(
    message="What topics are available?",
    session_id="quick_session",
    qa_orchestrator=qa_orchestrator,
    integration_manager=integration_manager
)
```

## Query Types and Processing

### 🔍 Discovery Queries
**Examples**: "What topics are available?", "Show me the clusters", "What documents are in the system?"

**Processing**: Uses `ClusterDiscoveryService` and `DocumentDiscoveryService` to provide system overview.

**Response**: Lists available clusters, documents, and system statistics.

### 📚 Subject-Level Queries
**Examples**: "What are the main concepts in AI?", "Explain machine learning", "Tell me about neural networks"

**Processing**: Uses `MultiLevelSearchOrchestrator` with subject-level preference and fallback mechanisms.

**Response**: Comprehensive answers from consolidated subject knowledge, with fallback to document/chunk-level information if needed.

### 📄 Document-Level Queries
**Examples**: "What does document X say about Y?", "Show me information from the research paper"

**Processing**: Uses document-level search through the multi-level orchestrator.

**Response**: Information organized by document with source attribution.

### 🔎 Chunk-Level Queries
**Examples**: "How does neural network training work?", "What is backpropagation?"

**Processing**: Uses chunk-level search for specific, detailed information.

**Response**: Specific answers with detailed source information.

### 🛠️ System Queries
**Examples**: "Help", "What can you do?", "How do I use this system?"

**Processing**: Provides system capabilities and usage information.

**Response**: Help text and system information.

## Configuration

### AgentChat Configuration

```python
agentchat_config = {
    "max_conversation_length": 50,        # Maximum conversation history length
    "context_window_size": 10,            # Context window for multi-turn conversations
    "enable_multi_turn": True,            # Enable multi-turn conversation support
    "enable_subject_fallback": True,      # Enable fallback to chunk-level when subject knowledge insufficient
    "enable_discovery_mode": True,        # Enable discovery query processing
    "response_format": "conversational",  # Response format style
    "include_sources": True,              # Include sources in responses
    "include_confidence": True           # Include confidence scores in responses
}

integration = AgentChatLERKIntegration(
    agentchat_config=agentchat_config
)
```

## Session Management

### Session Lifecycle

```python
# Sessions are automatically created when first message is processed
response = await integration.process_chat_message(
    message="Hello!",
    session_id="new_session",
    user_context={"user_type": "student"}
)

# Get session information
session_info = integration.get_session_info("new_session")
print(f"Session created: {session_info['created_at']}")
print(f"Messages processed: {session_info['query_count']}")
print(f"Conversation length: {session_info['conversation_length']}")

# Get all active sessions
active_sessions = integration.get_active_sessions()
print(f"Active sessions: {len(active_sessions)}")
```

### Conversation Context

The integration maintains conversation history and uses it for context-aware responses:

```python
# First message
response1 = await integration.process_chat_message(
    message="Tell me about machine learning",
    session_id="context_session"
)

# Follow-up message (uses context)
response2 = await integration.process_chat_message(
    message="What about deep learning?",
    session_id="context_session"  # Same session ID
)
# The system understands "deep learning" in the context of the previous ML discussion
```

## Response Format

### Standard Response Structure

```python
{
    "message": "Machine learning is a subset of artificial intelligence...",
    "type": "assistant",
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {
        "query_type": "subject_level",
        "confidence": 0.92,
        "sources_count": 3,
        "fallback_used": False,
        "lerk_system": True,
        "session_id": "user_session_001"
    },
    "sources": [
        {
            "content": "Machine learning algorithms can be supervised or unsupervised...",
            "metadata": {
                "document_id": "doc_001",
                "document_title": "Introduction to ML",
                "similarity_score": 0.95,
                "knowledge_type": "subject"
            }
        }
    ],
    "confidence_score": 0.92
}
```

### Error Response Structure

```python
{
    "message": "I encountered an error processing your request. Please try rephrasing your question.",
    "type": "assistant",
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {
        "error": True,
        "error_message": "Specific error details"
    }
}
```

## Advanced Features

### Custom Query Classification

```python
# Override default query classification
class CustomAgentChatIntegration(AgentChatLERKIntegration):
    async def _classify_query(self, message: str) -> str:
        # Custom classification logic
        if "custom_keyword" in message.lower():
            return "custom_type"
        return await super()._classify_query(message)
```

### Custom Response Formatting

```python
# Override response formatting
class CustomAgentChatIntegration(AgentChatLERKIntegration):
    def _format_response_for_agentchat(self, response: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        # Custom formatting logic
        formatted = super()._format_response_for_agentchat(response, query_type)
        formatted["custom_field"] = "custom_value"
        return formatted
```

## Integration with AgentChat Platform

### Webhook Integration

```python
# Example webhook handler for AgentChat
async def agentchat_webhook_handler(request):
    data = await request.json()
    
    message = data.get("message", "")
    session_id = data.get("session_id", "default")
    user_context = data.get("user_context", {})
    
    # Process through LERK System
    response = await process_chat_message_with_lerk(
        message=message,
        session_id=session_id,
        user_context=user_context
    )
    
    return {
        "response": response["message"],
        "metadata": response["metadata"],
        "sources": response.get("sources", [])
    }
```

### API Integration

```python
# Example API endpoint
from fastapi import FastAPI
from integration import AgentChatLERKIntegration

app = FastAPI()
integration = AgentChatLERKIntegration()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = await integration.process_chat_message(
        message=request.message,
        session_id=request.session_id,
        user_context=request.user_context
    )
    return response
```

## Error Handling

The integration includes comprehensive error handling:

- **Import Errors**: Graceful handling when LERK modules are not available
- **Processing Errors**: Fallback responses when query processing fails
- **Session Errors**: Error handling for session management issues
- **Classification Errors**: Fallback to chunk-level processing when classification fails

## Performance Considerations

- **Session Memory**: Conversation history is automatically trimmed to prevent memory issues
- **Async Processing**: All operations are asynchronous for better performance
- **Caching**: Integration leverages LERK System's built-in caching mechanisms
- **Connection Pooling**: Reuses LERK System connections for efficiency

## Testing

```python
# Run integration tests
pytest tests/integration_test/test_agentchat_integration.py

# Run example demonstrations
python integration/agentchat_example.py
```

## Dependencies

- **LERK System**: All core modules (qa_agent, retriever, consolidation, clustering)
- **AgentChat**: External platform integration
- **AsyncIO**: For asynchronous processing
- **Logging**: For comprehensive logging

## Examples

See `integration/agentchat_example.py` for comprehensive examples including:
- Basic integration setup
- Query classification demonstration
- Response formatting examples
- Session management
- Multi-turn conversations
- Error handling scenarios

## Support

For issues related to AgentChat integration:
1. Check the LERK System logs for detailed error information
2. Verify all LERK System modules are properly installed
3. Ensure AgentChat platform connectivity
4. Review session management and conversation history

The AgentChat integration provides a powerful, conversational interface to the LERK System's sophisticated knowledge retrieval capabilities, making complex knowledge queries accessible through natural language interaction.
