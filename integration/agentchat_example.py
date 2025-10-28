"""
AgentChat Integration Example for LERK System

This example demonstrates how to integrate AgentChat with the LERK System's
QA agent capabilities, providing a conversational interface for knowledge retrieval.
"""

import asyncio
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_agentchat_integration():
    """Demonstrate AgentChat integration with LERK System."""
    print("🤖 AgentChat-LERK System Integration Demonstration")
    print("=" * 60)
    
    try:
        # Import the integration components
        from integration.agentchat_integration import (
            AgentChatLERKIntegration,
            create_agentchat_integration,
            process_chat_message_with_lerk
        )
        
        # Import LERK System components
        from qa_agent import QAOrchestrator
        from integration import LERKIntegrationManager
        
        print("\n🔧 Step 1: Initialize LERK System Components")
        print("-" * 50)
        
        # Create QA orchestrator
        qa_orchestrator = QAOrchestrator()
        print("✅ QA Orchestrator initialized")
        
        # Create integration manager
        integration_manager = LERKIntegrationManager()
        print("✅ Integration Manager initialized")
        
        print("\n🤖 Step 2: Create AgentChat Integration")
        print("-" * 50)
        
        # Create AgentChat integration with custom configuration
        agentchat_config = {
            "max_conversation_length": 30,
            "context_window_size": 5,
            "enable_multi_turn": True,
            "enable_subject_fallback": True,
            "enable_discovery_mode": True,
            "response_format": "conversational",
            "include_sources": True,
            "include_confidence": True
        }
        
        agentchat_integration = create_agentchat_integration(
            qa_orchestrator=qa_orchestrator,
            integration_manager=integration_manager,
            agentchat_config=agentchat_config
        )
        
        print("✅ AgentChat Integration created")
        print(f"📊 Configuration: {agentchat_config}")
        
        print("\n💬 Step 3: Simulate Chat Conversations")
        print("-" * 50)
        
        # Simulate different types of conversations
        conversations = [
            {
                "session_id": "demo_session_001",
                "messages": [
                    "Hello! What can you help me with?",
                    "What topics are available in the knowledge base?",
                    "What are the main concepts in machine learning?",
                    "Can you tell me more about neural networks?",
                    "What does document X say about deep learning?"
                ]
            },
            {
                "session_id": "demo_session_002", 
                "messages": [
                    "Help me understand artificial intelligence",
                    "Show me the available clusters",
                    "How does supervised learning work?"
                ]
            }
        ]
        
        for conversation in conversations:
            session_id = conversation["session_id"]
            print(f"\n🗣️  Session: {session_id}")
            print("-" * 30)
            
            for i, message in enumerate(conversation["messages"], 1):
                print(f"\n👤 User: {message}")
                
                # Process message through AgentChat integration
                response = await agentchat_integration.process_chat_message(
                    message=message,
                    session_id=session_id,
                    user_context={"demo": True, "user_type": "researcher"}
                )
                
                print(f"🤖 Assistant: {response['message']}")
                
                # Show metadata if available
                if response.get("metadata"):
                    metadata = response["metadata"]
                    print(f"   📊 Query Type: {metadata.get('query_type', 'unknown')}")
                    print(f"   🎯 Confidence: {metadata.get('confidence', 0.0):.2f}")
                    if metadata.get("sources_count", 0) > 0:
                        print(f"   📚 Sources: {metadata['sources_count']} found")
                    if metadata.get("fallback_used"):
                        print(f"   ⚠️  Fallback used: {metadata.get('fallback_reason', 'unknown')}")
                
                # Show sources if available
                if response.get("sources"):
                    print(f"   📖 Sample Sources:")
                    for j, source in enumerate(response["sources"][:2], 1):
                        doc_title = source["metadata"].get("document_title", "Unknown")
                        similarity = source["metadata"].get("similarity_score", 0.0)
                        print(f"      {j}. {doc_title} (similarity: {similarity:.2f})")
        
        print("\n📊 Step 4: Session Management Demo")
        print("-" * 50)
        
        # Show session information
        for conversation in conversations:
            session_id = conversation["session_id"]
            session_info = agentchat_integration.get_session_info(session_id)
            
            print(f"\n📋 Session {session_id}:")
            print(f"   🕒 Created: {session_info.get('created_at', 'unknown')}")
            print(f"   💬 Messages: {session_info.get('query_count', 0)}")
            print(f"   📏 Conversation Length: {session_info.get('conversation_length', 0)}")
            print(f"   🕐 Last Activity: {session_info.get('last_activity', 'unknown')}")
        
        # Show active sessions
        active_sessions = agentchat_integration.get_active_sessions()
        print(f"\n🟢 Active Sessions: {len(active_sessions)}")
        for session_id in active_sessions:
            print(f"   - {session_id}")
        
        print("\n🚀 Step 5: Convenience Function Demo")
        print("-" * 50)
        
        # Demonstrate convenience function
        quick_response = await process_chat_message_with_lerk(
            message="What is the LERK System?",
            session_id="quick_demo",
            qa_orchestrator=qa_orchestrator,
            integration_manager=integration_manager,
            user_context={"quick_demo": True}
        )
        
        print(f"👤 Quick Question: What is the LERK System?")
        print(f"🤖 Quick Answer: {quick_response['message']}")
        
        print("\n✅ AgentChat Integration Demonstration Completed Successfully!")
        print("=" * 60)
        
        return {
            "success": True,
            "sessions_processed": len(conversations),
            "total_messages": sum(len(conv["messages"]) for conv in conversations),
            "active_sessions": len(active_sessions)
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all LERK System modules are properly installed")
        return None
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.error(f"AgentChat integration demonstration failed: {e}")
        return None


async def demonstrate_query_classification():
    """Demonstrate query classification capabilities."""
    print("\n🔍 Query Classification Demonstration")
    print("=" * 50)
    
    try:
        from integration.agentchat_integration import AgentChatLERKIntegration
        
        integration = AgentChatLERKIntegration()
        
        test_queries = [
            ("What topics are available?", "discovery"),
            ("Show me the clusters", "discovery"),
            ("What are the main concepts in AI?", "subject_level"),
            ("Explain machine learning", "subject_level"),
            ("What does document X say about Y?", "document_level"),
            ("How does neural network training work?", "chunk_level"),
            ("Help me understand this", "system"),
            ("What can you do?", "system")
        ]
        
        print("Testing query classification:")
        print("-" * 30)
        
        for query, expected_type in test_queries:
            classified_type = await integration._classify_query(query)
            status = "✅" if classified_type == expected_type else "❌"
            print(f"{status} '{query}' -> {classified_type} (expected: {expected_type})")
        
        print("\n✅ Query Classification Demo Completed!")
        
    except Exception as e:
        print(f"❌ Query classification demo failed: {e}")


async def demonstrate_response_formatting():
    """Demonstrate response formatting for AgentChat."""
    print("\n📝 Response Formatting Demonstration")
    print("=" * 50)
    
    try:
        from integration.agentchat_integration import AgentChatLERKIntegration
        
        integration = AgentChatLERKIntegration()
        
        # Mock response data
        mock_response = {
            "answer": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
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
            "confidence": 0.92,
            "metadata": {
                "query_type": "subject_level",
                "result_count": 3,
                "fallback_used": False
            }
        }
        
        # Format for AgentChat
        formatted = integration._format_response_for_agentchat(mock_response, "subject_level")
        
        print("Original Response:")
        print(f"  Answer: {mock_response['answer']}")
        print(f"  Confidence: {mock_response['confidence']}")
        print(f"  Sources: {len(mock_response['sources'])}")
        
        print("\nFormatted for AgentChat:")
        print(f"  Message: {formatted['message']}")
        print(f"  Type: {formatted['type']}")
        print(f"  Confidence Score: {formatted.get('confidence_score', 'N/A')}")
        print(f"  Sources Count: {formatted['metadata'].get('sources_count', 'N/A')}")
        print(f"  LERK System: {formatted['metadata'].get('lerk_system', False)}")
        
        print("\n✅ Response Formatting Demo Completed!")
        
    except Exception as e:
        print(f"❌ Response formatting demo failed: {e}")


if __name__ == "__main__":
    print("Starting AgentChat-LERK System Integration Examples...")
    
    # Run the demonstrations
    asyncio.run(demonstrate_agentchat_integration())
    asyncio.run(demonstrate_query_classification())
    asyncio.run(demonstrate_response_formatting())
    
    print("\n🎉 All AgentChat integration examples completed!")
