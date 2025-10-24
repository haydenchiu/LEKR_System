"""
Example usage of the QA agent module.

This module demonstrates how to use the QA orchestrator to ask questions
and get comprehensive answers from the LERK System knowledge base.
"""

import asyncio
import logging
from pathlib import Path
import os

# Ensure the project root is in the Python path for imports
if Path(__file__).parent.parent not in [Path(p) for p in map(Path, os.sys.path)]:
    os.sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_agent import (
    QAOrchestrator, 
    DEFAULT_QA_CONFIG, 
    FAST_QA_CONFIG, 
    HIGH_QUALITY_QA_CONFIG,
    ACADEMIC_QA_CONFIG,
    CONVERSATIONAL_QA_CONFIG
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_basic_qa():
    """Demonstrate basic question answering functionality."""
    logger.info("\n--- Basic QA Demonstration ---")
    
    # Initialize QA orchestrator with default config
    qa_system = QAOrchestrator(config=DEFAULT_QA_CONFIG)
    
    # Ask a question
    question = "What is the attention mechanism in transformers?"
    logger.info(f"Question: {question}")
    
    try:
        response = await qa_system.aask_question(question)
        logger.info(f"Answer: {response['answer']}")
        logger.info(f"Sources: {response['sources']}")
        logger.info(f"Session ID: {response['session_id']}")
    except Exception as e:
        logger.error(f"Error during QA: {e}")


async def demonstrate_conversational_qa():
    """Demonstrate conversational QA with session management."""
    logger.info("\n--- Conversational QA Demonstration ---")
    
    # Initialize with conversational config
    qa_system = QAOrchestrator(config=CONVERSATIONAL_QA_CONFIG)
    
    # Create a session
    session_id = "demo_session_1"
    
    # First question
    question1 = "What is machine learning?"
    logger.info(f"Question 1: {question1}")
    
    try:
        response1 = await qa_system.aask_question(question1, session_id=session_id)
        logger.info(f"Answer 1: {response1['answer']}")
        
        # Follow-up question (should use context)
        question2 = "How does it relate to deep learning?"
        logger.info(f"Question 2: {question2}")
        
        response2 = await qa_system.aask_question(question2, session_id=session_id)
        logger.info(f"Answer 2: {response2['answer']}")
        
        # Get conversation history
        history = qa_system.get_conversation_history(session_id)
        logger.info(f"Conversation history: {len(history)} messages")
        
        # Get session stats
        stats = qa_system.get_session_stats(session_id)
        logger.info(f"Session stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error during conversational QA: {e}")


async def demonstrate_different_configs():
    """Demonstrate different QA configurations."""
    logger.info("\n--- Different Configurations Demonstration ---")
    
    question = "Explain the transformer architecture"
    
    configs = [
        ("Fast", FAST_QA_CONFIG),
        ("High Quality", HIGH_QUALITY_QA_CONFIG),
        ("Academic", ACADEMIC_QA_CONFIG),
        ("Conversational", CONVERSATIONAL_QA_CONFIG)
    ]
    
    for config_name, config in configs:
        logger.info(f"\n--- {config_name} Configuration ---")
        qa_system = QAOrchestrator(config=config)
        
        try:
            response = await qa_system.aask_question(question)
            logger.info(f"Answer ({config_name}): {response['answer'][:200]}...")
            logger.info(f"Answer length: {len(response['answer'])}")
            logger.info(f"Sources: {len(response['sources'])}")
        except Exception as e:
            logger.error(f"Error with {config_name} config: {e}")


async def demonstrate_batch_questions():
    """Demonstrate asking multiple questions."""
    logger.info("\n--- Batch Questions Demonstration ---")
    
    qa_system = QAOrchestrator(config=DEFAULT_QA_CONFIG)
    
    questions = [
        "What is the attention mechanism?",
        "How do transformers work?",
        "What are the advantages of transformers over RNNs?",
        "What is BERT and how does it work?",
        "Explain the encoder-decoder architecture"
    ]
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\nQuestion {i}: {question}")
        try:
            response = await qa_system.aask_question(question)
            logger.info(f"Answer {i}: {response['answer'][:150]}...")
        except Exception as e:
            logger.error(f"Error with question {i}: {e}")


async def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    logger.info("\n--- Error Handling Demonstration ---")
    
    qa_system = QAOrchestrator(config=DEFAULT_QA_CONFIG)
    
    # Test invalid questions
    invalid_questions = [
        "",  # Empty question
        "a",  # Too short
        "What is the meaning of life?" * 100,  # Too long
    ]
    
    for question in invalid_questions:
        try:
            response = await qa_system.aask_question(question)
            logger.info(f"Unexpected success for invalid question: {question[:50]}")
        except Exception as e:
            logger.info(f"Correctly caught error for invalid question: {type(e).__name__}: {e}")


async def demonstrate_system_stats():
    """Demonstrate system statistics and monitoring."""
    logger.info("\n--- System Statistics Demonstration ---")
    
    qa_system = QAOrchestrator(config=DEFAULT_QA_CONFIG)
    
    # Ask a few questions to generate some activity
    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?"
    ]
    
    for question in questions:
        try:
            await qa_system.aask_question(question)
        except Exception as e:
            logger.error(f"Error asking question: {e}")
    
    # Get system statistics
    stats = qa_system.get_system_stats()
    logger.info(f"System stats: {stats}")


async def main():
    """Main demonstration function."""
    logger.info("Starting QA Agent Module Examples...")
    
    try:
        await demonstrate_basic_qa()
        await demonstrate_conversational_qa()
        await demonstrate_different_configs()
        await demonstrate_batch_questions()
        await demonstrate_error_handling()
        await demonstrate_system_stats()
        
        logger.info("\n✅ All QA Agent examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())
