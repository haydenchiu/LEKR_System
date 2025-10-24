"""
Example usage of the retriever module.

This module demonstrates how to use the different retriever types
for various retrieval scenarios in the LERK System.
"""

import asyncio
import logging
from typing import List, Dict, Any

# Import retriever components
from retriever import (
    SemanticRetriever, HybridRetriever, ContextRetriever,
    DEFAULT_RETRIEVER_CONFIG, FAST_RETRIEVER_CONFIG, HIGH_QUALITY_RETRIEVER_CONFIG,
    format_retrieval_results, calculate_relevance_score, filter_by_metadata
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_semantic_retrieval():
    """Demonstrate semantic retrieval functionality."""
    logger.info("=== Semantic Retrieval Demo ===")
    
    try:
        # Initialize semantic retriever
        retriever = SemanticRetriever(config=FAST_RETRIEVER_CONFIG)
        
        # Example queries
        queries = [
            "machine learning algorithms",
            "neural network architectures", 
            "natural language processing"
        ]
        
        for query in queries:
            logger.info(f"Query: {query}")
            
            # Retrieve documents
            results = retriever.get_relevant_documents(query, limit=3)
            
            # Format and display results
            formatted_results = format_retrieval_results(results, "standard")
            for i, result in enumerate(formatted_results, 1):
                logger.info(f"  {i}. {result['title']} (Score: {result['similarity_score']:.3f})")
            
            logger.info("")
        
    except Exception as e:
        logger.error(f"Semantic retrieval demo failed: {e}")


def demonstrate_hybrid_retrieval():
    """Demonstrate hybrid retrieval functionality."""
    logger.info("=== Hybrid Retrieval Demo ===")
    
    try:
        # Initialize hybrid retriever
        retriever = HybridRetriever(config=HIGH_QUALITY_RETRIEVER_CONFIG)
        
        # Example query
        query = "deep learning for computer vision"
        logger.info(f"Query: {query}")
        
        # Retrieve with different weight combinations
        weight_combinations = [
            (0.8, 0.2),  # Semantic-heavy
            (0.5, 0.5),  # Balanced
            (0.2, 0.8),  # Keyword-heavy
        ]
        
        for semantic_weight, keyword_weight in weight_combinations:
            logger.info(f"Weights: Semantic={semantic_weight}, Keyword={keyword_weight}")
            
            results = retriever.search_with_weights(
                query=query,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                limit=3
            )
            
            formatted_results = format_retrieval_results(results, "minimal")
            for i, result in enumerate(formatted_results, 1):
                logger.info(f"  {i}. {result['title']} (Score: {result['score']:.3f})")
            
            logger.info("")
        
    except Exception as e:
        logger.error(f"Hybrid retrieval demo failed: {e}")


def demonstrate_context_aware_retrieval():
    """Demonstrate context-aware retrieval functionality."""
    logger.info("=== Context-Aware Retrieval Demo ===")
    
    try:
        # Initialize context-aware retriever
        retriever = ContextRetriever(config=HIGH_QUALITY_RETRIEVER_CONFIG)
        
        # Start a session
        session_id = retriever.start_session("demo_session")
        logger.info(f"Started session: {session_id}")
        
        # Simulate a conversation
        conversation_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "What are neural networks?",
            "Explain backpropagation"
        ]
        
        for i, query in enumerate(conversation_queries, 1):
            logger.info(f"Turn {i}: {query}")
            
            # Retrieve documents
            results = retriever.get_relevant_documents(query, limit=2)
            
            # Add to context
            retriever.add_to_context(query, results)
            
            # Display results
            formatted_results = format_retrieval_results(results, "minimal")
            for j, result in enumerate(formatted_results, 1):
                logger.info(f"  {j}. {result['title']} (Score: {result['score']:.3f})")
            
            # Show context summary
            context_summary = retriever.get_context_summary()
            logger.info(f"  Context: {context_summary['context_entries']} entries")
            logger.info("")
        
        # Demonstrate feedback integration
        logger.info("=== Feedback Integration Demo ===")
        
        feedback = {
            "preferred_categories": ["AI", "Machine Learning"],
            "liked_documents": ["doc_1", "doc_3"],
            "disliked_documents": ["doc_2"]
        }
        
        retriever.update_user_preferences(feedback)
        
        # Retrieve with feedback
        final_query = "What are the latest advances in AI?"
        results = retriever.retrieve_with_feedback(
            query=final_query,
            feedback=feedback,
            limit=3
        )
        
        logger.info(f"Final query: {final_query}")
        formatted_results = format_retrieval_results(results, "minimal")
        for i, result in enumerate(formatted_results, 1):
            logger.info(f"  {i}. {result['title']} (Score: {result['score']:.3f})")
        
    except Exception as e:
        logger.error(f"Context-aware retrieval demo failed: {e}")


def demonstrate_batch_retrieval():
    """Demonstrate batch retrieval functionality."""
    logger.info("=== Batch Retrieval Demo ===")
    
    try:
        # Initialize retriever
        retriever = SemanticRetriever(config=FAST_RETRIEVER_CONFIG)
        
        # Batch queries
        queries = [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural networks",
            "natural language processing"
        ]
        
        logger.info(f"Processing {len(queries)} queries in batch...")
        
        # Perform batch retrieval
        batch_results = retriever.batch_retrieve(queries, limit=2)
        
        # Display results
        for i, (query, results) in enumerate(zip(queries, batch_results), 1):
            logger.info(f"{i}. Query: {query}")
            formatted_results = format_retrieval_results(results, "minimal")
            for j, result in enumerate(formatted_results, 1):
                logger.info(f"   {j}. {result['title']} (Score: {result['score']:.3f})")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Batch retrieval demo failed: {e}")


def demonstrate_metadata_filtering():
    """Demonstrate metadata filtering functionality."""
    logger.info("=== Metadata Filtering Demo ===")
    
    try:
        # Initialize retriever
        retriever = SemanticRetriever(config=DEFAULT_RETRIEVER_CONFIG)
        
        # Example query
        query = "machine learning"
        logger.info(f"Query: {query}")
        
        # Different filter scenarios
        filter_scenarios = [
            {"category": "AI"},  # Filter by category
            {"quality_score": {"min": 0.8}},  # High quality only
            {"subject": "Computer Science"},  # Specific subject
            {"category": "AI", "quality_score": {"min": 0.7}}  # Combined filters
        ]
        
        for i, filters in enumerate(filter_scenarios, 1):
            logger.info(f"Scenario {i}: Filters = {filters}")
            
            results = retriever.get_relevant_documents(query, filters=filters, limit=2)
            
            formatted_results = format_retrieval_results(results, "minimal")
            for j, result in enumerate(formatted_results, 1):
                logger.info(f"  {j}. {result['title']} (Score: {result['score']:.3f})")
            
            logger.info("")
        
    except Exception as e:
        logger.error(f"Metadata filtering demo failed: {e}")


def demonstrate_retriever_statistics():
    """Demonstrate retriever statistics and monitoring."""
    logger.info("=== Retriever Statistics Demo ===")
    
    try:
        # Initialize different retriever types
        retrievers = {
            "Semantic": SemanticRetriever(config=FAST_RETRIEVER_CONFIG),
            "Hybrid": HybridRetriever(config=HIGH_QUALITY_RETRIEVER_CONFIG),
            "Context-Aware": ContextRetriever(config=HIGH_QUALITY_RETRIEVER_CONFIG)
        }
        
        # Display statistics for each retriever
        for name, retriever in retrievers.items():
            logger.info(f"=== {name} Retriever Statistics ===")
            stats = retriever.get_retrieval_stats()
            
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            logger.info("")
        
    except Exception as e:
        logger.error(f"Statistics demo failed: {e}")


async def main():
    """Main demonstration function."""
    logger.info("Starting LERK Retriever Module Demo")
    logger.info("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_semantic_retrieval()
        demonstrate_hybrid_retrieval()
        demonstrate_context_aware_retrieval()
        demonstrate_batch_retrieval()
        demonstrate_metadata_filtering()
        demonstrate_retriever_statistics()
        
        logger.info("=" * 50)
        logger.info("LERK Retriever Module Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
