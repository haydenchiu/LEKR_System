"""
Comprehensive example demonstrating subject-level retrieval and multi-level search.

This example shows how the LERK System handles subject-level queries about
document clusters with fallback mechanisms and discovery capabilities.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

from retriever import (
    MultiLevelSearchOrchestrator,
    SubjectRetriever,
    ClusterDiscoveryService,
    DocumentDiscoveryService,
    create_multi_level_orchestrator,
    create_subject_retriever,
    create_cluster_discovery_service,
    create_document_discovery_service,
    RetrieverConfig,
    DEFAULT_RETRIEVER_CONFIG
)

# Import dynamic clustering from clustering module
from clustering import (
    DynamicClusteringManager,
    create_dynamic_clustering_manager,
    process_new_documents_async
)


def demonstrate_subject_level_queries():
    """Demonstrate subject-level query handling."""
    print("🔍 Subject-Level Query Demonstration")
    print("=" * 50)
    
    # Create subject retriever
    subject_retriever = create_subject_retriever()
    
    # Example subject-level queries
    subject_queries = [
        "What are the main concepts in machine learning?",
        "Give me an overview of artificial intelligence",
        "What are the fundamentals of deep learning?",
        "Explain the principles of neural networks",
        "What topics are available in the system?"
    ]
    
    for query in subject_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 30)
        
        try:
            # Search subject knowledge
            results = subject_retriever.get_relevant_documents(query)
            
            print(f"✅ Found {len(results)} results")
            
            # Show results with fallback information
            for i, doc in enumerate(results[:3]):  # Show top 3 results
                print(f"\n  Result {i+1}:")
                print(f"    Type: {doc.metadata.get('result_type', 'unknown')}")
                print(f"    Priority: {doc.metadata.get('result_priority', 'unknown')}")
                print(f"    Method: {doc.metadata.get('search_method', 'unknown')}")
                print(f"    Similarity: {doc.metadata.get('similarity_score', 0):.3f}")
                
                if doc.metadata.get('fallback_reason'):
                    print(f"    Fallback Reason: {doc.metadata['fallback_reason']}")
                
                # Show content preview
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"    Content: {content_preview}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


def demonstrate_discovery_capabilities():
    """Demonstrate cluster and document discovery."""
    print("\n\n🔍 Discovery Capabilities Demonstration")
    print("=" * 50)
    
    # Create discovery services
    cluster_discovery = create_cluster_discovery_service()
    document_discovery = create_document_discovery_service()
    
    # Get available clusters
    print("\n📊 Available Document Clusters:")
    print("-" * 30)
    
    try:
        clusters_info = cluster_discovery.get_all_clusters()
        print(f"Total clusters: {clusters_info.get('cluster_count', 0)}")
        
        for cluster_id, cluster_data in clusters_info.get('clusters', {}).items():
            print(f"\n  Cluster: {cluster_data.get('label', cluster_id)}")
            print(f"    ID: {cluster_id}")
            print(f"    Documents: {cluster_data.get('document_count', 0)}")
            print(f"    Quality: {cluster_data.get('quality_score', 0):.2f}")
            print(f"    Topics: {', '.join(cluster_data.get('main_topics', []))}")
            
    except Exception as e:
        print(f"❌ Error getting clusters: {e}")
    
    # Get available documents
    print("\n📄 Available Documents:")
    print("-" * 30)
    
    try:
        documents_info = document_discovery.get_all_documents()
        print(f"Total documents: {documents_info.get('document_count', 0)}")
        
        for doc_id, doc_data in documents_info.get('documents', {}).items():
            print(f"\n  Document: {doc_data.get('title', doc_id)}")
            print(f"    ID: {doc_id}")
            print(f"    Type: {doc_data.get('file_type', 'unknown')}")
            print(f"    Chunks: {doc_data.get('chunk_count', 0)}")
            print(f"    Cluster: {doc_data.get('cluster_id', 'unassigned')}")
            
    except Exception as e:
        print(f"❌ Error getting documents: {e}")


def demonstrate_multi_level_search():
    """Demonstrate multi-level search orchestration."""
    print("\n\n🔍 Multi-Level Search Demonstration")
    print("=" * 50)
    
    # Create multi-level orchestrator
    orchestrator = create_multi_level_orchestrator()
    
    # Example queries for different search levels
    test_queries = [
        {
            "query": "What are the main concepts in machine learning?",
            "level": "subject",
            "description": "Subject-level query"
        },
        {
            "query": "Find specific details about neural network architectures",
            "level": "chunk",
            "description": "Chunk-level query"
        },
        {
            "query": "Compare machine learning and deep learning approaches",
            "level": "all",
            "description": "Multi-level comparison query"
        },
        {
            "query": "What topics are available in the system?",
            "level": "discovery",
            "description": "Discovery query"
        }
    ]
    
    for test_case in test_queries:
        print(f"\n📝 {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Level: {test_case['level']}")
        print("-" * 40)
        
        try:
            # Perform multi-level search
            results = orchestrator.search(
                query=test_case['query'],
                search_level=test_case['level'],
                include_discovery=True
            )
            
            print(f"✅ Search completed successfully")
            print(f"Total results: {results.get('query_metadata', {}).get('total_results', 0)}")
            
            # Show level statistics
            level_stats = results.get('level_statistics', {})
            if level_stats:
                print(f"Level breakdown:")
                for level, count in level_stats.items():
                    print(f"  {level}: {count} results")
            
            # Show combined results
            combined_results = results.get('combined_results', [])
            if combined_results:
                print(f"\nTop results:")
                for i, doc in enumerate(combined_results[:3]):
                    print(f"  {i+1}. {doc.metadata.get('result_type', 'unknown')} "
                          f"(priority: {doc.metadata.get('result_priority', 'unknown')}, "
                          f"similarity: {doc.metadata.get('similarity_score', 0):.3f})")
            
            # Show discovery information if available
            discovery = results.get('discovery', {})
            if discovery:
                print(f"\nDiscovery info:")
                clusters = discovery.get('clusters', {})
                if clusters:
                    print(f"  Available clusters: {clusters.get('cluster_count', 0)}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


async def demonstrate_dynamic_clustering():
    """Demonstrate dynamic clustering and subject knowledge updates."""
    print("\n\n🔍 Dynamic Clustering Demonstration")
    print("=" * 50)
    
    # Create dynamic clustering manager
    clustering_manager = create_dynamic_clustering_manager()
    
    # Simulate new documents being added
    new_documents = [
        {
            "document_id": "new_doc_001",
            "title": "Advanced Machine Learning Techniques",
            "file_type": "pdf",
            "content": "This document covers advanced machine learning techniques including ensemble methods, deep learning, and reinforcement learning.",
            "chunks": [
                {
                    "chunk_id": "chunk_001",
                    "content": "Ensemble methods combine multiple models to improve performance",
                    "chunk_index": 0
                },
                {
                    "chunk_id": "chunk_002", 
                    "content": "Deep learning uses neural networks with multiple layers",
                    "chunk_index": 1
                }
            ]
        },
        {
            "document_id": "new_doc_002",
            "title": "Natural Language Processing Fundamentals",
            "file_type": "pdf",
            "content": "This document introduces natural language processing concepts and techniques.",
            "chunks": [
                {
                    "chunk_id": "chunk_003",
                    "content": "NLP involves processing and understanding human language",
                    "chunk_index": 0
                },
                {
                    "chunk_id": "chunk_004",
                    "content": "Tokenization is the process of breaking text into tokens",
                    "chunk_index": 1
                }
            ]
        },
        {
            "document_id": "new_doc_003",
            "title": "Computer Vision Applications",
            "file_type": "pdf",
            "content": "This document covers computer vision applications and techniques.",
            "chunks": [
                {
                    "chunk_id": "chunk_005",
                    "content": "Computer vision enables machines to interpret visual information",
                    "chunk_index": 0
                },
                {
                    "chunk_id": "chunk_006",
                    "content": "Image classification is a common computer vision task",
                    "chunk_index": 1
                }
            ]
        }
    ]
    
    print(f"📄 Processing {len(new_documents)} new documents...")
    
    try:
        # Process new documents with different strategies
        strategies = ["auto", "incremental", "hybrid", "full_reclustering"]
        
        for strategy in strategies:
            print(f"\n🔄 Strategy: {strategy}")
            print("-" * 20)
            
            update_results = await clustering_manager.process_new_documents(
                new_documents=new_documents,
                update_strategy=strategy
            )
            
            print(f"Success: {update_results.get('success', False)}")
            print(f"Strategy used: {update_results.get('strategy', 'unknown')}")
            
            # Show clustering changes
            clustering_changes = update_results.get('clustering_changes', {})
            print(f"New clusters: {clustering_changes.get('new_clusters', 0)}")
            print(f"Updated clusters: {clustering_changes.get('updated_clusters', 0)}")
            print(f"Assigned documents: {clustering_changes.get('assigned_documents', 0)}")
            
            # Show subject knowledge changes
            subject_changes = update_results.get('subject_knowledge_changes', {})
            print(f"Updated subjects: {subject_changes.get('updated_subjects', 0)}")
            print(f"New subjects: {subject_changes.get('new_subjects', 0)}")
            
            # Show quality metrics
            quality_metrics = update_results.get('quality_metrics', {})
            if quality_metrics:
                print(f"Quality metrics: {quality_metrics}")
            
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_fallback_mechanisms():
    """Demonstrate fallback mechanisms when subject knowledge is insufficient."""
    print("\n\n🔍 Fallback Mechanisms Demonstration")
    print("=" * 50)
    
    # Create subject retriever
    subject_retriever = create_subject_retriever()
    
    # Test queries that might trigger fallbacks
    fallback_test_queries = [
        {
            "query": "What is the exact formula for backpropagation?",
            "description": "Specific technical detail (likely to fallback to chunks)"
        },
        {
            "query": "Show me the specific code implementation of gradient descent",
            "description": "Very specific technical request (likely to fallback)"
        },
        {
            "query": "What are the main concepts in quantum computing?",
            "description": "Topic not in system (likely to fallback or return empty)"
        }
    ]
    
    for test_case in fallback_test_queries:
        print(f"\n📝 {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print("-" * 40)
        
        try:
            results = subject_retriever.get_relevant_documents(test_case['query'])
            
            print(f"✅ Found {len(results)} results")
            
            # Analyze fallback behavior
            subject_results = [doc for doc in results if doc.metadata.get('result_type') == 'subject_knowledge']
            fallback_results = [doc for doc in results if doc.metadata.get('result_type') == 'fallback']
            
            print(f"Subject knowledge results: {len(subject_results)}")
            print(f"Fallback results: {len(fallback_results)}")
            
            if fallback_results:
                print("🔄 Fallback triggered:")
                for doc in fallback_results[:2]:  # Show first 2 fallback results
                    print(f"  - {doc.metadata.get('search_method', 'unknown')} "
                          f"(reason: {doc.metadata.get('fallback_reason', 'unknown')})")
            
            # Show result quality
            if results:
                avg_similarity = sum(doc.metadata.get('similarity_score', 0) for doc in results) / len(results)
                print(f"Average similarity: {avg_similarity:.3f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


def demonstrate_query_classification():
    """Demonstrate query classification and intent detection."""
    print("\n\n🔍 Query Classification Demonstration")
    print("=" * 50)
    
    # Create multi-level orchestrator
    orchestrator = create_multi_level_orchestrator()
    
    # Test queries with different intents
    classification_test_queries = [
        "What are the main concepts in machine learning?",
        "Compare supervised and unsupervised learning",
        "Why does gradient descent work?",
        "Find the exact quote about neural networks",
        "What topics are available in the system?",
        "Summarize the key findings about deep learning",
        "How do convolutional neural networks work?",
        "Show me specific examples of machine learning algorithms"
    ]
    
    for query in classification_test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 30)
        
        try:
            # Get search capabilities to access classification
            capabilities = orchestrator.get_search_capabilities()
            
            # Simulate query classification (this would be done internally)
            query_lower = query.lower()
            
            # Determine search level
            if any(word in query_lower for word in ["main concepts", "overview", "what is"]):
                search_level = "subject"
            elif any(word in query_lower for word in ["compare", "difference", "versus"]):
                search_level = "comparison"
            elif any(word in query_lower for word in ["exact", "specific", "quote"]):
                search_level = "chunk"
            elif any(word in query_lower for word in ["what topics", "available", "show me"]):
                search_level = "discovery"
            else:
                search_level = "auto"
            
            print(f"Detected search level: {search_level}")
            
            # Determine intent
            if any(word in query_lower for word in ["compare", "difference", "versus"]):
                intent = "comparison"
            elif any(word in query_lower for word in ["why", "how", "explain"]):
                intent = "explanation"
            elif any(word in query_lower for word in ["summarize", "overview"]):
                intent = "synthesis"
            else:
                intent = "information_retrieval"
            
            print(f"Detected intent: {intent}")
            
            # Determine complexity
            if len(query.split()) > 10 or intent in ["comparison", "explanation"]:
                complexity = "complex"
            else:
                complexity = "simple"
            
            print(f"Query complexity: {complexity}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 LERK System Subject-Level Retrieval Demonstration")
    print("=" * 60)
    print("This demonstration shows how the LERK System handles subject-level")
    print("queries about document clusters with comprehensive fallback mechanisms.")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_subject_level_queries()
    demonstrate_discovery_capabilities()
    demonstrate_multi_level_search()
    
    # Run async demonstrations
    print("\n🔄 Running async demonstrations...")
    asyncio.run(demonstrate_dynamic_clustering())
    
    demonstrate_fallback_mechanisms()
    demonstrate_query_classification()
    
    print("\n\n✅ Demonstration completed!")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("• Subject-level query handling with fallback mechanisms")
    print("• Cluster and document discovery capabilities")
    print("• Multi-level search orchestration")
    print("• Dynamic clustering and subject knowledge updates")
    print("• Query classification and intent detection")
    print("• Comprehensive error handling and recovery")
    print("=" * 60)


if __name__ == "__main__":
    main()
