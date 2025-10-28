#!/usr/bin/env python3
"""
LERK System - Qdrant Indexing Example

This script demonstrates how to use the Qdrant indexing functionality
to index processed documents with enrichments and logic extractions
into Qdrant vector database for semantic search.

Usage:
    python retriever/example_indexing.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from retriever import QdrantIndexer, EmbeddingStrategy, RetrieverConfig, DEFAULT_RETRIEVER_CONFIG
    from retriever.exceptions import VectorSearchError, DatabaseConnectionError
except ImportError as e:
    print(f"Failed to import retriever module: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_processed_document():
    """Create a sample processed document for testing."""
    sample_data = {
        'file_path': 'sample_document.pdf',
        'processing_timestamp': datetime.utcnow().isoformat(),
        'chunks': [
            {
                'id': 'chunk_1',
                'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.',
                'content_type': 'text',
                'chunk_index': 0,
                'enrichment': {
                    'summary': 'Introduction to machine learning as a subset of AI',
                    'keywords': ['machine learning', 'artificial intelligence', 'computers', 'data', 'programming'],
                    'hypothetical_questions': [
                        'What is machine learning?',
                        'How does machine learning relate to AI?',
                        'What makes machine learning different from traditional programming?'
                    ],
                    'table_summary': None
                },
                'logic_extraction': {
                    'claims': [
                        {
                            'id': 'claim_1',
                            'statement': 'Machine learning is a subset of artificial intelligence',
                            'type': 'factual',
                            'confidence': 0.95,
                            'derived_from': None
                        },
                        {
                            'id': 'claim_2',
                            'statement': 'Machine learning enables computers to learn from data',
                            'type': 'factual',
                            'confidence': 0.90,
                            'derived_from': ['claim_1']
                        }
                    ],
                    'logical_relations': [
                        {
                            'premise': 'claim_1',
                            'conclusion': 'claim_2',
                            'relation_type': 'causal',
                            'certainty': 0.85
                        }
                    ],
                    'assumptions': [
                        'Computers can learn from data',
                        'Learning from data is valuable'
                    ],
                    'constraints': [
                        'Requires data to learn from',
                        'Limited by quality of data'
                    ],
                    'open_questions': [
                        'What are the limitations of machine learning?',
                        'How does machine learning scale with data size?'
                    ]
                },
                'metadata': {
                    'file_type': 'pdf',
                    'page_number': 1,
                    'section': 'introduction'
                }
            },
            {
                'id': 'chunk_2',
                'content': 'There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.',
                'content_type': 'text',
                'chunk_index': 1,
                'enrichment': {
                    'summary': 'Overview of three main types of machine learning',
                    'keywords': ['supervised learning', 'unsupervised learning', 'reinforcement learning', 'types'],
                    'hypothetical_questions': [
                        'What are the main types of machine learning?',
                        'What is supervised learning?',
                        'What is unsupervised learning?',
                        'What is reinforcement learning?'
                    ],
                    'table_summary': None
                },
                'logic_extraction': {
                    'claims': [
                        {
                            'id': 'claim_3',
                            'statement': 'There are three main types of machine learning',
                            'type': 'factual',
                            'confidence': 0.98,
                            'derived_from': None
                        },
                        {
                            'id': 'claim_4',
                            'statement': 'Supervised learning is a type of machine learning',
                            'type': 'factual',
                            'confidence': 0.95,
                            'derived_from': ['claim_3']
                        },
                        {
                            'id': 'claim_5',
                            'statement': 'Unsupervised learning is a type of machine learning',
                            'type': 'factual',
                            'confidence': 0.95,
                            'derived_from': ['claim_3']
                        }
                    ],
                    'logical_relations': [
                        {
                            'premise': 'claim_3',
                            'conclusion': 'claim_4',
                            'relation_type': 'supportive',
                            'certainty': 0.95
                        },
                        {
                            'premise': 'claim_3',
                            'conclusion': 'claim_5',
                            'relation_type': 'supportive',
                            'certainty': 0.95
                        }
                    ],
                    'assumptions': [
                        'These three types cover most machine learning approaches',
                        'Each type has distinct characteristics'
                    ],
                    'constraints': [
                        'Limited to three main categories',
                        'May not cover all specialized approaches'
                    ],
                    'open_questions': [
                        'Are there other important types of machine learning?',
                        'How do these types differ in practice?'
                    ]
                },
                'metadata': {
                    'file_type': 'pdf',
                    'page_number': 1,
                    'section': 'types'
                }
            }
        ],
        'processing_stats': {
            'total_chunks': 2,
            'enriched_chunks': 2,
            'logic_chunks': 2
        }
    }
    
    return sample_data


def demonstrate_embedding_strategies():
    """Demonstrate different embedding strategies."""
    print("\n" + "="*60)
    print("EMBEDDING STRATEGIES DEMONSTRATION")
    print("="*60)
    
    sample_chunk = {
        'content': 'Machine learning is a subset of artificial intelligence.',
        'enrichment': {
            'summary': 'Introduction to machine learning',
            'keywords': ['machine learning', 'AI', 'artificial intelligence'],
            'hypothetical_questions': ['What is machine learning?', 'How does ML relate to AI?'],
            'table_summary': None
        },
        'logic_extraction': {
            'claims': [
                {
                    'id': 'claim_1',
                    'statement': 'Machine learning is a subset of AI',
                    'type': 'factual',
                    'confidence': 0.95,
                    'derived_from': None
                }
            ],
            'logical_relations': [
                {
                    'premise': 'claim_1',
                    'conclusion': 'claim_2',
                    'relation_type': 'causal',
                    'certainty': 0.85
                }
            ],
            'assumptions': ['AI can be subdivided into categories'],
            'constraints': ['Limited to subset relationship'],
            'open_questions': ['What other AI subsets exist?']
        }
    }
    
    strategies = {
        'content_only': {
            'include_base_content': True,
            'include_enrichments': False,
            'include_logic_extractions': False,
            'combination_strategy': 'concatenate'
        },
        'content_and_enrichments': {
            'include_base_content': True,
            'include_enrichments': True,
            'include_logic_extractions': False,
            'combination_strategy': 'structured'
        },
        'full_enriched': {
            'include_base_content': True,
            'include_enrichments': True,
            'include_logic_extractions': True,
            'combination_strategy': 'structured'
        }
    }
    
    for strategy_name, config in strategies.items():
        print(f"\n{strategy_name.upper()} Strategy:")
        print("-" * 40)
        
        strategy = EmbeddingStrategy(config)
        embedding_text = strategy.create_embedding_text(sample_chunk)
        
        print(f"Embedding Text ({len(embedding_text)} chars):")
        print(f"'{embedding_text}'")


def demonstrate_indexing():
    """Demonstrate Qdrant indexing functionality."""
    print("\n" + "="*60)
    print("QDRANT INDEXING DEMONSTRATION")
    print("="*60)
    
    # Create sample processed document
    sample_data = create_sample_processed_document()
    
    # Save sample document
    sample_file = Path("temp_sample_processed.json")
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample processed document: {sample_file}")
    
    try:
        # Create indexer with custom configuration
        config = RetrieverConfig(
            collection_name='lerk_demo',
            host='localhost',
            port=6333,
            embedding_model='all-MiniLM-L6-v2',
            embedding_dimension=384
        )
        
        # Create embedding strategy
        strategy = EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': True,
            'include_logic_extractions': True,
            'combination_strategy': 'structured',
            'max_text_length': 512
        })
        
        # Create indexer
        indexer = QdrantIndexer(config, strategy)
        
        print("\nIndexing sample document...")
        
        # Index the sample document
        result = indexer.index_processed_document(str(sample_file))
        
        if result['success']:
            print(f"✅ Successfully indexed {result['indexed_count']} chunks")
            print(f"📁 File: {result['file_path']}")
        else:
            print(f"❌ Indexing failed: {result['error']}")
        
        # Get collection stats
        print("\nCollection Statistics:")
        stats = indexer.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Demonstrate individual chunk indexing
        print("\nDemonstrating individual chunk indexing...")
        
        chunk_data = {
            'id': 'demo_chunk',
            'document_id': 'demo_document',
            'content': 'This is a demonstration chunk for testing.',
            'content_type': 'text',
            'chunk_index': 0,
            'enrichment': {
                'summary': 'Demo chunk for testing purposes',
                'keywords': ['demo', 'test', 'chunk']
            },
            'logic_extraction': {
                'claims': [{'text': 'This is a test chunk', 'confidence': 0.8}]
            },
            'metadata': {'test': True}
        }
        
        point_id = indexer.index_chunk(chunk_data)
        print(f"✅ Indexed individual chunk with ID: {point_id}")
        
        # Demonstrate batch indexing
        print("\nDemonstrating batch indexing...")
        
        batch_chunks = [
            {
                'id': f'batch_chunk_{i}',
                'document_id': 'batch_document',
                'content': f'This is batch chunk number {i} for testing.',
                'content_type': 'text',
                'chunk_index': i,
                'enrichment': {
                    'summary': f'Batch chunk {i} summary',
                    'keywords': ['batch', 'test', f'chunk_{i}']
                },
                'metadata': {'batch': True, 'index': i}
            }
            for i in range(3)
        ]
        
        point_ids = indexer.index_chunks_batch(batch_chunks)
        print(f"✅ Indexed {len(point_ids)} chunks in batch")
        
        # Demonstrate deletion
        print("\nDemonstrating chunk deletion...")
        
        # Delete the individual chunk
        deleted = indexer.delete_chunk('demo_chunk')
        if deleted:
            print("✅ Successfully deleted demo chunk")
        else:
            print("⚠️ Demo chunk not found for deletion")
        
        # Delete all batch chunks
        deleted_count = indexer.delete_document_chunks('batch_document')
        print(f"✅ Deleted {deleted_count} chunks for batch document")
        
    except DatabaseConnectionError as e:
        print(f"❌ Database connection error: {e}")
        print("Make sure Qdrant is running on localhost:6333")
    except VectorSearchError as e:
        print(f"❌ Vector search error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Clean up sample file
        if sample_file.exists():
            sample_file.unlink()
            print(f"\n🧹 Cleaned up sample file: {sample_file}")


def demonstrate_search_simulation():
    """Demonstrate how indexed content would be searched."""
    print("\n" + "="*60)
    print("SEARCH SIMULATION")
    print("="*60)
    
    # Simulate search queries and show what would be found
    queries = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "How does AI relate to machine learning?",
        "What is supervised learning?"
    ]
    
    print("Simulated search queries and expected results:")
    print("-" * 50)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # This would normally use the retriever to search Qdrant
        # For demo purposes, we'll show what would be found
        if "machine learning" in query.lower():
            print("  Expected results:")
            print("    - Chunk 1: Machine learning definition and AI relationship")
            print("    - Chunk 2: Three types of machine learning")
        
        if "types" in query.lower():
            print("  Expected results:")
            print("    - Chunk 2: Supervised, unsupervised, and reinforcement learning")
        
        if "supervised" in query.lower():
            print("  Expected results:")
            print("    - Chunk 2: Supervised learning as a type of ML")


def main():
    """Main demonstration function."""
    print("LERK System - Qdrant Indexing Demonstration")
    print("=" * 60)
    
    print("\nThis demonstration shows:")
    print("1. Different embedding strategies for chunk content")
    print("2. Qdrant indexing with enrichments and logic extractions")
    print("3. Index management (creation, deletion, statistics)")
    print("4. Search simulation")
    
    # Demonstrate embedding strategies
    demonstrate_embedding_strategies()
    
    # Demonstrate indexing
    demonstrate_indexing()
    
    # Demonstrate search simulation
    demonstrate_search_simulation()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("2. Run this script to test indexing")
    print("3. Use SemanticRetriever to search indexed content")
    print("4. Integrate with QA agent for question answering")


if __name__ == "__main__":
    main()
