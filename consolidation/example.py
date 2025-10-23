"""
Example usage of the consolidation module.

This module demonstrates how to use the consolidation functionality
to consolidate knowledge from document chunks and subjects.
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import consolidation module components
from consolidation.document_consolidator import DocumentConsolidator
from consolidation.subject_consolidator import SubjectConsolidator
from consolidation.knowledge_storage import KnowledgeStorageManager
from consolidation.cluster_integration import (
    ClusterBasedSubjectConsolidator,
    IntegratedConsolidationPipeline
)
from consolidation.models import DocumentKnowledge, SubjectKnowledge, KeyConcept, KnowledgeRelation
from consolidation.config import (
    ConsolidationConfig,
    DEFAULT_CONSOLIDATION_CONFIG,
    FAST_CONSOLIDATION_CONFIG,
    HIGH_QUALITY_CONSOLIDATION_CONFIG
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_chunks() -> List[Dict[str, Any]]:
    """Create sample document chunks for demonstration."""
    return [
        {
            "chunk_id": "chunk_1",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
            "metadata": {"page": 1, "section": "introduction"}
        },
        {
            "chunk_id": "chunk_2", 
            "content": "Deep learning uses neural networks with multiple layers to process data. Convolutional neural networks are particularly effective for image recognition tasks, while recurrent neural networks excel at sequence processing.",
            "metadata": {"page": 2, "section": "deep_learning"}
        },
        {
            "chunk_id": "chunk_3",
            "content": "Natural language processing combines computational linguistics with machine learning to help computers understand human language. Key techniques include tokenization, part-of-speech tagging, and named entity recognition.",
            "metadata": {"page": 3, "section": "nlp"}
        },
        {
            "chunk_id": "chunk_4",
            "content": "Computer vision enables machines to interpret and analyze visual information from the world. It involves image processing, object detection, and pattern recognition techniques.",
            "metadata": {"page": 4, "section": "computer_vision"}
        },
        {
            "chunk_id": "chunk_5",
            "content": "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, analysis, and visualization.",
            "metadata": {"page": 5, "section": "data_science"}
        }
    ]


def create_sample_logic_data() -> List[Dict[str, Any]]:
    """Create sample logic extraction data for demonstration."""
    return [
        {
            "chunk_id": "chunk_1",
            "claims": [
                {"id": "claim_1", "statement": "Machine learning is a subset of AI", "type": "factual", "confidence": 0.9},
                {"id": "claim_2", "statement": "ML includes supervised, unsupervised, and reinforcement learning", "type": "factual", "confidence": 0.8}
            ],
            "relations": [
                {"premise": "claim_1", "conclusion": "claim_2", "type": "hierarchical", "strength": 0.8}
            ]
        },
        {
            "chunk_id": "chunk_2",
            "claims": [
                {"id": "claim_3", "statement": "Deep learning uses neural networks with multiple layers", "type": "factual", "confidence": 0.9},
                {"id": "claim_4", "statement": "CNNs are effective for image recognition", "type": "factual", "confidence": 0.8}
            ],
            "relations": [
                {"premise": "claim_3", "conclusion": "claim_4", "type": "causal", "strength": 0.7}
            ]
        }
    ]


def demonstrate_document_consolidation():
    """Demonstrate document-level consolidation."""
    logger.info("=== Document Consolidation Demo ===")
    
    # Create sample data
    chunks = create_sample_chunks()
    logic_data = create_sample_logic_data()
    
    # Initialize document consolidator
    config = FAST_CONSOLIDATION_CONFIG
    consolidator = DocumentConsolidator(config)
    
    try:
        # Consolidate document knowledge
        document_knowledge = consolidator.consolidate_document(
            document_id="doc_ai_fundamentals",
            document_title="AI Fundamentals",
            chunks=chunks,
            chunk_logic_data=logic_data
        )
        
        # Display results
        logger.info(f"Document: {document_knowledge.title}")
        logger.info(f"Summary: {document_knowledge.summary}")
        logger.info(f"Key Concepts: {len(document_knowledge.key_concepts)}")
        logger.info(f"Relations: {len(document_knowledge.knowledge_relations)}")
        logger.info(f"Quality Score: {document_knowledge.quality_score:.3f}")
        
        # Show sample concepts
        for i, concept in enumerate(document_knowledge.key_concepts[:3]):
            logger.info(f"  Concept {i+1}: {concept.name} (confidence: {concept.confidence:.3f})")
        
        return document_knowledge
        
    except Exception as e:
        logger.error(f"Document consolidation failed: {e}")
        return None


def demonstrate_subject_consolidation():
    """Demonstrate subject-level consolidation."""
    logger.info("=== Subject Consolidation Demo ===")
    
    # Create sample document knowledge
    doc1 = DocumentKnowledge(
        document_id="doc_ml_basics",
        title="Machine Learning Basics",
        summary="Introduction to machine learning concepts and algorithms",
        key_concepts=[
            KeyConcept(
                concept_id="concept_1",
                name="Supervised Learning",
                description="Learning with labeled training data",
                category="machine_learning",
                confidence=0.9
            ),
            KeyConcept(
                concept_id="concept_2", 
                name="Neural Networks",
                description="Computational models inspired by biological neural networks",
                category="deep_learning",
                confidence=0.8
            )
        ],
        knowledge_relations=[],
        main_themes=["machine_learning", "algorithms"],
        knowledge_graph={},
        quality_score=0.8
    )
    
    doc2 = DocumentKnowledge(
        document_id="doc_dl_advanced",
        title="Advanced Deep Learning",
        summary="Advanced topics in deep learning and neural networks",
        key_concepts=[
            KeyConcept(
                concept_id="concept_3",
                name="Deep Neural Networks",
                description="Neural networks with multiple hidden layers",
                category="deep_learning", 
                confidence=0.9
            ),
            KeyConcept(
                concept_id="concept_4",
                name="Backpropagation",
                description="Algorithm for training neural networks",
                category="deep_learning",
                confidence=0.8
            )
        ],
        knowledge_relations=[],
        main_themes=["deep_learning", "neural_networks"],
        knowledge_graph={},
        quality_score=0.85
    )
    
    # Initialize subject consolidator
    config = DEFAULT_CONSOLIDATION_CONFIG
    consolidator = SubjectConsolidator(config)
    
    try:
        # Consolidate subject knowledge
        subject_knowledge = consolidator.consolidate_subject(
            subject_id="subject_ai_ml",
            subject_name="Artificial Intelligence and Machine Learning",
            document_knowledge=[doc1, doc2],
            subject_description="Comprehensive overview of AI and ML concepts"
        )
        
        # Display results
        logger.info(f"Subject: {subject_knowledge.name}")
        logger.info(f"Description: {subject_knowledge.description}")
        logger.info(f"Core Concepts: {len(subject_knowledge.core_concepts)}")
        logger.info(f"Relations: {len(subject_knowledge.knowledge_relations)}")
        logger.info(f"Expertise Level: {subject_knowledge.expertise_level}")
        logger.info(f"Quality Score: {subject_knowledge.quality_score:.3f}")
        
        # Show sample concepts
        for i, concept in enumerate(subject_knowledge.core_concepts[:3]):
            logger.info(f"  Concept {i+1}: {concept.name} (confidence: {concept.confidence:.3f})")
        
        return subject_knowledge
        
    except Exception as e:
        logger.error(f"Subject consolidation failed: {e}")
        return None


def demonstrate_knowledge_storage():
    """Demonstrate knowledge storage and retrieval."""
    logger.info("=== Knowledge Storage Demo ===")
    
    # Initialize storage manager
    config = DEFAULT_CONSOLIDATION_CONFIG
    storage = KnowledgeStorageManager(config)
    
    try:
        # Create sample document knowledge
        doc_knowledge = DocumentKnowledge(
            document_id="doc_sample",
            title="Sample Document",
            summary="A sample document for demonstration",
            key_concepts=[
                KeyConcept(
                    concept_id="sample_concept",
                    name="Sample Concept",
                    description="A sample concept for testing",
                    category="general",
                    confidence=0.8
                )
            ],
            knowledge_relations=[],
            main_themes=["sample", "testing"],
            knowledge_graph={},
            quality_score=0.8
        )
        
        # Save document knowledge
        storage_id = storage.save_document_knowledge(doc_knowledge)
        logger.info(f"Saved document knowledge with storage ID: {storage_id}")
        
        # Retrieve document knowledge
        retrieved_doc = storage.retrieve_document_knowledge("doc_sample")
        if retrieved_doc:
            logger.info(f"Retrieved document: {retrieved_doc.title}")
        else:
            logger.warning("Failed to retrieve document knowledge")
        
        # Demonstrate similarity search
        search_results = storage.search_documents_by_similarity(
            "machine learning algorithms",
            limit=5,
            min_similarity=0.3
        )
        logger.info(f"Found {len(search_results)} similar documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Knowledge storage failed: {e}")
        return False


def demonstrate_cluster_based_consolidation():
    """Demonstrate cluster-based subject consolidation."""
    logger.info("=== Cluster-Based Consolidation Demo ===")
    
    try:
        # Create sample documents
        documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Deep learning uses neural networks with multiple layers to process complex data patterns.",
            "Natural language processing combines computational linguistics with machine learning to understand human language.",
            "Computer vision enables machines to interpret and analyze visual information from images and videos.",
            "Data science combines statistics, programming, and domain expertise to extract insights from large datasets."
        ]
        
        document_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        
        # Create sample document chunks (simplified)
        document_chunks_map = {
            doc_id: [{"content": doc, "metadata": {"page": 1}}] 
            for doc_id, doc in zip(document_ids, documents)
        }
        
        # Initialize integrated pipeline
        pipeline = IntegratedConsolidationPipeline(DEFAULT_CONSOLIDATION_CONFIG)
        
        # Process documents to subjects
        doc_knowledge_list, subject_knowledge_list = pipeline.process_documents_to_subjects(
            documents=documents,
            document_ids=document_ids,
            document_chunks_map=document_chunks_map
        )
        
        # Display results
        logger.info(f"Processed {len(doc_knowledge_list)} documents")
        logger.info(f"Created {len(subject_knowledge_list)} subjects")
        
        for i, subject in enumerate(subject_knowledge_list):
            logger.info(f"Subject {i+1}: {subject.name}")
            logger.info(f"  Description: {subject.description}")
            logger.info(f"  Core concepts: {len(subject.core_concepts)}")
            logger.info(f"  Quality score: {subject.quality_score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Cluster-based consolidation failed: {e}")
        return False


def demonstrate_consolidation_pipeline():
    """Demonstrate the complete consolidation pipeline."""
    logger.info("=== Complete Consolidation Pipeline Demo ===")
    
    try:
        # Step 1: Document consolidation
        logger.info("Step 1: Document Consolidation")
        doc_knowledge = demonstrate_document_consolidation()
        
        if not doc_knowledge:
            logger.error("Document consolidation failed")
            return False
        
        # Step 2: Subject consolidation (with multiple documents)
        logger.info("Step 2: Subject Consolidation")
        subject_knowledge = demonstrate_subject_consolidation()
        
        if not subject_knowledge:
            logger.error("Subject consolidation failed")
            return False
        
        # Step 3: Knowledge storage
        logger.info("Step 3: Knowledge Storage")
        storage_success = demonstrate_knowledge_storage()
        
        if not storage_success:
            logger.error("Knowledge storage failed")
            return False
        
        logger.info("=== Consolidation Pipeline Completed Successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Consolidation pipeline failed: {e}")
        return False


def demonstrate_different_configurations():
    """Demonstrate different consolidation configurations."""
    logger.info("=== Configuration Comparison Demo ===")
    
    chunks = create_sample_chunks()
    
    configs = {
        "Fast": FAST_CONSOLIDATION_CONFIG,
        "Default": DEFAULT_CONSOLIDATION_CONFIG,
        "High Quality": HIGH_QUALITY_CONSOLIDATION_CONFIG
    }
    
    results = {}
    
    for config_name, config in configs.items():
        logger.info(f"Testing {config_name} configuration...")
        
        try:
            consolidator = DocumentConsolidator(config)
            doc_knowledge = consolidator.consolidate_document(
                document_id=f"doc_{config_name.lower().replace(' ', '_')}",
                document_title=f"Document with {config_name} Config",
                chunks=chunks
            )
            
            results[config_name] = {
                "num_concepts": len(doc_knowledge.key_concepts),
                "num_relations": len(doc_knowledge.knowledge_relations),
                "quality_score": doc_knowledge.quality_score,
                "processing_time": "N/A"  # Would need timing in real implementation
            }
            
            logger.info(f"  Concepts: {len(doc_knowledge.key_concepts)}")
            logger.info(f"  Relations: {len(doc_knowledge.knowledge_relations)}")
            logger.info(f"  Quality: {doc_knowledge.quality_score:.3f}")
            
        except Exception as e:
            logger.error(f"  {config_name} configuration failed: {e}")
            results[config_name] = {"error": str(e)}
    
    # Display comparison
    logger.info("\n=== Configuration Comparison ===")
    for config_name, result in results.items():
        if "error" in result:
            logger.info(f"{config_name}: ERROR - {result['error']}")
        else:
            logger.info(f"{config_name}: {result['num_concepts']} concepts, "
                       f"{result['num_relations']} relations, "
                       f"quality {result['quality_score']:.3f}")
    
    return results


def main():
    """Main demonstration function."""
    logger.info("Starting Consolidation Module Demonstration")
    logger.info("=" * 50)
    
    try:
        # Demonstrate individual components
        demonstrate_document_consolidation()
        logger.info("")
        
        demonstrate_subject_consolidation()
        logger.info("")
        
        demonstrate_knowledge_storage()
        logger.info("")
        
        # Demonstrate cluster-based consolidation
        demonstrate_cluster_based_consolidation()
        logger.info("")
        
        # Demonstrate complete pipeline
        demonstrate_consolidation_pipeline()
        logger.info("")
        
        # Demonstrate different configurations
        demonstrate_different_configurations()
        
        logger.info("=" * 50)
        logger.info("Consolidation Module Demonstration Completed")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")


if __name__ == "__main__":
    main()
