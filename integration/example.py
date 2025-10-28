"""
LERK System Integration Example

This example demonstrates the clean integration between clustering,
consolidation, and retriever modules using the integration layer.
"""

import asyncio
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_clean_integration():
    """Demonstrate clean integration between modules."""
    print("🔗 LERK System Clean Integration Demonstration")
    print("=" * 60)
    
    try:
        # Import the integration manager
        from integration import (
            LERKIntegrationManager,
            create_integration_manager,
            process_documents_with_integration
        )
        
        # Import individual modules
        from clustering import DynamicClusteringManager
        from consolidation import ClusterBasedSubjectConsolidator
        from retriever import SubjectRetriever
        
        print("\n📋 Step 1: Initialize Module Managers")
        print("-" * 40)
        
        # Create individual module managers
        clustering_manager = DynamicClusteringManager()
        consolidation_manager = ClusterBasedSubjectConsolidator()
        retriever_manager = SubjectRetriever()
        
        print("✅ Clustering manager initialized")
        print("✅ Consolidation manager initialized") 
        print("✅ Retriever manager initialized")
        
        print("\n🔗 Step 2: Create Integration Manager")
        print("-" * 40)
        
        # Create integration manager
        integration_manager = create_integration_manager(
            clustering_manager=clustering_manager,
            consolidation_manager=consolidation_manager,
            retriever_manager=retriever_manager
        )
        
        print("✅ Integration manager created")
        
        # Check integration status
        status = integration_manager.get_integration_status()
        print(f"📊 Integration health: {status['integration_health']}")
        print(f"📊 Managers available: {status['managers_available']}")
        
        print("\n📄 Step 3: Prepare Sample Data")
        print("-" * 40)
        
        # Create sample documents
        new_documents = [
            {
                "document_id": "doc_001",
                "content": "Machine learning algorithms can be supervised or unsupervised.",
                "file_type": "text"
            },
            {
                "document_id": "doc_002", 
                "content": "Deep learning uses neural networks with multiple layers.",
                "file_type": "text"
            },
            {
                "document_id": "doc_003",
                "content": "Natural language processing helps computers understand human language.",
                "file_type": "text"
            }
        ]
        
        # Create sample document knowledge map
        document_knowledge_map = {
            "doc_001": {
                "document_id": "doc_001",
                "core_concepts": ["machine learning", "algorithms", "supervised", "unsupervised"],
                "key_insights": ["ML algorithms can be categorized by learning type"]
            },
            "doc_002": {
                "document_id": "doc_002",
                "core_concepts": ["deep learning", "neural networks", "layers"],
                "key_insights": ["Deep learning uses multi-layer neural networks"]
            },
            "doc_003": {
                "document_id": "doc_003", 
                "core_concepts": ["natural language processing", "computers", "human language"],
                "key_insights": ["NLP enables computer understanding of human language"]
            }
        }
        
        print(f"✅ Created {len(new_documents)} sample documents")
        print(f"✅ Created document knowledge map with {len(document_knowledge_map)} entries")
        
        print("\n⚡ Step 4: Process Documents Through Integration Pipeline")
        print("-" * 40)
        
        # Process documents through the complete integration pipeline
        integration_results = await integration_manager.process_new_documents_integrated(
            new_documents=new_documents,
            document_knowledge_map=document_knowledge_map,
            update_strategy="auto"
        )
        
        print(f"✅ Integration processing completed: {integration_results['success']}")
        
        # Display results
        print("\n📊 Step 5: Integration Results")
        print("-" * 40)
        
        clustering_results = integration_results.get("clustering_results", {})
        consolidation_results = integration_results.get("consolidation_results", {})
        retriever_notification = integration_results.get("retriever_notification", {})
        
        print(f"🔵 Clustering Results:")
        print(f"   - Success: {clustering_results.get('success', False)}")
        print(f"   - Strategy: {clustering_results.get('strategy', 'unknown')}")
        print(f"   - New clusters: {clustering_results.get('clustering_changes', {}).get('new_clusters', 0)}")
        print(f"   - Updated clusters: {clustering_results.get('clustering_changes', {}).get('updated_clusters', 0)}")
        
        print(f"\n🟢 Consolidation Results:")
        print(f"   - Updated subjects: {len(consolidation_results.get('updated_subjects', []))}")
        print(f"   - New subjects: {len(consolidation_results.get('new_subjects', []))}")
        print(f"   - Deleted subjects: {len(consolidation_results.get('deleted_subjects', []))}")
        
        print(f"\n🟡 Retriever Notification:")
        print(f"   - Success: {retriever_notification.get('success', False)}")
        print(f"   - Cache updated: {retriever_notification.get('cache_updated', False)}")
        print(f"   - Indices refreshed: {retriever_notification.get('indices_refreshed', False)}")
        
        if integration_results.get("errors"):
            print(f"\n⚠️  Errors:")
            for error in integration_results["errors"]:
                print(f"   - {error}")
        
        print("\n🎯 Step 6: Demonstrate Individual Module Interfaces")
        print("-" * 40)
        
        # Demonstrate individual module interfaces
        from integration import (
            ClusteringIntegrationInterface,
            ConsolidationIntegrationInterface,
            RetrieverIntegrationInterface
        )
        
        # Test clustering interface
        print("🔵 Testing Clustering Interface:")
        clustering_interface_results = await ClusteringIntegrationInterface.process_documents_for_clustering(
            new_documents[:2], "incremental"
        )
        print(f"   - Clustering interface success: {clustering_interface_results.get('success', False)}")
        
        # Test consolidation interface
        print("\n🟢 Testing Consolidation Interface:")
        consolidation_interface_results = await ConsolidationIntegrationInterface.handle_clustering_updates(
            clustering_results, document_knowledge_map
        )
        print(f"   - Consolidation interface success: {not consolidation_interface_results.get('error')}")
        
        # Test retriever interface
        print("\n🟡 Testing Retriever Interface:")
        retriever_interface_results = RetrieverIntegrationInterface.search_subject_knowledge(
            "machine learning concepts"
        )
        print(f"   - Retriever interface results: {len(retriever_interface_results)} results")
        
        print("\n✅ Clean Integration Demonstration Completed Successfully!")
        print("=" * 60)
        
        return integration_results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all modules are properly installed and available")
        return None
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.error(f"Integration demonstration failed: {e}")
        return None


async def demonstrate_convenience_function():
    """Demonstrate the convenience function for integrated processing."""
    print("\n🚀 Convenience Function Demonstration")
    print("=" * 50)
    
    try:
        from integration import process_documents_with_integration
        
        # Sample data
        documents = [
            {"document_id": "conv_001", "content": "Artificial intelligence is transforming industries."},
            {"document_id": "conv_002", "content": "Machine learning models require training data."}
        ]
        
        knowledge_map = {
            "conv_001": {"document_id": "conv_001", "concepts": ["AI", "industries"]},
            "conv_002": {"document_id": "conv_002", "concepts": ["ML", "training", "data"]}
        }
        
        # Process using convenience function
        results = await process_documents_with_integration(
            new_documents=documents,
            document_knowledge_map=knowledge_map,
            update_strategy="hybrid"
        )
        
        print(f"✅ Convenience function completed: {results.get('success', False)}")
        print(f"📊 Clustering success: {results.get('clustering_results', {}).get('success', False)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Convenience function failed: {e}")
        return None


if __name__ == "__main__":
    print("Starting LERK System Integration Examples...")
    
    # Run the demonstrations
    asyncio.run(demonstrate_clean_integration())
    asyncio.run(demonstrate_convenience_function())
    
    print("\n🎉 All integration examples completed!")
