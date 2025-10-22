#!/usr/bin/env python3
"""
Example script demonstrating the enrichment module usage.

This script shows how to use the enrichment module to enrich document chunks
with summaries, keywords, questions, and table data.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Any

# Import enrichment module components
from enrichment.enricher import DocumentEnricher, add_enrichment_to_chunk, add_enrichment_to_chunk_async
from enrichment.models import ChunkEnrichment
from enrichment.config import EnrichmentConfig, DEFAULT_ENRICHMENT_CONFIG, FAST_ENRICHMENT_CONFIG, HIGH_QUALITY_ENRICHMENT_CONFIG
from enrichment.utils import process_chunks_concurrently, enrich_and_extract_logic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_chunks() -> List[Any]:
    """
    Create sample chunks for demonstration.
    
    Returns:
        List of sample chunks
    """
    # This is a mock implementation - in real usage, chunks would come from parsing
    class MockChunk:
        def __init__(self, text: str, is_table: bool = False):
            self.text = text
            self.metadata = MockMetadata(is_table)
    
    class MockMetadata:
        def __init__(self, is_table: bool):
            self.is_table = is_table
            if is_table:
                self.text_as_html = "<table><tr><td>Sample</td><td>Data</td></tr></table>"
        
        def to_dict(self):
            result = {"is_table": self.is_table}
            if self.is_table:
                result["text_as_html"] = self.text_as_html
            return result
    
    chunks = [
        MockChunk("This is a sample text chunk about machine learning and artificial intelligence.", False),
        MockChunk("Another text chunk discussing natural language processing and transformer models.", False),
        MockChunk("", True),  # Table chunk
    ]
    
    return chunks


def demonstrate_sync_enrichment():
    """Demonstrate synchronous enrichment processing."""
    print("=== Synchronous Enrichment Demo ===")
    
    # Create sample chunks
    chunks = create_sample_chunks()
    print(f"Created {len(chunks)} sample chunks")
    
    # Create enricher with default config
    enricher = DocumentEnricher()
    
    # Enrich chunks one by one
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            print(f"Enriching chunk {i+1}...")
            enriched_chunk = enricher.enrich_chunk(chunk)
            enriched_chunks.append(enriched_chunk)
            
            # Display enrichment results
            if hasattr(enriched_chunk, 'enrichment'):
                enrichment = enriched_chunk.enrichment
                print(f"  Summary: {enrichment.summary}")
                print(f"  Keywords: {enrichment.keywords}")
                print(f"  Questions: {enrichment.hypothetical_questions}")
                if enrichment.table_summary:
                    print(f"  Table Summary: {enrichment.table_summary}")
                print()
            
        except Exception as e:
            print(f"  Error enriching chunk {i+1}: {e}")
    
    # Get enrichment statistics
    stats = enricher.get_enrichment_stats(enriched_chunks)
    print(f"Enrichment Statistics: {stats}")


async def demonstrate_async_enrichment():
    """Demonstrate asynchronous enrichment processing."""
    print("\n=== Asynchronous Enrichment Demo ===")
    
    # Create sample chunks
    chunks = create_sample_chunks()
    print(f"Created {len(chunks)} sample chunks")
    
    # Create enricher with fast config
    config = FAST_ENRICHMENT_CONFIG
    enricher = DocumentEnricher(config=config)
    
    try:
        # Enrich chunks asynchronously
        print("Enriching chunks asynchronously...")
        enriched_chunks = await enricher.enrich_chunks_async(chunks)
        
        # Display results
        for i, chunk in enumerate(enriched_chunks):
            if hasattr(chunk, 'enrichment'):
                enrichment = chunk.enrichment
                print(f"Chunk {i+1} - Summary: {enrichment.summary}")
        
        # Get statistics
        stats = enricher.get_enrichment_stats(enriched_chunks)
        print(f"Async Enrichment Statistics: {stats}")
        
    except Exception as e:
        print(f"Error in async enrichment: {e}")


def demonstrate_configurations():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Demo ===")
    
    # Default configuration
    default_config = DEFAULT_ENRICHMENT_CONFIG
    print(f"Default config - Model: {default_config.model_name}, Batch size: {default_config.batch_size}")
    
    # Fast configuration
    fast_config = FAST_ENRICHMENT_CONFIG
    print(f"Fast config - Model: {fast_config.model_name}, Batch size: {fast_config.batch_size}")
    
    # High quality configuration
    hq_config = HIGH_QUALITY_ENRICHMENT_CONFIG
    print(f"High quality config - Model: {hq_config.model_name}, Batch size: {hq_config.batch_size}")
    
    # Custom configuration
    custom_config = EnrichmentConfig(
        model_name="gpt-4o",
        temperature=0.1,
        batch_size=3,
        max_keywords=10,
        max_questions=7
    )
    print(f"Custom config - Model: {custom_config.model_name}, Keywords: {custom_config.max_keywords}")


def demonstrate_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n=== Convenience Functions Demo ===")
    
    # Create a sample chunk
    chunks = create_sample_chunks()
    chunk = chunks[0]
    
    try:
        # Use convenience function
        enriched_chunk = add_enrichment_to_chunk(chunk)
        
        if hasattr(enriched_chunk, 'enrichment'):
            enrichment = enriched_chunk.enrichment
            print(f"Convenience function result - Summary: {enrichment.summary}")
        
    except Exception as e:
        print(f"Error with convenience function: {e}")


async def demonstrate_batch_processing():
    """Demonstrate batch processing with concurrent execution."""
    print("\n=== Batch Processing Demo ===")
    
    # Create multiple chunks
    chunks = create_sample_chunks() * 3  # Create more chunks for batch processing
    print(f"Created {len(chunks)} chunks for batch processing")
    
    # Create enricher
    enricher = DocumentEnricher()
    
    try:
        # Process chunks in batches
        enriched_chunks = await process_chunks_concurrently(
            chunks,
            enricher.enrich_chunk_async,
            batch_size=2
        )
        
        print(f"Successfully processed {len(enriched_chunks)} chunks")
        
        # Show statistics
        stats = enricher.get_enrichment_stats(enriched_chunks)
        print(f"Batch processing statistics: {stats}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")


def main():
    """Main function to run all demonstrations."""
    print("LERK System - Enrichment Module Demo")
    print("=" * 50)
    
    try:
        # Run synchronous demo
        demonstrate_sync_enrichment()
        
        # Run configuration demo
        demonstrate_configurations()
        
        # Run convenience functions demo
        demonstrate_convenience_functions()
        
        # Run async demo
        asyncio.run(demonstrate_async_enrichment())
        
        # Run batch processing demo
        asyncio.run(demonstrate_batch_processing())
        
        print("\n=== Demo Complete ===")
        print("All enrichment demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error in main demo: {e}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    main()
