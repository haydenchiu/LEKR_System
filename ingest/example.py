"""
Example usage of the ingestion module.
"""

import logging
from pathlib import Path

from .orchestrator import DocumentIngestionOrchestrator
from .config import IngestionConfig, DEFAULT_CONFIG, LARGE_DOCUMENT_CONFIG
from .utils import get_ingestion_summary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_file():
    """Example of ingesting a single file."""
    print("=== Single File Ingestion Example ===")
    
    # Create orchestrator with default config
    orchestrator = DocumentIngestionOrchestrator()
    
    # Example file path (replace with actual file)
    file_path = "example.pdf"
    
    if Path(file_path).exists():
        result = orchestrator.ingest_file(file_path)
        
        if result["success"]:
            print(f"✅ Successfully ingested {file_path}")
            print(f"   Elements: {len(result['elements'])}")
            print(f"   Chunks: {result['statistics']['total_chunks']}")
            print(f"   Text chunks: {result['statistics']['text_chunks']}")
            print(f"   Table chunks: {result['statistics']['table_chunks']}")
        else:
            print(f"❌ Failed to ingest {file_path}: {result.get('error', 'Unknown error')}")
    else:
        print(f"File {file_path} not found. Please provide a valid file path.")


def example_multiple_files():
    """Example of ingesting multiple files."""
    print("\n=== Multiple Files Ingestion Example ===")
    
    # Create orchestrator with custom config
    config = LARGE_DOCUMENT_CONFIG
    orchestrator = DocumentIngestionOrchestrator(
        parsing_strategy=config.parsing.strategy,
        max_partition=config.parsing.max_partition,
        max_characters=config.chunking.max_characters,
        combine_text_under_n_chars=config.chunking.combine_text_under_n_chars,
        new_after_n_chars=config.chunking.new_after_n_chars
    )
    
    # Example file paths (replace with actual files)
    file_paths = ["document1.pdf", "document2.html", "document3.txt"]
    
    # Filter existing files
    existing_files = [f for f in file_paths if Path(f).exists()]
    
    if existing_files:
        results = orchestrator.ingest_multiple_files(existing_files)
        summary = get_ingestion_summary(results)
        
        print(f"📊 Ingestion Summary:")
        print(f"   Total files: {summary['total_files']}")
        print(f"   Successful: {summary['successful_files']}")
        print(f"   Failed: {summary['failed_files']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        print(f"   Total chunks: {summary['total_chunks']}")
        print(f"   Text chunks: {summary['total_text_chunks']}")
        print(f"   Table chunks: {summary['total_table_chunks']}")
        
        # Show details for each file
        for result in results:
            if result["success"]:
                print(f"   ✅ {result['file_path']}: {result['statistics']['total_chunks']} chunks")
            else:
                print(f"   ❌ {result['file_path']}: {result.get('error', 'Unknown error')}")
    else:
        print("No existing files found. Please provide valid file paths.")


def example_url_ingestion():
    """Example of ingesting content from a URL."""
    print("\n=== URL Ingestion Example ===")
    
    orchestrator = DocumentIngestionOrchestrator()
    
    # Example URL
    url = "https://example.com/article"
    
    try:
        result = orchestrator.ingest_url(url)
        
        if result["success"]:
            print(f"✅ Successfully ingested URL: {url}")
            print(f"   Elements: {len(result['elements'])}")
            print(f"   Chunks: {result['statistics']['total_chunks']}")
        else:
            print(f"❌ Failed to ingest URL: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error ingesting URL: {e}")


def example_text_ingestion():
    """Example of ingesting text content directly."""
    print("\n=== Text Ingestion Example ===")
    
    orchestrator = DocumentIngestionOrchestrator()
    
    # Example text content
    text_content = """
    This is a sample document for testing the ingestion system.
    
    It contains multiple paragraphs and should be chunked appropriately.
    
    The system should handle various types of content including:
    - Plain text
    - Structured content
    - Multiple paragraphs
    """
    
    result = orchestrator.ingest_text(text_content, "text/plain")
    
    if result["success"]:
        print(f"✅ Successfully ingested text content")
        print(f"   Elements: {len(result['elements'])}")
        print(f"   Chunks: {result['statistics']['total_chunks']}")
        
        # Show first chunk content
        if result["chunks"]:
            first_chunk = result["chunks"][0]
            content = first_chunk.text if hasattr(first_chunk, 'text') else str(first_chunk)
            print(f"   First chunk preview: {content[:100]}...")
    else:
        print(f"❌ Failed to ingest text: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    print("LERK System - Ingestion Module Examples")
    print("=" * 50)
    
    # Run examples
    example_single_file()
    example_multiple_files()
    example_url_ingestion()
    example_text_ingestion()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
