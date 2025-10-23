"""
Example usage of the logic_extractor module.

This module demonstrates how to use the logic extraction functionality
to extract logical and causal structure from document chunks.
"""

import asyncio
import logging

# Import logic_extractor module components
from logic_extractor.extractor import LogicExtractor
from logic_extractor.config import (
    DEFAULT_LOGIC_EXTRACTION_CONFIG,
    FAST_LOGIC_EXTRACTION_CONFIG
)
from unstructured.documents.elements import Text, Table, Element
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_chunks():
    """Creates sample unstructured elements for testing."""
    text_element = Text(
        text="This is a sample text chunk about AI. It discusses machine learning and neural networks."
    )
    table_html = (
        "<table><thead><tr><th>Header 1</th><th>Header 2</th></tr></thead>"
        "<tbody><tr><td>Row 1 Col 1</td><td>Row 1 Col 2</td></tr></tbody></table>"
    )
    table_element = Table(
        text="Sample table summary.",
        metadata={"text_as_html": table_html}
    )

    # Simulate a more complex chunking scenario
    elements = partition(
        filename=str(__file__).replace("example.py", "sample_document.txt")
    )
    chunks = chunk_by_title(elements)

    return [text_element, table_element, chunks[0] if chunks else text_element]


async def demonstrate_async_logic_extraction():
    """Demonstrate asynchronous logic extraction."""
    logger.info("Demonstrating asynchronous logic extraction...")
    chunks = create_sample_chunks()

    extractor = LogicExtractor(config=FAST_LOGIC_EXTRACTION_CONFIG)
    extracted_chunks = await extractor.extract_logic_from_chunks_async(chunks)

    for i, chunk in enumerate(extracted_chunks):
        logger.info(f"--- Extracted Logic for Chunk {i+1} ---")
        if hasattr(chunk, 'logic') and chunk.logic:
            logger.info(f"Claims: {len(chunk.logic.claims)}")
            logger.info(f"Relations: {len(chunk.logic.logical_relations)}")
            if chunk.logic.claims:
                logger.info(f"Sample Claim: {chunk.logic.claims[0].statement}")
        else:
            logger.info("No logic extracted.")


def demonstrate_sync_logic_extraction():
    """Demonstrate synchronous logic extraction."""
    logger.info("Demonstrating synchronous logic extraction...")
    chunks = create_sample_chunks()

    extractor = LogicExtractor(config=DEFAULT_LOGIC_EXTRACTION_CONFIG)
    extracted_chunks = extractor.extract_logic_from_chunks(chunks)

    for i, chunk in enumerate(extracted_chunks):
        logger.info(f"--- Extracted Logic for Chunk {i+1} ---")
        if hasattr(chunk, 'logic') and chunk.logic:
            logger.info(f"Claims: {len(chunk.logic.claims)}")
            logger.info(f"Relations: {len(chunk.logic.logical_relations)}")
            if chunk.logic.claims:
                logger.info(f"Sample Claim: {chunk.logic.claims[0].statement}")
        else:
            logger.info("No logic extracted.")


async def main():
    """Main function to run the examples."""
    logger.info("Starting logic_extractor example...")
    demonstrate_sync_logic_extraction()
    await demonstrate_async_logic_extraction()
    logger.info("Logic extraction example finished.")


if __name__ == "__main__":
    asyncio.run(main())