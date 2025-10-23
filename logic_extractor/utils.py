"""
Utility functions for logic processing and extraction.

This module provides utility functions for processing chunks, validating logic extraction
results, and performing various operations on extracted logical structures.
"""

import asyncio
import logging
from typing import List, Callable, Any, Dict, Awaitable
from .extractor import LogicExtractor
from .config import LogicExtractionConfig, DEFAULT_LOGIC_EXTRACTION_CONFIG

logger = logging.getLogger(__name__)


def is_table_chunk(chunk: Any) -> bool:
    """
    Check if a chunk is a table chunk based on its metadata.

    Args:
        chunk: Document chunk to check

    Returns:
        True if chunk is a table, False otherwise
    """
    try:
        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'to_dict'):
            metadata = chunk.metadata.to_dict()
            return "text_as_html" in metadata
        return False
    except Exception as e:
        logger.warning(f"Error checking if chunk is table: {e}")
        return False


def get_chunk_content(chunk: Any, is_table: bool = None) -> str:
    """
    Get the content of a chunk, handling both text and table chunks.

    Args:
        chunk: Document chunk
        is_table: Whether chunk is a table (optional, will be determined if not provided)

    Returns:
        Chunk content as string
    """
    try:
        if is_table is None:
            is_table = is_table_chunk(chunk)

        if is_table:
            # For table chunks, get HTML content
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'text_as_html'):
                return chunk.metadata.text_as_html
            elif hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'to_dict'):
                metadata = chunk.metadata.to_dict()
                return metadata.get('text_as_html', '')
        else:
            # For text chunks, get text content
            if hasattr(chunk, 'text') and chunk.text is not None:
                return chunk.text
            elif hasattr(chunk, 'content') and chunk.content is not None:
                return chunk.content
            else:
                # Fallback for text chunks
                return str(chunk)

        # Fallback
        return str(chunk)

    except Exception as e:
        logger.warning(f"Error getting chunk content: {e}")
        return str(chunk)


def get_chunk_metadata(chunk: Any) -> Dict[str, Any]:
    """
    Extract metadata from a chunk.

    Args:
        chunk: Document chunk

    Returns:
        Dictionary of metadata
    """
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'to_dict'):
        return chunk.metadata.to_dict()
    elif hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
        return chunk.metadata
    return {}


def is_logic_extracted_chunk(chunk: Any) -> bool:
    """
    Check if a chunk has had logic extracted.

    Args:
        chunk: Document chunk to check

    Returns:
        True if chunk has logic data, False otherwise
    """
    return hasattr(chunk, 'logic') and chunk.logic is not None


def get_logic_data(chunk: Any) -> Dict[str, Any]:
    """
    Get the logic data from a chunk.

    Args:
        chunk: Logic-extracted document chunk

    Returns:
        Dictionary of logic data
    """
    if is_logic_extracted_chunk(chunk):
        if hasattr(chunk.logic, 'model_dump'):
            return chunk.logic.model_dump()
        elif hasattr(chunk.logic, 'dict'):
            return chunk.logic.dict()
        else:
            return dict(chunk.logic)
    return {}


async def process_chunks_concurrently(
    chunks: List[Any],
    func: Callable[[Any], Awaitable[Any]],
    batch_size: int = 5
) -> List[Any]:
    """
    Process chunks concurrently in batches.

    Args:
        chunks: List of chunks to process
        func: Async function to apply to each chunk
        batch_size: Number of chunks to process in each batch

    Returns:
        List of processed chunks
    """
    processed_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tasks = [func(chunk) for chunk in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for j, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"Error processing chunk {i+j}: {res}")
                processed_chunks.append(batch[j])  # Append original chunk on error
            else:
                processed_chunks.append(res)
    return processed_chunks


async def extract_logic_from_chunk(
    chunk: Any,
    logic_extraction_config: LogicExtractionConfig = DEFAULT_LOGIC_EXTRACTION_CONFIG,
) -> Any:
    """
    Orchestrates the logic extraction for a single chunk.
    """
    extractor = LogicExtractor(config=logic_extraction_config)
    extracted_chunk = await extractor.extract_logic_from_chunk_async(chunk)
    return extracted_chunk