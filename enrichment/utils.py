"""
Utility functions for document enrichment.

Provides helper functions for chunk processing, content extraction,
and batch processing operations.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Awaitable, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def is_table_chunk(chunk: Any) -> bool:
    """
    Check if a chunk is a table chunk.
    
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
    Get metadata from a chunk.
    
    Args:
        chunk: Document chunk
        
    Returns:
        Dictionary of chunk metadata
    """
    try:
        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'to_dict'):
            return chunk.metadata.to_dict()
        elif hasattr(chunk, 'metadata'):
            return dict(chunk.metadata)
        else:
            return {}
    except Exception as e:
        logger.warning(f"Error getting chunk metadata: {e}")
        return {}


def is_enriched_chunk(chunk: Any) -> bool:
    """
    Check if a chunk has been enriched.
    
    Args:
        chunk: Document chunk to check
        
    Returns:
        True if chunk has enrichment data, False otherwise
    """
    return hasattr(chunk, 'enrichment') and chunk.enrichment is not None


def get_enrichment_data(chunk: Any) -> Dict[str, Any]:
    """
    Get enrichment data from a chunk.
    
    Args:
        chunk: Document chunk
        
    Returns:
        Dictionary of enrichment data
    """
    if is_enriched_chunk(chunk):
        if hasattr(chunk.enrichment, 'model_dump'):
            return chunk.enrichment.model_dump()
        elif hasattr(chunk.enrichment, 'dict'):
            return chunk.enrichment.dict()
        else:
            return dict(chunk.enrichment)
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
    results = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        try:
            processed = await asyncio.gather(*(func(chunk) for chunk in batch))
            results.extend(processed)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add original chunks if processing fails
            results.extend(batch)
    
    return results


async def enrich_and_extract_logic(chunk: Any, enricher: Any = None, logic_extractor: Any = None) -> Any:
    """
    Enrich a chunk and extract logic from it.
    
    Args:
        chunk: Document chunk to process
        enricher: Document enricher instance
        logic_extractor: Logic extractor instance
        
    Returns:
        Processed chunk with enrichment and logic data
    """
    try:
        # First enrich the chunk
        if enricher:
            chunk = await enricher.enrich_chunk_async(chunk)
        else:
            from .enricher import add_enrichment_to_chunk_async
            chunk = await add_enrichment_to_chunk_async(chunk)
        
        # Then extract logic (if logic extractor is available)
        if logic_extractor:
            chunk = await logic_extractor.extract_logic_async(chunk)
        
        return chunk
        
    except Exception as e:
        logger.error(f"Error in enrich_and_extract_logic: {e}")
        return chunk


def validate_chunk(chunk: Any) -> bool:
    """
    Validate that a chunk has the required attributes for enrichment.
    
    Args:
        chunk: Document chunk to validate
        
    Returns:
        True if chunk is valid, False otherwise
    """
    try:
        # Check if chunk has text content
        content = get_chunk_content(chunk)
        if not content or len(content.strip()) == 0:
            return False
        
        # Check if chunk has metadata
        metadata = get_chunk_metadata(chunk)
        if not metadata:
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating chunk: {e}")
        return False


def filter_valid_chunks(chunks: List[Any]) -> List[Any]:
    """
    Filter out invalid chunks from a list.
    
    Args:
        chunks: List of chunks to filter
        
    Returns:
        List of valid chunks
    """
    return [chunk for chunk in chunks if validate_chunk(chunk)]


def get_chunk_statistics(chunks: List[Any]) -> Dict[str, Any]:
    """
    Get statistics about a list of chunks.
    
    Args:
        chunks: List of chunks to analyze
        
    Returns:
        Dictionary with chunk statistics
    """
    total_chunks = len(chunks)
    table_chunks = sum(1 for chunk in chunks if is_table_chunk(chunk))
    text_chunks = total_chunks - table_chunks
    enriched_chunks = sum(1 for chunk in chunks if is_enriched_chunk(chunk))
    
    # Calculate content lengths
    content_lengths = []
    for chunk in chunks:
        try:
            content = get_chunk_content(chunk)
            content_lengths.append(len(content))
        except:
            content_lengths.append(0)
    
    avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    max_content_length = max(content_lengths) if content_lengths else 0
    min_content_length = min(content_lengths) if content_lengths else 0
    
    return {
        "total_chunks": total_chunks,
        "table_chunks": table_chunks,
        "text_chunks": text_chunks,
        "enriched_chunks": enriched_chunks,
        "enrichment_rate": enriched_chunks / total_chunks if total_chunks > 0 else 0,
        "avg_content_length": avg_content_length,
        "max_content_length": max_content_length,
        "min_content_length": min_content_length
    }
