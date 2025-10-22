"""
Utility functions for the ingestion module.
"""

import logging
from typing import List, Dict, Any, Union
from pathlib import Path
import mimetypes

from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)


def get_file_type(file_path: Union[str, Path]) -> str:
    """
    Get the MIME type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    file_path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


def is_supported_file_type(file_path: Union[str, Path]) -> bool:
    """
    Check if a file type is supported for parsing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file type is supported
    """
    supported_types = {
        "application/pdf",
        "text/html",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/csv",
        "application/json",
        "application/xml",
        "text/markdown"
    }
    
    file_type = get_file_type(file_path)
    return file_type in supported_types


def get_chunk_content(chunk: Element) -> str:
    """
    Get the content of a chunk, handling both text and table chunks.
    
    Args:
        chunk: Document chunk element
        
    Returns:
        Content string
    """
    metadata = chunk.metadata.to_dict()
    
    if "text_as_html" in metadata:
        # This is a table chunk
        return metadata["text_as_html"]
    else:
        # This is a text chunk
        return getattr(chunk, "text", "")


def is_table_chunk(chunk: Element) -> bool:
    """
    Check if a chunk is a table chunk.
    
    Args:
        chunk: Document chunk element
        
    Returns:
        True if chunk is a table
    """
    metadata = chunk.metadata.to_dict()
    return "text_as_html" in metadata


def get_chunk_metadata(chunk: Element) -> Dict[str, Any]:
    """
    Get metadata from a chunk.
    
    Args:
        chunk: Document chunk element
        
    Returns:
        Dictionary of metadata
    """
    metadata = chunk.metadata.to_dict()
    
    # Clean up metadata
    cleaned_metadata = {}
    for key, value in metadata.items():
        if key != "orig_elements":  # Skip large binary data
            cleaned_metadata[key] = value
    
    # Add chunk type information
    cleaned_metadata["is_table"] = is_table_chunk(chunk)
    
    return cleaned_metadata


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """
    Validate and normalize a file path.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Normalized Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    return path.resolve()


def get_ingestion_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get a summary of ingestion results.
    
    Args:
        results: List of ingestion results
        
    Returns:
        Summary dictionary
    """
    total_files = len(results)
    successful_files = sum(1 for r in results if r.get("success", False))
    failed_files = total_files - successful_files
    
    total_chunks = sum(r.get("statistics", {}).get("total_chunks", 0) for r in results)
    total_text_chunks = sum(r.get("statistics", {}).get("text_chunks", 0) for r in results)
    total_table_chunks = sum(r.get("statistics", {}).get("table_chunks", 0) for r in results)
    
    return {
        "total_files": total_files,
        "successful_files": successful_files,
        "failed_files": failed_files,
        "success_rate": successful_files / total_files if total_files > 0 else 0,
        "total_chunks": total_chunks,
        "total_text_chunks": total_text_chunks,
        "total_table_chunks": total_table_chunks
    }


def log_ingestion_progress(current: int, total: int, file_path: str) -> None:
    """
    Log ingestion progress.
    
    Args:
        current: Current file number
        total: Total number of files
        file_path: Current file being processed
    """
    progress = (current / total) * 100 if total > 0 else 0
    logger.info(f"Processing file {current}/{total} ({progress:.1f}%): {file_path}")
