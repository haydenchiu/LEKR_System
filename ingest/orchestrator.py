"""
Document ingestion orchestrator for the LERK System.

Coordinates the parsing and chunking of documents.
"""

import logging
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

from .parsing import DocumentParser
from .chunking import DocumentChunker
from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)


class DocumentIngestionOrchestrator:
    """Orchestrates the document ingestion pipeline."""
    
    def __init__(self,
                 # Parsing parameters
                 parsing_strategy: str = "hi_res",
                 skip_infer_table_types: Optional[List[str]] = None,
                 max_partition: Optional[int] = None,
                 # Chunking parameters
                 max_characters: int = 2048,
                 combine_text_under_n_chars: int = 256,
                 new_after_n_chars: int = 1800):
        """
        Initialize the document ingestion orchestrator.
        
        Args:
            parsing_strategy: Strategy for document parsing
            skip_infer_table_types: Table types to skip during parsing
            max_partition: Maximum number of partitions to process
            max_characters: Maximum size of chunks
            combine_text_under_n_chars: Combine small text elements under this size
            new_after_n_chars: Start new chunk if current exceeds this size
        """
        self.parser = DocumentParser(
            strategy=parsing_strategy,
            skip_infer_table_types=skip_infer_table_types,
            max_partition=max_partition
        )
        self.chunker = DocumentChunker(
            max_characters=max_characters,
            combine_text_under_n_chars=combine_text_under_n_chars,
            new_after_n_chars=new_after_n_chars
        )
    
    def ingest_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Ingest a single file through the complete pipeline.
        
        Args:
            file_path: Path to the file to ingest
            
        Returns:
            Dictionary containing ingestion results
        """
        try:
            logger.info(f"Starting ingestion of file: {file_path}")
            
            # Parse the file
            elements = self.parser.parse_file(file_path)
            
            # Chunk the elements
            chunks = self.chunker.chunk_elements(elements)
            
            # Get statistics
            stats = self.chunker.get_chunk_statistics(chunks)
            
            result = {
                "file_path": str(file_path),
                "elements": elements,
                "chunks": chunks,
                "statistics": stats,
                "success": True
            }
            
            logger.info(f"Successfully ingested {file_path}: {stats}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "elements": [],
                "chunks": [],
                "statistics": {"total_chunks": 0, "text_chunks": 0, "table_chunks": 0},
                "success": False,
                "error": str(e)
            }
    
    def ingest_url(self, url: str) -> Dict[str, Any]:
        """
        Ingest content from a URL.
        
        Args:
            url: URL to ingest
            
        Returns:
            Dictionary containing ingestion results
        """
        try:
            logger.info(f"Starting ingestion of URL: {url}")
            
            # Parse the URL
            elements = self.parser.parse_url(url)
            
            # Chunk the elements
            chunks = self.chunker.chunk_elements(elements)
            
            # Get statistics
            stats = self.chunker.get_chunk_statistics(chunks)
            
            result = {
                "url": url,
                "elements": elements,
                "chunks": chunks,
                "statistics": stats,
                "success": True
            }
            
            logger.info(f"Successfully ingested {url}: {stats}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest {url}: {e}")
            return {
                "url": url,
                "elements": [],
                "chunks": [],
                "statistics": {"total_chunks": 0, "text_chunks": 0, "table_chunks": 0},
                "success": False,
                "error": str(e)
            }
    
    def ingest_text(self, text: str, filetype: str = "text/plain") -> Dict[str, Any]:
        """
        Ingest text content directly.
        
        Args:
            text: Text content to ingest
            filetype: MIME type of the content
            
        Returns:
            Dictionary containing ingestion results
        """
        try:
            logger.info(f"Starting ingestion of text content")
            
            # Parse the text
            elements = self.parser.parse_text(text, filetype)
            
            # Chunk the elements
            chunks = self.chunker.chunk_elements(elements)
            
            # Get statistics
            stats = self.chunker.get_chunk_statistics(chunks)
            
            result = {
                "content_type": "text",
                "filetype": filetype,
                "elements": elements,
                "chunks": chunks,
                "statistics": stats,
                "success": True
            }
            
            logger.info(f"Successfully ingested text content: {stats}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest text content: {e}")
            return {
                "content_type": "text",
                "filetype": filetype,
                "elements": [],
                "chunks": [],
                "statistics": {"total_chunks": 0, "text_chunks": 0, "table_chunks": 0},
                "success": False,
                "error": str(e)
            }
    
    def ingest_multiple_files(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Ingest multiple files.
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            List of ingestion results
        """
        results = []
        for file_path in file_paths:
            result = self.ingest_file(file_path)
            results.append(result)
        return results
    
    def get_supported_file_types(self) -> List[str]:
        """
        Get list of supported file types.
        
        Returns:
            List of supported MIME types
        """
        return [
            "application/pdf",
            "text/html",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/csv",
            "application/json",
            "application/xml"
        ]
