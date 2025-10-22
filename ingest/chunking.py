"""
Document chunking module for the LERK System.

Handles chunking of parsed document elements into manageable pieces.
"""

import logging
from typing import List, Optional
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunker for document elements using title-based chunking."""
    
    def __init__(self,
                 max_characters: int = 2048,
                 combine_text_under_n_chars: int = 256,
                 new_after_n_chars: int = 1800):
        """
        Initialize the document chunker.
        
        Args:
            max_characters: Maximum size of a chunk
            combine_text_under_n_chars: Combine small text elements under this size
            new_after_n_chars: Start a new chunk if current one exceeds this size
        """
        self.max_characters = max_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_chars = new_after_n_chars
    
    def chunk_elements(self, elements: List[Element]) -> List[Element]:
        """
        Convert a list of elements to a list of chunks.
        
        Args:
            elements: List of parsed document elements
            
        Returns:
            List of chunked elements
        """
        try:
            logger.info(f"Chunking {len(elements)} elements")
            chunks = chunk_by_title(
                elements,
                max_characters=self.max_characters,
                combine_text_under_n_chars=self.combine_text_under_n_chars,
                new_after_n_chars=self.new_after_n_chars
            )
            logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking elements: {e}")
            raise
    
    def get_chunk_statistics(self, chunks: List[Element]) -> dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunked elements
            
        Returns:
            Dictionary with chunk statistics
        """
        text_chunks = 0
        table_chunks = 0
        
        for chunk in chunks:
            if "text_as_html" in chunk.metadata.to_dict():
                table_chunks += 1
            else:
                text_chunks += 1
        
        return {
            "total_chunks": len(chunks),
            "text_chunks": text_chunks,
            "table_chunks": table_chunks
        }


def elements_to_chunks(elements: List[Element], 
                      max_characters: int = 2048,
                      combine_text_under_n_chars: int = 256,
                      new_after_n_chars: int = 1800) -> List[Element]:
    """
    Convenience function to chunk elements.
    
    Args:
        elements: List of parsed document elements
        max_characters: Maximum size of a chunk
        combine_text_under_n_chars: Combine small text elements under this size
        new_after_n_chars: Start a new chunk if current one exceeds this size
        
    Returns:
        List of chunked elements
    """
    chunker = DocumentChunker(
        max_characters=max_characters,
        combine_text_under_n_chars=combine_text_under_n_chars,
        new_after_n_chars=new_after_n_chars
    )
    return chunker.chunk_elements(elements)
