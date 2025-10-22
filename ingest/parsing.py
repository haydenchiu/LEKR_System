"""
Document parsing module for the LERK System.

Handles parsing of various document types using the unstructured library.
"""

import logging
from typing import List, Optional, Union
from pathlib import Path
from urllib.parse import urlparse

from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import Element
from .exceptions import FileNotFoundError, ParsingError

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parser for various document types using unstructured library."""
    
    def __init__(self, 
                 strategy: str = "hi_res",
                 skip_infer_table_types: Optional[List[str]] = None,
                 max_partition: Optional[int] = None):
        """
        Initialize the document parser.
        
        Args:
            strategy: Parsing strategy for the unstructured library
            skip_infer_table_types: List of table types to skip inference for
            max_partition: Maximum number of partitions to process
        """
        self.strategy = strategy
        self.skip_infer_table_types = skip_infer_table_types or []
        self.max_partition = max_partition
    
    def parse_file(self, file_path: Union[str, Path]) -> List[Element]:
        """
        Parse a file and return a list of table-aware elements.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of parsed elements
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If parsing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            logger.info(f"Parsing file: {file_path}")
            elements = partition(
                filename=str(file_path),
                skip_infer_table_types=self.skip_infer_table_types,
                strategy=self.strategy,
                max_partition=self.max_partition
            )
            logger.info(f"Successfully parsed {len(elements)} elements from {file_path}")
            return elements
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            raise
    
    def parse_url(self, url: str) -> List[Element]:
        """
        Parse content from a URL.
        
        Args:
            url: URL to parse
            
        Returns:
            List of parsed elements
        """
        try:
            logger.info(f"Parsing URL: {url}")
            elements = partition(url=url)
            logger.info(f"Successfully parsed {len(elements)} elements from {url}")
            return elements
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            raise
    
    def parse_text(self, text: str, filetype: str = "text/plain") -> List[Element]:
        """
        Parse text content directly.
        
        Args:
            text: Text content to parse
            filetype: MIME type of the content
            
        Returns:
            List of parsed elements
        """
        try:
            logger.info(f"Parsing text content of type: {filetype}")
            elements = partition_text(text=text)
            logger.info(f"Successfully parsed {len(elements)} elements from text")
            return elements
        except Exception as e:
            logger.error(f"Error parsing text: {e}")
            raise
    
    def get_file_type(self, file_path: Union[str, Path]) -> str:
        """
        Get the MIME type of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        import mimetypes
        file_path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"


def parse_file(file_path: Union[str, Path], 
               strategy: str = "hi_res",
               skip_infer_table_types: Optional[List[str]] = None,
               max_partition: Optional[int] = None) -> List[Element]:
    """
    Convenience function to parse a file.
    
    Args:
        file_path: Path to the file to parse
        strategy: Parsing strategy
        skip_infer_table_types: List of table types to skip
        max_partition: Maximum number of partitions
        
    Returns:
        List of parsed elements
    """
    parser = DocumentParser(
        strategy=strategy,
        skip_infer_table_types=skip_infer_table_types,
        max_partition=max_partition
    )
    return parser.parse_file(file_path)
