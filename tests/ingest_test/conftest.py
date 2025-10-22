"""
Pytest configuration and fixtures for ingestion module tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from unstructured.documents.elements import Element, Text, Title, Table


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """
    This is a sample document for testing.
    
    It contains multiple paragraphs with different content.
    
    The document has various sections and subsections.
    
    This is another paragraph with more content.
    """


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing."""
    return """
    <html>
    <body>
        <h1>Sample Document</h1>
        <p>This is a paragraph with some content.</p>
        <table>
            <tr><th>Header 1</th><th>Header 2</th></tr>
            <tr><td>Data 1</td><td>Data 2</td></tr>
        </table>
        <p>Another paragraph after the table.</p>
    </body>
    </html>
    """


@pytest.fixture
def sample_elements():
    """Sample unstructured elements for testing."""
    elements = []
    
    # Add text elements
    elements.append(Text("This is a sample text element."))
    elements.append(Title("Sample Title"))
    elements.append(Text("This is another text element with more content."))
    
    # Add table element
    table_html = "<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>"
    table_element = Table(text="Header\nData")
    table_element.metadata.text_as_html = table_html
    elements.append(table_element)
    
    return elements


@pytest.fixture
def sample_chunks():
    """Sample chunked elements for testing."""
    chunks = []
    
    # Text chunk
    text_chunk = Text("This is a sample text chunk with some content.")
    text_chunk.metadata.filetype = "text/plain"
    chunks.append(text_chunk)
    
    # Table chunk
    table_chunk = Table(text="Header\nData")
    table_chunk.metadata.text_as_html = "<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>"
    table_chunk.metadata.filetype = "text/html"
    chunks.append(table_chunk)
    
    return chunks


@pytest.fixture
def temp_file(sample_text_content):
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    # For testing purposes, we'll create a simple text file with .pdf extension
    # In real tests, you might want to use actual PDF files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        f.write("Sample PDF content")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_parser():
    """Mock parser for testing."""
    parser = Mock()
    parser.parse_file.return_value = []
    parser.parse_url.return_value = []
    parser.parse_text.return_value = []
    parser.get_file_type.return_value = "text/plain"
    return parser


@pytest.fixture
def mock_chunker():
    """Mock chunker for testing."""
    chunker = Mock()
    chunker.chunk_elements.return_value = []
    chunker.get_chunk_statistics.return_value = {
        "total_chunks": 0,
        "text_chunks": 0,
        "table_chunks": 0
    }
    return chunker


@pytest.fixture
def sample_ingestion_result():
    """Sample ingestion result for testing."""
    return {
        "file_path": "test.txt",
        "elements": [],
        "chunks": [],
        "statistics": {
            "total_chunks": 5,
            "text_chunks": 4,
            "table_chunks": 1
        },
        "success": True
    }


@pytest.fixture
def sample_ingestion_results():
    """Sample multiple ingestion results for testing."""
    return [
        {
            "file_path": "test1.txt",
            "elements": [],
            "chunks": [],
            "statistics": {
                "total_chunks": 3,
                "text_chunks": 3,
                "table_chunks": 0
            },
            "success": True
        },
        {
            "file_path": "test2.html",
            "elements": [],
            "chunks": [],
            "statistics": {
                "total_chunks": 2,
                "text_chunks": 1,
                "table_chunks": 1
            },
            "success": True
        },
        {
            "file_path": "test3.pdf",
            "elements": [],
            "chunks": [],
            "statistics": {
                "total_chunks": 0,
                "text_chunks": 0,
                "table_chunks": 0
            },
            "success": False,
            "error": "File not found"
        }
    ]


@pytest.fixture
def mock_partition():
    """Mock the unstructured partition function."""
    def _mock_partition(*args, **kwargs):
        # Return sample elements based on input
        if 'filename' in kwargs:
            # File parsing
            elements = [
                Text("Sample text from file"),
                Title("Sample Title")
            ]
        elif 'url' in kwargs:
            # URL parsing
            elements = [
                Text("Sample text from URL"),
                Title("URL Title")
            ]
        else:
            # Text parsing
            elements = [
                Text("Sample text content")
            ]
        
        return elements
    
    return _mock_partition


@pytest.fixture
def mock_chunk_by_title():
    """Mock the chunk_by_title function."""
    def _mock_chunk_by_title(elements, **kwargs):
        # Return chunks based on input elements
        chunks = []
        for i, element in enumerate(elements):
            chunk = Text(f"Chunk {i}: {element.text if hasattr(element, 'text') else str(element)}")
            chunk.metadata = element.metadata
            chunks.append(chunk)
        return chunks
    
    return _mock_chunk_by_title
