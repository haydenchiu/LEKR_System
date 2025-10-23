"""
Core logic extraction functionality.

This module provides the main LogicExtractor class and convenience functions for
extracting logical and causal structure from document chunks using LLMs.
"""

from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from unstructured.documents.elements import Element

from .models import LogicExtractionSchemaLiteChunk
from .prompts import generate_logic_extraction_prompt
from .config import LogicExtractionConfig, DEFAULT_LOGIC_EXTRACTION_CONFIG
from .exceptions import (
    LLMInvocationError, 
    InvalidChunkError, 
    MissingAPIKeyError, 
    ChunkProcessingError,
    LogicExtractionError
)


def _initialize_llm(config: LogicExtractionConfig) -> Runnable:
    """Initializes the LLM with structured output."""
    try:
        return ChatOpenAI(
            model=config.model_name, 
            temperature=config.temperature
        ).with_structured_output(LogicExtractionSchemaLiteChunk)
    except Exception as e:
        raise MissingAPIKeyError(f"Failed to initialize LLM. Ensure OPENAI_API_KEY is set: {e}")


def add_logic_extraction_to_chunk(
    chunk: Element, 
    config: LogicExtractionConfig = DEFAULT_LOGIC_EXTRACTION_CONFIG
) -> Element:
    """Add logic extraction to a chunk that follows the LogicExtractionSchemaLiteChunk schema."""
    if not isinstance(chunk, Element):
        raise InvalidChunkError("Input must be an Unstructured Element.")

    logic_extraction_llm = _initialize_llm(config)
    is_table = "text_as_html" in chunk.metadata.to_dict()
    content = chunk.text if not is_table else chunk.metadata.text_as_html

    try:
        logic = logic_extraction_llm.invoke(
            generate_logic_extraction_prompt(content, is_table), 
            config={"timeout": config.timeout}
        )
        chunk.logic = logic
        return chunk
    except Exception as e:
        raise LLMInvocationError(f"Failed to extract logic from chunk with LLM: {e}")


async def add_logic_extraction_to_chunk_async(
    chunk: Element, 
    config: LogicExtractionConfig = DEFAULT_LOGIC_EXTRACTION_CONFIG
) -> Element:
    """Add logic extraction to a chunk asynchronously."""
    if not isinstance(chunk, Element):
        raise InvalidChunkError("Input must be an Unstructured Element.")

    logic_extraction_llm = _initialize_llm(config)
    is_table = "text_as_html" in chunk.metadata.to_dict()
    content = chunk.text if not is_table else chunk.metadata.text_as_html

    try:
        logic = await logic_extraction_llm.ainvoke(
            generate_logic_extraction_prompt(content, is_table), 
            config={"timeout": config.timeout}
        )
        chunk.logic = logic
        return chunk
    except Exception as e:
        raise LLMInvocationError(f"Failed to extract logic from chunk with LLM asynchronously: {e}")


class LogicExtractor:
    """
    Orchestrates the extraction of logical and causal structure from document chunks using LLMs.
    """
    
    def __init__(
        self, 
        config: LogicExtractionConfig = DEFAULT_LOGIC_EXTRACTION_CONFIG,
        llm: Optional[Runnable] = None
    ):
        """
        Initialize the LogicExtractor.
        
        Args:
            config: Configuration for the logic extraction process
            llm: Optional pre-configured LLM instance
        """
        self.config = config
        self.llm = llm or self._create_llm()
    
    def _create_llm(self) -> Runnable:
        """Initializes the LLM with structured output."""
        try:
            return ChatOpenAI(
                model=self.config.model_name, 
                temperature=self.config.temperature
            ).with_structured_output(LogicExtractionSchemaLiteChunk)
        except Exception as e:
            raise LogicExtractionError(f"Failed to create LLM: {e}")
    
    def extract_logic(self, chunk: Element) -> Element:
        """
        Extract logical structure from a single document chunk synchronously.
        
        Args:
            chunk: Document chunk to extract logic from
            
        Returns:
            Chunk with logic extraction results attached
            
        Raises:
            InvalidChunkError: If chunk is not a valid Element
            ChunkProcessingError: If logic extraction fails
        """
        if not isinstance(chunk, Element):
            raise InvalidChunkError("Input must be an Unstructured Element.")
        
        try:
            is_table = "text_as_html" in chunk.metadata.to_dict()
            content = chunk.text if not is_table else chunk.metadata.text_as_html
            
            logic = self.llm.invoke(
                generate_logic_extraction_prompt(content, is_table), 
                config={"timeout": self.config.timeout}
            )
            chunk.logic = logic
            return chunk
        except Exception as e:
            raise ChunkProcessingError(f"Failed to extract logic from chunk: {e}")
    
    async def extract_logic_async(self, chunk: Element) -> Element:
        """
        Extract logical structure from a single document chunk asynchronously.
        
        Args:
            chunk: Document chunk to extract logic from
            
        Returns:
            Chunk with logic extraction results attached
            
        Raises:
            InvalidChunkError: If chunk is not a valid Element
            ChunkProcessingError: If logic extraction fails
        """
        if not isinstance(chunk, Element):
            raise InvalidChunkError("Input must be an Unstructured Element.")
        
        try:
            is_table = "text_as_html" in chunk.metadata.to_dict()
            content = chunk.text if not is_table else chunk.metadata.text_as_html
            
            logic = await self.llm.ainvoke(
                generate_logic_extraction_prompt(content, is_table), 
                config={"timeout": self.config.timeout}
            )
            chunk.logic = logic
            return chunk
        except Exception as e:
            raise ChunkProcessingError(f"Failed to extract logic from chunk asynchronously: {e}")
    
    def extract_logic_batch(self, chunks: List[Element]) -> List[Element]:
        """
        Extract logical structure from a list of document chunks synchronously.
        
        Args:
            chunks: List of document chunks to extract logic from
            
        Returns:
            List of chunks with logic extraction results attached
        """
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                processed_chunks.append(self.extract_logic(chunk))
            except ChunkProcessingError as e:
                print(f"Error extracting logic from chunk {i+1}: {e}")
                processed_chunks.append(chunk)  # Return original chunk on error
        return processed_chunks
    
    async def extract_logic_batch_async(self, chunks: List[Element]) -> List[Element]:
        """
        Extract logical structure from a list of document chunks asynchronously using concurrent processing.
        
        Args:
            chunks: List of document chunks to extract logic from
            
        Returns:
            List of chunks with logic extraction results attached
        """
        # Always use async processing for batch async operations
        
        from .utils import process_chunks_concurrently
        
        return await process_chunks_concurrently(
            chunks, 
            self.extract_logic_async, 
            batch_size=self.config.batch_size
        )
    
    def get_extraction_stats(self, chunks: List[Element]) -> Dict[str, Any]:
        """
        Get statistics about logic extraction results.
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            Dictionary with extraction statistics
        """
        total_chunks = len(chunks)
        chunks_with_logic = sum(1 for chunk in chunks if hasattr(chunk, 'logic') and chunk.logic is not None)
        
        total_claims = 0
        total_relations = 0
        total_assumptions = 0
        total_constraints = 0
        total_questions = 0
        
        for chunk in chunks:
            if hasattr(chunk, 'logic') and chunk.logic is not None:
                logic = chunk.logic
                total_claims += len(logic.claims)
                total_relations += len(logic.logical_relations)
                total_assumptions += len(logic.assumptions or [])
                total_constraints += len(logic.constraints or [])
                total_questions += len(logic.open_questions or [])
        
        return {
            "total_chunks": total_chunks,
            "chunks_with_logic": chunks_with_logic,
            "extraction_rate": chunks_with_logic / total_chunks if total_chunks > 0 else 0.0,
            "total_claims": total_claims,
            "total_relations": total_relations,
            "total_assumptions": total_assumptions,
            "total_constraints": total_constraints,
            "total_questions": total_questions,
            "avg_claims_per_chunk": total_claims / chunks_with_logic if chunks_with_logic > 0 else 0.0,
            "avg_relations_per_chunk": total_relations / chunks_with_logic if chunks_with_logic > 0 else 0.0
        }
