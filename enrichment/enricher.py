"""
Document enrichment processing.

Handles the enrichment of document chunks using LLM-based processing
to add summaries, keywords, questions, and table data.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

from .models import ChunkEnrichment
from .prompts import generate_enrichment_prompt
from .config import EnrichmentConfig
from .utils import is_table_chunk, get_chunk_content
from .exceptions import (
    EnrichmentError,
    LLMInvocationError,
    ChunkProcessingError
)

logger = logging.getLogger(__name__)


class DocumentEnricher:
    """Enriches document chunks with summaries, keywords, questions, and table data."""
    
    def __init__(
        self, 
        config: Optional[EnrichmentConfig] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the document enricher.
        
        Args:
            config: Enrichment configuration
            llm: Language model for enrichment (optional)
        """
        self.config = config or EnrichmentConfig.default()
        self.llm = llm or self._create_llm()
        
    def _create_llm(self) -> BaseLanguageModel:
        """Create the language model for enrichment."""
        try:
            llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.llm_parameters
            )
            return llm.with_structured_output(ChunkEnrichment)
        except Exception as e:
            raise EnrichmentError(f"Failed to create LLM: {e}")
    
    def enrich_chunk(self, chunk: Any) -> Any:
        """
        Enrich a single chunk with summary, keywords, questions, and table data.
        
        Args:
            chunk: Document chunk to enrich
            
        Returns:
            Enriched chunk with enrichment data
            
        Raises:
            ChunkProcessingError: If chunk processing fails
        """
        try:
            is_table = is_table_chunk(chunk)
            content = get_chunk_content(chunk, is_table)
            
            prompt = generate_enrichment_prompt(content, is_table)
            
            logger.debug(f"Enriching chunk (table: {is_table})")
            enrichment = self.llm.invoke(prompt)
            
            # Add enrichment to chunk
            chunk.enrichment = enrichment
            logger.debug(f"Successfully enriched chunk")
            
            return chunk
            
        except Exception as e:
            error_msg = f"Failed to enrich chunk: {e}"
            logger.error(error_msg)
            raise ChunkProcessingError(error_msg) from e
    
    async def enrich_chunk_async(self, chunk: Any) -> Any:
        """
        Asynchronously enrich a single chunk.
        
        Args:
            chunk: Document chunk to enrich
            
        Returns:
            Enriched chunk with enrichment data
            
        Raises:
            ChunkProcessingError: If chunk processing fails
        """
        try:
            is_table = is_table_chunk(chunk)
            content = get_chunk_content(chunk, is_table)
            
            prompt = generate_enrichment_prompt(content, is_table)
            
            logger.debug(f"Async enriching chunk (table: {is_table})")
            enrichment = await self.llm.ainvoke(prompt)
            
            # Add enrichment to chunk
            chunk.enrichment = enrichment
            logger.debug(f"Successfully async enriched chunk")
            
            return chunk
            
        except Exception as e:
            error_msg = f"Failed to async enrich chunk: {e}"
            logger.error(error_msg)
            raise ChunkProcessingError(error_msg) from e
    
    def enrich_chunks(self, chunks: List[Any]) -> List[Any]:
        """
        Enrich multiple chunks sequentially.
        
        Args:
            chunks: List of document chunks to enrich
            
        Returns:
            List of enriched chunks
            
        Raises:
            ChunkProcessingError: If chunk processing fails
        """
        enriched_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Enriching chunk {i+1}/{len(chunks)}")
                enriched_chunk = self.enrich_chunk(chunk)
                enriched_chunks.append(enriched_chunk)
                
            except ChunkProcessingError as e:
                logger.error(f"Failed to enrich chunk {i+1}: {e}")
                if self.config.max_retries > 0:
                    # Retry logic could be added here
                    raise
                else:
                    # Skip failed chunk
                    enriched_chunks.append(chunk)
        
        return enriched_chunks
    
    async def enrich_chunks_async(self, chunks: List[Any]) -> List[Any]:
        """
        Enrich multiple chunks asynchronously in batches.
        
        Args:
            chunks: List of document chunks to enrich
            
        Returns:
            List of enriched chunks
        """
        if not self.config.enable_async_processing:
            return self.enrich_chunks(chunks)
        
        return await process_chunks_concurrently(
            chunks, 
            self.enrich_chunk_async, 
            batch_size=self.config.batch_size
        )
    
    def get_enrichment_stats(self, chunks: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about enrichment processing.
        
        Args:
            chunks: List of enriched chunks
            
        Returns:
            Dictionary with enrichment statistics
        """
        total_chunks = len(chunks)
        enriched_chunks = sum(1 for chunk in chunks if hasattr(chunk, 'enrichment'))
        table_chunks = sum(1 for chunk in chunks if is_table_chunk(chunk))
        
        return {
            "total_chunks": total_chunks,
            "enriched_chunks": enriched_chunks,
            "table_chunks": table_chunks,
            "text_chunks": total_chunks - table_chunks,
            "enrichment_rate": enriched_chunks / total_chunks if total_chunks > 0 else 0
        }


def add_enrichment_to_chunk(chunk: Any, enricher: Optional[DocumentEnricher] = None) -> Any:
    """
    Add enrichment to a chunk (convenience function).
    
    Args:
        chunk: Document chunk to enrich
        enricher: Document enricher instance (optional)
        
    Returns:
        Enriched chunk
    """
    if enricher is None:
        enricher = DocumentEnricher()
    
    return enricher.enrich_chunk(chunk)


async def add_enrichment_to_chunk_async(chunk: Any, enricher: Optional[DocumentEnricher] = None) -> Any:
    """
    Add enrichment to a chunk asynchronously (convenience function).
    
    Args:
        chunk: Document chunk to enrich
        enricher: Document enricher instance (optional)
        
    Returns:
        Enriched chunk
    """
    if enricher is None:
        enricher = DocumentEnricher()
    
    return await enricher.enrich_chunk_async(chunk)
