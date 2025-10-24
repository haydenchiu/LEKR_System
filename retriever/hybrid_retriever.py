"""
Hybrid retriever combining semantic and keyword search.

This module implements hybrid retrieval functionality that combines
semantic vector search with traditional keyword-based search to provide
more comprehensive and accurate results.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_core.documents import Document
try:
    from langchain.retrievers import BM25Retriever
except ImportError:
    # Mock BM25Retriever for testing
    class BM25Retriever:
        def __init__(self, *args, **kwargs):
            pass
        
        @classmethod
        def from_documents(cls, documents):
            return cls()
        
        def get_relevant_documents(self, query):
            return []
try:
    import numpy as np
except ImportError:
    # Mock numpy for testing
    class np:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def dot(a, b):
            return [[0.0] * len(b[0]) for _ in range(len(a))]

from .base_retriever import LERKBaseRetriever
from .config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG, SearchStrategy
from .exceptions import VectorSearchError, InvalidQueryError
from .utils import merge_retrieval_results

logger = logging.getLogger(__name__)


class HybridRetriever(LERKBaseRetriever):
    """
    Hybrid retriever that combines semantic and keyword search.
    
    This retriever uses both vector similarity search and traditional
    keyword-based search (BM25) to find relevant documents, then
    combines and ranks the results for better accuracy.
    """
    
    def __init__(
        self, 
        config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
        vector_store = None,
        embedding_model = None,
        bm25_retriever: Optional[BM25Retriever] = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            config: Configuration for the retriever
            vector_store: Pre-initialized vector store (optional)
            embedding_model: Pre-initialized embedding model (optional)
            bm25_retriever: Pre-initialized BM25 retriever (optional)
        """
        # Ensure hybrid search strategy
        if config.search_strategy != SearchStrategy.HYBRID:
            config = config.update(search_strategy=SearchStrategy.HYBRID)
            
        super().__init__(config, vector_store, embedding_model)
        self.bm25_retriever = bm25_retriever
        self._documents_for_bm25: List[Document] = []
        self._bm25_initialized = False
    
    def _initialize_bm25(self) -> None:
        """Initialize the BM25 retriever with documents from the vector store."""
        if self._bm25_initialized:
            return
            
        try:
            # Get all documents from the vector store for BM25 indexing
            # This is a simplified approach - in production, you might want to
            # maintain a separate document collection for BM25
            all_docs = self.vector_store.similarity_search(
                query="",  # Empty query to get all documents
                k=10000  # Large number to get all documents
            )
            
            if not all_docs:
                logger.warning("No documents found for BM25 initialization")
                return
            
            # Initialize BM25 retriever
            self.bm25_retriever = BM25Retriever.from_documents(all_docs)
            self.bm25_retriever.k = self.config.max_results * 2  # Get more results for combination
            
            self._documents_for_bm25 = all_docs
            self._bm25_initialized = True
            logger.info(f"Initialized BM25 retriever with {len(all_docs)} documents")
            
        except Exception as e:
            logger.warning(f"Failed to initialize BM25 retriever: {e}")
            # Continue without BM25 - will use only semantic search
    
    def _retrieve_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search (semantic + keyword).
        
        Args:
            query: Search query
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents from hybrid search
        """
        try:
            # Initialize BM25 if not already done
            if not self._bm25_initialized:
                self._initialize_bm25()
            
            # Perform semantic search
            semantic_docs = self._semantic_search(query, filters, **kwargs)
            
            # Perform keyword search if BM25 is available
            keyword_docs = []
            if self.bm25_retriever is not None:
                keyword_docs = self._keyword_search(query, filters, **kwargs)
            
            # Combine results
            if keyword_docs:
                combined_docs = self._combine_results(semantic_docs, keyword_docs)
            else:
                combined_docs = semantic_docs
                logger.info("Using semantic search only (BM25 not available)")
            
            # Add hybrid search metadata
            for doc in combined_docs:
                doc.metadata['search_method'] = 'hybrid'
                doc.metadata['semantic_weight'] = self.config.semantic_weight
                doc.metadata['keyword_weight'] = self.config.keyword_weight
            
            logger.info(f"Hybrid search found {len(combined_docs)} documents for query: {query[:50]}...")
            return combined_docs
            
        except Exception as e:
            raise VectorSearchError(f"Hybrid search failed: {e}") from e
    
    def _semantic_search(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Perform semantic search using vector similarity."""
        try:
            # Use similarity search with vector store
            documents = self.vector_store.similarity_search_with_score(
                query=query,
                k=self.config.max_results * 2,  # Get more results for combination
                filter=filters
            )
            
            # Process semantic results
            results = []
            for doc, score in documents:
                # Convert score to similarity (higher is better)
                similarity_score = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
                
                doc.metadata['semantic_score'] = similarity_score
                doc.metadata['semantic_search_score'] = score
                doc.metadata['search_type'] = 'semantic'
                
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Perform keyword search using BM25."""
        try:
            if self.bm25_retriever is None:
                return []
            
            # Perform BM25 search
            documents = self.bm25_retriever.get_relevant_documents(query)
            
            # Process keyword results
            results = []
            for doc in documents:
                # BM25 scores are typically in the metadata
                bm25_score = doc.metadata.get('score', 0.0)
                
                # Normalize BM25 score to 0-1 range (approximate)
                normalized_score = min(1.0, bm25_score / 10.0) if bm25_score > 0 else 0.0
                
                doc.metadata['keyword_score'] = normalized_score
                doc.metadata['bm25_score'] = bm25_score
                doc.metadata['search_type'] = 'keyword'
                
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_results(
        self, 
        semantic_docs: List[Document], 
        keyword_docs: List[Document]
    ) -> List[Document]:
        """Combine semantic and keyword search results."""
        try:
            # Create a mapping of document IDs to documents
            doc_map = {}
            
            # Add semantic results
            for doc in semantic_docs:
                doc_id = doc.metadata.get('document_id', id(doc))
                doc_map[doc_id] = doc
                
                # Initialize combined score with semantic weight
                semantic_score = doc.metadata.get('semantic_score', 0.0)
                doc.metadata['combined_score'] = self.config.semantic_weight * semantic_score
            
            # Add keyword results and update combined scores
            for doc in keyword_docs:
                doc_id = doc.metadata.get('document_id', id(doc))
                keyword_score = doc.metadata.get('keyword_score', 0.0)
                
                if doc_id in doc_map:
                    # Document exists in both results - combine scores
                    existing_doc = doc_map[doc_id]
                    existing_doc.metadata['keyword_score'] = keyword_score
                    existing_doc.metadata['search_type'] = 'hybrid'
                    
                    # Update combined score
                    semantic_score = existing_doc.metadata.get('semantic_score', 0.0)
                    combined_score = (
                        self.config.semantic_weight * semantic_score +
                        self.config.keyword_weight * keyword_score
                    )
                    existing_doc.metadata['combined_score'] = combined_score
                else:
                    # Document only in keyword results
                    doc.metadata['semantic_score'] = 0.0
                    doc.metadata['combined_score'] = self.config.keyword_weight * keyword_score
                    doc.metadata['search_type'] = 'keyword_only'
                    doc_map[doc_id] = doc
            
            # Convert back to list and sort by combined score
            combined_docs = list(doc_map.values())
            combined_docs.sort(key=lambda x: x.metadata.get('combined_score', 0), reverse=True)
            
            logger.info(f"Combined {len(semantic_docs)} semantic + {len(keyword_docs)} keyword = {len(combined_docs)} unique documents")
            return combined_docs
            
        except Exception as e:
            logger.error(f"Failed to combine results: {e}")
            # Return semantic results as fallback
            return semantic_docs
    
    def search_with_weights(
        self, 
        query: str,
        semantic_weight: float = None,
        keyword_weight: float = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform hybrid search with custom weights.
        
        Args:
            query: Search query
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of documents with custom weighting
        """
        # Store original weights
        original_semantic_weight = self.config.semantic_weight
        original_keyword_weight = self.config.keyword_weight
        
        try:
            # Update weights temporarily
            if semantic_weight is not None:
                self.config.semantic_weight = semantic_weight
            if keyword_weight is not None:
                self.config.keyword_weight = keyword_weight
            
            # Perform search with custom weights
            results = self._retrieve_documents(query, filters, **kwargs)
            
            return results
            
        finally:
            # Restore original weights
            self.config.semantic_weight = original_semantic_weight
            self.config.keyword_weight = original_keyword_weight
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid retriever."""
        base_stats = super().get_retrieval_stats()
        base_stats.update({
            "retriever_type": "hybrid",
            "bm25_initialized": self._bm25_initialized,
            "bm25_documents": len(self._documents_for_bm25),
            "semantic_weight": self.config.semantic_weight,
            "keyword_weight": self.config.keyword_weight
        })
        return base_stats
    
    def __str__(self) -> str:
        """String representation of the hybrid retriever."""
        return f"HybridRetriever(config={self.config}, bm25_initialized={self._bm25_initialized})"
    
    def __repr__(self) -> str:
        """Detailed representation of the hybrid retriever."""
        return (f"HybridRetriever(config={self.config}, "
                f"bm25_initialized={self._bm25_initialized}, "
                f"bm25_docs={len(self._documents_for_bm25)})")
