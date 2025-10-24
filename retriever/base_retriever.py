"""
Base retriever class with LERK-specific functionality.

This module provides the base retriever class that implements common functionality
for all retriever types in the LERK System, including result formatting,
relevance scoring, and metadata handling.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

try:
    from langchain.retrievers.base import BaseRetriever
    from langchain_core.documents import Document
    from langchain.vectorstores import Qdrant
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    # Create mock classes for missing dependencies
    class BaseRetriever:
        def __init__(self):
            pass
    
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class Qdrant:
        def __init__(self, *args, **kwargs):
            pass
    
    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass
    
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
    
    class np:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def dot(a, b):
            return [[0.0] * len(b[0]) for _ in range(len(a))]
        
        @staticmethod
        def linalg():
            class norm:
                @staticmethod
                def __call__(vec):
                    return 1.0
            return type('linalg', (), {'norm': norm})()

from .config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG
from .exceptions import (
    RetrievalError, VectorSearchError, InvalidQueryError, 
    MissingEmbeddingModelError, DatabaseConnectionError
)
from .utils import (
    format_retrieval_results, calculate_relevance_score, 
    filter_by_metadata, rank_by_quality
)

logger = logging.getLogger(__name__)


class LERKBaseRetriever(BaseRetriever, ABC):
    """
    Base retriever class for LERK System with common functionality.
    
    This class provides the foundation for all retriever implementations
    in the LERK System, including result processing, relevance scoring,
    and metadata handling.
    """
    
    def __init__(
        self, 
        config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
        vector_store: Optional[Qdrant] = None,
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize the base retriever.
        
        Args:
            config: Configuration for the retriever
            vector_store: Pre-initialized vector store (optional)
            embedding_model: Pre-initialized embedding model (optional)
        """
        super().__init__()
        self.config = config
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self._client: Optional[QdrantClient] = None
        self._initialized = False
        
    def _initialize(self) -> None:
        """Initialize the retriever components."""
        if self._initialized:
            return
            
        try:
            # Initialize embedding model
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Initialized embedding model: {self.config.embedding_model}")
            
            # Initialize vector store
            if self.vector_store is None:
                self._initialize_vector_store()
                
            self._initialized = True
            logger.info("Base retriever initialized successfully")
            
        except Exception as e:
            raise RetrievalError(f"Failed to initialize retriever: {e}") from e
    
    def _initialize_vector_store(self) -> None:
        """Initialize the vector store connection."""
        try:
            # Initialize Qdrant client
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port
            )
            
            # Initialize LangChain Qdrant wrapper
            self.vector_store = Qdrant(
                client=self._client,
                collection_name=self.config.collection_name,
                embeddings=self.embedding_model
            )
            
            logger.info(f"Initialized vector store: {self.config.host}:{self.config.port}")
            
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to initialize vector store: {e}") from e
    
    @abstractmethod
    def _retrieve_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents based on the query.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieved documents
        """
        pass
    
    def get_relevant_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Get relevant documents for a query.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of relevant documents
        """
        if not self._initialized:
            self._initialize()
            
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")
            
        try:
            # Retrieve documents using the specific implementation
            documents = self._retrieve_documents(query, filters, **kwargs)
            
            # Apply similarity threshold filtering
            if self.config.similarity_threshold > 0:
                documents = self._filter_by_similarity(documents)
            
            # Apply metadata filtering
            if self.config.enable_metadata_filtering and filters:
                documents = filter_by_metadata(documents, filters)
            
            # Apply default filters
            if self.config.default_filters:
                documents = filter_by_metadata(documents, self.config.default_filters)
            
            # Rank results
            documents = self._rank_results(documents, query)
            
            # Apply diversity filtering if enabled
            if self.config.enable_diversity:
                documents = self._apply_diversity_filtering(documents)
            
            # Limit results
            documents = documents[:self.config.max_results]
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve documents: {e}") from e
    
    def _filter_by_similarity(self, documents: List[Document]) -> List[Document]:
        """Filter documents by similarity threshold."""
        filtered_docs = []
        for doc in documents:
            # Extract similarity score from metadata
            similarity = doc.metadata.get('similarity_score', 0.0)
            if similarity >= self.config.similarity_threshold:
                filtered_docs.append(doc)
        return filtered_docs
    
    def _rank_results(self, documents: List[Document], query: str) -> List[Document]:
        """Rank results based on the configured ranking method."""
        if not documents:
            return documents
            
        try:
            # Calculate relevance scores
            for doc in documents:
                relevance_score = calculate_relevance_score(doc, query)
                doc.metadata['relevance_score'] = relevance_score
            
            # Apply ranking based on method
            if self.config.ranking_method.value == "relevance":
                documents.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
            elif self.config.ranking_method.value == "quality":
                documents = rank_by_quality(documents)
            elif self.config.ranking_method.value == "combined":
                documents = self._combined_ranking(documents)
            elif self.config.ranking_method.value == "recency":
                documents = self._recency_ranking(documents)
            
            return documents
            
        except Exception as e:
            logger.warning(f"Failed to rank results: {e}")
            return documents
    
    def _combined_ranking(self, documents: List[Document]) -> List[Document]:
        """Apply combined relevance and quality ranking."""
        for doc in documents:
            relevance_score = doc.metadata.get('relevance_score', 0)
            quality_score = doc.metadata.get('quality_score', 0)
            
            combined_score = (
                self.config.relevance_weight * relevance_score +
                self.config.quality_weight * quality_score
            )
            doc.metadata['combined_score'] = combined_score
        
        documents.sort(key=lambda x: x.metadata.get('combined_score', 0), reverse=True)
        return documents
    
    def _recency_ranking(self, documents: List[Document]) -> List[Document]:
        """Apply recency-based ranking."""
        for doc in documents:
            # Extract timestamp from metadata
            timestamp = doc.metadata.get('timestamp', 0)
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp).timestamp()
                except:
                    timestamp = 0
            doc.metadata['recency_score'] = timestamp
        
        documents.sort(key=lambda x: x.metadata.get('recency_score', 0), reverse=True)
        return documents
    
    def _apply_diversity_filtering(self, documents: List[Document]) -> List[Document]:
        """Apply diversity filtering to avoid similar results."""
        if len(documents) <= 1:
            return documents
            
        try:
            # Extract embeddings for diversity calculation
            embeddings = []
            for doc in documents:
                embedding = doc.metadata.get('embedding')
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Generate embedding if not available
                    embedding = self.embedding_model.encode(doc.page_content)
                    embeddings.append(embedding)
                    doc.metadata['embedding'] = embedding.tolist()
            
            if not embeddings:
                return documents
            
            # Calculate pairwise similarities
            embeddings = np.array(embeddings)
            similarity_matrix = np.dot(embeddings, embeddings.T)
            
            # Select diverse documents
            selected_docs = [documents[0]]  # Always include the first (highest ranked)
            selected_indices = [0]
            
            for i in range(1, len(documents)):
                # Check similarity with already selected documents
                max_similarity = 0
                for selected_idx in selected_indices:
                    similarity = similarity_matrix[i, selected_idx]
                    max_similarity = max(max_similarity, similarity)
                
                # Add if not too similar to existing selections
                if max_similarity < self.config.diversity_threshold:
                    selected_docs.append(documents[i])
                    selected_indices.append(i)
            
            return selected_docs
            
        except Exception as e:
            logger.warning(f"Failed to apply diversity filtering: {e}")
            return documents
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[List[Document]]:
        """
        Retrieve documents for multiple queries in batch.
        
        Args:
            queries: List of search queries
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of document lists, one per query
        """
        if not self._initialized:
            self._initialize()
            
        results = []
        for i in range(0, len(queries), self.config.batch_size):
            batch_queries = queries[i:i + self.config.batch_size]
            batch_results = []
            
            for query in batch_queries:
                try:
                    docs = self.get_relevant_documents(query, filters, **kwargs)
                    batch_results.append(docs)
                except Exception as e:
                    logger.error(f"Failed to retrieve for query '{query}': {e}")
                    batch_results.append([])
            
            results.extend(batch_results)
        
        return results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever."""
        return {
            "config": self.config.model_dump(),
            "initialized": self._initialized,
            "vector_store_connected": self._client is not None,
            "embedding_model_loaded": self.embedding_model is not None
        }
    
    def __str__(self) -> str:
        """String representation of the retriever."""
        return f"LERKBaseRetriever(config={self.config}, initialized={self._initialized})"
    
    def __repr__(self) -> str:
        """Detailed representation of the retriever."""
        return (f"LERKBaseRetriever(config={self.config}, "
                f"vector_store={self.vector_store is not None}, "
                f"embedding_model={self.embedding_model is not None})")
