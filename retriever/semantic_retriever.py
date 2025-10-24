"""
Semantic retriever using vector similarity search.

This module implements semantic retrieval functionality that uses vector
similarity search to find relevant documents based on semantic meaning
rather than exact keyword matches.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document

from .base_retriever import LERKBaseRetriever
from .config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG, SearchStrategy
from .exceptions import VectorSearchError, InvalidQueryError

logger = logging.getLogger(__name__)


class SemanticRetriever(LERKBaseRetriever):
    """
    Semantic retriever that uses vector similarity search.
    
    This retriever finds relevant documents by comparing the semantic
    meaning of the query with stored document embeddings using cosine
    similarity or other distance metrics.
    """
    
    def __init__(
        self, 
        config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
        vector_store = None,
        embedding_model = None
    ):
        """
        Initialize the semantic retriever.
        
        Args:
            config: Configuration for the retriever
            vector_store: Pre-initialized vector store (optional)
            embedding_model: Pre-initialized embedding model (optional)
        """
        # Ensure semantic search strategy
        if config.search_strategy != SearchStrategy.SEMANTIC:
            config = config.update(search_strategy=SearchStrategy.SEMANTIC)
            
        super().__init__(config, vector_store, embedding_model)
        self._search_method = "similarity"
    
    def _retrieve_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents using semantic similarity search.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents
        """
        try:
            # Use similarity search with vector store
            documents = self.vector_store.similarity_search_with_score(
                query=query,
                k=self.config.max_results * 2,  # Get more results for filtering
                filter=filters
            )
            
            # Extract documents and scores
            results = []
            for doc, score in documents:
                # Convert score to similarity (higher is better)
                similarity_score = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
                
                # Add similarity score to metadata
                doc.metadata['similarity_score'] = similarity_score
                doc.metadata['search_score'] = score
                doc.metadata['search_method'] = self._search_method
                
                results.append(doc)
            
            logger.info(f"Semantic search found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            raise VectorSearchError(f"Semantic search failed: {e}") from e
    
    def similarity_search_with_threshold(
        self, 
        query: str, 
        threshold: float = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search with a custom threshold.
        
        Args:
            query: Search query
            threshold: Similarity threshold (overrides config)
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of documents above the threshold
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
            
        # Get documents
        documents = self._retrieve_documents(query, filters, **kwargs)
        
        # Filter by threshold
        filtered_docs = [
            doc for doc in documents 
            if doc.metadata.get('similarity_score', 0) >= threshold
        ]
        
        logger.info(f"Threshold filtering: {len(documents)} -> {len(filtered_docs)} documents")
        return filtered_docs
    
    def get_similar_documents(
        self, 
        document_id: str,
        limit: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Find documents similar to a specific document.
        
        Args:
            document_id: ID of the reference document
            limit: Maximum number of results
            filters: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        if limit is None:
            limit = self.config.max_results
            
        try:
            # First, retrieve the reference document
            ref_docs = self.vector_store.similarity_search(
                query="",  # Empty query to get all documents
                k=1000,  # Large number to find the reference
                filter={"document_id": document_id}
            )
            
            if not ref_docs:
                logger.warning(f"Reference document {document_id} not found")
                return []
            
            ref_doc = ref_docs[0]
            
            # Use the reference document's content for similarity search
            similar_docs = self.vector_store.similarity_search_with_score(
                query=ref_doc.page_content,
                k=limit + 1,  # +1 to exclude the reference document itself
                filter=filters
            )
            
            # Filter out the reference document itself
            results = []
            for doc, score in similar_docs:
                if doc.metadata.get('document_id') != document_id:
                    similarity_score = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
                    doc.metadata['similarity_score'] = similarity_score
                    doc.metadata['search_score'] = score
                    doc.metadata['reference_document_id'] = document_id
                    results.append(doc)
            
            return results[:limit]
            
        except Exception as e:
            raise VectorSearchError(f"Failed to find similar documents: {e}") from e
    
    def semantic_search_by_concept(
        self, 
        concept: str,
        concept_type: str = "key_concept",
        limit: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for documents containing a specific concept.
        
        Args:
            concept: Concept to search for
            concept_type: Type of concept (key_concept, relation, etc.)
            limit: Maximum number of results
            filters: Optional metadata filters
            
        Returns:
            List of documents containing the concept
        """
        if limit is None:
            limit = self.config.max_results
            
        # Build concept-specific filters
        concept_filters = filters or {}
        concept_filters[f"{concept_type}_name"] = concept
        
        try:
            # Search for documents with the concept
            documents = self.vector_store.similarity_search_with_score(
                query=concept,
                k=limit * 2,  # Get more results for filtering
                filter=concept_filters
            )
            
            # Process results
            results = []
            for doc, score in documents:
                similarity_score = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
                doc.metadata['similarity_score'] = similarity_score
                doc.metadata['search_score'] = score
                doc.metadata['concept_search'] = concept
                doc.metadata['concept_type'] = concept_type
                results.append(doc)
            
            # Filter by similarity threshold
            filtered_results = [
                doc for doc in results 
                if doc.metadata.get('similarity_score', 0) >= self.config.similarity_threshold
            ]
            
            logger.info(f"Concept search for '{concept}': {len(results)} -> {len(filtered_results)} documents")
            return filtered_results[:limit]
            
        except Exception as e:
            raise VectorSearchError(f"Concept search failed: {e}") from e
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic retriever."""
        base_stats = super().get_retrieval_stats()
        base_stats.update({
            "retriever_type": "semantic",
            "search_method": self._search_method,
            "similarity_threshold": self.config.similarity_threshold
        })
        return base_stats
    
    def __str__(self) -> str:
        """String representation of the semantic retriever."""
        return f"SemanticRetriever(config={self.config}, initialized={self._initialized})"
    
    def __repr__(self) -> str:
        """Detailed representation of the semantic retriever."""
        return (f"SemanticRetriever(config={self.config}, "
                f"search_method={self._search_method}, "
                f"initialized={self._initialized})")
