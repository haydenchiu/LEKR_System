"""
Context-aware retriever with conversation history.

This module implements context-aware retrieval functionality that considers
conversation history and user intent to provide more relevant and personalized
search results.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import deque
import json

from .base_retriever import LERKBaseRetriever
from .config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG, SearchStrategy
from .exceptions import VectorSearchError, InvalidQueryError
from .utils import calculate_relevance_score

logger = logging.getLogger(__name__)


class ContextRetriever(LERKBaseRetriever):
    """
    Context-aware retriever that considers conversation history.
    
    This retriever maintains a context window of previous queries and
    uses this information to improve retrieval relevance and provide
    more personalized results.
    """
    
    def __init__(
        self, 
        config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
        vector_store = None,
        embedding_model = None
    ):
        """
        Initialize the context-aware retriever.
        
        Args:
            config: Configuration for the retriever
            vector_store: Pre-initialized vector store (optional)
            embedding_model: Pre-initialized embedding model (optional)
        """
        # Ensure context-aware search strategy
        if config.search_strategy != SearchStrategy.CONTEXT_AWARE:
            config = config.update(search_strategy=SearchStrategy.CONTEXT_AWARE)
            
        super().__init__(config, vector_store, embedding_model)
        self._context_history: deque = deque(maxlen=config.context_window_size)
        self._user_preferences: Dict[str, Any] = {}
        self._session_id: str = None
        self._context_embeddings: List[List[float]] = []
    
    def start_session(self, session_id: str = None) -> str:
        """
        Start a new retrieval session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._session_id = session_id
        self._context_history.clear()
        self._context_embeddings.clear()
        
        logger.info(f"Started new retrieval session: {session_id}")
        return session_id
    
    def add_to_context(self, query: str, results: List[Dict[str, Any]] = None) -> None:
        """
        Add a query and its results to the context history.
        
        Args:
            query: The query that was processed
            results: Optional results from the query
        """
        context_entry = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results) if results else 0,
            "session_id": self._session_id
        }
        
        if results:
            context_entry["top_results"] = [
                {
                    "document_id": r.get("document_id"),
                    "title": r.get("title", "")[:100],
                    "relevance_score": r.get("relevance_score", 0)
                }
                for r in results[:3]  # Store top 3 results
            ]
        
        self._context_history.append(context_entry)
        
        # Generate and store embedding for context
        try:
            embedding = self.embedding_model.encode(query)
            self._context_embeddings.append(embedding.tolist())
        except Exception as e:
            logger.warning(f"Failed to generate context embedding: {e}")
    
    def _retrieve_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with context awareness.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents with context information
        """
        try:
            # Build context-aware query
            enhanced_query = self._build_context_aware_query(query)
            
            # Perform base retrieval
            documents = self.vector_store.similarity_search_with_score(
                query=enhanced_query,
                k=self.config.max_results * 2,  # Get more results for context filtering
                filter=filters
            )
            
            # Process results with context awareness
            results = []
            for doc, score in documents:
                # Calculate base similarity score
                similarity_score = 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score)
                
                # Apply context-aware scoring
                context_score = self._calculate_context_score(doc, query)
                
                # Combine scores
                final_score = (
                    0.7 * similarity_score +  # Base similarity
                    0.3 * context_score       # Context relevance
                )
                
                doc.metadata['similarity_score'] = similarity_score
                doc.metadata['context_score'] = context_score
                doc.metadata['final_score'] = final_score
                doc.metadata['search_score'] = score
                doc.metadata['search_method'] = 'context_aware'
                doc.metadata['session_id'] = self._session_id
                
                results.append(doc)
            
            # Sort by final score
            results.sort(key=lambda x: x.metadata.get('final_score', 0), reverse=True)
            
            logger.info(f"Context-aware search found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            raise VectorSearchError(f"Context-aware search failed: {e}") from e
    
    def _build_context_aware_query(self, query: str) -> str:
        """Build a context-aware query by incorporating history."""
        if not self._context_history:
            return query
        
        try:
            # Extract relevant context from history
            context_queries = []
            for i, entry in enumerate(self._context_history):
                # Apply decay factor based on recency
                decay = self.config.context_decay_factor ** i
                if decay > 0.1:  # Only include if decay is significant
                    context_queries.append(entry["query"])
            
            # Build enhanced query
            if context_queries:
                context_text = " ".join(context_queries[-3:])  # Use last 3 queries
                enhanced_query = f"{query} {context_text}"
            else:
                enhanced_query = query
            
            logger.debug(f"Enhanced query: {enhanced_query[:100]}...")
            return enhanced_query
            
        except Exception as e:
            logger.warning(f"Failed to build context-aware query: {e}")
            return query
    
    def _calculate_context_score(self, doc: Dict[str, Any], query: str) -> float:
        """Calculate context relevance score for a document."""
        try:
            context_score = 0.0
            
            # Check if document was recently accessed
            if self._context_history:
                doc_id = doc.metadata.get('document_id')
                for entry in self._context_history:
                    if entry.get("top_results"):
                        for result in entry["top_results"]:
                            if result.get("document_id") == doc_id:
                                # Document was in recent results
                                context_score += 0.3
                                break
            
            # Check for topic continuity
            if self._context_embeddings:
                try:
                    # Generate embedding for current query
                    query_embedding = self.embedding_model.encode(query)
                    
                    # Calculate similarity with context embeddings
                    max_similarity = 0.0
                    for context_embedding in self._context_embeddings:
                        similarity = self._cosine_similarity(query_embedding, context_embedding)
                        max_similarity = max(max_similarity, similarity)
                    
                    # Add topic continuity score
                    context_score += 0.2 * max_similarity
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate topic continuity: {e}")
            
            # Check for user preferences
            if self._user_preferences:
                doc_category = doc.metadata.get('category', '')
                preferred_categories = self._user_preferences.get('categories', [])
                if doc_category in preferred_categories:
                    context_score += 0.2
            
            return min(1.0, context_score)
            
        except Exception as e:
            logger.warning(f"Failed to calculate context score: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.warning(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def update_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update user preferences for personalized retrieval.
        
        Args:
            preferences: Dictionary of user preferences
        """
        self._user_preferences.update(preferences)
        logger.info(f"Updated user preferences: {list(preferences.keys())}")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context."""
        return {
            "session_id": self._session_id,
            "context_entries": len(self._context_history),
            "context_window_size": self.config.context_window_size,
            "user_preferences": self._user_preferences,
            "recent_queries": [entry["query"] for entry in list(self._context_history)[-3:]]
        }
    
    def clear_context(self) -> None:
        """Clear the context history."""
        self._context_history.clear()
        self._context_embeddings.clear()
        logger.info("Cleared context history")
    
    def retrieve_with_feedback(
        self, 
        query: str,
        feedback: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with user feedback integration.
        
        Args:
            query: Search query
            feedback: User feedback on previous results
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of documents with feedback-informed ranking
        """
        try:
            # Update preferences based on feedback
            if feedback.get("preferred_categories"):
                self.update_user_preferences({
                    "categories": feedback["preferred_categories"]
                })
            
            # Perform context-aware retrieval
            results = self._retrieve_documents(query, filters, **kwargs)
            
            # Apply feedback-based adjustments
            for doc in results:
                doc_id = doc.metadata.get('document_id')
                
                # Boost score if document was previously liked
                if doc_id in feedback.get("liked_documents", []):
                    doc.metadata['final_score'] *= 1.2
                
                # Reduce score if document was previously disliked
                if doc_id in feedback.get("disliked_documents", []):
                    doc.metadata['final_score'] *= 0.8
            
            # Re-sort by adjusted scores
            results.sort(key=lambda x: x.metadata.get('final_score', 0), reverse=True)
            
            # Add to context
            self.add_to_context(query, results)
            
            return results
            
        except Exception as e:
            raise VectorSearchError(f"Feedback-informed retrieval failed: {e}") from e
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the context-aware retriever."""
        base_stats = super().get_retrieval_stats()
        base_stats.update({
            "retriever_type": "context_aware",
            "session_id": self._session_id,
            "context_entries": len(self._context_history),
            "context_embeddings": len(self._context_embeddings),
            "user_preferences": len(self._user_preferences),
            "context_window_size": self.config.context_window_size,
            "context_decay_factor": self.config.context_decay_factor
        })
        return base_stats
    
    def __str__(self) -> str:
        """String representation of the context-aware retriever."""
        return f"ContextRetriever(config={self.config}, session={self._session_id})"
    
    def __repr__(self) -> str:
        """Detailed representation of the context-aware retriever."""
        return (f"ContextRetriever(config={self.config}, "
                f"session={self._session_id}, "
                f"context_entries={len(self._context_history)})")
