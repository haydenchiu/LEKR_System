"""
Subject-level retriever for LERK System.

This module implements subject-level retrieval functionality that handles
queries about document clusters and consolidated subject knowledge,
with comprehensive fallback mechanisms and cluster awareness.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    import numpy as np
except ImportError as e:
    # Create mock classes for missing dependencies
    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass
    
    class Distance:
        COSINE = "Cosine"
    
    class VectorParams:
        def __init__(self, *args, **kwargs):
            pass
    
    class PointStruct:
        def __init__(self, *args, **kwargs):
            pass
    
    class Filter:
        def __init__(self, *args, **kwargs):
            pass
    
    class FieldCondition:
        def __init__(self, *args, **kwargs):
            pass
    
    class MatchValue:
        def __init__(self, *args, **kwargs):
            pass
    
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
    
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class np:
        @staticmethod
        def array(data):
            return data

from .base_retriever import LERKBaseRetriever
from .config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG, SearchStrategy
from .exceptions import VectorSearchError, InvalidQueryError, DatabaseConnectionError

logger = logging.getLogger(__name__)


class SubjectRetriever(LERKBaseRetriever):
    """
    Subject-level retriever for consolidated knowledge queries.
    
    This retriever handles queries about document clusters and subject knowledge,
    with fallback mechanisms when subject knowledge is insufficient.
    """
    
    def __init__(
        self, 
        config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
        vector_store = None,
        embedding_model = None,
        knowledge_storage_manager = None,
        chunk_retriever = None
    ):
        """
        Initialize the subject retriever.
        
        Args:
            config: Configuration for the retriever
            vector_store: Pre-initialized vector store (optional)
            embedding_model: Pre-initialized embedding model (optional)
            knowledge_storage_manager: Knowledge storage manager (optional)
            chunk_retriever: Fallback chunk retriever (optional)
        """
        # Ensure subject search strategy
        if config.search_strategy != SearchStrategy.SUBJECT:
            config = config.update(search_strategy=SearchStrategy.SUBJECT)
            
        super().__init__(config, vector_store, embedding_model)
        self.knowledge_storage_manager = knowledge_storage_manager
        self.chunk_retriever = chunk_retriever
        self._cluster_metadata: Dict[str, Any] = {}
        self._subject_maturity_threshold = 0.7  # Minimum quality for subject knowledge
        self._fallback_threshold = 3  # Minimum results before fallback
    
    def _initialize(self) -> None:
        """Initialize the subject retriever components."""
        if self._initialized:
            return
        
        try:
            # Initialize base components
            super()._initialize()
            
            # Load cluster metadata
            self._load_cluster_metadata()
            
            # Initialize knowledge storage if not provided
            if self.knowledge_storage_manager is None:
                try:
                    from consolidation import KnowledgeStorageManager
                    self.knowledge_storage_manager = KnowledgeStorageManager()
                except ImportError:
                    logger.warning("KnowledgeStorageManager not available - subject retrieval limited")
            
            # Initialize fallback chunk retriever
            if self.chunk_retriever is None:
                try:
                    from .semantic_retriever import SemanticRetriever
                    self.chunk_retriever = SemanticRetriever(self.config)
                except ImportError:
                    logger.warning("Chunk retriever not available - no fallback mechanism")
            
            self._initialized = True
            logger.info("Subject retriever initialized successfully")
            
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to initialize subject retriever: {e}") from e
    
    def _load_cluster_metadata(self) -> None:
        """Load cluster metadata for cluster-aware retrieval."""
        try:
            # This would typically load from database or configuration
            # For now, we'll create a mock structure
            self._cluster_metadata = {
                "clusters": {},
                "cluster_labels": {},
                "cluster_topics": {},
                "last_updated": datetime.utcnow().isoformat()
            }
            logger.info("Cluster metadata loaded")
        except Exception as e:
            logger.warning(f"Failed to load cluster metadata: {e}")
            self._cluster_metadata = {"clusters": {}, "cluster_labels": {}}
    
    def _retrieve_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve subject knowledge with fallback mechanisms.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents (subject knowledge or chunks)
        """
        try:
            # Step 1: Classify query intent
            query_intent = self._classify_query_intent(query)
            logger.info(f"Query intent: {query_intent}")
            
            # Step 2: Search subject knowledge
            subject_results = self._search_subject_knowledge(query, filters, query_intent)
            
            # Step 3: Check if subject knowledge is mature enough
            mature_results = self._filter_mature_subject_knowledge(subject_results)
            
            # Step 4: Apply fallback if needed
            if len(mature_results) < self._fallback_threshold:
                logger.info(f"Subject knowledge insufficient ({len(mature_results)} results), applying fallback")
                fallback_results = self._apply_fallback_search(query, filters, query_intent)
                return self._combine_results(mature_results, fallback_results, query_intent)
            
            # Step 5: Enhance results with cluster context
            enhanced_results = self._enhance_with_cluster_context(mature_results, query_intent)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Subject retrieval failed: {e}")
            # Emergency fallback to chunk search
            if self.chunk_retriever:
                logger.info("Applying emergency fallback to chunk search")
                return self.chunk_retriever.get_relevant_documents(query, filters)
            else:
                raise VectorSearchError(f"Subject retrieval failed and no fallback available: {e}") from e
    
    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent to determine search strategy.
        
        Args:
            query: User query
            
        Returns:
            Query intent classification
        """
        query_lower = query.lower()
        
        intent = {
            "search_level": "subject",
            "target_clusters": [],
            "subject_focus": False,
            "document_focus": False,
            "chunk_focus": False,
            "cluster_specific": False,
            "discovery_query": False,
            "comparison_query": False
        }
        
        # Subject-level indicators
        subject_indicators = [
            "main concepts", "overview", "summary", "fundamentals",
            "principles", "key topics", "core ideas", "general",
            "what is", "explain", "describe", "tell me about"
        ]
        
        # Document-level indicators
        document_indicators = [
            "specific document", "in this paper", "author states",
            "according to", "document mentions", "paper says"
        ]
        
        # Chunk-level indicators
        chunk_indicators = [
            "specific detail", "exact quote", "precise information",
            "specific example", "particular case", "exactly"
        ]
        
        # Discovery indicators
        discovery_indicators = [
            "what topics", "what subjects", "what clusters", "what documents",
            "list", "show me", "available", "in the system"
        ]
        
        # Comparison indicators
        comparison_indicators = [
            "compare", "difference", "versus", "vs", "contrast",
            "similarities", "differences"
        ]
        
        # Classify intent
        if any(indicator in query_lower for indicator in discovery_indicators):
            intent["discovery_query"] = True
            intent["search_level"] = "discovery"
        elif any(indicator in query_lower for indicator in comparison_indicators):
            intent["comparison_query"] = True
            intent["search_level"] = "comparison"
        elif any(indicator in query_lower for indicator in subject_indicators):
            intent["subject_focus"] = True
            intent["search_level"] = "subject"
        elif any(indicator in query_lower for indicator in document_indicators):
            intent["document_focus"] = True
            intent["search_level"] = "document"
        elif any(indicator in query_lower for indicator in chunk_indicators):
            intent["chunk_focus"] = True
            intent["search_level"] = "chunk"
        
        # Extract target clusters from query
        intent["target_clusters"] = self._extract_target_clusters(query)
        
        return intent
    
    def _extract_target_clusters(self, query: str) -> List[str]:
        """
        Extract target cluster IDs from query.
        
        Args:
            query: User query
            
        Returns:
            List of target cluster IDs
        """
        target_clusters = []
        query_lower = query.lower()
        
        # Simple keyword-based cluster matching
        # In a real implementation, this would use more sophisticated NLP
        cluster_keywords = {
            "machine learning": ["ml_cluster", "ai_cluster"],
            "artificial intelligence": ["ai_cluster", "ml_cluster"],
            "deep learning": ["dl_cluster", "ml_cluster"],
            "neural networks": ["nn_cluster", "dl_cluster"],
            "natural language processing": ["nlp_cluster", "ai_cluster"],
            "computer vision": ["cv_cluster", "ai_cluster"],
            "data science": ["ds_cluster", "ml_cluster"],
            "statistics": ["stats_cluster", "ds_cluster"]
        }
        
        for keyword, cluster_ids in cluster_keywords.items():
            if keyword in query_lower:
                target_clusters.extend(cluster_ids)
        
        return list(set(target_clusters))  # Remove duplicates
    
    def _search_subject_knowledge(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]], 
        query_intent: Dict[str, Any]
    ) -> List[Document]:
        """
        Search consolidated subject knowledge.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            query_intent: Query intent classification
            
        Returns:
            List of subject knowledge documents
        """
        try:
            # Build search filters
            search_filters = self._build_subject_filters(filters, query_intent)
            
            # Search in Qdrant for subject knowledge
            if self.vector_store:
                documents = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=self.config.max_results * 2,
                    filter=search_filters
                )
                
                # Convert to Document objects
                results = []
                for doc, score in documents:
                    if doc.metadata.get("knowledge_type") == "subject":
                        # Convert subject knowledge to document format
                        subject_doc = self._convert_subject_to_document(doc, score)
                        results.append(subject_doc)
                
                logger.info(f"Found {len(results)} subject knowledge results")
                return results
            
            # Fallback to knowledge storage manager
            elif self.knowledge_storage_manager:
                return self._search_subject_knowledge_fallback(query, filters)
            
            else:
                logger.warning("No subject knowledge search method available")
                return []
                
        except Exception as e:
            logger.error(f"Subject knowledge search failed: {e}")
            return []
    
    def _build_subject_filters(
        self, 
        filters: Optional[Dict[str, Any]], 
        query_intent: Dict[str, Any]
    ) -> Optional[Filter]:
        """
        Build filters for subject knowledge search.
        
        Args:
            filters: User-provided filters
            query_intent: Query intent classification
            
        Returns:
            Qdrant filter object
        """
        filter_conditions = []
        
        # Always filter for subject knowledge
        filter_conditions.append(
            FieldCondition(
                key="knowledge_type",
                match=MatchValue(value="subject")
            )
        )
        
        # Filter by target clusters if specified
        if query_intent.get("target_clusters"):
            cluster_filter = FieldCondition(
                key="cluster_id",
                match=MatchValue(value=query_intent["target_clusters"][0])
            )
            filter_conditions.append(cluster_filter)
        
        # Apply user filters
        if filters:
            for key, value in filters.items():
                if key in ["cluster_id", "subject_id", "expertise_level", "domain_tags"]:
                    filter_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
        
        if filter_conditions:
            return Filter(must=filter_conditions)
        
        return None
    
    def _convert_subject_to_document(self, doc: Document, score: float) -> Document:
        """
        Convert subject knowledge to document format.
        
        Args:
            doc: Subject knowledge document
            score: Similarity score
            
        Returns:
            Converted document
        """
        # Extract subject information
        subject_id = doc.metadata.get("subject_id", "unknown")
        subject_name = doc.metadata.get("name", "Unknown Subject")
        description = doc.metadata.get("description", "")
        core_concepts = doc.metadata.get("core_concepts", [])
        quality_score = doc.metadata.get("quality_score", 0.0)
        
        # Create comprehensive content
        content_parts = [
            f"Subject: {subject_name}",
            f"Description: {description}",
            f"Core Concepts: {', '.join([c.get('name', '') for c in core_concepts])}",
            f"Quality Score: {quality_score:.2f}",
            f"Subject ID: {subject_id}"
        ]
        
        content = "\n".join(content_parts)
        
        # Create enhanced metadata
        enhanced_metadata = {
            **doc.metadata,
            "similarity_score": 1.0 - score if score <= 1.0 else 1.0 / (1.0 + score),
            "search_score": score,
            "search_method": "subject",
            "knowledge_type": "subject",
            "content_type": "subject_knowledge",
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
        return Document(
            page_content=content,
            metadata=enhanced_metadata
        )
    
    def _filter_mature_subject_knowledge(self, results: List[Document]) -> List[Document]:
        """
        Filter results to only include mature subject knowledge.
        
        Args:
            results: List of subject knowledge results
            
        Returns:
            Filtered list of mature subject knowledge
        """
        mature_results = []
        
        for doc in results:
            quality_score = doc.metadata.get("quality_score", 0.0)
            concept_count = len(doc.metadata.get("core_concepts", []))
            document_sources = len(doc.metadata.get("document_sources", []))
            
            # Check maturity criteria
            is_mature = (
                quality_score >= self._subject_maturity_threshold and
                concept_count >= 3 and  # At least 3 core concepts
                document_sources >= 2   # At least 2 source documents
            )
            
            if is_mature:
                mature_results.append(doc)
                logger.debug(f"Subject {doc.metadata.get('subject_id')} is mature (quality: {quality_score:.2f})")
            else:
                logger.debug(f"Subject {doc.metadata.get('subject_id')} is not mature (quality: {quality_score:.2f})")
        
        logger.info(f"Filtered to {len(mature_results)} mature subject knowledge results")
        return mature_results
    
    def _apply_fallback_search(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]], 
        query_intent: Dict[str, Any]
    ) -> List[Document]:
        """
        Apply fallback search when subject knowledge is insufficient.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            query_intent: Query intent classification
            
        Returns:
            List of fallback results
        """
        fallback_results = []
        
        try:
            # Strategy 1: Search document clusters
            if query_intent.get("target_clusters"):
                cluster_results = self._search_document_clusters(query, query_intent["target_clusters"])
                fallback_results.extend(cluster_results)
            
            # Strategy 2: Search individual chunks
            if self.chunk_retriever and len(fallback_results) < self._fallback_threshold:
                chunk_results = self.chunk_retriever.get_relevant_documents(query, filters)
                fallback_results.extend(chunk_results)
            
            # Strategy 3: Search all available knowledge
            if len(fallback_results) < self._fallback_threshold:
                all_results = self._search_all_knowledge(query, filters)
                fallback_results.extend(all_results)
            
            logger.info(f"Fallback search found {len(fallback_results)} results")
            return fallback_results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def _search_document_clusters(
        self, 
        query: str, 
        target_clusters: List[str]
    ) -> List[Document]:
        """
        Search within specific document clusters.
        
        Args:
            query: Search query
            target_clusters: List of target cluster IDs
            
        Returns:
            List of cluster-based results
        """
        cluster_results = []
        
        try:
            for cluster_id in target_clusters:
                # Search chunks within this cluster
                cluster_filter = {
                    "cluster_id": cluster_id,
                    "knowledge_type": "chunk"
                }
                
                if self.chunk_retriever:
                    cluster_docs = self.chunk_retriever.get_relevant_documents(query, cluster_filter)
                    
                    # Add cluster context to results
                    for doc in cluster_docs:
                        doc.metadata["cluster_id"] = cluster_id
                        doc.metadata["search_method"] = "cluster_fallback"
                        doc.metadata["fallback_reason"] = "insufficient_subject_knowledge"
                    
                    cluster_results.extend(cluster_docs)
            
            logger.info(f"Cluster search found {len(cluster_results)} results")
            return cluster_results
            
        except Exception as e:
            logger.error(f"Cluster search failed: {e}")
            return []
    
    def _search_all_knowledge(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]]
    ) -> List[Document]:
        """
        Search all available knowledge types.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            
        Returns:
            List of all knowledge results
        """
        all_results = []
        
        try:
            if self.vector_store:
                # Search all knowledge types
                documents = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=self.config.max_results * 3
                )
                
                for doc, score in documents:
                    doc.metadata["search_method"] = "all_knowledge_fallback"
                    doc.metadata["fallback_reason"] = "insufficient_subject_knowledge"
                    all_results.append(doc)
            
            logger.info(f"All knowledge search found {len(all_results)} results")
            return all_results
            
        except Exception as e:
            logger.error(f"All knowledge search failed: {e}")
            return []
    
    def _combine_results(
        self, 
        subject_results: List[Document], 
        fallback_results: List[Document], 
        query_intent: Dict[str, Any]
    ) -> List[Document]:
        """
        Combine subject and fallback results with appropriate ranking.
        
        Args:
            subject_results: Subject knowledge results
            fallback_results: Fallback search results
            query_intent: Query intent classification
            
        Returns:
            Combined and ranked results
        """
        combined_results = []
        
        # Add subject results with high priority
        for doc in subject_results:
            doc.metadata["result_priority"] = "high"
            doc.metadata["result_type"] = "subject_knowledge"
            combined_results.append(doc)
        
        # Add fallback results with lower priority
        for doc in fallback_results:
            doc.metadata["result_priority"] = "medium"
            doc.metadata["result_type"] = "fallback"
            combined_results.append(doc)
        
        # Sort by priority and similarity score
        combined_results.sort(
            key=lambda x: (
                0 if x.metadata.get("result_priority") == "high" else 1,
                -x.metadata.get("similarity_score", 0)
            )
        )
        
        # Limit results
        combined_results = combined_results[:self.config.max_results]
        
        logger.info(f"Combined {len(subject_results)} subject + {len(fallback_results)} fallback = {len(combined_results)} total results")
        return combined_results
    
    def _enhance_with_cluster_context(
        self, 
        results: List[Document], 
        query_intent: Dict[str, Any]
    ) -> List[Document]:
        """
        Enhance results with cluster context information.
        
        Args:
            results: List of results to enhance
            query_intent: Query intent classification
            
        Returns:
            Enhanced results with cluster context
        """
        enhanced_results = []
        
        for doc in results:
            # Add cluster context if available
            cluster_id = doc.metadata.get("cluster_id")
            if cluster_id and cluster_id in self._cluster_metadata.get("clusters", {}):
                cluster_info = self._cluster_metadata["clusters"][cluster_id]
                doc.metadata["cluster_context"] = {
                    "cluster_label": cluster_info.get("label", ""),
                    "cluster_topics": cluster_info.get("topics", []),
                    "document_count": cluster_info.get("document_count", 0),
                    "last_updated": cluster_info.get("last_updated", "")
                }
            
            enhanced_results.append(doc)
        
        return enhanced_results
    
    def get_available_clusters(self) -> Dict[str, Any]:
        """
        Get information about available document clusters.
        
        Returns:
            Dictionary with cluster information
        """
        try:
            if not self._initialized:
                self._initialize()
            
            return {
                "clusters": self._cluster_metadata.get("clusters", {}),
                "cluster_count": len(self._cluster_metadata.get("clusters", {})),
                "last_updated": self._cluster_metadata.get("last_updated", ""),
                "cluster_labels": self._cluster_metadata.get("cluster_labels", {}),
                "cluster_topics": self._cluster_metadata.get("cluster_topics", {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get available clusters: {e}")
            return {"clusters": {}, "cluster_count": 0, "error": str(e)}
    
    def get_available_documents(self) -> Dict[str, Any]:
        """
        Get information about available documents.
        
        Returns:
            Dictionary with document information
        """
        try:
            if not self._initialized:
                self._initialize()
            
            # This would typically query the database
            # For now, return mock data
            return {
                "documents": {},
                "document_count": 0,
                "last_updated": datetime.utcnow().isoformat(),
                "note": "Document discovery not yet implemented"
            }
            
        except Exception as e:
            logger.error(f"Failed to get available documents: {e}")
            return {"documents": {}, "document_count": 0, "error": str(e)}
    
    def _search_subject_knowledge_fallback(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]]
    ) -> List[Document]:
        """
        Fallback method for searching subject knowledge.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            
        Returns:
            List of subject knowledge documents
        """
        # This would use the knowledge storage manager
        # For now, return empty list
        logger.warning("Subject knowledge fallback search not implemented")
        return []


# Convenience functions
def create_subject_retriever(config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG) -> SubjectRetriever:
    """Create a subject retriever instance."""
    return SubjectRetriever(config)


def search_subject_knowledge(
    query: str, 
    config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
    filters: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Search subject knowledge for a query.
    
    Args:
        query: Search query
        config: Retriever configuration
        filters: Optional metadata filters
        
    Returns:
        List of subject knowledge results
    """
    retriever = SubjectRetriever(config)
    return retriever.get_relevant_documents(query, filters)
