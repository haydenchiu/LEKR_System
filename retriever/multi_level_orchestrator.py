"""
Multi-level search orchestrator for LERK System.

This module orchestrates search across chunks, documents, and subjects,
providing intelligent query routing and result combination.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from langchain_core.documents import Document
    import numpy as np
except ImportError as e:
    # Create mock classes for missing dependencies
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class np:
        @staticmethod
        def array(data):
            return data

from .subject_retriever import SubjectRetriever
from .semantic_retriever import SemanticRetriever
from .hybrid_retriever import HybridRetriever
from .context_retriever import ContextRetriever
from .discovery_service import ClusterDiscoveryService, DocumentDiscoveryService
from .config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG, SearchStrategy
from .exceptions import RetrievalError, InvalidQueryError

logger = logging.getLogger(__name__)


class MultiLevelSearchOrchestrator:
    """
    Orchestrates search across multiple knowledge levels.
    """
    
    def __init__(
        self,
        config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
        vector_store=None,
        embedding_model=None,
        knowledge_storage_manager=None
    ):
        """
        Initialize the multi-level search orchestrator.
        
        Args:
            config: Configuration for the orchestrator
            vector_store: Qdrant vector store instance
            embedding_model: Embedding model instance
            knowledge_storage_manager: Knowledge storage manager
        """
        self.config = config
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.knowledge_storage_manager = knowledge_storage_manager
        
        # Initialize retrievers
        self.subject_retriever = None
        self.chunk_retriever = None
        self.document_retriever = None
        self.hybrid_retriever = None
        self.context_retriever = None
        
        # Initialize discovery services
        self.cluster_discovery = None
        self.document_discovery = None
        
        # Search configuration
        self.search_levels = ["subject", "document", "chunk"]
        self.fallback_thresholds = {
            "subject": 3,  # Minimum subject results before fallback
            "document": 5,  # Minimum document results before fallback
            "chunk": 10   # Minimum chunk results before fallback
        }
        
        self._initialized = False
    
    def _initialize(self) -> None:
        """Initialize all retrievers and services."""
        if self._initialized:
            return
        
        try:
            # Initialize retrievers
            self.subject_retriever = SubjectRetriever(
                config=self.config,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                knowledge_storage_manager=self.knowledge_storage_manager
            )
            
            self.chunk_retriever = SemanticRetriever(
                config=self.config,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model
            )
            
            self.hybrid_retriever = HybridRetriever(
                config=self.config,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model
            )
            
            self.context_retriever = ContextRetriever(
                config=self.config,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model
            )
            
            # Initialize discovery services
            self.cluster_discovery = ClusterDiscoveryService(
                vector_store=self.vector_store,
                knowledge_storage_manager=self.knowledge_storage_manager
            )
            
            self.document_discovery = DocumentDiscoveryService(
                vector_store=self.vector_store,
                knowledge_storage_manager=self.knowledge_storage_manager
            )
            
            self._initialized = True
            logger.info("Multi-level search orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-level search orchestrator: {e}")
            raise RetrievalError(f"Initialization failed: {e}") from e
    
    def search(
        self,
        query: str,
        search_level: str = "auto",
        filters: Optional[Dict[str, Any]] = None,
        include_discovery: bool = False,
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search across multiple knowledge levels.
        
        Args:
            query: Search query
            search_level: Search level ("auto", "subject", "document", "chunk", "all")
            filters: Optional metadata filters
            include_discovery: Whether to include discovery information
            max_results: Maximum number of results per level
            
        Returns:
            Comprehensive search results
        """
        try:
            if not self._initialized:
                self._initialize()
            
            # Validate query
            if not query or not query.strip():
                raise InvalidQueryError("Query cannot be empty")
            
            # Determine search level
            if search_level == "auto":
                search_level = self._determine_search_level(query)
            
            # Classify query intent
            query_intent = self._classify_query_intent(query)
            
            # Execute search based on level
            if search_level == "all":
                results = self._search_all_levels(query, filters, query_intent, max_results)
            elif search_level == "subject":
                results = self._search_subject_level(query, filters, query_intent, max_results)
            elif search_level == "document":
                results = self._search_document_level(query, filters, query_intent, max_results)
            elif search_level == "chunk":
                results = self._search_chunk_level(query, filters, query_intent, max_results)
            else:
                raise InvalidQueryError(f"Unknown search level: {search_level}")
            
            # Add discovery information if requested
            if include_discovery:
                results["discovery"] = self._get_discovery_information()
            
            # Add query metadata
            results["query_metadata"] = {
                "original_query": query,
                "search_level": search_level,
                "query_intent": query_intent,
                "timestamp": datetime.utcnow().isoformat(),
                "total_results": self._count_total_results(results)
            }
            
            logger.info(f"Multi-level search completed: {search_level} level, "
                       f"{results['query_metadata']['total_results']} total results")
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-level search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_metadata": {
                    "original_query": query,
                    "search_level": search_level,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    def _determine_search_level(self, query: str) -> str:
        """
        Automatically determine the appropriate search level.
        
        Args:
            query: User query
            
        Returns:
            Recommended search level
        """
        query_lower = query.lower()
        
        # Subject-level indicators
        subject_indicators = [
            "main concepts", "overview", "summary", "fundamentals",
            "principles", "key topics", "core ideas", "general",
            "what is", "explain", "describe", "tell me about",
            "basics", "introduction", "foundation"
        ]
        
        # Document-level indicators
        document_indicators = [
            "specific document", "in this paper", "author states",
            "according to", "document mentions", "paper says",
            "research shows", "study found", "article discusses"
        ]
        
        # Chunk-level indicators
        chunk_indicators = [
            "specific detail", "exact quote", "precise information",
            "specific example", "particular case", "exactly",
            "quote", "citation", "reference"
        ]
        
        # Discovery indicators
        discovery_indicators = [
            "what topics", "what subjects", "what clusters", "what documents",
            "list", "show me", "available", "in the system",
            "what do you have", "what's available"
        ]
        
        # Check for discovery queries first
        if any(indicator in query_lower for indicator in discovery_indicators):
            return "discovery"
        
        # Check for subject-level queries
        elif any(indicator in query_lower for indicator in subject_indicators):
            return "subject"
        
        # Check for document-level queries
        elif any(indicator in query_lower for indicator in document_indicators):
            return "document"
        
        # Check for chunk-level queries
        elif any(indicator in query_lower for indicator in chunk_indicators):
            return "chunk"
        
        # Default to subject-level for general queries
        else:
            return "subject"
    
    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent for better search routing.
        
        Args:
            query: User query
            
        Returns:
            Query intent classification
        """
        query_lower = query.lower()
        
        intent = {
            "primary_intent": "information_retrieval",
            "secondary_intents": [],
            "target_clusters": [],
            "complexity": "simple",
            "requires_reasoning": False,
            "requires_comparison": False,
            "requires_synthesis": False
        }
        
        # Primary intent classification
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "contrast"]):
            intent["primary_intent"] = "comparison"
            intent["requires_comparison"] = True
            intent["complexity"] = "complex"
        
        elif any(word in query_lower for word in ["why", "how", "explain", "reason"]):
            intent["primary_intent"] = "explanation"
            intent["requires_reasoning"] = True
            intent["complexity"] = "complex"
        
        elif any(word in query_lower for word in ["summarize", "synthesize", "overview"]):
            intent["primary_intent"] = "synthesis"
            intent["requires_synthesis"] = True
            intent["complexity"] = "complex"
        
        elif any(word in query_lower for word in ["what", "list", "show", "find"]):
            intent["primary_intent"] = "information_retrieval"
            intent["complexity"] = "simple"
        
        # Extract target clusters
        intent["target_clusters"] = self._extract_target_clusters(query)
        
        # Determine complexity
        if len(query.split()) > 10 or intent["requires_reasoning"] or intent["requires_comparison"]:
            intent["complexity"] = "complex"
        
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
        cluster_keywords = {
            "machine learning": ["ml_cluster", "ai_cluster"],
            "artificial intelligence": ["ai_cluster", "ml_cluster"],
            "deep learning": ["dl_cluster", "ml_cluster"],
            "neural networks": ["nn_cluster", "dl_cluster"],
            "natural language processing": ["nlp_cluster", "ai_cluster"],
            "computer vision": ["cv_cluster", "ai_cluster"],
            "data science": ["ds_cluster", "ml_cluster"],
            "statistics": ["stats_cluster", "ds_cluster"],
            "algorithms": ["ml_cluster", "ai_cluster"],
            "models": ["ml_cluster", "dl_cluster"]
        }
        
        for keyword, cluster_ids in cluster_keywords.items():
            if keyword in query_lower:
                target_clusters.extend(cluster_ids)
        
        return list(set(target_clusters))  # Remove duplicates
    
    def _search_all_levels(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        query_intent: Dict[str, Any],
        max_results: Optional[int]
    ) -> Dict[str, Any]:
        """
        Search across all knowledge levels.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            query_intent: Query intent classification
            max_results: Maximum number of results per level
            
        Returns:
            Combined results from all levels
        """
        try:
            results = {
                "success": True,
                "search_level": "all",
                "subject_results": [],
                "document_results": [],
                "chunk_results": [],
                "combined_results": [],
                "level_statistics": {}
            }
            
            # Search each level
            if self.subject_retriever:
                subject_results = self.subject_retriever.get_relevant_documents(
                    query, filters
                )[:max_results or self.fallback_thresholds["subject"]]
                results["subject_results"] = subject_results
                results["level_statistics"]["subject_count"] = len(subject_results)
            
            if self.chunk_retriever:
                chunk_results = self.chunk_retriever.get_relevant_documents(
                    query, filters
                )[:max_results or self.fallback_thresholds["chunk"]]
                results["chunk_results"] = chunk_results
                results["level_statistics"]["chunk_count"] = len(chunk_results)
            
            # Combine and rank results
            results["combined_results"] = self._combine_all_level_results(results, query_intent)
            
            return results
            
        except Exception as e:
            logger.error(f"All-level search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_level": "all",
                "subject_results": [],
                "document_results": [],
                "chunk_results": [],
                "combined_results": []
            }
    
    def _search_subject_level(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        query_intent: Dict[str, Any],
        max_results: Optional[int]
    ) -> Dict[str, Any]:
        """
        Search at subject level with fallback.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            query_intent: Query intent classification
            max_results: Maximum number of results
            
        Returns:
            Subject-level search results
        """
        try:
            results = {
                "success": True,
                "search_level": "subject",
                "subject_results": [],
                "fallback_results": [],
                "combined_results": []
            }
            
            # Primary subject search
            if self.subject_retriever:
                subject_results = self.subject_retriever.get_relevant_documents(
                    query, filters
                )[:max_results or self.fallback_thresholds["subject"]]
                results["subject_results"] = subject_results
            
            # Check if fallback is needed
            if len(results["subject_results"]) < self.fallback_thresholds["subject"]:
                logger.info("Subject results insufficient, applying fallback")
                
                # Fallback to chunk search
                if self.chunk_retriever:
                    fallback_results = self.chunk_retriever.get_relevant_documents(
                        query, filters
                    )[:max_results or self.fallback_thresholds["chunk"]]
                    results["fallback_results"] = fallback_results
                    
                    # Mark fallback results
                    for doc in results["fallback_results"]:
                        doc.metadata["search_method"] = "subject_fallback"
                        doc.metadata["fallback_reason"] = "insufficient_subject_knowledge"
            
            # Combine results
            results["combined_results"] = self._combine_subject_results(results, query_intent)
            
            return results
            
        except Exception as e:
            logger.error(f"Subject-level search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_level": "subject",
                "subject_results": [],
                "fallback_results": [],
                "combined_results": []
            }
    
    def _search_document_level(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        query_intent: Dict[str, Any],
        max_results: Optional[int]
    ) -> Dict[str, Any]:
        """
        Search at document level with fallback.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            query_intent: Query intent classification
            max_results: Maximum number of results
            
        Returns:
            Document-level search results
        """
        try:
            results = {
                "success": True,
                "search_level": "document",
                "document_results": [],
                "fallback_results": [],
                "combined_results": []
            }
            
            # Document-level search would be implemented here
            # For now, fallback to chunk search
            if self.chunk_retriever:
                document_results = self.chunk_retriever.get_relevant_documents(
                    query, filters
                )[:max_results or self.fallback_thresholds["document"]]
                results["document_results"] = document_results
                
                # Mark as document-level results
                for doc in results["document_results"]:
                    doc.metadata["search_method"] = "document_level"
            
            # Check if fallback is needed
            if len(results["document_results"]) < self.fallback_thresholds["document"]:
                logger.info("Document results insufficient, applying fallback")
                
                # Fallback to chunk search
                if self.chunk_retriever:
                    fallback_results = self.chunk_retriever.get_relevant_documents(
                        query, filters
                    )[:max_results or self.fallback_thresholds["chunk"]]
                    results["fallback_results"] = fallback_results
                    
                    # Mark fallback results
                    for doc in results["fallback_results"]:
                        doc.metadata["search_method"] = "document_fallback"
                        doc.metadata["fallback_reason"] = "insufficient_document_knowledge"
            
            # Combine results
            results["combined_results"] = self._combine_document_results(results, query_intent)
            
            return results
            
        except Exception as e:
            logger.error(f"Document-level search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_level": "document",
                "document_results": [],
                "fallback_results": [],
                "combined_results": []
            }
    
    def _search_chunk_level(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        query_intent: Dict[str, Any],
        max_results: Optional[int]
    ) -> Dict[str, Any]:
        """
        Search at chunk level.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            query_intent: Query intent classification
            max_results: Maximum number of results
            
        Returns:
            Chunk-level search results
        """
        try:
            results = {
                "success": True,
                "search_level": "chunk",
                "chunk_results": [],
                "combined_results": []
            }
            
            # Chunk-level search
            if self.chunk_retriever:
                chunk_results = self.chunk_retriever.get_relevant_documents(
                    query, filters
                )[:max_results or self.fallback_thresholds["chunk"]]
                results["chunk_results"] = chunk_results
                
                # Mark as chunk-level results
                for doc in results["chunk_results"]:
                    doc.metadata["search_method"] = "chunk_level"
            
            # Combine results
            results["combined_results"] = self._combine_chunk_results(results, query_intent)
            
            return results
            
        except Exception as e:
            logger.error(f"Chunk-level search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_level": "chunk",
                "chunk_results": [],
                "combined_results": []
            }
    
    def _combine_all_level_results(
        self,
        results: Dict[str, Any],
        query_intent: Dict[str, Any]
    ) -> List[Document]:
        """
        Combine results from all levels with intelligent ranking.
        
        Args:
            results: Results from all levels
            query_intent: Query intent classification
            
        Returns:
            Combined and ranked results
        """
        try:
            combined_results = []
            
            # Add subject results with highest priority
            for doc in results.get("subject_results", []):
                doc.metadata["result_priority"] = "high"
                doc.metadata["result_type"] = "subject_knowledge"
                doc.metadata["search_level"] = "subject"
                combined_results.append(doc)
            
            # Add document results with medium priority
            for doc in results.get("document_results", []):
                doc.metadata["result_priority"] = "medium"
                doc.metadata["result_type"] = "document_knowledge"
                doc.metadata["search_level"] = "document"
                combined_results.append(doc)
            
            # Add chunk results with lower priority
            for doc in results.get("chunk_results", []):
                doc.metadata["result_priority"] = "low"
                doc.metadata["result_type"] = "chunk_content"
                doc.metadata["search_level"] = "chunk"
                combined_results.append(doc)
            
            # Sort by priority and similarity score
            combined_results.sort(
                key=lambda x: (
                    0 if x.metadata.get("result_priority") == "high" else
                    1 if x.metadata.get("result_priority") == "medium" else 2,
                    -x.metadata.get("similarity_score", 0)
                )
            )
            
            # Apply diversity filtering
            combined_results = self._apply_diversity_filtering(combined_results)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to combine all-level results: {e}")
            return []
    
    def _combine_subject_results(
        self,
        results: Dict[str, Any],
        query_intent: Dict[str, Any]
    ) -> List[Document]:
        """Combine subject and fallback results."""
        try:
            combined_results = []
            
            # Add subject results first
            for doc in results.get("subject_results", []):
                doc.metadata["result_priority"] = "high"
                doc.metadata["result_type"] = "subject_knowledge"
                combined_results.append(doc)
            
            # Add fallback results
            for doc in results.get("fallback_results", []):
                doc.metadata["result_priority"] = "medium"
                doc.metadata["result_type"] = "fallback"
                combined_results.append(doc)
            
            # Sort by priority and similarity
            combined_results.sort(
                key=lambda x: (
                    0 if x.metadata.get("result_priority") == "high" else 1,
                    -x.metadata.get("similarity_score", 0)
                )
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to combine subject results: {e}")
            return []
    
    def _combine_document_results(
        self,
        results: Dict[str, Any],
        query_intent: Dict[str, Any]
    ) -> List[Document]:
        """Combine document and fallback results."""
        try:
            combined_results = []
            
            # Add document results first
            for doc in results.get("document_results", []):
                doc.metadata["result_priority"] = "high"
                doc.metadata["result_type"] = "document_knowledge"
                combined_results.append(doc)
            
            # Add fallback results
            for doc in results.get("fallback_results", []):
                doc.metadata["result_priority"] = "medium"
                doc.metadata["result_type"] = "fallback"
                combined_results.append(doc)
            
            # Sort by priority and similarity
            combined_results.sort(
                key=lambda x: (
                    0 if x.metadata.get("result_priority") == "high" else 1,
                    -x.metadata.get("similarity_score", 0)
                )
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to combine document results: {e}")
            return []
    
    def _combine_chunk_results(
        self,
        results: Dict[str, Any],
        query_intent: Dict[str, Any]
    ) -> List[Document]:
        """Combine chunk results."""
        try:
            combined_results = []
            
            # Add chunk results
            for doc in results.get("chunk_results", []):
                doc.metadata["result_priority"] = "high"
                doc.metadata["result_type"] = "chunk_content"
                combined_results.append(doc)
            
            # Sort by similarity score
            combined_results.sort(
                key=lambda x: -x.metadata.get("similarity_score", 0)
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to combine chunk results: {e}")
            return []
    
    def _apply_diversity_filtering(self, results: List[Document]) -> List[Document]:
        """
        Apply diversity filtering to avoid duplicate results.
        
        Args:
            results: List of results to filter
            
        Returns:
            Filtered results with diversity
        """
        try:
            if not results:
                return results
            
            filtered_results = []
            seen_sources = set()
            
            for doc in results:
                # Check for source diversity
                source_id = doc.metadata.get("document_id") or doc.metadata.get("subject_id")
                if source_id and source_id not in seen_sources:
                    filtered_results.append(doc)
                    seen_sources.add(source_id)
                elif not source_id:
                    # Include results without source ID
                    filtered_results.append(doc)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to apply diversity filtering: {e}")
            return results
    
    def _get_discovery_information(self) -> Dict[str, Any]:
        """Get discovery information about available clusters and documents."""
        try:
            discovery_info = {
                "clusters": {},
                "documents": {},
                "last_updated": datetime.utcnow().isoformat()
            }
            
            if self.cluster_discovery:
                discovery_info["clusters"] = self.cluster_discovery.get_all_clusters()
            
            if self.document_discovery:
                discovery_info["documents"] = self.document_discovery.get_all_documents()
            
            return discovery_info
            
        except Exception as e:
            logger.error(f"Failed to get discovery information: {e}")
            return {"error": str(e)}
    
    def _count_total_results(self, results: Dict[str, Any]) -> int:
        """Count total results across all levels."""
        try:
            total = 0
            total += len(results.get("subject_results", []))
            total += len(results.get("document_results", []))
            total += len(results.get("chunk_results", []))
            total += len(results.get("fallback_results", []))
            return total
            
        except Exception as e:
            logger.error(f"Failed to count total results: {e}")
            return 0
    
    def get_search_capabilities(self) -> Dict[str, Any]:
        """Get information about search capabilities."""
        try:
            return {
                "available_levels": self.search_levels,
                "fallback_thresholds": self.fallback_thresholds,
                "retrievers_available": {
                    "subject_retriever": self.subject_retriever is not None,
                    "chunk_retriever": self.chunk_retriever is not None,
                    "hybrid_retriever": self.hybrid_retriever is not None,
                    "context_retriever": self.context_retriever is not None
                },
                "discovery_services": {
                    "cluster_discovery": self.cluster_discovery is not None,
                    "document_discovery": self.document_discovery is not None
                },
                "query_classification": True,
                "multi_level_search": True,
                "fallback_mechanisms": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get search capabilities: {e}")
            return {"error": str(e)}


# Convenience functions
def create_multi_level_orchestrator(
    config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
    vector_store=None,
    embedding_model=None,
    knowledge_storage_manager=None
) -> MultiLevelSearchOrchestrator:
    """Create a multi-level search orchestrator instance."""
    return MultiLevelSearchOrchestrator(
        config, vector_store, embedding_model, knowledge_storage_manager
    )


def search_multi_level(
    query: str,
    search_level: str = "auto",
    config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
    filters: Optional[Dict[str, Any]] = None,
    include_discovery: bool = False
) -> Dict[str, Any]:
    """
    Perform multi-level search.
    
    Args:
        query: Search query
        search_level: Search level ("auto", "subject", "document", "chunk", "all")
        config: Retriever configuration
        filters: Optional metadata filters
        include_discovery: Whether to include discovery information
        
    Returns:
        Multi-level search results
    """
    orchestrator = MultiLevelSearchOrchestrator(config)
    return orchestrator.search(query, search_level, filters, include_discovery)
