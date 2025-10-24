"""
Utility functions for retrieval processing.

This module provides utility functions for processing retrieval results,
calculating relevance scores, filtering by metadata, and other common
retrieval operations.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import re
import json

logger = logging.getLogger(__name__)


def format_retrieval_results(
    documents: List[Dict[str, Any]], 
    format_type: str = "standard"
) -> List[Dict[str, Any]]:
    """
    Format retrieval results for different output formats.
    
    Args:
        documents: List of retrieved documents
        format_type: Format type ("standard", "minimal", "detailed")
        
    Returns:
        Formatted results
    """
    if not documents:
        return []
    
    try:
        if format_type == "minimal":
            return [
                {
                    "document_id": doc.metadata.get("document_id"),
                    "title": doc.metadata.get("title", ""),
                    "score": doc.metadata.get("similarity_score", 0)
                }
                for doc in documents
            ]
        
        elif format_type == "detailed":
            return [
                {
                    "document_id": doc.metadata.get("document_id"),
                    "title": doc.metadata.get("title", ""),
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "similarity_score": doc.metadata.get("similarity_score", 0),
                    "relevance_score": doc.metadata.get("relevance_score", 0),
                    "quality_score": doc.metadata.get("quality_score", 0),
                    "search_method": doc.metadata.get("search_method", "unknown"),
                    "metadata": doc.metadata,
                    "timestamp": datetime.now().isoformat()
                }
                for doc in documents
            ]
        
        else:  # standard format
            return [
                {
                    "document_id": doc.metadata.get("document_id"),
                    "title": doc.metadata.get("title", ""),
                    "summary": doc.metadata.get("summary", ""),
                    "similarity_score": doc.metadata.get("similarity_score", 0),
                    "relevance_score": doc.metadata.get("relevance_score", 0),
                    "quality_score": doc.metadata.get("quality_score", 0),
                    "search_method": doc.metadata.get("search_method", "unknown"),
                    "category": doc.metadata.get("category", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "timestamp": datetime.now().isoformat()
                }
                for doc in documents
            ]
            
    except Exception as e:
        logger.error(f"Failed to format results: {e}")
        return []


def calculate_relevance_score(doc: Dict[str, Any], query: str) -> float:
    """
    Calculate relevance score for a document given a query.
    
    Args:
        doc: Document with metadata
        query: Search query
        
    Returns:
        Relevance score between 0 and 1
    """
    try:
        score = 0.0
        
        # Base similarity score
        similarity_score = doc.metadata.get("similarity_score", 0.0)
        score += 0.4 * similarity_score
        
        # Content relevance (simple keyword matching)
        content = doc.page_content.lower()
        query_terms = query.lower().split()
        
        # Count query term matches
        term_matches = sum(1 for term in query_terms if term in content)
        term_score = min(1.0, term_matches / len(query_terms)) if query_terms else 0.0
        score += 0.3 * term_score
        
        # Title relevance
        title = doc.metadata.get("title", "").lower()
        title_matches = sum(1 for term in query_terms if term in title)
        title_score = min(1.0, title_matches / len(query_terms)) if query_terms else 0.0
        score += 0.2 * title_score
        
        # Quality score influence
        quality_score = doc.metadata.get("quality_score", 0.0)
        score += 0.1 * quality_score
        
        return min(1.0, score)
        
    except Exception as e:
        logger.warning(f"Failed to calculate relevance score: {e}")
        return 0.0


def filter_by_metadata(
    documents: List[Dict[str, Any]], 
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter documents by metadata criteria.
    
    Args:
        documents: List of documents to filter
        filters: Filter criteria
        
    Returns:
        Filtered documents
    """
    if not filters:
        return documents
    
    try:
        filtered_docs = []
        
        for doc in documents:
            include = True
            
            for key, value in filters.items():
                doc_value = doc.metadata.get(key)
                
                if isinstance(value, list):
                    # Check if document value is in the list
                    if doc_value not in value:
                        include = False
                        break
                elif isinstance(value, dict):
                    # Range filtering
                    if "min" in value and doc_value < value["min"]:
                        include = False
                        break
                    if "max" in value and doc_value > value["max"]:
                        include = False
                        break
                else:
                    # Exact match
                    if doc_value != value:
                        include = False
                        break
            
            if include:
                filtered_docs.append(doc)
        
        logger.info(f"Metadata filtering: {len(documents)} -> {len(filtered_docs)} documents")
        return filtered_docs
        
    except Exception as e:
        logger.error(f"Failed to filter by metadata: {e}")
        return documents


def rank_by_quality(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank documents by quality score.
    
    Args:
        documents: List of documents to rank
        
    Returns:
        Documents ranked by quality
    """
    try:
        # Sort by quality score (descending)
        ranked_docs = sorted(
            documents, 
            key=lambda x: x.metadata.get("quality_score", 0), 
            reverse=True
        )
        
        logger.info(f"Ranked {len(ranked_docs)} documents by quality")
        return ranked_docs
        
    except Exception as e:
        logger.error(f"Failed to rank by quality: {e}")
        return documents


def batch_retrieve(
    retriever,
    queries: List[str],
    batch_size: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[List[Dict[str, Any]]]:
    """
    Perform batch retrieval for multiple queries.
    
    Args:
        retriever: Retriever instance
        queries: List of queries
        batch_size: Batch size for processing
        filters: Optional metadata filters
        **kwargs: Additional parameters
        
    Returns:
        List of results for each query
    """
    try:
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_results = []
            
            for query in batch_queries:
                try:
                    docs = retriever.get_relevant_documents(query, filters, **kwargs)
                    batch_results.append(docs)
                except Exception as e:
                    logger.error(f"Failed to retrieve for query '{query}': {e}")
                    batch_results.append([])
            
            results.extend(batch_results)
        
        logger.info(f"Batch retrieval completed: {len(queries)} queries processed")
        return results
        
    except Exception as e:
        logger.error(f"Batch retrieval failed: {e}")
        return [[] for _ in queries]


def merge_retrieval_results(
    results_list: List[List[Dict[str, Any]]],
    merge_strategy: str = "union"
) -> List[Dict[str, Any]]:
    """
    Merge multiple retrieval result lists.
    
    Args:
        results_list: List of result lists to merge
        merge_strategy: Strategy for merging ("union", "intersection", "weighted")
        
    Returns:
        Merged results
    """
    if not results_list:
        return []
    
    try:
        if merge_strategy == "union":
            # Union: combine all results, remove duplicates
            seen_ids = set()
            merged_results = []
            
            for results in results_list:
                for doc in results:
                    doc_id = doc.metadata.get("document_id", id(doc))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        merged_results.append(doc)
            
            return merged_results
        
        elif merge_strategy == "intersection":
            # Intersection: only documents that appear in all results
            if len(results_list) < 2:
                return results_list[0] if results_list else []
            
            # Get document IDs from first result
            first_ids = {doc.metadata.get("document_id", id(doc)) for doc in results_list[0]}
            
            # Find intersection
            for results in results_list[1:]:
                current_ids = {doc.metadata.get("document_id", id(doc)) for doc in results}
                first_ids = first_ids.intersection(current_ids)
            
            # Return documents with IDs in intersection
            intersection_results = []
            for results in results_list[0]:
                doc_id = results.metadata.get("document_id", id(results))
                if doc_id in first_ids:
                    intersection_results.append(results)
            
            return intersection_results
        
        elif merge_strategy == "weighted":
            # Weighted: combine scores from multiple retrievals
            doc_scores = {}
            
            for i, results in enumerate(results_list):
                weight = 1.0 / len(results_list)  # Equal weight for each result set
                
                for doc in results:
                    doc_id = doc.metadata.get("document_id", id(doc))
                    score = doc.metadata.get("similarity_score", 0)
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {"doc": doc, "total_score": 0, "count": 0}
                    
                    doc_scores[doc_id]["total_score"] += weight * score
                    doc_scores[doc_id]["count"] += 1
            
            # Create merged results with averaged scores
            merged_results = []
            for doc_id, data in doc_scores.items():
                doc = data["doc"]
                avg_score = data["total_score"] / data["count"]
                doc.metadata["merged_score"] = avg_score
                doc.metadata["source_count"] = data["count"]
                merged_results.append(doc)
            
            # Sort by merged score
            merged_results.sort(key=lambda x: x.metadata.get("merged_score", 0), reverse=True)
            return merged_results
        
        else:
            logger.warning(f"Unknown merge strategy: {merge_strategy}")
            return results_list[0] if results_list else []
            
    except Exception as e:
        logger.error(f"Failed to merge results: {e}")
        return results_list[0] if results_list else []


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple heuristics.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    try:
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in keywords[:max_keywords]]
        
    except Exception as e:
        logger.warning(f"Failed to extract keywords: {e}")
        return []


def calculate_diversity_score(documents: List[Dict[str, Any]]) -> float:
    """
    Calculate diversity score for a set of documents.
    
    Args:
        documents: List of documents
        
    Returns:
        Diversity score between 0 and 1
    """
    if len(documents) <= 1:
        return 1.0
    
    try:
        # Extract categories/subjects for diversity calculation
        categories = [doc.metadata.get("category", "") for doc in documents]
        subjects = [doc.metadata.get("subject", "") for doc in documents]
        
        # Calculate category diversity
        unique_categories = len(set(categories))
        category_diversity = unique_categories / len(categories)
        
        # Calculate subject diversity
        unique_subjects = len(set(subjects))
        subject_diversity = unique_subjects / len(subjects)
        
        # Combined diversity score
        diversity_score = (category_diversity + subject_diversity) / 2
        
        return min(1.0, diversity_score)
        
    except Exception as e:
        logger.warning(f"Failed to calculate diversity score: {e}")
        return 0.0


def validate_query(query: str) -> bool:
    """
    Validate a search query.
    
    Args:
        query: Query to validate
        
    Returns:
        True if query is valid, False otherwise
    """
    if not query or not query.strip():
        return False
    
    # Check minimum length
    if len(query.strip()) < 2:
        return False
    
    # Check for excessive length
    if len(query) > 1000:
        return False
    
    # Check for only special characters
    if not re.search(r'[a-zA-Z0-9]', query):
        return False
    
    return True


def normalize_score(score: float, min_score: float = 0.0, max_score: float = 1.0) -> float:
    """
    Normalize a score to the 0-1 range.
    
    Args:
        score: Score to normalize
        min_score: Minimum possible score
        max_score: Maximum possible score
        
    Returns:
        Normalized score
    """
    try:
        if max_score == min_score:
            return 0.0
        
        normalized = (score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))
        
    except Exception as e:
        logger.warning(f"Failed to normalize score: {e}")
        return 0.0
