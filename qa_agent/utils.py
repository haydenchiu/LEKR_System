"""
Utility functions for the QA agent module.

This module provides helper functions for tasks such as answer formatting,
source extraction, question validation, and session management.
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from .exceptions import ValidationError, SessionError

logger = logging.getLogger(__name__)


def format_answer(answer: str, style: str = "detailed", max_length: int = 2000, 
                 include_sources: bool = True) -> str:
    """
    Format an answer based on the specified style and constraints.
    
    Args:
        answer: The raw answer text
        style: The formatting style (concise, detailed, academic)
        max_length: Maximum length of the formatted answer
        include_sources: Whether to include source citations
        
    Returns:
        Formatted answer string
    """
    if not answer:
        return "I apologize, but I couldn't generate a proper response."
    
    # Truncate if too long
    if len(answer) > max_length:
        answer = answer[:max_length-3] + "..."
    
    # Apply style-specific formatting
    if style == "concise":
        # Remove excessive whitespace and make more concise
        answer = re.sub(r'\s+', ' ', answer.strip())
        # Remove redundant phrases
        answer = re.sub(r'\b(Furthermore|Moreover|Additionally|In addition)\b', '', answer, flags=re.IGNORECASE)
        
    elif style == "academic":
        # Ensure proper academic formatting
        if not answer.endswith('.'):
            answer += '.'
        # Add formal language markers if needed
        if not any(word in answer.lower() for word in ['according to', 'research shows', 'studies indicate']):
            if 'the' in answer.lower()[:50]:  # If it starts with "the"
                answer = "According to the available information, " + answer.lower()
    
    elif style == "detailed":
        # Ensure comprehensive formatting
        if len(answer.split()) < 50:  # If too short
            answer = f"Based on the available information: {answer}"
    
    return answer.strip()


def extract_sources(answer: str) -> List[Dict[str, Any]]:
    """
    Extract source citations from an answer.
    
    Args:
        answer: The answer text containing source citations
        
    Returns:
        List of source dictionaries
    """
    sources = []
    
    # Look for common citation patterns
    citation_patterns = [
        r'\[(\d+)\]',  # [1], [2], etc.
        r'\(([^)]+)\)',  # (Author, Year)
        r'Source:\s*([^\n]+)',  # Source: ...
        r'According to\s+([^,]+)',  # According to ...
    ]
    
    for pattern in citation_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        for match in matches:
            sources.append({
                "type": "citation",
                "text": match.strip(),
                "pattern": pattern
            })
    
    # Look for document references in metadata
    if "metadata" in answer.lower():
        # Extract any document references
        doc_refs = re.findall(r'document[^:]*:\s*([^\n]+)', answer, re.IGNORECASE)
        for ref in doc_refs:
            sources.append({
                "type": "document",
                "text": ref.strip(),
                "pattern": "document_reference"
            })
    
    return sources


def validate_question(question: str) -> bool:
    """
    Validate a user question.
    
    Args:
        question: The question to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If question is invalid
    """
    if not question:
        raise ValidationError("Question cannot be empty")
    
    if not isinstance(question, str):
        raise ValidationError("Question must be a string")
    
    if len(question.strip()) < 3:
        raise ValidationError("Question must be at least 3 characters long")
    
    if len(question) > 1000:
        raise ValidationError("Question is too long (maximum 1000 characters)")
    
    # Check for potentially problematic content
    if any(word in question.lower() for word in ['<script', 'javascript:', 'eval(']):
        raise ValidationError("Question contains potentially harmful content")
    
    return True


def create_session() -> str:
    """
    Create a new session ID.
    
    Returns:
        Unique session ID string
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Created new session: {session_id}")
    return session_id


def cleanup_session(session_id: str):
    """
    Clean up a session.
    
    Args:
        session_id: The session ID to clean up
        
    Raises:
        SessionError: If cleanup fails
    """
    try:
        # In a real implementation, this would clean up session data
        # For now, just log the cleanup
        logger.info(f"Cleaned up session: {session_id}")
    except Exception as e:
        logger.error(f"Failed to cleanup session {session_id}: {e}")
        raise SessionError(f"Failed to cleanup session: {e}")


def extract_keywords(question: str) -> List[str]:
    """
    Extract keywords from a question for better retrieval.
    
    Args:
        question: The user's question
        
    Returns:
        List of extracted keywords
    """
    # Simple keyword extraction (in a real system, you might use NLP libraries)
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which'}
    
    # Extract words
    words = re.findall(r'\b\w+\b', question.lower())
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords


def calculate_answer_quality(answer: str, question: str) -> Dict[str, Any]:
    """
    Calculate quality metrics for an answer.
    
    Args:
        answer: The generated answer
        question: The original question
        
    Returns:
        Dictionary containing quality metrics
    """
    if not answer or not question:
        return {"quality_score": 0.0, "metrics": {}}
    
    metrics = {}
    
    # Length metrics
    metrics["answer_length"] = len(answer)
    metrics["question_length"] = len(question)
    metrics["length_ratio"] = len(answer) / len(question) if len(question) > 0 else 0
    
    # Content metrics
    metrics["has_sources"] = bool(re.search(r'\[.*?\]|\(.*?\)|source:', answer, re.IGNORECASE))
    metrics["has_numbers"] = bool(re.search(r'\d+', answer))
    metrics["has_questions"] = bool(re.search(r'\?', answer))
    
    # Calculate overall quality score (simple heuristic)
    quality_score = 0.0
    
    # Length appropriateness (not too short, not too long)
    if 50 <= len(answer) <= 2000:
        quality_score += 0.3
    
    # Has sources
    if metrics["has_sources"]:
        quality_score += 0.2
    
    # Contains relevant keywords from question
    question_keywords = extract_keywords(question)
    answer_lower = answer.lower()
    keyword_matches = sum(1 for keyword in question_keywords if keyword in answer_lower)
    if question_keywords:
        keyword_score = keyword_matches / len(question_keywords)
        quality_score += keyword_score * 0.3
    
    # Not just a question back
    if not (answer.strip().endswith('?') and len(answer.split()) < 10):
        quality_score += 0.2
    
    metrics["quality_score"] = min(quality_score, 1.0)
    
    return {
        "quality_score": metrics["quality_score"],
        "metrics": metrics
    }


def format_conversation_history(messages: List[Dict[str, Any]]) -> str:
    """
    Format conversation history for display.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Formatted conversation history string
    """
    if not messages:
        return "No conversation history available."
    
    formatted_history = []
    for i, msg in enumerate(messages, 1):
        msg_type = msg.get("type", "Unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        if msg_type == "HumanMessage":
            formatted_history.append(f"{i}. User: {content}")
        elif msg_type == "AIMessage":
            formatted_history.append(f"{i}. Assistant: {content}")
        else:
            formatted_history.append(f"{i}. {msg_type}: {content}")
    
    return "\n".join(formatted_history)


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent potential issues.
    
    Args:
        text: The input text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove potentially harmful characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Limit length
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    return text.strip()
