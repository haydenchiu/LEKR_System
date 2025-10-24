"""
QA Orchestrator Module

This module provides the main interface for question answering in the LERK System.
It orchestrates the entire QA workflow, managing sessions, context, and providing
a unified interface for asking questions and getting answers.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage

from .config import QAConfig, DEFAULT_QA_CONFIG
from .exceptions import QAError, SessionError, ValidationError, TimeoutError
from .supervisor_agent import SupervisorAgent
from .utils import format_answer, extract_sources, validate_question, create_session, cleanup_session

logger = logging.getLogger(__name__)


class QAOrchestrator:
    """
    Main orchestrator for question answering in the LERK System.
    
    This class provides a unified interface for asking questions and getting
    comprehensive answers, managing conversation context, and handling sessions.
    """

    def __init__(self, config: QAConfig = DEFAULT_QA_CONFIG):
        self.config = config
        self.supervisor_agent = SupervisorAgent(config)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._initialize_logging()

    def _initialize_logging(self):
        """Initialize logging configuration."""
        if self.config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def ask_question(self, question: str, session_id: Optional[str] = None, 
                    context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ask a question and get a comprehensive answer.
        
        Args:
            question: The user's question
            session_id: Optional session ID for conversation context
            context: Optional additional context for the question
            
        Returns:
            Dictionary containing the answer and metadata
            
        Raises:
            ValidationError: If question is invalid
            QAError: If question answering fails
        """
        # Validate question
        validate_question(question)
        
        # Create or get session
        if not session_id:
            session_id = create_session()
        
        # Ensure session exists
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0,
                "context": []
            }
        
        try:
            # Update session activity
            self.active_sessions[session_id]["last_activity"] = datetime.now()
            self.active_sessions[session_id]["message_count"] += 1
            
            # Add context if provided
            if context:
                self.active_sessions[session_id]["context"].extend(context)
            
            # Ask the question using the supervisor agent
            answer = self.supervisor_agent.ask_question(question, session_id)
            
            # Format the answer based on configuration
            formatted_answer = format_answer(
                answer, 
                style=self.config.answer_style,
                max_length=self.config.max_answer_length,
                include_sources=self.config.include_sources
            )
            
            # Extract sources if available
            sources = extract_sources(answer) if self.config.include_sources else []
            
            # Prepare response
            response = {
                "answer": formatted_answer,
                "sources": sources,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "context_used": len(self.active_sessions[session_id]["context"]) > 0,
                "message_count": self.active_sessions[session_id]["message_count"]
            }
            
            logger.info(f"Generated answer for session {session_id}: {question[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error during question answering: {e}")
            raise QAError(f"Failed to answer question: {e}")

    async def aask_question(self, question: str, session_id: Optional[str] = None,
                           context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Asynchronously ask a question and get a comprehensive answer.
        
        Args:
            question: The user's question
            session_id: Optional session ID for conversation context
            context: Optional additional context for the question
            
        Returns:
            Dictionary containing the answer and metadata
            
        Raises:
            ValidationError: If question is invalid
            QAError: If question answering fails
        """
        # Validate question
        validate_question(question)
        
        # Create or get session
        if not session_id:
            session_id = create_session()
        
        # Ensure session exists
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0,
                "context": []
            }
        
        try:
            # Update session activity
            self.active_sessions[session_id]["last_activity"] = datetime.now()
            self.active_sessions[session_id]["message_count"] += 1
            
            # Add context if provided
            if context:
                self.active_sessions[session_id]["context"].extend(context)
            
            # Ask the question using the supervisor agent
            answer = await self.supervisor_agent.aask_question(question, session_id)
            
            # Format the answer based on configuration
            formatted_answer = format_answer(
                answer, 
                style=self.config.answer_style,
                max_length=self.config.max_answer_length,
                include_sources=self.config.include_sources
            )
            
            # Extract sources if available
            sources = extract_sources(answer) if self.config.include_sources else []
            
            # Prepare response
            response = {
                "answer": formatted_answer,
                "sources": sources,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "context_used": len(self.active_sessions[session_id]["context"]) > 0,
                "message_count": self.active_sessions[session_id]["message_count"]
            }
            
            logger.info(f"Async generated answer for session {session_id}: {question[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error during async question answering: {e}")
            raise QAError(f"Failed to answer question: {e}")

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of conversation messages
            
        Raises:
            SessionError: If session retrieval fails
        """
        if session_id not in self.active_sessions:
            raise SessionError(f"Session {session_id} not found")
        
        try:
            return self.supervisor_agent.get_conversation_history(session_id)
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            raise SessionError(f"Failed to retrieve conversation history: {e}")

    def clear_session(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: The session ID
            
        Raises:
            SessionError: If session clearing fails
        """
        if session_id not in self.active_sessions:
            raise SessionError(f"Session {session_id} not found")
        
        try:
            self.supervisor_agent.clear_session(session_id)
            # Remove from active sessions
            del self.active_sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            raise SessionError(f"Failed to clear session: {e}")

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            if current_time - session_data["last_activity"] > timedelta(seconds=self.config.session_timeout):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            try:
                cleanup_session(session_id)
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session {session_id}: {e}")

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            Dictionary containing session statistics
            
        Raises:
            SessionError: If session not found
        """
        if session_id not in self.active_sessions:
            raise SessionError(f"Session {session_id} not found")
        
        session_data = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session_data["created_at"].isoformat(),
            "last_activity": session_data["last_activity"].isoformat(),
            "message_count": session_data["message_count"],
            "context_length": len(session_data["context"]),
            "is_active": True
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get overall system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        self.cleanup_expired_sessions()
        
        return {
            "active_sessions": len(self.active_sessions),
            "total_sessions_created": sum(1 for _ in self.active_sessions.values()),
            "config": self.config.to_dict(),
            "agent_stats": self.supervisor_agent.get_agent_stats(),
            "system_uptime": datetime.now().isoformat()
        }

    def update_config(self, **kwargs):
        """
        Update the orchestrator configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config = self.config.update(**kwargs)
        self.supervisor_agent.update_config(**kwargs)
        logger.info(f"QAOrchestrator configuration updated: {kwargs}")

    def __str__(self) -> str:
        """String representation of the QA orchestrator."""
        return f"QAOrchestrator(sessions={len(self.active_sessions)}, config={self.config})"

    def __repr__(self) -> str:
        """Detailed string representation of the QA orchestrator."""
        return (f"QAOrchestrator(config={self.config}, "
                f"active_sessions={len(self.active_sessions)}, "
                f"supervisor={self.supervisor_agent})")
