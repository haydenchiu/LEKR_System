"""
AgentChat Integration for LERK System QA Agent

This module provides integration between AgentChat and the LERK System's
QA agent module, enabling conversational interfaces for knowledge retrieval
and question answering.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import asyncio

logger = logging.getLogger(__name__)


class AgentChatLERKIntegration:
    """
    Integration between AgentChat and LERK System QA Agent.
    
    This class provides a bridge between AgentChat's conversational interface
    and the LERK System's sophisticated QA capabilities, including subject-level
    retrieval, multi-level search, and knowledge consolidation.
    """
    
    def __init__(
        self,
        qa_orchestrator=None,
        integration_manager=None,
        agentchat_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AgentChat-LERK integration.
        
        Args:
            qa_orchestrator: QAOrchestrator instance from qa_agent module
            integration_manager: LERKIntegrationManager instance
            agentchat_config: AgentChat configuration
        """
        self.qa_orchestrator = qa_orchestrator
        self.integration_manager = integration_manager
        self.agentchat_config = agentchat_config or self._get_default_agentchat_config()
        
        # Session management
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Query classification and routing
        self._query_classifiers = {
            "subject_level": self._is_subject_level_query,
            "document_level": self._is_document_level_query,
            "chunk_level": self._is_chunk_level_query,
            "discovery": self._is_discovery_query,
            "system": self._is_system_query
        }
    
    def _get_default_agentchat_config(self) -> Dict[str, Any]:
        """Get default AgentChat configuration."""
        return {
            "max_conversation_length": 50,
            "context_window_size": 10,
            "enable_multi_turn": True,
            "enable_subject_fallback": True,
            "enable_discovery_mode": True,
            "response_format": "conversational",
            "include_sources": True,
            "include_confidence": True
        }
    
    async def process_chat_message(
        self,
        message: str,
        session_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message through the LERK System.
        
        Args:
            message: User's chat message
            session_id: Unique session identifier
            user_context: Optional user context and preferences
            
        Returns:
            Response dictionary with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing chat message for session {session_id}")
            
            # Initialize session if needed
            if session_id not in self._active_sessions:
                await self._initialize_session(session_id, user_context)
            
            # Classify query type
            query_type = await self._classify_query(message)
            
            # Get conversation context
            conversation_context = self._get_conversation_context(session_id)
            
            # Process through appropriate QA pipeline
            response = await self._process_query_by_type(
                message, query_type, conversation_context, session_id
            )
            
            # Update conversation history
            self._update_conversation_history(session_id, message, response)
            
            # Format response for AgentChat
            formatted_response = self._format_response_for_agentchat(response, query_type)
            
            logger.info(f"Successfully processed chat message for session {session_id}")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to process chat message: {e}")
            return self._create_error_response(str(e))
    
    async def _initialize_session(
        self,
        session_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize a new chat session."""
        try:
            self._active_sessions[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "user_context": user_context or {},
                "query_count": 0,
                "preferred_search_level": "auto",
                "last_activity": datetime.utcnow().isoformat()
            }
            
            self._conversation_history[session_id] = []
            
            logger.info(f"Initialized session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize session {session_id}: {e}")
    
    async def _classify_query(self, message: str) -> str:
        """
        Classify the type of query to determine processing strategy.
        
        Args:
            message: User's message
            
        Returns:
            Query type classification
        """
        try:
            message_lower = message.lower()
            
            # Check for discovery queries
            if any(keyword in message_lower for keyword in [
                "what topics", "what subjects", "what clusters", "what documents",
                "show me", "list", "available", "what's in"
            ]):
                return "discovery"
            
            # Check for system queries
            if any(keyword in message_lower for keyword in [
                "help", "how to", "what can you", "capabilities", "commands"
            ]):
                return "system"
            
            # Check for subject-level queries (high-level concepts)
            if any(keyword in message_lower for keyword in [
                "overview", "summary", "main concepts", "key ideas", "general",
                "what is", "explain", "describe", "tell me about"
            ]):
                return "subject_level"
            
            # Check for document-level queries
            if any(keyword in message_lower for keyword in [
                "document", "paper", "article", "report", "in the document"
            ]):
                return "document_level"
            
            # Default to chunk-level for specific questions
            return "chunk_level"
            
        except Exception as e:
            logger.error(f"Failed to classify query: {e}")
            return "chunk_level"
    
    def _get_conversation_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation context for the session."""
        try:
            return self._conversation_history.get(session_id, [])[-self.agentchat_config["context_window_size"]:]
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return []
    
    async def _process_query_by_type(
        self,
        message: str,
        query_type: str,
        conversation_context: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Process query based on its classification.
        
        Args:
            message: User's message
            query_type: Classified query type
            conversation_context: Previous conversation context
            session_id: Session identifier
            
        Returns:
            Processing results
        """
        try:
            if query_type == "discovery":
                return await self._process_discovery_query(message, session_id)
            
            elif query_type == "system":
                return await self._process_system_query(message, session_id)
            
            elif query_type == "subject_level":
                return await self._process_subject_level_query(message, conversation_context, session_id)
            
            elif query_type == "document_level":
                return await self._process_document_level_query(message, conversation_context, session_id)
            
            else:  # chunk_level
                return await self._process_chunk_level_query(message, conversation_context, session_id)
                
        except Exception as e:
            logger.error(f"Failed to process {query_type} query: {e}")
            return self._create_error_response(str(e))
    
    async def _process_discovery_query(self, message: str, session_id: str) -> Dict[str, Any]:
        """Process discovery queries (what topics/clusters are available)."""
        try:
            # Use discovery services from retriever module
            from retriever import ClusterDiscoveryService, DocumentDiscoveryService
            
            cluster_service = ClusterDiscoveryService()
            document_service = DocumentDiscoveryService()
            
            # Get available clusters and documents
            clusters = cluster_service.get_all_clusters()
            documents = document_service.get_all_documents()
            
            response = {
                "query_type": "discovery",
                "answer": f"I found {clusters.get('cluster_count', 0)} subject clusters and {documents.get('document_count', 0)} documents in the knowledge base.",
                "sources": {
                    "clusters": clusters.get("clusters", []),
                    "documents": documents.get("documents", [])
                },
                "confidence": 1.0,
                "metadata": {
                    "discovery_type": "system_overview",
                    "session_id": session_id
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Discovery query processing failed: {e}")
            return self._create_error_response(str(e))
    
    async def _process_system_query(self, message: str, session_id: str) -> Dict[str, Any]:
        """Process system help and capability queries."""
        try:
            help_response = {
                "query_type": "system",
                "answer": """I'm the LERK System QA Agent, powered by advanced knowledge retrieval and consolidation. Here's what I can help you with:

🔍 **Discovery**: Ask "What topics are available?" or "Show me the clusters"
📚 **Subject-Level Questions**: "What are the main concepts in machine learning?"
📄 **Document-Specific**: "What does document X say about Y?"
🔎 **Specific Questions**: "How does neural network training work?"

I can search across multiple levels of knowledge and provide comprehensive answers with sources and confidence scores.""",
                "sources": [],
                "confidence": 1.0,
                "metadata": {
                    "help_type": "capabilities",
                    "session_id": session_id
                }
            }
            
            return help_response
            
        except Exception as e:
            logger.error(f"System query processing failed: {e}")
            return self._create_error_response(str(e))
    
    async def _process_subject_level_query(
        self,
        message: str,
        conversation_context: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """Process subject-level queries using our sophisticated retrieval system."""
        try:
            # Use multi-level search orchestrator for intelligent routing
            from retriever import MultiLevelSearchOrchestrator
            
            orchestrator = MultiLevelSearchOrchestrator()
            
            # Search with subject-level preference
            results = orchestrator.search(
                query=message,
                search_level="subject",
                include_discovery=True,
                max_results=10
            )
            
            # Check if fallback was used
            fallback_used = any(
                doc.metadata.get("fallback_reason") for doc in results
            )
            
            # Generate comprehensive answer
            answer = self._generate_subject_level_answer(results, fallback_used)
            
            response = {
                "query_type": "subject_level",
                "answer": answer,
                "sources": [self._format_source(doc) for doc in results[:5]],
                "confidence": self._calculate_response_confidence(results),
                "metadata": {
                    "fallback_used": fallback_used,
                    "result_count": len(results),
                    "session_id": session_id
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Subject-level query processing failed: {e}")
            return self._create_error_response(str(e))
    
    async def _process_document_level_query(
        self,
        message: str,
        conversation_context: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """Process document-level queries."""
        try:
            # Use document-level search
            from retriever import MultiLevelSearchOrchestrator
            
            orchestrator = MultiLevelSearchOrchestrator()
            
            results = orchestrator.search(
                query=message,
                search_level="document",
                max_results=8
            )
            
            answer = self._generate_document_level_answer(results)
            
            response = {
                "query_type": "document_level",
                "answer": answer,
                "sources": [self._format_source(doc) for doc in results[:5]],
                "confidence": self._calculate_response_confidence(results),
                "metadata": {
                    "result_count": len(results),
                    "session_id": session_id
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Document-level query processing failed: {e}")
            return self._create_error_response(str(e))
    
    async def _process_chunk_level_query(
        self,
        message: str,
        conversation_context: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """Process chunk-level queries for specific information."""
        try:
            # Use chunk-level search
            from retriever import MultiLevelSearchOrchestrator
            
            orchestrator = MultiLevelSearchOrchestrator()
            
            results = orchestrator.search(
                query=message,
                search_level="chunk",
                max_results=6
            )
            
            answer = self._generate_chunk_level_answer(results)
            
            response = {
                "query_type": "chunk_level",
                "answer": answer,
                "sources": [self._format_source(doc) for doc in results[:4]],
                "confidence": self._calculate_response_confidence(results),
                "metadata": {
                    "result_count": len(results),
                    "session_id": session_id
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Chunk-level query processing failed: {e}")
            return self._create_error_response(str(e))
    
    def _generate_subject_level_answer(self, results: List[Any], fallback_used: bool) -> str:
        """Generate a comprehensive subject-level answer."""
        try:
            if not results:
                return "I couldn't find sufficient information to provide a comprehensive answer about this topic. The knowledge base may not have enough consolidated subject knowledge yet."
            
            # Group results by knowledge type
            subject_results = [r for r in results if r.metadata.get("knowledge_type") == "subject"]
            chunk_results = [r for r in results if r.metadata.get("knowledge_type") == "chunk"]
            
            answer_parts = []
            
            if subject_results:
                answer_parts.append("Based on the consolidated knowledge in the system:")
                for result in subject_results[:2]:
                    answer_parts.append(f"• {result.page_content[:200]}...")
            
            if chunk_results:
                if fallback_used:
                    answer_parts.append("\nI also found some specific information from individual documents:")
                else:
                    answer_parts.append("\nAdditional specific details:")
                
                for result in chunk_results[:3]:
                    answer_parts.append(f"• {result.page_content[:150]}...")
            
            if fallback_used:
                answer_parts.append("\n*Note: The subject knowledge is still being consolidated, so I've included information from individual documents.*")
            
            return "\n".join(answer_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate subject-level answer: {e}")
            return "I found some information but had trouble formatting a comprehensive answer."
    
    def _generate_document_level_answer(self, results: List[Any]) -> str:
        """Generate a document-level answer."""
        try:
            if not results:
                return "I couldn't find any documents matching your query."
            
            answer_parts = ["Based on the documents in the knowledge base:"]
            
            for i, result in enumerate(results[:3], 1):
                doc_title = result.metadata.get("document_title", f"Document {i}")
                answer_parts.append(f"\n**{doc_title}:**")
                answer_parts.append(f"{result.page_content[:200]}...")
            
            return "\n".join(answer_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate document-level answer: {e}")
            return "I found some documents but had trouble formatting the answer."
    
    def _generate_chunk_level_answer(self, results: List[Any]) -> str:
        """Generate a chunk-level answer."""
        try:
            if not results:
                return "I couldn't find specific information matching your query."
            
            answer_parts = ["Here's what I found:"]
            
            for i, result in enumerate(results[:3], 1):
                answer_parts.append(f"\n{i}. {result.page_content[:250]}...")
            
            return "\n".join(answer_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate chunk-level answer: {e}")
            return "I found some information but had trouble formatting the answer."
    
    def _format_source(self, doc: Any) -> Dict[str, Any]:
        """Format a document as a source."""
        try:
            return {
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": {
                    "document_id": doc.metadata.get("document_id", "unknown"),
                    "document_title": doc.metadata.get("document_title", "Untitled"),
                    "similarity_score": doc.metadata.get("similarity_score", 0.0),
                    "knowledge_type": doc.metadata.get("knowledge_type", "unknown"),
                    "fallback_reason": doc.metadata.get("fallback_reason")
                }
            }
        except Exception as e:
            logger.error(f"Failed to format source: {e}")
            return {"content": "Source formatting error", "metadata": {}}
    
    def _calculate_response_confidence(self, results: List[Any]) -> float:
        """Calculate confidence score for the response."""
        try:
            if not results:
                return 0.0
            
            # Calculate average similarity score
            similarity_scores = [
                r.metadata.get("similarity_score", 0.0) for r in results
            ]
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            
            # Adjust based on result count and quality
            confidence = avg_similarity
            if len(results) >= 3:
                confidence += 0.1
            if len(results) >= 5:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _format_response_for_agentchat(
        self,
        response: Dict[str, Any],
        query_type: str
    ) -> Dict[str, Any]:
        """Format response for AgentChat interface."""
        try:
            formatted = {
                "message": response["answer"],
                "type": "assistant",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "query_type": query_type,
                    "confidence": response.get("confidence", 0.0),
                    "sources_count": len(response.get("sources", [])),
                    "lerk_system": True
                }
            }
            
            # Add sources if configured
            if self.agentchat_config.get("include_sources", True):
                formatted["sources"] = response.get("sources", [])
            
            # Add confidence if configured
            if self.agentchat_config.get("include_confidence", True):
                formatted["confidence_score"] = response.get("confidence", 0.0)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format response for AgentChat: {e}")
            return {
                "message": "I encountered an error processing your request.",
                "type": "assistant",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"error": True}
            }
    
    def _update_conversation_history(
        self,
        session_id: str,
        user_message: str,
        assistant_response: Dict[str, Any]
    ) -> None:
        """Update conversation history for the session."""
        try:
            if session_id not in self._conversation_history:
                self._conversation_history[session_id] = []
            
            # Add user message
            self._conversation_history[session_id].append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Add assistant response
            self._conversation_history[session_id].append({
                "role": "assistant",
                "content": assistant_response["answer"],
                "metadata": assistant_response.get("metadata", {}),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Trim conversation history if too long
            max_length = self.agentchat_config.get("max_conversation_length", 50)
            if len(self._conversation_history[session_id]) > max_length:
                self._conversation_history[session_id] = self._conversation_history[session_id][-max_length:]
            
            # Update session activity
            if session_id in self._active_sessions:
                self._active_sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
                self._active_sessions[session_id]["query_count"] += 1
            
        except Exception as e:
            logger.error(f"Failed to update conversation history: {e}")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "message": f"I encountered an error: {error_message}. Please try rephrasing your question.",
            "type": "assistant",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "error": True,
                "error_message": error_message
            }
        }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        try:
            if session_id not in self._active_sessions:
                return {"error": "Session not found"}
            
            session_info = self._active_sessions[session_id].copy()
            session_info["conversation_length"] = len(self._conversation_history.get(session_id, []))
            
            return session_info
            
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return {"error": str(e)}
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        try:
            return list(self._active_sessions.keys())
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []


# Convenience functions
def create_agentchat_integration(
    qa_orchestrator=None,
    integration_manager=None,
    agentchat_config: Optional[Dict[str, Any]] = None
) -> AgentChatLERKIntegration:
    """Create an AgentChat-LERK integration instance."""
    return AgentChatLERKIntegration(
        qa_orchestrator=qa_orchestrator,
        integration_manager=integration_manager,
        agentchat_config=agentchat_config
    )


async def process_chat_message_with_lerk(
    message: str,
    session_id: str,
    qa_orchestrator=None,
    integration_manager=None,
    user_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a chat message through the LERK System.
    
    This is a convenience function for direct integration with AgentChat.
    """
    integration = AgentChatLERKIntegration(
        qa_orchestrator=qa_orchestrator,
        integration_manager=integration_manager
    )
    
    return await integration.process_chat_message(message, session_id, user_context)


__version__ = "1.0.0"
__author__ = "LERK System Team"

# Public API
__all__ = [
    "AgentChatLERKIntegration",
    "create_agentchat_integration",
    "process_chat_message_with_lerk",
]
