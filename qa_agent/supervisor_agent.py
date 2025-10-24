"""
Supervisor Agent Module

This module implements a supervisor agent that orchestrates the QA workflow,
coordinating between the retrieval agent and answer generation to provide
comprehensive responses to user questions.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
try:
    from langgraph_supervisor import create_supervisor
except ImportError:
    # Mock for testing purposes
    from unittest.mock import Mock
    def create_supervisor(*args, **kwargs):
        return Mock()
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver

from .config import QAConfig, DEFAULT_QA_CONFIG
from .exceptions import QAError, AnswerGenerationError, SessionError
from .retrieval_agent import RetrievalAgent

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor agent that orchestrates the QA workflow.
    
    This agent coordinates between retrieval and answer generation to provide
    comprehensive responses to user questions, maintaining conversation context
    and ensuring high-quality answers.
    """

    def __init__(self, config: QAConfig = DEFAULT_QA_CONFIG):
        self.config = config
        self.retrieval_agent = RetrievalAgent(config)
        self.llm = None
        self.supervisor = None
        self.memory = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize the supervisor components."""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                timeout=self.config.timeout
            )
            
            # Initialize memory for conversation context
            self.memory = MemorySaver()
            
            # Create the supervisor
            self._create_supervisor()
            
            logger.info("SupervisorAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize supervisor components: {e}")
            raise QAError(f"Failed to initialize supervisor: {e}")

    def _create_supervisor(self):
        """Create the LangGraph supervisor."""
        try:
            # Create system message
            system_message = SystemMessage(
                content=(
                    "You are a helpful question-answering assistant that can answer questions "
                    "about processed documents. You have access to a retrieval system that can "
                    "find relevant information from the knowledge base.\n\n"
                    "INSTRUCTIONS:\n"
                    "- Use the retrieval_agent to find relevant documents for the user's question\n"
                    "- Generate comprehensive, accurate answers based on the retrieved information\n"
                    "- If no relevant documents are found, clearly state this\n"
                    "- Always cite your sources when providing information\n"
                    "- Maintain conversation context when appropriate\n"
                    "- If the question is unclear, ask for clarification\n"
                    "- Respond in a helpful, professional manner\n"
                )
            )
            
            # Create the supervisor using langgraph_supervisor
            self.supervisor = create_supervisor(
                agents=[self.retrieval_agent.agent],
                model=self.llm,
                prompt=system_message.content,
                add_handoff_back_messages=True,
                output_mode="full_history",
            ).compile(checkpointer=self.memory)
            
            logger.info("Supervisor created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create supervisor: {e}")
            raise QAError(f"Failed to create supervisor: {e}")

    def ask_question(self, question: str, session_id: Optional[str] = None) -> str:
        """
        Ask a question and get an answer.
        
        Args:
            question: The user's question
            session_id: Optional session ID for conversation context
            
        Returns:
            The generated answer
            
        Raises:
            AnswerGenerationError: If answer generation fails
            SessionError: If session management fails
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        try:
            # Prepare the input message
            input_message = HumanMessage(content=question)
            
            # Prepare configuration for session management
            config = {}
            if session_id:
                config = {"configurable": {"thread_id": session_id}}
            
            # Invoke the supervisor
            result = self.supervisor.invoke(
                {"messages": [input_message]},
                config=config
            )
            
            # Extract the answer from the result
            if "messages" in result and result["messages"]:
                # Get the last AI message
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    answer = last_message.content
                else:
                    # If not an AI message, get the content anyway
                    answer = str(last_message.content) if hasattr(last_message, 'content') else str(last_message)
            else:
                answer = "I apologize, but I couldn't generate a proper response."
            
            logger.info(f"Generated answer for question: {question[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error during question answering: {e}")
            raise AnswerGenerationError(f"Failed to generate answer: {e}")

    async def aask_question(self, question: str, session_id: Optional[str] = None) -> str:
        """
        Asynchronously ask a question and get an answer.
        
        Args:
            question: The user's question
            session_id: Optional session ID for conversation context
            
        Returns:
            The generated answer
            
        Raises:
            AnswerGenerationError: If answer generation fails
            SessionError: If session management fails
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        try:
            # Prepare the input message
            input_message = HumanMessage(content=question)
            
            # Prepare configuration for session management
            config = {}
            if session_id:
                config = {"configurable": {"thread_id": session_id}}
            
            # Invoke the supervisor asynchronously
            result = await self.supervisor.ainvoke(
                {"messages": [input_message]},
                config=config
            )
            
            # Extract the answer from the result
            if "messages" in result and result["messages"]:
                # Get the last AI message
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    answer = last_message.content
                else:
                    # If not an AI message, get the content anyway
                    answer = str(last_message.content) if hasattr(last_message, 'content') else str(last_message)
            else:
                answer = "I apologize, but I couldn't generate a proper response."
            
            logger.info(f"Async generated answer for question: {question[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error during async question answering: {e}")
            raise AnswerGenerationError(f"Failed to generate answer: {e}")

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
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.supervisor.get_state(config)
            
            if state and "messages" in state.values:
                messages = []
                for msg in state.values["messages"]:
                    messages.append({
                        "type": type(msg).__name__,
                        "content": msg.content if hasattr(msg, 'content') else str(msg),
                        "timestamp": getattr(msg, 'timestamp', None)
                    })
                return messages
            else:
                return []
                
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
        try:
            config = {"configurable": {"thread_id": session_id}}
            # Clear the session state
            self.supervisor.update_state(config, {"messages": []})
            logger.info(f"Cleared session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            raise SessionError(f"Failed to clear session: {e}")

    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the supervisor agent.
        
        Returns:
            Dictionary containing agent statistics
        """
        return {
            "llm_model": self.config.llm_model,
            "temperature": self.config.llm_temperature,
            "max_answer_length": self.config.max_answer_length,
            "answer_style": self.config.answer_style,
            "include_sources": self.config.include_sources,
            "session_timeout": self.config.session_timeout,
            "context_memory": self.config.enable_context_memory,
            "retrieval_stats": self.retrieval_agent.get_retrieval_stats()
        }

    def update_config(self, **kwargs):
        """
        Update the agent configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config = self.config.update(**kwargs)
        # Update retrieval agent config
        self.retrieval_agent.update_config(**kwargs)
        logger.info(f"SupervisorAgent configuration updated: {kwargs}")

    def __str__(self) -> str:
        """String representation of the supervisor agent."""
        return f"SupervisorAgent(model={self.config.llm_model}, style={self.config.answer_style})"

    def __repr__(self) -> str:
        """Detailed string representation of the supervisor agent."""
        return (f"SupervisorAgent(config={self.config}, "
                f"llm={self.llm}, supervisor={self.supervisor is not None})")
