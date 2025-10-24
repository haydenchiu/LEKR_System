#!/usr/bin/env python3
"""
LERK System - QA Service
This service provides question answering capabilities.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qa_agent import QAOrchestrator, DEFAULT_QA_CONFIG, HIGH_QUALITY_QA_CONFIG, FAST_QA_CONFIG
from retriever import SemanticRetriever, HybridRetriever, ContextRetriever, DEFAULT_RETRIEVER_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/qa_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QAService:
    """Service for question answering capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the QA service.
        
        Args:
            config: Service configuration
        """
        self.config = config or self._load_config()
        self.qa_orchestrator = QAOrchestrator(self.config['qa'])
        self.retriever = self._initialize_retriever()
        
        # Service statistics
        self.stats = {
            'questions_answered': 0,
            'sessions_created': 0,
            'retrieval_operations': 0,
            'start_time': None,
            'last_activity': None
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        return {
            'qa': DEFAULT_QA_CONFIG,
            'retriever': DEFAULT_RETRIEVER_CONFIG,
            'service': {
                'name': 'qa-service',
                'version': '1.0.0',
                'max_sessions': int(os.getenv('MAX_SESSIONS', '1000')),
                'session_timeout': int(os.getenv('SESSION_TIMEOUT', '3600')),
                'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3'))
            }
        }
    
    def _initialize_retriever(self):
        """Initialize the retriever based on configuration."""
        retriever_type = self.config['service'].get('retriever_type', 'semantic')
        
        if retriever_type == 'semantic':
            return SemanticRetriever(self.config['retriever'])
        elif retriever_type == 'hybrid':
            return HybridRetriever(self.config['retriever'])
        elif retriever_type == 'context':
            return ContextRetriever(self.config['retriever'])
        else:
            return SemanticRetriever(self.config['retriever'])
    
    async def ask_question(
        self, 
        question: str, 
        session_id: Optional[str] = None,
        answer_style: str = "concise",
        context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using the LERK system.
        
        Args:
            question: The question to answer
            session_id: Session ID for conversation context
            answer_style: Style of answer (concise, detailed, academic, conversational)
            context: Additional context for the question
            
        Returns:
            Answer with sources and metadata
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            start_time = time.time()
            
            # Use QA orchestrator to answer question
            response = self.qa_orchestrator.ask_question(
                question=question,
                session_id=session_id,
                answer_style=answer_style
            )
            
            # Add context if provided
            if context:
                response['context'] = context
            
            # Update statistics
            self.stats['questions_answered'] += 1
            self.stats['last_activity'] = datetime.utcnow().isoformat()
            
            if session_id:
                self.stats['sessions_created'] += 1
            
            processing_time = time.time() - start_time
            response['processing_time'] = processing_time
            
            logger.info(f"Question answered in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'question': question,
                'session_id': session_id,
                'processing_time': time.time() - start_time
            }
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session.
        
        Args:
            session_id: Session ID to get info for
            
        Returns:
            Session information
        """
        try:
            # Get session from QA orchestrator
            session_info = self.qa_orchestrator.get_session_info(session_id)
            
            return {
                'success': True,
                'session_id': session_id,
                'session_info': session_info,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    async def clear_session(self, session_id: str) -> Dict[str, Any]:
        """
        Clear a session and its context.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            Clear operation result
        """
        try:
            # Clear session from QA orchestrator
            self.qa_orchestrator.clear_session(session_id)
            
            return {
                'success': True,
                'message': f"Session {session_id} cleared successfully",
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    async def search_documents(
        self, 
        query: str, 
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search documents using the retriever.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Search results
        """
        try:
            logger.info(f"Searching documents for: {query[:100]}...")
            start_time = time.time()
            
            # Use retriever to search documents
            results = self.retriever.search(
                query=query,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            # Update statistics
            self.stats['retrieval_operations'] += 1
            self.stats['last_activity'] = datetime.utcnow().isoformat()
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'results_count': len(results),
                'processing_time': processing_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'processing_time': time.time() - start_time
            }
    
    async def get_knowledge_summary(self, subject: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of available knowledge.
        
        Args:
            subject: Optional subject to filter by
            
        Returns:
            Knowledge summary
        """
        try:
            logger.info("Generating knowledge summary")
            start_time = time.time()
            
            # Get knowledge summary from QA orchestrator
            summary = self.qa_orchestrator.get_knowledge_summary(subject=subject)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'summary': summary,
                'subject': subject,
                'processing_time': processing_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Knowledge summary generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'subject': subject,
                'processing_time': time.time() - start_time
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            'service_name': self.config['service']['name'],
            'service_version': self.config['service']['version'],
            'status': 'running',
            'statistics': self.stats,
            'configuration': self.config['service'],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get service health check."""
        try:
            # Check if QA orchestrator is healthy
            self.qa_orchestrator.get_session_info('test')
            
            # Check if retriever is healthy
            self.retriever.get_retrieval_stats()
            
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'components': {
                    'qa_orchestrator': 'healthy',
                    'retriever': 'healthy'
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }


async def main():
    """Main entry point for the QA service."""
    try:
        # Initialize service
        service = QAService()
        
        # Start service
        logger.info("Starting LERK QA Service")
        service.stats['start_time'] = datetime.utcnow().isoformat()
        
        # Example usage
        if len(sys.argv) > 1:
            # Answer a question
            question = sys.argv[1]
            session_id = sys.argv[2] if len(sys.argv) > 2 else None
            
            result = await service.ask_question(question, session_id)
            print(json.dumps(result, indent=2))
        else:
            # Service mode - keep running
            logger.info("QA service running in service mode")
            while True:
                await asyncio.sleep(60)  # Check every minute
                
    except KeyboardInterrupt:
        logger.info("QA service stopped by user")
    except Exception as e:
        logger.error(f"QA service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
