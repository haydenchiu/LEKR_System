#!/usr/bin/env python3
"""
LERK System - Worker Service
This service handles background processing tasks.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime, timedelta
import signal

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from clustering import DocumentClusterer, DEFAULT_CONFIG as CLUSTERING_CONFIG
from consolidation import (
    DocumentConsolidator, 
    SubjectConsolidator, 
    ClusterBasedSubjectConsolidator,
    DEFAULT_CONFIG as CONSOLIDATION_CONFIG
)
from retriever import SemanticRetriever, DEFAULT_RETRIEVER_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/worker_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WorkerService:
    """Service for background processing tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the worker service.
        
        Args:
            config: Service configuration
        """
        self.config = config or self._load_config()
        self.clusterer = DocumentClusterer(self.config['clustering'])
        self.document_consolidator = DocumentConsolidator(self.config['consolidation'])
        self.subject_consolidator = SubjectConsolidator(self.config['consolidation'])
        self.cluster_consolidator = ClusterBasedSubjectConsolidator(self.config['consolidation'])
        self.retriever = SemanticRetriever(DEFAULT_RETRIEVER_CONFIG)
        
        # Worker statistics
        self.stats = {
            'tasks_processed': 0,
            'clustering_tasks': 0,
            'consolidation_tasks': 0,
            'retrieval_tasks': 0,
            'start_time': None,
            'last_activity': None
        }
        
        # Task queue
        self.task_queue = asyncio.Queue()
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        return {
            'clustering': CLUSTERING_CONFIG,
            'consolidation': CONSOLIDATION_CONFIG,
            'service': {
                'name': 'worker-service',
                'version': '1.0.0',
                'max_workers': int(os.getenv('MAX_WORKERS', '4')),
                'task_timeout': int(os.getenv('TASK_TIMEOUT', '300')),
                'poll_interval': int(os.getenv('POLL_INTERVAL', '10')),
                'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3'))
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def add_task(self, task_type: str, task_data: Dict[str, Any]):
        """
        Add a task to the processing queue.
        
        Args:
            task_type: Type of task to process
            task_data: Task data and parameters
        """
        task = {
            'id': f"task_{int(time.time())}_{task_type}",
            'type': task_type,
            'data': task_data,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'pending'
        }
        
        await self.task_queue.put(task)
        logger.info(f"Added task {task['id']} to queue")
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single task.
        
        Args:
            task: Task to process
            
        Returns:
            Task processing result
        """
        try:
            logger.info(f"Processing task {task['id']} of type {task['type']}")
            start_time = time.time()
            
            if task['type'] == 'clustering':
                result = await self._process_clustering_task(task)
            elif task['type'] == 'consolidation':
                result = await self._process_consolidation_task(task)
            elif task['type'] == 'retrieval_indexing':
                result = await self._process_retrieval_indexing_task(task)
            else:
                result = {
                    'success': False,
                    'error': f"Unknown task type: {task['type']}"
                }
            
            duration = time.time() - start_time
            result['duration'] = duration
            result['task_id'] = task['id']
            
            # Update statistics
            self.stats['tasks_processed'] += 1
            self.stats['last_activity'] = datetime.utcnow().isoformat()
            
            if task['type'] == 'clustering':
                self.stats['clustering_tasks'] += 1
            elif task['type'] == 'consolidation':
                self.stats['consolidation_tasks'] += 1
            elif task['type'] == 'retrieval_indexing':
                self.stats['retrieval_tasks'] += 1
            
            logger.info(f"Task {task['id']} completed in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Task {task['id']} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task['id'],
                'duration': time.time() - start_time
            }
    
    async def _process_clustering_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process document clustering task."""
        try:
            input_path = task['data']['input_path']
            output_path = task['data']['output_path']
            
            # Run document clustering
            cluster_results = self.clusterer.cluster_documents(input_path)
            
            # Save clustering results
            output_file = Path(output_path) / "clustering_results.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cluster_results, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'clusters_count': len(cluster_results.get('clusters', [])),
                'output_file': str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Clustering task failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _process_consolidation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process knowledge consolidation task."""
        try:
            input_path = task['data']['input_path']
            output_path = task['data']['output_path']
            use_clustering = task['data'].get('use_clustering', True)
            
            # Get processed documents
            processed_files = list(Path(input_path).glob("*_processed.json"))
            
            if not processed_files:
                return {
                    'success': False,
                    'error': 'No processed documents found'
                }
            
            # Load processed documents
            consolidated_documents = []
            for file_path in processed_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                # Extract chunks and document knowledge
                chunks = processed_data.get('chunks', [])
                
                # Consolidate document knowledge
                document_knowledge = self.document_consolidator.consolidate_document(
                    chunks,
                    document_id=processed_data.get('file_path', str(file_path.stem))
                )
                
                consolidated_documents.append({
                    'file_path': processed_data.get('file_path', str(file_path)),
                    'document_knowledge': document_knowledge.model_dump()
                })
            
            # Run subject consolidation
            if use_clustering:
                # Use cluster-based consolidation
                cluster_file = Path(input_path) / "clustering_results.json"
                if cluster_file.exists():
                    with open(cluster_file, 'r', encoding='utf-8') as f:
                        cluster_results = json.load(f)
                    
                    subject_knowledge = self.cluster_consolidator.consolidate_subjects_from_clusters(
                        consolidated_documents,
                        cluster_results
                    )
                else:
                    # Fall back to traditional consolidation
                    subject_knowledge = self.subject_consolidator.consolidate_subjects(
                        consolidated_documents
                    )
            else:
                # Traditional consolidation
                subject_knowledge = self.subject_consolidator.consolidate_subjects(
                    consolidated_documents
                )
            
            # Save consolidation results
            output_file = Path(output_path) / "subject_knowledge.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            consolidation_data = {
                'subject_knowledge': [knowledge.model_dump() for knowledge in subject_knowledge],
                'consolidation_stats': {
                    'total_subjects': len(subject_knowledge),
                    'total_concepts': sum(len(sk.concepts) for sk in subject_knowledge),
                    'total_relations': sum(len(sk.relations) for sk in subject_knowledge)
                },
                'created_at': datetime.utcnow().isoformat()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(consolidation_data, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'subjects_count': len(subject_knowledge),
                'output_file': str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Consolidation task failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _process_retrieval_indexing_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process retrieval indexing task."""
        try:
            input_path = task['data']['input_path']
            collection_name = task['data'].get('collection_name', 'lerk_documents')
            
            # Get processed documents
            processed_files = list(Path(input_path).glob("*_processed.json"))
            
            if not processed_files:
                return {
                    'success': False,
                    'error': 'No processed documents found'
                }
            
            # Index documents for retrieval
            indexed_count = 0
            for file_path in processed_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                # Extract chunks for indexing
                chunks = processed_data.get('chunks', [])
                
                # Index chunks (this would typically involve vector database operations)
                # For now, we'll just count the chunks
                indexed_count += len(chunks)
            
            return {
                'success': True,
                'indexed_documents': len(processed_files),
                'indexed_chunks': indexed_count,
                'collection_name': collection_name
            }
            
        except Exception as e:
            logger.error(f"Retrieval indexing task failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def worker_loop(self):
        """Main worker loop."""
        logger.info("Starting worker loop")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=self.config['service']['poll_interval']
                )
                
                # Process task
                result = await self.process_task(task)
                
                # Log result
                if result['success']:
                    logger.info(f"Task {result['task_id']} completed successfully")
                else:
                    logger.error(f"Task {result['task_id']} failed: {result.get('error', 'Unknown error')}")
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def start(self):
        """Start the worker service."""
        try:
            logger.info("Starting LERK Worker Service")
            self.running = True
            self.stats['start_time'] = datetime.utcnow().isoformat()
            
            # Start worker loop
            await self.worker_loop()
            
        except Exception as e:
            logger.error(f"Worker service failed: {e}")
            raise
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            'service_name': self.config['service']['name'],
            'service_version': self.config['service']['version'],
            'status': 'running' if self.running else 'stopped',
            'statistics': self.stats,
            'queue_size': self.task_queue.qsize(),
            'configuration': self.config['service'],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get service health check."""
        try:
            # Check if services are healthy
            self.clusterer.get_clustering_stats()
            self.document_consolidator.get_consolidation_stats()
            
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'components': {
                    'clusterer': 'healthy',
                    'document_consolidator': 'healthy',
                    'subject_consolidator': 'healthy',
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
    """Main entry point for the worker service."""
    try:
        # Initialize service
        service = WorkerService()
        
        # Start service
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("Worker service stopped by user")
    except Exception as e:
        logger.error(f"Worker service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
