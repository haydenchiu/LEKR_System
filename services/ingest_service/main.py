#!/usr/bin/env python3
"""
LERK System - Ingest Service
This service handles document ingestion and initial processing.
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

from ingest import DocumentIngestionOrchestrator, DEFAULT_CONFIG
from enrichment import DocumentEnricher, DEFAULT_CONFIG as ENRICHMENT_CONFIG
from logic_extractor import LogicExtractor, DEFAULT_CONFIG as LOGIC_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingest_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IngestService:
    """Service for document ingestion and processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ingest service.
        
        Args:
            config: Service configuration
        """
        self.config = config or self._load_config()
        self.ingestion_orchestrator = DocumentIngestionOrchestrator(self.config['ingestion'])
        self.enricher = DocumentEnricher(self.config['enrichment'])
        self.logic_extractor = LogicExtractor(self.config['logic'])
        
        # Service statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'enrichments_completed': 0,
            'logic_extractions_completed': 0,
            'start_time': None,
            'last_activity': None
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        return {
            'ingestion': DEFAULT_CONFIG,
            'enrichment': ENRICHMENT_CONFIG,
            'logic': LOGIC_CONFIG,
            'service': {
                'name': 'ingest-service',
                'version': '1.0.0',
                'max_workers': int(os.getenv('MAX_WORKERS', '4')),
                'batch_size': int(os.getenv('BATCH_SIZE', '10')),
                'timeout': int(os.getenv('TIMEOUT', '300')),
                'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3'))
            }
        }
    
    async def process_document(self, file_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to input document
            output_path: Path to save processed document
            
        Returns:
            Processing result with statistics
        """
        try:
            logger.info(f"Processing document: {file_path}")
            start_time = time.time()
            
            # Step 1: Document Ingestion
            logger.info("Starting document ingestion")
            ingestion_result = self.ingestion_orchestrator.ingest_document(file_path)
            
            if not ingestion_result['success']:
                return {
                    'success': False,
                    'error': f"Ingestion failed: {ingestion_result['error']}",
                    'file_path': file_path,
                    'duration': time.time() - start_time
                }
            
            chunks = ingestion_result['chunks']
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 2: Chunk Enrichment
            logger.info("Starting chunk enrichment")
            enriched_chunks = []
            for chunk in chunks:
                try:
                    enriched_chunk = await self.enricher.enrich_chunk_async(chunk)
                    enriched_chunks.append(enriched_chunk)
                except Exception as e:
                    logger.warning(f"Failed to enrich chunk: {e}")
                    enriched_chunks.append(chunk)  # Keep original chunk
            
            # Step 3: Logic Extraction
            logger.info("Starting logic extraction")
            logic_chunks = []
            for chunk in enriched_chunks:
                try:
                    logic_chunk = await self.logic_extractor.extract_logic_async(chunk)
                    logic_chunks.append(logic_chunk)
                except Exception as e:
                    logger.warning(f"Failed to extract logic from chunk: {e}")
                    logic_chunks.append(chunk)  # Keep original chunk
            
            # Step 4: Save processed document
            output_file = Path(output_path) / f"{Path(file_path).stem}_processed.json"
            processed_data = {
                'file_path': file_path,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'chunks': [chunk.model_dump() for chunk in logic_chunks],
                'processing_stats': {
                    'total_chunks': len(chunks),
                    'enriched_chunks': len(enriched_chunks),
                    'logic_chunks': len(logic_chunks)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['enrichments_completed'] += len(enriched_chunks)
            self.stats['logic_extractions_completed'] += len(logic_chunks)
            self.stats['last_activity'] = datetime.utcnow().isoformat()
            
            duration = time.time() - start_time
            logger.info(f"Document processed successfully in {duration:.2f} seconds")
            
            return {
                'success': True,
                'file_path': file_path,
                'output_file': str(output_file),
                'chunks_count': len(chunks),
                'enriched_chunks_count': len(enriched_chunks),
                'logic_chunks_count': len(logic_chunks),
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path,
                'duration': time.time() - start_time
            }
    
    async def process_batch(self, file_paths: List[str], output_path: str) -> Dict[str, Any]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            output_path: Path to save processed documents
            
        Returns:
            Batch processing results
        """
        try:
            logger.info(f"Processing batch of {len(file_paths)} documents")
            start_time = time.time()
            
            # Create output directory
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Process documents in parallel
            tasks = []
            for file_path in file_paths:
                task = asyncio.create_task(self.process_document(file_path, output_path))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful = 0
            failed = 0
            total_chunks = 0
            total_enriched = 0
            total_logic = 0
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"Task failed with exception: {result}")
                elif result['success']:
                    successful += 1
                    total_chunks += result['chunks_count']
                    total_enriched += result['enriched_chunks_count']
                    total_logic += result['logic_chunks_count']
                else:
                    failed += 1
                    logger.error(f"Task failed: {result.get('error', 'Unknown error')}")
            
            duration = time.time() - start_time
            
            return {
                'success': failed == 0,
                'total_documents': len(file_paths),
                'successful_documents': successful,
                'failed_documents': failed,
                'total_chunks': total_chunks,
                'total_enriched': total_enriched,
                'total_logic': total_logic,
                'duration': duration,
                'throughput': len(file_paths) / duration if duration > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_documents': len(file_paths),
                'successful_documents': 0,
                'failed_documents': len(file_paths),
                'duration': time.time() - start_time
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
            # Test ingestion orchestrator
            self.ingestion_orchestrator.get_ingestion_stats()
            
            # Test enricher
            self.enricher.get_enrichment_stats()
            
            # Test logic extractor
            self.logic_extractor.get_extraction_stats()
            
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'components': {
                    'ingestion_orchestrator': 'healthy',
                    'enricher': 'healthy',
                    'logic_extractor': 'healthy'
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
    """Main entry point for the ingest service."""
    try:
        # Initialize service
        service = IngestService()
        
        # Start service
        logger.info("Starting LERK Ingest Service")
        service.stats['start_time'] = datetime.utcnow().isoformat()
        
        # Example usage
        if len(sys.argv) > 1:
            # Process single document
            file_path = sys.argv[1]
            output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
            
            result = await service.process_document(file_path, output_path)
            print(json.dumps(result, indent=2))
        else:
            # Service mode - keep running
            logger.info("Ingest service running in service mode")
            while True:
                await asyncio.sleep(60)  # Check every minute
                
    except KeyboardInterrupt:
        logger.info("Ingest service stopped by user")
    except Exception as e:
        logger.error(f"Ingest service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
