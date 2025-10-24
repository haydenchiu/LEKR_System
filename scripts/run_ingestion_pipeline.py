#!/usr/bin/env python3
"""
LERK System - Document Ingestion Pipeline
This script orchestrates the complete document ingestion pipeline.
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingest import (
    DocumentIngestionOrchestrator,
    DEFAULT_CONFIG,
    LARGE_DOCUMENT_CONFIG,
    FAST_CONFIG,
    HIGH_QUALITY_CONFIG
)
from enrichment import DocumentEnricher, DEFAULT_CONFIG as ENRICHMENT_CONFIG
from logic_extractor import LogicExtractor, DEFAULT_CONFIG as LOGIC_CONFIG
from clustering import DocumentClusterer, DEFAULT_CONFIG as CLUSTERING_CONFIG
from consolidation import (
    DocumentConsolidator,
    SubjectConsolidator,
    KnowledgeStorage,
    DEFAULT_CONFIG as CONSOLIDATION_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the complete document ingestion pipeline."""
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        config_preset: str = "default",
        max_workers: int = 4,
        enable_async: bool = True
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            input_path: Path to input documents
            output_path: Path to save processed documents
            config_preset: Configuration preset to use
            max_workers: Maximum number of worker threads
            enable_async: Whether to use async processing
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.max_workers = max_workers
        self.enable_async = enable_async
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_preset)
        
        # Initialize components
        self.ingestion_orchestrator = DocumentIngestionOrchestrator(self.config['ingestion'])
        self.enricher = DocumentEnricher(self.config['enrichment'])
        self.logic_extractor = LogicExtractor(self.config['logic'])
        self.clusterer = DocumentClusterer(self.config['clustering'])
        self.document_consolidator = DocumentConsolidator(self.config['consolidation'])
        self.subject_consolidator = SubjectConsolidator(self.config['consolidation'])
        self.knowledge_storage = KnowledgeStorage(self.config['consolidation'])
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'enriched_chunks': 0,
            'logic_extracted_chunks': 0,
            'clustered_documents': 0,
            'consolidated_documents': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _load_config(self, preset: str) -> Dict[str, Any]:
        """Load configuration based on preset."""
        configs = {
            'default': {
                'ingestion': DEFAULT_CONFIG,
                'enrichment': ENRICHMENT_CONFIG,
                'logic': LOGIC_CONFIG,
                'clustering': CLUSTERING_CONFIG,
                'consolidation': CONSOLIDATION_CONFIG
            },
            'fast': {
                'ingestion': FAST_CONFIG,
                'enrichment': ENRICHMENT_CONFIG,
                'logic': LOGIC_CONFIG,
                'clustering': CLUSTERING_CONFIG,
                'consolidation': CONSOLIDATION_CONFIG
            },
            'high_quality': {
                'ingestion': HIGH_QUALITY_CONFIG,
                'enrichment': ENRICHMENT_CONFIG,
                'logic': LOGIC_CONFIG,
                'clustering': CLUSTERING_CONFIG,
                'consolidation': CONSOLIDATION_CONFIG
            },
            'large_document': {
                'ingestion': LARGE_DOCUMENT_CONFIG,
                'enrichment': ENRICHMENT_CONFIG,
                'logic': LOGIC_CONFIG,
                'clustering': CLUSTERING_CONFIG,
                'consolidation': CONSOLIDATION_CONFIG
            }
        }
        
        if preset not in configs:
            raise ValueError(f"Unknown config preset: {preset}")
        
        return configs[preset]
    
    def _get_supported_files(self) -> List[Path]:
        """Get list of supported files from input path."""
        supported_extensions = {'.pdf', '.docx', '.doc', '.html', '.htm', '.txt', '.md'}
        files = []
        
        if self.input_path.is_file():
            if self.input_path.suffix.lower() in supported_extensions:
                files.append(self.input_path)
        else:
            for ext in supported_extensions:
                files.extend(self.input_path.rglob(f'*{ext}'))
        
        return files
    
    def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file through the complete pipeline."""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Step 1: Document Ingestion
            logger.info(f"Ingesting document: {file_path}")
            ingestion_result = self.ingestion_orchestrator.ingest_document(str(file_path))
            
            if not ingestion_result['success']:
                raise Exception(f"Ingestion failed: {ingestion_result['error']}")
            
            chunks = ingestion_result['chunks']
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            
            # Step 2: Chunk Enrichment
            logger.info(f"Enriching {len(chunks)} chunks")
            enriched_chunks = []
            for chunk in chunks:
                try:
                    enriched_chunk = self.enricher.enrich_chunk(chunk)
                    enriched_chunks.append(enriched_chunk)
                except Exception as e:
                    logger.warning(f"Failed to enrich chunk: {e}")
                    enriched_chunks.append(chunk)  # Keep original chunk
            
            # Step 3: Logic Extraction
            logger.info(f"Extracting logic from {len(enriched_chunks)} chunks")
            logic_chunks = []
            for chunk in enriched_chunks:
                try:
                    logic_chunk = self.logic_extractor.extract_logic(chunk)
                    logic_chunks.append(logic_chunk)
                except Exception as e:
                    logger.warning(f"Failed to extract logic from chunk: {e}")
                    logic_chunks.append(chunk)  # Keep original chunk
            
            # Step 4: Document Consolidation
            logger.info(f"Consolidating document knowledge")
            document_knowledge = self.document_consolidator.consolidate_document(
                logic_chunks, 
                document_id=str(file_path.stem)
            )
            
            # Step 5: Save processed document
            output_file = self.output_path / f"{file_path.stem}_processed.json"
            self._save_processed_document(
                output_file,
                {
                    'file_path': str(file_path),
                    'chunks': [chunk.model_dump() for chunk in logic_chunks],
                    'document_knowledge': document_knowledge.model_dump(),
                    'processing_stats': {
                        'total_chunks': len(chunks),
                        'enriched_chunks': len(enriched_chunks),
                        'logic_chunks': len(logic_chunks)
                    }
                }
            )
            
            return {
                'success': True,
                'file_path': str(file_path),
                'chunks_count': len(chunks),
                'enriched_chunks_count': len(enriched_chunks),
                'logic_chunks_count': len(logic_chunks),
                'output_file': str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                'success': False,
                'file_path': str(file_path),
                'error': str(e)
            }
    
    def _save_processed_document(self, output_file: Path, data: Dict[str, Any]):
        """Save processed document to file."""
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _process_files_parallel(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process files in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Exception processing {file_path}: {e}")
                    results.append({
                        'success': False,
                        'file_path': str(file_path),
                        'error': str(e)
                    })
        
        return results
    
    async def _process_files_async(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process files asynchronously."""
        tasks = []
        for file_path in files:
            task = asyncio.create_task(self._process_single_file_async(file_path))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'file_path': str(files[i]),
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_file_async(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file asynchronously."""
        # For now, run the synchronous version in a thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_single_file, file_path)
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete ingestion pipeline."""
        logger.info("Starting LERK ingestion pipeline")
        self.stats['start_time'] = time.time()
        
        try:
            # Get input files
            files = self._get_supported_files()
            self.stats['total_files'] = len(files)
            
            if not files:
                logger.warning("No supported files found in input path")
                return self._get_pipeline_summary()
            
            logger.info(f"Found {len(files)} files to process")
            
            # Process files
            if self.enable_async:
                logger.info("Processing files asynchronously")
                results = asyncio.run(self._process_files_async(files))
            else:
                logger.info(f"Processing files in parallel with {self.max_workers} workers")
                results = self._process_files_parallel(files)
            
            # Update statistics
            for result in results:
                if result['success']:
                    self.stats['processed_files'] += 1
                    self.stats['total_chunks'] += result.get('chunks_count', 0)
                    self.stats['enriched_chunks'] += result.get('enriched_chunks_count', 0)
                    self.stats['logic_extracted_chunks'] += result.get('logic_chunks_count', 0)
                else:
                    self.stats['failed_files'] += 1
                    logger.error(f"Failed to process {result['file_path']}: {result.get('error', 'Unknown error')}")
            
            # Run clustering and subject consolidation
            logger.info("Running document clustering")
            cluster_results = self.clusterer.cluster_documents(self.output_path)
            self.stats['clustered_documents'] = len(cluster_results.get('clusters', []))
            
            logger.info("Running subject consolidation")
            subject_knowledge = self.subject_consolidator.consolidate_subjects(
                self.output_path,
                cluster_results
            )
            self.stats['consolidated_documents'] = len(subject_knowledge)
            
            # Store knowledge in database
            logger.info("Storing knowledge in database")
            self.knowledge_storage.store_knowledge(subject_knowledge)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            self.stats['end_time'] = time.time()
        
        return self._get_pipeline_summary()
    
    def _get_pipeline_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        duration = 0
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
        
        return {
            'success': self.stats['failed_files'] == 0,
            'statistics': self.stats,
            'duration_seconds': duration,
            'throughput_files_per_second': self.stats['processed_files'] / duration if duration > 0 else 0,
            'throughput_chunks_per_second': self.stats['total_chunks'] / duration if duration > 0 else 0
        }


def main():
    """Main entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(description='LERK Document Ingestion Pipeline')
    parser.add_argument('input_path', help='Path to input documents (file or directory)')
    parser.add_argument('output_path', help='Path to save processed documents')
    parser.add_argument('--config', default='default', 
                       choices=['default', 'fast', 'high_quality', 'large_document'],
                       help='Configuration preset to use')
    parser.add_argument('--workers', type=int, default=4,
                       help='Maximum number of worker threads')
    parser.add_argument('--no-async', action='store_true',
                       help='Disable async processing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run pipeline
    pipeline = IngestionPipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        config_preset=args.config,
        max_workers=args.workers,
        enable_async=not args.no_async
    )
    
    try:
        summary = pipeline.run_pipeline()
        
        # Print summary
        print("\n" + "="*50)
        print("LERK INGESTION PIPELINE SUMMARY")
        print("="*50)
        print(f"Success: {summary['success']}")
        print(f"Total files: {summary['statistics']['total_files']}")
        print(f"Processed files: {summary['statistics']['processed_files']}")
        print(f"Failed files: {summary['statistics']['failed_files']}")
        print(f"Total chunks: {summary['statistics']['total_chunks']}")
        print(f"Enriched chunks: {summary['statistics']['enriched_chunks']}")
        print(f"Logic extracted chunks: {summary['statistics']['logic_extracted_chunks']}")
        print(f"Clustered documents: {summary['statistics']['clustered_documents']}")
        print(f"Consolidated documents: {summary['statistics']['consolidated_documents']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Throughput: {summary['throughput_files_per_second']:.2f} files/sec")
        print("="*50)
        
        if not summary['success']:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
