#!/usr/bin/env python3
"""
LERK System - Knowledge Consolidation Pipeline
This script orchestrates the knowledge consolidation pipeline.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from consolidation import (
    DocumentConsolidator,
    SubjectConsolidator,
    ClusterBasedSubjectConsolidator,
    KnowledgeStorage,
    DEFAULT_CONFIG as CONSOLIDATION_CONFIG,
    FAST_CONFIG as FAST_CONSOLIDATION_CONFIG,
    HIGH_QUALITY_CONFIG as HIGH_QUALITY_CONSOLIDATION_CONFIG
)
from clustering import DocumentClusterer, DEFAULT_CONFIG as CLUSTERING_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/consolidation_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConsolidationPipeline:
    """Orchestrates the knowledge consolidation pipeline."""
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        config_preset: str = "default",
        enable_clustering: bool = True
    ):
        """
        Initialize the consolidation pipeline.
        
        Args:
            input_path: Path to processed documents
            output_path: Path to save consolidated knowledge
            config_preset: Configuration preset to use
            enable_clustering: Whether to enable document clustering
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.enable_clustering = enable_clustering
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_preset)
        
        # Initialize components
        self.document_consolidator = DocumentConsolidator(self.config['consolidation'])
        self.subject_consolidator = SubjectConsolidator(self.config['consolidation'])
        self.cluster_consolidator = ClusterBasedSubjectConsolidator(self.config['consolidation'])
        self.knowledge_storage = KnowledgeStorage(self.config['consolidation'])
        
        if self.enable_clustering:
            self.clusterer = DocumentClusterer(self.config['clustering'])
        
        # Statistics
        self.stats = {
            'total_documents': 0,
            'processed_documents': 0,
            'failed_documents': 0,
            'total_chunks': 0,
            'consolidated_documents': 0,
            'clustered_documents': 0,
            'subject_knowledge_items': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _load_config(self, preset: str) -> Dict[str, Any]:
        """Load configuration based on preset."""
        configs = {
            'default': {
                'consolidation': CONSOLIDATION_CONFIG,
                'clustering': CLUSTERING_CONFIG
            },
            'fast': {
                'consolidation': FAST_CONSOLIDATION_CONFIG,
                'clustering': CLUSTERING_CONFIG
            },
            'high_quality': {
                'consolidation': HIGH_QUALITY_CONSOLIDATION_CONFIG,
                'clustering': CLUSTERING_CONFIG
            }
        }
        
        if preset not in configs:
            raise ValueError(f"Unknown config preset: {preset}")
        
        return configs[preset]
    
    def _get_processed_documents(self) -> List[Path]:
        """Get list of processed document files."""
        processed_files = []
        
        if self.input_path.is_file():
            if self.input_path.suffix == '.json':
                processed_files.append(self.input_path)
        else:
            processed_files.extend(self.input_path.rglob('*_processed.json'))
        
        return processed_files
    
    def _load_processed_document(self, file_path: Path) -> Dict[str, Any]:
        """Load a processed document from file."""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_consolidated_knowledge(self, output_file: Path, data: Dict[str, Any]):
        """Save consolidated knowledge to file."""
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def run_document_consolidation(self) -> List[Dict[str, Any]]:
        """Run document-level consolidation."""
        logger.info("Starting document-level consolidation")
        
        processed_files = self._get_processed_documents()
        self.stats['total_documents'] = len(processed_files)
        
        if not processed_files:
            logger.warning("No processed documents found")
            return []
        
        consolidated_documents = []
        
        for file_path in processed_files:
            try:
                logger.info(f"Consolidating document: {file_path}")
                
                # Load processed document
                processed_doc = self._load_processed_document(file_path)
                
                # Extract chunks and document knowledge
                chunks = processed_doc.get('chunks', [])
                existing_knowledge = processed_doc.get('document_knowledge', {})
                
                if not chunks:
                    logger.warning(f"No chunks found in {file_path}")
                    continue
                
                # Consolidate document knowledge
                document_knowledge = self.document_consolidator.consolidate_document(
                    chunks,
                    document_id=processed_doc.get('file_path', str(file_path.stem))
                )
                
                # Save consolidated document
                output_file = self.output_path / f"{file_path.stem}_consolidated.json"
                consolidated_data = {
                    'file_path': processed_doc.get('file_path', str(file_path)),
                    'document_knowledge': document_knowledge.model_dump(),
                    'consolidation_stats': {
                        'total_chunks': len(chunks),
                        'concepts_count': len(document_knowledge.concepts),
                        'relations_count': len(document_knowledge.relations)
                    }
                }
                
                self._save_consolidated_knowledge(output_file, consolidated_data)
                consolidated_documents.append(consolidated_data)
                
                self.stats['processed_documents'] += 1
                self.stats['total_chunks'] += len(chunks)
                self.stats['consolidated_documents'] += 1
                
                logger.info(f"Consolidated document: {file_path} -> {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to consolidate {file_path}: {e}")
                self.stats['failed_documents'] += 1
        
        return consolidated_documents
    
    def run_clustering(self, consolidated_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run document clustering."""
        if not self.enable_clustering:
            logger.info("Clustering disabled, skipping clustering step")
            return {}
        
        logger.info("Starting document clustering")
        
        try:
            # Prepare documents for clustering
            documents = []
            for doc in consolidated_documents:
                # Extract text content from document knowledge
                concepts = doc['document_knowledge'].get('concepts', [])
                text_content = ' '.join([concept.get('description', '') for concept in concepts])
                
                if text_content.strip():
                    documents.append({
                        'id': doc['file_path'],
                        'content': text_content,
                        'metadata': {
                            'file_path': doc['file_path'],
                            'concepts_count': len(concepts)
                        }
                    })
            
            if not documents:
                logger.warning("No documents with content found for clustering")
                return {}
            
            # Run clustering
            cluster_results = self.clusterer.cluster_documents(documents)
            self.stats['clustered_documents'] = len(cluster_results.get('clusters', []))
            
            # Save clustering results
            clustering_output = self.output_path / "clustering_results.json"
            self._save_consolidated_knowledge(clustering_output, cluster_results)
            
            logger.info(f"Clustering completed: {len(cluster_results.get('clusters', []))} clusters")
            return cluster_results
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {}
    
    def run_subject_consolidation(
        self, 
        consolidated_documents: List[Dict[str, Any]], 
        cluster_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run subject-level consolidation."""
        logger.info("Starting subject-level consolidation")
        
        try:
            if cluster_results and self.enable_clustering:
                # Use cluster-based subject consolidation
                logger.info("Using cluster-based subject consolidation")
                subject_knowledge = self.cluster_consolidator.consolidate_subjects_from_clusters(
                    consolidated_documents,
                    cluster_results
                )
            else:
                # Use traditional subject consolidation
                logger.info("Using traditional subject consolidation")
                subject_knowledge = self.subject_consolidator.consolidate_subjects(
                    consolidated_documents
                )
            
            # Save subject knowledge
            subject_output = self.output_path / "subject_knowledge.json"
            subject_data = {
                'subject_knowledge': [knowledge.model_dump() for knowledge in subject_knowledge],
                'consolidation_stats': {
                    'total_subjects': len(subject_knowledge),
                    'total_concepts': sum(len(sk.concepts) for sk in subject_knowledge),
                    'total_relations': sum(len(sk.relations) for sk in subject_knowledge)
                }
            }
            
            self._save_consolidated_knowledge(subject_output, subject_data)
            self.stats['subject_knowledge_items'] = len(subject_knowledge)
            
            logger.info(f"Subject consolidation completed: {len(subject_knowledge)} subjects")
            return subject_data
            
        except Exception as e:
            logger.error(f"Subject consolidation failed: {e}")
            return {}
    
    def store_knowledge(self, subject_data: Dict[str, Any]):
        """Store consolidated knowledge in database."""
        logger.info("Storing knowledge in database")
        
        try:
            if not subject_data or 'subject_knowledge' not in subject_data:
                logger.warning("No subject knowledge to store")
                return
            
            # Store knowledge using knowledge storage
            self.knowledge_storage.store_knowledge(subject_data['subject_knowledge'])
            
            logger.info("Knowledge stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete consolidation pipeline."""
        logger.info("Starting LERK consolidation pipeline")
        self.stats['start_time'] = time.time()
        
        try:
            # Step 1: Document-level consolidation
            consolidated_documents = self.run_document_consolidation()
            
            if not consolidated_documents:
                logger.warning("No documents were consolidated")
                return self._get_pipeline_summary()
            
            # Step 2: Document clustering (optional)
            cluster_results = self.run_clustering(consolidated_documents)
            
            # Step 3: Subject-level consolidation
            subject_data = self.run_subject_consolidation(consolidated_documents, cluster_results)
            
            # Step 4: Store knowledge in database
            self.store_knowledge(subject_data)
            
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
            'success': self.stats['failed_documents'] == 0,
            'statistics': self.stats,
            'duration_seconds': duration,
            'throughput_documents_per_second': self.stats['processed_documents'] / duration if duration > 0 else 0
        }


def main():
    """Main entry point for the consolidation pipeline."""
    parser = argparse.ArgumentParser(description='LERK Knowledge Consolidation Pipeline')
    parser.add_argument('input_path', help='Path to processed documents')
    parser.add_argument('output_path', help='Path to save consolidated knowledge')
    parser.add_argument('--config', default='default', 
                       choices=['default', 'fast', 'high_quality'],
                       help='Configuration preset to use')
    parser.add_argument('--no-clustering', action='store_true',
                       help='Disable document clustering')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run pipeline
    pipeline = ConsolidationPipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        config_preset=args.config,
        enable_clustering=not args.no_clustering
    )
    
    try:
        summary = pipeline.run_pipeline()
        
        # Print summary
        print("\n" + "="*50)
        print("LERK CONSOLIDATION PIPELINE SUMMARY")
        print("="*50)
        print(f"Success: {summary['success']}")
        print(f"Total documents: {summary['statistics']['total_documents']}")
        print(f"Processed documents: {summary['statistics']['processed_documents']}")
        print(f"Failed documents: {summary['statistics']['failed_documents']}")
        print(f"Total chunks: {summary['statistics']['total_chunks']}")
        print(f"Consolidated documents: {summary['statistics']['consolidated_documents']}")
        print(f"Clustered documents: {summary['statistics']['clustered_documents']}")
        print(f"Subject knowledge items: {summary['statistics']['subject_knowledge_items']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Throughput: {summary['throughput_documents_per_second']:.2f} documents/sec")
        print("="*50)
        
        if not summary['success']:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
