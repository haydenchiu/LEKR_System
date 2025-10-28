"""
Qdrant indexing service for LERK System.

This module provides functionality for indexing document chunks with enrichments
and logic extractions into Qdrant vector database for semantic search.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import json

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    # Create mock classes for missing dependencies
    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass
    
    class Distance:
        COSINE = "Cosine"
    
    class VectorParams:
        def __init__(self, *args, **kwargs):
            pass
    
    class PointStruct:
        def __init__(self, *args, **kwargs):
            pass
    
    class Filter:
        def __init__(self, *args, **kwargs):
            pass
    
    class FieldCondition:
        def __init__(self, *args, **kwargs):
            pass
    
    class MatchValue:
        def __init__(self, *args, **kwargs):
            pass
    
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
    
    class np:
        @staticmethod
        def array(data):
            return data

from .config import RetrieverConfig, DEFAULT_RETRIEVER_CONFIG
from .exceptions import DatabaseConnectionError, VectorSearchError, MissingEmbeddingModelError

logger = logging.getLogger(__name__)


class EmbeddingStrategy:
    """Strategy for creating embeddings from chunk data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding strategy.
        
        Args:
            config: Configuration for embedding generation
        """
        self.config = config
        self.include_base_content = config.get('include_base_content', True)
        self.include_enrichments = config.get('include_enrichments', True)
        self.include_logic_extractions = config.get('include_logic_extractions', True)
        self.content_weight = config.get('content_weight', 1.0)
        self.enrichment_weight = config.get('enrichment_weight', 0.8)
        self.logic_weight = config.get('logic_weight', 0.6)
        self.combination_strategy = config.get('combination_strategy', 'structured')
        self.max_text_length = config.get('max_text_length', 512)
    
    def create_embedding_text(self, chunk_data: Dict[str, Any]) -> str:
        """
        Create text for embedding from chunk data.
        
        Args:
            chunk_data: Chunk data including content, enrichment, and logic
            
        Returns:
            Text to be embedded
        """
        embedding_parts = []
        
        # Base content
        if self.include_base_content and chunk_data.get('content'):
            content_text = chunk_data['content']
            if self.combination_strategy == 'structured':
                embedding_parts.append(f"Content: {content_text}")
            else:
                embedding_parts.append(content_text)
        
        # Enrichment data
        if self.include_enrichments and chunk_data.get('enrichment'):
            enrichment = chunk_data['enrichment']
            
            if enrichment.get('summary'):
                summary_text = enrichment['summary']
                if self.combination_strategy == 'structured':
                    embedding_parts.append(f"Summary: {summary_text}")
                else:
                    embedding_parts.append(summary_text)
            
            if enrichment.get('keywords'):
                keywords_text = ', '.join(enrichment['keywords'])
                if self.combination_strategy == 'structured':
                    embedding_parts.append(f"Keywords: {keywords_text}")
                else:
                    embedding_parts.append(keywords_text)
            
            if enrichment.get('table_summary'):
                table_text = enrichment['table_summary']
                if self.combination_strategy == 'structured':
                    embedding_parts.append(f"Table: {table_text}")
                else:
                    embedding_parts.append(table_text)
        
        # Logic extraction data
        if self.include_logic_extractions and chunk_data.get('logic_extraction'):
            logic = chunk_data['logic_extraction']
            
            if logic.get('claims'):
                claims_text = ', '.join([claim.get('text', '') for claim in logic['claims'] if claim.get('text')])
                if claims_text and self.combination_strategy == 'structured':
                    embedding_parts.append(f"Claims: {claims_text}")
                elif claims_text:
                    embedding_parts.append(claims_text)
            
            if logic.get('relations'):
                relations_text = ', '.join([rel.get('description', '') for rel in logic['relations'] if rel.get('description')])
                if relations_text and self.combination_strategy == 'structured':
                    embedding_parts.append(f"Relations: {relations_text}")
                elif relations_text:
                    embedding_parts.append(relations_text)
        
        # Combine all parts
        full_text = "\n".join(embedding_parts)
        
        # Truncate if too long
        if len(full_text) > self.max_text_length:
            full_text = full_text[:self.max_text_length].rsplit(' ', 1)[0] + "..."
        
        return full_text


class QdrantIndexer:
    """
    Qdrant indexing service for LERK System.
    
    This class handles indexing document chunks with enrichments and logic
    extractions into Qdrant vector database for semantic search.
    """
    
    def __init__(
        self, 
        config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
        embedding_strategy: Optional[EmbeddingStrategy] = None
    ):
        """
        Initialize the Qdrant indexer.
        
        Args:
            config: Configuration for the retriever
            embedding_strategy: Strategy for creating embeddings
        """
        self.config = config
        self.embedding_strategy = embedding_strategy or EmbeddingStrategy({
            'include_base_content': True,
            'include_enrichments': True,
            'include_logic_extractions': True,
            'combination_strategy': 'structured',
            'max_text_length': 512
        })
        
        self._client: Optional[QdrantClient] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._initialized = False
    
    def _initialize(self) -> None:
        """Initialize the indexer components."""
        if self._initialized:
            return
        
        try:
            # Initialize Qdrant client
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port
            )
            
            # Initialize embedding model
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Ensure collection exists
            self._ensure_collection_exists()
            
            self._initialized = True
            logger.info(f"Qdrant indexer initialized: {self.config.host}:{self.config.port}")
            
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to initialize Qdrant indexer: {e}") from e
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists."""
        try:
            collections = self._client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.config.collection_name not in collection_names:
                self._client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.config.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.config.collection_name}")
                
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to ensure collection exists: {e}") from e
    
    def index_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """
        Index a single chunk into Qdrant.
        
        Args:
            chunk_data: Chunk data including content, enrichment, and logic
            
        Returns:
            Point ID in Qdrant
            
        Raises:
            VectorSearchError: If indexing fails
        """
        if not self._initialized:
            self._initialize()
        
        try:
            # Generate point ID
            point_id = str(uuid.uuid4())
            
            # Create embedding text
            embedding_text = self.embedding_strategy.create_embedding_text(chunk_data)
            
            # Generate embedding vector
            embedding_vector = self._embedding_model.encode(embedding_text).tolist()
            
            # Prepare payload
            payload = {
                'chunk_id': chunk_data.get('id', point_id),
                'document_id': chunk_data.get('document_id'),
                'content': chunk_data.get('content', ''),
                'content_type': chunk_data.get('content_type', 'text'),
                'chunk_index': chunk_data.get('chunk_index', 0),
                'enrichment': chunk_data.get('enrichment'),
                'logic_extraction': chunk_data.get('logic_extraction'),
                'metadata': chunk_data.get('metadata', {}),
                'embedding_text': embedding_text,  # Store for debugging
                'indexed_at': datetime.utcnow().isoformat()
            }
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload=payload
            )
            
            # Upsert point
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=[point]
            )
            
            logger.debug(f"Indexed chunk {point_id} into Qdrant")
            return point_id
            
        except Exception as e:
            raise VectorSearchError(f"Failed to index chunk: {e}") from e
    
    def index_chunks_batch(self, chunks_data: List[Dict[str, Any]]) -> List[str]:
        """
        Index multiple chunks into Qdrant in batch.
        
        Args:
            chunks_data: List of chunk data
            
        Returns:
            List of point IDs in Qdrant
            
        Raises:
            VectorSearchError: If batch indexing fails
        """
        if not self._initialized:
            self._initialize()
        
        try:
            points = []
            point_ids = []
            
            for chunk_data in chunks_data:
                # Generate point ID
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Create embedding text
                embedding_text = self.embedding_strategy.create_embedding_text(chunk_data)
                
                # Generate embedding vector
                embedding_vector = self._embedding_model.encode(embedding_text).tolist()
                
                # Prepare payload
                payload = {
                    'chunk_id': chunk_data.get('id', point_id),
                    'document_id': chunk_data.get('document_id'),
                    'content': chunk_data.get('content', ''),
                    'content_type': chunk_data.get('content_type', 'text'),
                    'chunk_index': chunk_data.get('chunk_index', 0),
                    'enrichment': chunk_data.get('enrichment'),
                    'logic_extraction': chunk_data.get('logic_extraction'),
                    'metadata': chunk_data.get('metadata', {}),
                    'embedding_text': embedding_text,
                    'indexed_at': datetime.utcnow().isoformat()
                }
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding_vector,
                    payload=payload
                )
                points.append(point)
            
            # Batch upsert
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
            
            logger.info(f"Indexed {len(chunks_data)} chunks into Qdrant")
            return point_ids
            
        except Exception as e:
            raise VectorSearchError(f"Failed to index chunks batch: {e}") from e
    
    def index_processed_document(self, processed_file_path: str) -> Dict[str, Any]:
        """
        Index a processed document file into Qdrant.
        
        Args:
            processed_file_path: Path to processed document JSON file
            
        Returns:
            Indexing results
            
        Raises:
            VectorSearchError: If indexing fails
        """
        try:
            # Load processed document
            with open(processed_file_path, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            
            chunks = processed_data.get('chunks', [])
            if not chunks:
                return {
                    'success': False,
                    'error': 'No chunks found in processed document',
                    'indexed_count': 0
                }
            
            # Prepare chunks data
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': chunk.get('id', f"chunk_{i}"),
                    'document_id': processed_data.get('file_path', 'unknown'),
                    'content': chunk.get('content', ''),
                    'content_type': chunk.get('content_type', 'text'),
                    'chunk_index': i,
                    'enrichment': chunk.get('enrichment'),
                    'logic_extraction': chunk.get('logic_extraction'),
                    'metadata': chunk.get('metadata', {})
                }
                chunks_data.append(chunk_data)
            
            # Index chunks
            point_ids = self.index_chunks_batch(chunks_data)
            
            return {
                'success': True,
                'indexed_count': len(point_ids),
                'point_ids': point_ids,
                'file_path': processed_file_path
            }
            
        except Exception as e:
            raise VectorSearchError(f"Failed to index processed document: {e}") from e
    
    def index_processed_documents_batch(self, processed_dir: str) -> Dict[str, Any]:
        """
        Index all processed documents in a directory.
        
        Args:
            processed_dir: Directory containing processed document files
            
        Returns:
            Batch indexing results
        """
        try:
            processed_dir_path = Path(processed_dir)
            processed_files = list(processed_dir_path.glob("*_processed.json"))
            
            if not processed_files:
                return {
                    'success': False,
                    'error': 'No processed documents found',
                    'total_indexed': 0,
                    'files_processed': 0
                }
            
            total_indexed = 0
            files_processed = 0
            errors = []
            
            for file_path in processed_files:
                try:
                    result = self.index_processed_document(str(file_path))
                    if result['success']:
                        total_indexed += result['indexed_count']
                        files_processed += 1
                    else:
                        errors.append(f"{file_path}: {result['error']}")
                except Exception as e:
                    errors.append(f"{file_path}: {str(e)}")
            
            return {
                'success': len(errors) == 0,
                'total_indexed': total_indexed,
                'files_processed': files_processed,
                'total_files': len(processed_files),
                'errors': errors
            }
            
        except Exception as e:
            raise VectorSearchError(f"Failed to index processed documents batch: {e}") from e
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk from Qdrant.
        
        Args:
            chunk_id: ID of chunk to delete
            
        Returns:
            True if deletion successful
            
        Raises:
            VectorSearchError: If deletion fails
        """
        if not self._initialized:
            self._initialize()
        
        try:
            # Find points with matching chunk_id
            search_result = self._client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchValue(value=chunk_id)
                        )
                    ]
                ),
                limit=100
            )
            
            if search_result[0]:  # Points found
                point_ids = [point.id for point in search_result[0]]
                self._client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=point_ids
                )
                logger.info(f"Deleted chunk {chunk_id} from Qdrant")
                return True
            else:
                logger.warning(f"Chunk {chunk_id} not found in Qdrant")
                return False
                
        except Exception as e:
            raise VectorSearchError(f"Failed to delete chunk: {e}") from e
    
    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document from Qdrant.
        
        Args:
            document_id: ID of document whose chunks to delete
            
        Returns:
            Number of chunks deleted
            
        Raises:
            VectorSearchError: If deletion fails
        """
        if not self._initialized:
            self._initialize()
        
        try:
            # Find points with matching document_id
            search_result = self._client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000
            )
            
            if search_result[0]:  # Points found
                point_ids = [point.id for point in search_result[0]]
                self._client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=point_ids
                )
                deleted_count = len(point_ids)
                logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
                return deleted_count
            else:
                logger.warning(f"No chunks found for document {document_id}")
                return 0
                
        except Exception as e:
            raise VectorSearchError(f"Failed to delete document chunks: {e}") from e
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Qdrant collection.
        
        Returns:
            Collection statistics
        """
        if not self._initialized:
            self._initialize()
        
        try:
            collection_info = self._client.get_collection(self.config.collection_name)
            
            return {
                'collection_name': self.config.collection_name,
                'points_count': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance,
                'status': collection_info.status
            }
            
        except Exception as e:
            raise VectorSearchError(f"Failed to get collection stats: {e}") from e
    
    def reindex_collection(self, processed_dir: str) -> Dict[str, Any]:
        """
        Reindex entire collection by clearing and re-adding all documents.
        
        Args:
            processed_dir: Directory containing processed documents
            
        Returns:
            Reindexing results
        """
        try:
            # Clear existing collection
            self._client.delete_collection(self.config.collection_name)
            logger.info(f"Cleared collection {self.config.collection_name}")
            
            # Recreate collection
            self._ensure_collection_exists()
            
            # Reindex all documents
            result = self.index_processed_documents_batch(processed_dir)
            
            return {
                'success': result['success'],
                'total_indexed': result['total_indexed'],
                'files_processed': result['files_processed'],
                'total_files': result['total_files'],
                'errors': result.get('errors', [])
            }
            
        except Exception as e:
            raise VectorSearchError(f"Failed to reindex collection: {e}") from e


# Convenience functions
def create_indexer(config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG) -> QdrantIndexer:
    """Create a Qdrant indexer instance."""
    return QdrantIndexer(config)


def index_processed_documents(
    processed_dir: str, 
    config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG
) -> Dict[str, Any]:
    """
    Index all processed documents in a directory.
    
    Args:
        processed_dir: Directory containing processed documents
        config: Retriever configuration
        
    Returns:
        Indexing results
    """
    indexer = QdrantIndexer(config)
    return indexer.index_processed_documents_batch(processed_dir)
