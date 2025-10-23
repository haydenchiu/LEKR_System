"""
Knowledge storage for agentic Q&A retrieval.

This module implements storage functionality for saving consolidated knowledge
into databases and providing retrieval capabilities for agentic Q&A systems.
"""

import logging
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import sqlalchemy as sa
    from sqlalchemy import create_engine, Column, String, Text, Float, DateTime, JSON
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
except ImportError as e:
    raise ImportError(
        "Required storage dependencies not installed. "
        "Please install: pip install sentence-transformers scikit-learn sqlalchemy"
    ) from e

from .models import DocumentKnowledge, SubjectKnowledge, KnowledgeStorage
from .config import ConsolidationConfig, DEFAULT_CONSOLIDATION_CONFIG
from .exceptions import (
    StorageError, StorageBackendError, VectorSearchError, MissingAPIKeyError
)

logger = logging.getLogger(__name__)

# SQLAlchemy models
Base = declarative_base()


class DocumentKnowledgeDB(Base):
    """Database model for document knowledge."""
    __tablename__ = "document_knowledge"
    
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    key_concepts = Column(JSON, nullable=False)
    knowledge_relations = Column(JSON, nullable=False)
    main_themes = Column(JSON, nullable=False)
    knowledge_graph = Column(JSON, nullable=False)
    quality_score = Column(Float, nullable=False)
    consolidation_metadata = Column(JSON, nullable=False)
    embedding_vector = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class SubjectKnowledgeDB(Base):
    """Database model for subject knowledge."""
    __tablename__ = "subject_knowledge"
    
    id = Column(String, primary_key=True)
    subject_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    core_concepts = Column(JSON, nullable=False)
    knowledge_relations = Column(JSON, nullable=False)
    document_sources = Column(JSON, nullable=False)
    knowledge_hierarchy = Column(JSON, nullable=False)
    expertise_level = Column(String, nullable=False)
    domain_tags = Column(JSON, nullable=False)
    quality_score = Column(Float, nullable=False)
    consolidation_metadata = Column(JSON, nullable=False)
    embedding_vector = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class KnowledgeStorageManager:
    """
    Manages storage and retrieval of consolidated knowledge.
    
    This class provides functionality for storing document and subject knowledge
    in various backends and retrieving them for agentic Q&A systems.
    """
    
    def __init__(self, config: ConsolidationConfig = DEFAULT_CONSOLIDATION_CONFIG):
        """
        Initialize the KnowledgeStorageManager.
        
        Args:
            config: Configuration for storage parameters
        """
        self.config = config
        self.embedding_model: Optional[SentenceTransformer] = None
        self.engine = None
        self.Session = None
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage backend and embedding model."""
        try:
            # Initialize embedding model if vector search is enabled
            if self.config.enable_vector_search:
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Initialize storage backend
            if self.config.storage_backend == "sqlite":
                self._initialize_sqlite()
            elif self.config.storage_backend == "postgresql":
                self._initialize_postgresql()
            else:
                # Default to in-memory storage
                self._initialize_memory()
            
            logger.info(f"Initialized storage with backend: {self.config.storage_backend}")
            
        except Exception as e:
            raise StorageBackendError(f"Failed to initialize storage: {e}") from e
    
    def _initialize_sqlite(self) -> None:
        """Initialize SQLite storage."""
        db_path = Path("knowledge_storage.db")
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def _initialize_postgresql(self) -> None:
        """Initialize PostgreSQL storage."""
        # This would require additional configuration
        raise NotImplementedError("PostgreSQL storage not yet implemented")
    
    def _initialize_memory(self) -> None:
        """Initialize in-memory storage."""
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_document_knowledge(self, document_knowledge: DocumentKnowledge) -> str:
        """
        Save document knowledge to storage.
        
        Args:
            document_knowledge: Document knowledge to save
            
        Returns:
            Storage ID for the saved knowledge
            
        Raises:
            StorageError: If saving fails
        """
        try:
            # Generate storage ID
            storage_id = str(uuid.uuid4())
            
            # Generate embedding if enabled
            embedding_vector = None
            if self.embedding_model:
                embedding_vector = self._generate_document_embedding(document_knowledge)
            
            # Create database record
            db_record = DocumentKnowledgeDB(
                id=storage_id,
                document_id=document_knowledge.document_id,
                title=document_knowledge.title,
                summary=document_knowledge.summary,
                key_concepts=[c.model_dump() for c in document_knowledge.key_concepts],
                knowledge_relations=[r.model_dump() for r in document_knowledge.knowledge_relations],
                main_themes=document_knowledge.main_themes,
                knowledge_graph=document_knowledge.knowledge_graph,
                quality_score=document_knowledge.quality_score,
                consolidation_metadata=document_knowledge.consolidation_metadata,
                embedding_vector=embedding_vector.tolist() if embedding_vector is not None else None,
                created_at=document_knowledge.created_at,
                updated_at=document_knowledge.updated_at
            )
            
            # Save to database
            session = self.Session()
            session.add(db_record)
            session.commit()
            session.close()
            
            logger.info(f"Saved document knowledge {document_knowledge.document_id} with storage ID {storage_id}")
            return storage_id
            
        except Exception as e:
            raise StorageError(f"Failed to save document knowledge: {e}") from e
    
    def save_subject_knowledge(self, subject_knowledge: SubjectKnowledge) -> str:
        """
        Save subject knowledge to storage.
        
        Args:
            subject_knowledge: Subject knowledge to save
            
        Returns:
            Storage ID for the saved knowledge
            
        Raises:
            StorageError: If saving fails
        """
        try:
            # Generate storage ID
            storage_id = str(uuid.uuid4())
            
            # Generate embedding if enabled
            embedding_vector = None
            if self.embedding_model:
                embedding_vector = self._generate_subject_embedding(subject_knowledge)
            
            # Create database record
            db_record = SubjectKnowledgeDB(
                id=storage_id,
                subject_id=subject_knowledge.subject_id,
                name=subject_knowledge.name,
                description=subject_knowledge.description,
                core_concepts=[c.model_dump() for c in subject_knowledge.core_concepts],
                knowledge_relations=[r.model_dump() for r in subject_knowledge.knowledge_relations],
                document_sources=subject_knowledge.document_sources,
                knowledge_hierarchy=subject_knowledge.knowledge_hierarchy,
                expertise_level=subject_knowledge.expertise_level,
                domain_tags=subject_knowledge.domain_tags,
                quality_score=subject_knowledge.quality_score,
                consolidation_metadata=subject_knowledge.consolidation_metadata,
                embedding_vector=embedding_vector.tolist() if embedding_vector is not None else None,
                created_at=subject_knowledge.created_at,
                updated_at=subject_knowledge.updated_at
            )
            
            # Save to database
            session = self.Session()
            session.add(db_record)
            session.commit()
            session.close()
            
            logger.info(f"Saved subject knowledge {subject_knowledge.subject_id} with storage ID {storage_id}")
            return storage_id
            
        except Exception as e:
            raise StorageError(f"Failed to save subject knowledge: {e}") from e
    
    def retrieve_document_knowledge(self, document_id: str) -> Optional[DocumentKnowledge]:
        """
        Retrieve document knowledge by document ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document knowledge if found, None otherwise
        """
        try:
            session = self.Session()
            db_record = session.query(DocumentKnowledgeDB).filter_by(document_id=document_id).first()
            session.close()
            
            if not db_record:
                return None
            
            # Convert back to DocumentKnowledge
            document_knowledge = self._convert_db_to_document_knowledge(db_record)
            return document_knowledge
            
        except Exception as e:
            logger.error(f"Failed to retrieve document knowledge {document_id}: {e}")
            return None
    
    def retrieve_subject_knowledge(self, subject_id: str) -> Optional[SubjectKnowledge]:
        """
        Retrieve subject knowledge by subject ID.
        
        Args:
            subject_id: ID of the subject to retrieve
            
        Returns:
            Subject knowledge if found, None otherwise
        """
        try:
            session = self.Session()
            db_record = session.query(SubjectKnowledgeDB).filter_by(subject_id=subject_id).first()
            session.close()
            
            if not db_record:
                return None
            
            # Convert back to SubjectKnowledge
            subject_knowledge = self._convert_db_to_subject_knowledge(db_record)
            return subject_knowledge
            
        except Exception as e:
            logger.error(f"Failed to retrieve subject knowledge {subject_id}: {e}")
            return None
    
    def search_documents_by_similarity(
        self, 
        query: str, 
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[Tuple[DocumentKnowledge, float]]:
        """
        Search documents by similarity to a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (document knowledge, similarity score) tuples
        """
        try:
            if not self.embedding_model:
                raise VectorSearchError("Vector search not enabled")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Get all documents with embeddings
            session = self.Session()
            db_records = session.query(DocumentKnowledgeDB).filter(
                DocumentKnowledgeDB.embedding_vector.isnot(None)
            ).all()
            session.close()
            
            # Calculate similarities
            similarities = []
            for record in db_records:
                if record.embedding_vector:
                    doc_embedding = np.array(record.embedding_vector)
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                    
                    if similarity >= min_similarity:
                        document_knowledge = self._convert_db_to_document_knowledge(record)
                        similarities.append((document_knowledge, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            raise VectorSearchError(f"Failed to search documents by similarity: {e}") from e
    
    def search_subjects_by_similarity(
        self, 
        query: str, 
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[Tuple[SubjectKnowledge, float]]:
        """
        Search subjects by similarity to a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (subject knowledge, similarity score) tuples
        """
        try:
            if not self.embedding_model:
                raise VectorSearchError("Vector search not enabled")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Get all subjects with embeddings
            session = self.Session()
            db_records = session.query(SubjectKnowledgeDB).filter(
                SubjectKnowledgeDB.embedding_vector.isnot(None)
            ).all()
            session.close()
            
            # Calculate similarities
            similarities = []
            for record in db_records:
                if record.embedding_vector:
                    subject_embedding = np.array(record.embedding_vector)
                    similarity = cosine_similarity([query_embedding], [subject_embedding])[0][0]
                    
                    if similarity >= min_similarity:
                        subject_knowledge = self._convert_db_to_subject_knowledge(record)
                        similarities.append((subject_knowledge, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            raise VectorSearchError(f"Failed to search subjects by similarity: {e}") from e
    
    def get_all_documents(self) -> List[DocumentKnowledge]:
        """Get all stored document knowledge."""
        try:
            session = self.Session()
            db_records = session.query(DocumentKnowledgeDB).all()
            session.close()
            
            documents = [self._convert_db_to_document_knowledge(record) for record in db_records]
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []
    
    def get_all_subjects(self) -> List[SubjectKnowledge]:
        """Get all stored subject knowledge."""
        try:
            session = self.Session()
            db_records = session.query(SubjectKnowledgeDB).all()
            session.close()
            
            subjects = [self._convert_db_to_subject_knowledge(record) for record in db_records]
            return subjects
            
        except Exception as e:
            logger.error(f"Failed to get all subjects: {e}")
            return []
    
    def delete_document_knowledge(self, document_id: str) -> bool:
        """
        Delete document knowledge by document ID.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            session = self.Session()
            result = session.query(DocumentKnowledgeDB).filter_by(document_id=document_id).delete()
            session.commit()
            session.close()
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete document knowledge {document_id}: {e}")
            return False
    
    def delete_subject_knowledge(self, subject_id: str) -> bool:
        """
        Delete subject knowledge by subject ID.
        
        Args:
            subject_id: ID of the subject to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            session = self.Session()
            result = session.query(SubjectKnowledgeDB).filter_by(subject_id=subject_id).delete()
            session.commit()
            session.close()
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete subject knowledge {subject_id}: {e}")
            return False
    
    def _generate_document_embedding(self, document_knowledge: DocumentKnowledge) -> np.ndarray:
        """Generate embedding vector for document knowledge."""
        try:
            # Combine title, summary, and main themes
            text_content = f"{document_knowledge.title} {document_knowledge.summary} {' '.join(document_knowledge.main_themes)}"
            
            # Add concept names
            concept_names = [c.name for c in document_knowledge.key_concepts]
            text_content += f" {' '.join(concept_names)}"
            
            # Generate embedding
            embedding = self.embedding_model.encode([text_content])[0]
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to generate document embedding: {e}")
            return np.zeros(384)  # Default embedding size
    
    def _generate_subject_embedding(self, subject_knowledge: SubjectKnowledge) -> np.ndarray:
        """Generate embedding vector for subject knowledge."""
        try:
            # Combine name, description, and domain tags
            text_content = f"{subject_knowledge.name} {subject_knowledge.description} {' '.join(subject_knowledge.domain_tags)}"
            
            # Add concept names
            concept_names = [c.name for c in subject_knowledge.core_concepts]
            text_content += f" {' '.join(concept_names)}"
            
            # Generate embedding
            embedding = self.embedding_model.encode([text_content])[0]
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to generate subject embedding: {e}")
            return np.zeros(384)  # Default embedding size
    
    def _convert_db_to_document_knowledge(self, db_record: DocumentKnowledgeDB) -> DocumentKnowledge:
        """Convert database record to DocumentKnowledge."""
        from .models import KeyConcept, KnowledgeRelation
        
        # Convert concepts
        key_concepts = [KeyConcept(**concept_data) for concept_data in db_record.key_concepts]
        
        # Convert relations
        knowledge_relations = [KnowledgeRelation(**relation_data) for relation_data in db_record.knowledge_relations]
        
        return DocumentKnowledge(
            document_id=db_record.document_id,
            title=db_record.title,
            summary=db_record.summary,
            key_concepts=key_concepts,
            knowledge_relations=knowledge_relations,
            main_themes=db_record.main_themes,
            knowledge_graph=db_record.knowledge_graph,
            quality_score=db_record.quality_score,
            consolidation_metadata=db_record.consolidation_metadata,
            created_at=db_record.created_at,
            updated_at=db_record.updated_at
        )
    
    def _convert_db_to_subject_knowledge(self, db_record: SubjectKnowledgeDB) -> SubjectKnowledge:
        """Convert database record to SubjectKnowledge."""
        from .models import KeyConcept, KnowledgeRelation
        
        # Convert concepts
        core_concepts = [KeyConcept(**concept_data) for concept_data in db_record.core_concepts]
        
        # Convert relations
        knowledge_relations = [KnowledgeRelation(**relation_data) for relation_data in db_record.knowledge_relations]
        
        return SubjectKnowledge(
            subject_id=db_record.subject_id,
            name=db_record.name,
            description=db_record.description,
            core_concepts=core_concepts,
            knowledge_relations=knowledge_relations,
            document_sources=db_record.document_sources,
            knowledge_hierarchy=db_record.knowledge_hierarchy,
            expertise_level=db_record.expertise_level,
            domain_tags=db_record.domain_tags,
            quality_score=db_record.quality_score,
            consolidation_metadata=db_record.consolidation_metadata,
            created_at=db_record.created_at,
            updated_at=db_record.updated_at
        )
