#!/usr/bin/env python3
"""
LERK System - Database Initialization Script
This script initializes the database schema and creates necessary tables.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/db_init.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization and schema creation."""
    
    def __init__(self, database_url: str, drop_existing: bool = False):
        """
        Initialize the database initializer.
        
        Args:
            database_url: Database connection URL
            drop_existing: Whether to drop existing tables
        """
        self.database_url = database_url
        self.drop_existing = drop_existing
        self.engine = None
        self.session = None
        
    def connect(self):
        """Connect to the database."""
        try:
            logger.info(f"Connecting to database: {self.database_url}")
            self.engine = create_engine(self.database_url, echo=False)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection successful")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def create_tables(self):
        """Create all necessary tables."""
        try:
            logger.info("Creating database tables")
            
            # Create metadata
            metadata = MetaData()
            
            # Documents table
            documents_table = Table(
                'documents',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('file_path', String(500), nullable=False),
                Column('file_name', String(255), nullable=False),
                Column('file_type', String(50), nullable=False),
                Column('file_size', Integer, nullable=True),
                Column('uploaded_at', DateTime, default=datetime.utcnow),
                Column('processed_at', DateTime, nullable=True),
                Column('status', String(50), default='pending'),
                Column('metadata', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Chunks table
            chunks_table = Table(
                'chunks',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('document_id', String(255), nullable=False),
                Column('chunk_index', Integer, nullable=False),
                Column('content', Text, nullable=False),
                Column('content_type', String(50), nullable=False),
                Column('metadata', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Enrichments table
            enrichments_table = Table(
                'enrichments',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('chunk_id', String(255), nullable=False),
                Column('summary', Text, nullable=True),
                Column('keywords', JSON, nullable=True),
                Column('questions', JSON, nullable=True),
                Column('table_data', JSON, nullable=True),
                Column('metadata', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Logic extractions table
            logic_extractions_table = Table(
                'logic_extractions',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('chunk_id', String(255), nullable=False),
                Column('claims', JSON, nullable=True),
                Column('relations', JSON, nullable=True),
                Column('assumptions', JSON, nullable=True),
                Column('constraints', JSON, nullable=True),
                Column('open_questions', JSON, nullable=True),
                Column('metadata', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Document knowledge table
            document_knowledge_table = Table(
                'document_knowledge',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('document_id', String(255), nullable=False),
                Column('concepts', JSON, nullable=True),
                Column('relations', JSON, nullable=True),
                Column('metadata', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Subject knowledge table
            subject_knowledge_table = Table(
                'subject_knowledge',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('subject_name', String(255), nullable=False),
                Column('concepts', JSON, nullable=True),
                Column('relations', JSON, nullable=True),
                Column('metadata', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Clusters table
            clusters_table = Table(
                'clusters',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('cluster_name', String(255), nullable=False),
                Column('cluster_label', String(255), nullable=True),
                Column('document_ids', JSON, nullable=True),
                Column('centroid', JSON, nullable=True),
                Column('metadata', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # Vector embeddings table
            vector_embeddings_table = Table(
                'vector_embeddings',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('chunk_id', String(255), nullable=False),
                Column('embedding_model', String(100), nullable=False),
                Column('embedding_vector', JSON, nullable=False),
                Column('dimension', Integer, nullable=False),
                Column('metadata', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            )
            
            # QA sessions table
            qa_sessions_table = Table(
                'qa_sessions',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('user_id', String(255), nullable=True),
                Column('session_data', JSON, nullable=True),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
                Column('expires_at', DateTime, nullable=True)
            )
            
            # Processing jobs table
            processing_jobs_table = Table(
                'processing_jobs',
                metadata,
                Column('id', String(255), primary_key=True),
                Column('job_type', String(100), nullable=False),
                Column('status', String(50), default='pending'),
                Column('input_data', JSON, nullable=True),
                Column('output_data', JSON, nullable=True),
                Column('error_message', Text, nullable=True),
                Column('progress', Float, default=0.0),
                Column('created_at', DateTime, default=datetime.utcnow),
                Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
                Column('completed_at', DateTime, nullable=True)
            )
            
            # Create all tables
            if self.drop_existing:
                logger.info("Dropping existing tables")
                metadata.drop_all(self.engine)
            
            metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def create_indexes(self):
        """Create database indexes for better performance."""
        try:
            logger.info("Creating database indexes")
            
            with self.engine.connect() as conn:
                # Indexes for documents table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents(uploaded_at)"))
                
                # Indexes for chunks table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks(content_type)"))
                
                # Indexes for enrichments table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_enrichments_chunk_id ON enrichments(chunk_id)"))
                
                # Indexes for logic extractions table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_logic_extractions_chunk_id ON logic_extractions(chunk_id)"))
                
                # Indexes for document knowledge table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_document_knowledge_document_id ON document_knowledge(document_id)"))
                
                # Indexes for subject knowledge table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_subject_knowledge_subject_name ON subject_knowledge(subject_name)"))
                
                # Indexes for clusters table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_clusters_cluster_name ON clusters(cluster_name)"))
                
                # Indexes for vector embeddings table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vector_embeddings_chunk_id ON vector_embeddings(chunk_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vector_embeddings_model ON vector_embeddings(embedding_model)"))
                
                # Indexes for QA sessions table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_qa_sessions_user_id ON qa_sessions(user_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_qa_sessions_expires_at ON qa_sessions(expires_at)"))
                
                # Indexes for processing jobs table
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_processing_jobs_job_type ON processing_jobs(job_type)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs(status)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_processing_jobs_created_at ON processing_jobs(created_at)"))
            
            logger.info("Database indexes created successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create indexes: {e}")
            raise
    
    def create_views(self):
        """Create useful database views."""
        try:
            logger.info("Creating database views")
            
            with self.engine.connect() as conn:
                # View for document processing status
                conn.execute(text("""
                    CREATE OR REPLACE VIEW document_processing_status AS
                    SELECT 
                        d.id,
                        d.file_name,
                        d.file_type,
                        d.status,
                        d.uploaded_at,
                        d.processed_at,
                        COUNT(c.id) as chunk_count,
                        COUNT(e.id) as enrichment_count,
                        COUNT(le.id) as logic_extraction_count
                    FROM documents d
                    LEFT JOIN chunks c ON d.id = c.document_id
                    LEFT JOIN enrichments e ON c.id = e.chunk_id
                    LEFT JOIN logic_extractions le ON c.id = le.chunk_id
                    GROUP BY d.id, d.file_name, d.file_type, d.status, d.uploaded_at, d.processed_at
                """))
                
                # View for knowledge statistics
                conn.execute(text("""
                    CREATE OR REPLACE VIEW knowledge_statistics AS
                    SELECT 
                        COUNT(DISTINCT dk.id) as document_knowledge_count,
                        COUNT(DISTINCT sk.id) as subject_knowledge_count,
                        COUNT(DISTINCT c.id) as cluster_count,
                        COUNT(DISTINCT ve.id) as embedding_count
                    FROM document_knowledge dk
                    FULL OUTER JOIN subject_knowledge sk ON 1=1
                    FULL OUTER JOIN clusters c ON 1=1
                    FULL OUTER JOIN vector_embeddings ve ON 1=1
                """))
                
                # View for processing job statistics
                conn.execute(text("""
                    CREATE OR REPLACE VIEW processing_job_statistics AS
                    SELECT 
                        job_type,
                        status,
                        COUNT(*) as job_count,
                        AVG(progress) as avg_progress,
                        MIN(created_at) as earliest_job,
                        MAX(created_at) as latest_job
                    FROM processing_jobs
                    GROUP BY job_type, status
                """))
            
            logger.info("Database views created successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create views: {e}")
            raise
    
    def insert_initial_data(self):
        """Insert initial data into the database."""
        try:
            logger.info("Inserting initial data")
            
            Session = sessionmaker(bind=self.engine)
            session = Session()
            
            try:
                # Insert system configuration
                session.execute(text("""
                    INSERT INTO processing_jobs (id, job_type, status, input_data, created_at)
                    VALUES ('system_init', 'system', 'completed', '{"message": "Database initialized"}', NOW())
                    ON CONFLICT (id) DO NOTHING
                """))
                
                session.commit()
                logger.info("Initial data inserted successfully")
                
            except Exception as e:
                session.rollback()
                raise
            finally:
                session.close()
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to insert initial data: {e}")
            raise
    
    def verify_setup(self):
        """Verify that the database setup is correct."""
        try:
            logger.info("Verifying database setup")
            
            with self.engine.connect() as conn:
                # Check if all tables exist
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """))
                
                tables = [row[0] for row in result.fetchall()]
                expected_tables = [
                    'documents', 'chunks', 'enrichments', 'logic_extractions',
                    'document_knowledge', 'subject_knowledge', 'clusters',
                    'vector_embeddings', 'qa_sessions', 'processing_jobs'
                ]
                
                missing_tables = set(expected_tables) - set(tables)
                if missing_tables:
                    logger.error(f"Missing tables: {missing_tables}")
                    return False
                
                # Check if indexes exist
                result = conn.execute(text("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'public'
                    ORDER BY indexname
                """))
                
                indexes = [row[0] for row in result.fetchall()]
                expected_indexes = [
                    'idx_documents_status', 'idx_chunks_document_id',
                    'idx_enrichments_chunk_id', 'idx_logic_extractions_chunk_id'
                ]
                
                missing_indexes = set(expected_indexes) - set(indexes)
                if missing_indexes:
                    logger.warning(f"Missing indexes: {missing_indexes}")
                
                logger.info("Database setup verification completed")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database verification failed: {e}")
            return False
    
    def initialize_database(self):
        """Initialize the complete database."""
        try:
            logger.info("Starting database initialization")
            
            # Connect to database
            self.connect()
            
            # Create tables
            self.create_tables()
            
            # Create indexes
            self.create_indexes()
            
            # Create views
            self.create_views()
            
            # Insert initial data
            self.insert_initial_data()
            
            # Verify setup
            if self.verify_setup():
                logger.info("Database initialization completed successfully")
                return True
            else:
                logger.error("Database initialization verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False


def main():
    """Main entry point for database initialization."""
    parser = argparse.ArgumentParser(description='LERK Database Initialization')
    parser.add_argument('--database-url', 
                       default=os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/lerk_db'),
                       help='Database connection URL')
    parser.add_argument('--drop-existing', action='store_true',
                       help='Drop existing tables before creating new ones')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create database initializer
    initializer = DatabaseInitializer(
        database_url=args.database_url,
        drop_existing=args.drop_existing
    )
    
    try:
        success = initializer.initialize_database()
        
        if success:
            print("\n" + "="*50)
            print("DATABASE INITIALIZATION SUCCESSFUL")
            print("="*50)
            print(f"Database URL: {args.database_url}")
            print("Tables created: documents, chunks, enrichments, logic_extractions,")
            print("               document_knowledge, subject_knowledge, clusters,")
            print("               vector_embeddings, qa_sessions, processing_jobs")
            print("Indexes created: Performance indexes for all tables")
            print("Views created: document_processing_status, knowledge_statistics,")
            print("              processing_job_statistics")
            print("="*50)
        else:
            print("Database initialization failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
