#!/usr/bin/env python3
"""
LERK System - Database Migration Script
This script handles database migrations and schema updates.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/db_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handles database migrations and schema updates."""
    
    def __init__(self, database_url: str):
        """
        Initialize the database migrator.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
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
    
    def get_current_version(self) -> str:
        """Get the current database version."""
        try:
            with self.engine.connect() as conn:
                # Check if migrations table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'migrations'
                    )
                """))
                
                if not result.fetchone()[0]:
                    return "0.0.0"
                
                # Get latest migration version
                result = conn.execute(text("""
                    SELECT version FROM migrations 
                    ORDER BY applied_at DESC 
                    LIMIT 1
                """))
                
                row = result.fetchone()
                return row[0] if row else "0.0.0"
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to get current version: {e}")
            return "0.0.0"
    
    def create_migrations_table(self):
        """Create the migrations table if it doesn't exist."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS migrations (
                        id SERIAL PRIMARY KEY,
                        version VARCHAR(20) NOT NULL UNIQUE,
                        description TEXT,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        rollback_sql TEXT
                    )
                """))
            
            logger.info("Migrations table created/verified")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create migrations table: {e}")
            raise
    
    def get_available_migrations(self) -> List[Dict[str, Any]]:
        """Get list of available migrations."""
        migrations = [
            {
                "version": "1.0.0",
                "description": "Initial database schema",
                "sql": """
                    -- This migration is handled by init_db.py
                    -- No additional SQL needed
                """,
                "rollback_sql": """
                    -- Rollback would drop all tables
                    -- This should be done manually if needed
                """
            },
            {
                "version": "1.1.0",
                "description": "Add user management tables",
                "sql": """
                    CREATE TABLE users (
                        id VARCHAR(255) PRIMARY KEY,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        is_admin BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE user_sessions (
                        id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        session_token VARCHAR(255) UNIQUE NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                    
                    CREATE INDEX idx_users_username ON users(username);
                    CREATE INDEX idx_users_email ON users(email);
                    CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
                    CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
                    CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
                """,
                "rollback_sql": """
                    DROP TABLE IF EXISTS user_sessions;
                    DROP TABLE IF EXISTS users;
                """
            },
            {
                "version": "1.2.0",
                "description": "Add document versioning and audit trail",
                "sql": """
                    ALTER TABLE documents ADD COLUMN version INTEGER DEFAULT 1;
                    ALTER TABLE documents ADD COLUMN parent_document_id VARCHAR(255);
                    ALTER TABLE documents ADD COLUMN change_summary TEXT;
                    
                    CREATE TABLE document_versions (
                        id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        version INTEGER NOT NULL,
                        file_path VARCHAR(500) NOT NULL,
                        change_summary TEXT,
                        created_by VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                    );
                    
                    CREATE INDEX idx_documents_parent_id ON documents(parent_document_id);
                    CREATE INDEX idx_documents_version ON documents(version);
                    CREATE INDEX idx_document_versions_document_id ON document_versions(document_id);
                    CREATE INDEX idx_document_versions_version ON document_versions(version);
                """,
                "rollback_sql": """
                    DROP TABLE IF EXISTS document_versions;
                    ALTER TABLE documents DROP COLUMN IF EXISTS version;
                    ALTER TABLE documents DROP COLUMN IF EXISTS parent_document_id;
                    ALTER TABLE documents DROP COLUMN IF EXISTS change_summary;
                """
            },
            {
                "version": "1.3.0",
                "description": "Add advanced analytics and metrics",
                "sql": """
                    CREATE TABLE analytics_events (
                        id VARCHAR(255) PRIMARY KEY,
                        event_type VARCHAR(100) NOT NULL,
                        user_id VARCHAR(255),
                        document_id VARCHAR(255),
                        session_id VARCHAR(255),
                        event_data JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE system_metrics (
                        id VARCHAR(255) PRIMARY KEY,
                        metric_name VARCHAR(100) NOT NULL,
                        metric_value FLOAT NOT NULL,
                        metric_unit VARCHAR(50),
                        tags JSON,
                        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX idx_analytics_events_type ON analytics_events(event_type);
                    CREATE INDEX idx_analytics_events_user_id ON analytics_events(user_id);
                    CREATE INDEX idx_analytics_events_created_at ON analytics_events(created_at);
                    CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
                    CREATE INDEX idx_system_metrics_recorded_at ON system_metrics(recorded_at);
                """,
                "rollback_sql": """
                    DROP TABLE IF EXISTS system_metrics;
                    DROP TABLE IF EXISTS analytics_events;
                """
            },
            {
                "version": "1.4.0",
                "description": "Add document collaboration features",
                "sql": """
                    CREATE TABLE document_collaborators (
                        id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        permission_level VARCHAR(50) NOT NULL,
                        invited_by VARCHAR(255),
                        invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accepted_at TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        UNIQUE(document_id, user_id)
                    );
                    
                    CREATE TABLE document_comments (
                        id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        chunk_id VARCHAR(255),
                        comment_text TEXT NOT NULL,
                        parent_comment_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_comment_id) REFERENCES document_comments(id) ON DELETE CASCADE
                    );
                    
                    CREATE INDEX idx_document_collaborators_document_id ON document_collaborators(document_id);
                    CREATE INDEX idx_document_collaborators_user_id ON document_collaborators(user_id);
                    CREATE INDEX idx_document_comments_document_id ON document_comments(document_id);
                    CREATE INDEX idx_document_comments_user_id ON document_comments(user_id);
                    CREATE INDEX idx_document_comments_chunk_id ON document_comments(chunk_id);
                """,
                "rollback_sql": """
                    DROP TABLE IF EXISTS document_comments;
                    DROP TABLE IF EXISTS document_collaborators;
                """
            }
        ]
        
        return migrations
    
    def apply_migration(self, migration: Dict[str, Any]) -> bool:
        """Apply a single migration."""
        try:
            logger.info(f"Applying migration {migration['version']}: {migration['description']}")
            
            with self.engine.connect() as conn:
                # Start transaction
                trans = conn.begin()
                
                try:
                    # Execute migration SQL
                    if migration['sql'].strip():
                        conn.execute(text(migration['sql']))
                    
                    # Record migration
                    conn.execute(text("""
                        INSERT INTO migrations (version, description, rollback_sql)
                        VALUES (:version, :description, :rollback_sql)
                    """), {
                        'version': migration['version'],
                        'description': migration['description'],
                        'rollback_sql': migration['rollback_sql']
                    })
                    
                    # Commit transaction
                    trans.commit()
                    
                    logger.info(f"Migration {migration['version']} applied successfully")
                    return True
                    
                except Exception as e:
                    trans.rollback()
                    logger.error(f"Failed to apply migration {migration['version']}: {e}")
                    return False
                    
        except SQLAlchemyError as e:
            logger.error(f"Failed to apply migration {migration['version']}: {e}")
            return False
    
    def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration."""
        try:
            logger.info(f"Rolling back migration {version}")
            
            with self.engine.connect() as conn:
                # Get migration details
                result = conn.execute(text("""
                    SELECT rollback_sql FROM migrations 
                    WHERE version = :version
                """), {'version': version})
                
                row = result.fetchone()
                if not row:
                    logger.error(f"Migration {version} not found")
                    return False
                
                rollback_sql = row[0]
                
                # Start transaction
                trans = conn.begin()
                
                try:
                    # Execute rollback SQL
                    if rollback_sql and rollback_sql.strip():
                        conn.execute(text(rollback_sql))
                    
                    # Remove migration record
                    conn.execute(text("""
                        DELETE FROM migrations WHERE version = :version
                    """), {'version': version})
                    
                    # Commit transaction
                    trans.commit()
                    
                    logger.info(f"Migration {version} rolled back successfully")
                    return True
                    
                except Exception as e:
                    trans.rollback()
                    logger.error(f"Failed to rollback migration {version}: {e}")
                    return False
                    
        except SQLAlchemyError as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            return False
    
    def migrate_to_version(self, target_version: str) -> bool:
        """Migrate database to a specific version."""
        try:
            logger.info(f"Migrating database to version {target_version}")
            
            # Create migrations table if it doesn't exist
            self.create_migrations_table()
            
            # Get current version
            current_version = self.get_current_version()
            logger.info(f"Current version: {current_version}")
            
            # Get available migrations
            migrations = self.get_available_migrations()
            
            # Filter migrations to apply
            migrations_to_apply = [
                m for m in migrations 
                if self._version_compare(m['version'], current_version) > 0 
                and self._version_compare(m['version'], target_version) <= 0
            ]
            
            if not migrations_to_apply:
                logger.info("No migrations to apply")
                return True
            
            # Apply migrations
            for migration in migrations_to_apply:
                if not self.apply_migration(migration):
                    logger.error(f"Failed to apply migration {migration['version']}")
                    return False
            
            logger.info(f"Database migrated to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two version strings."""
        def version_tuple(v):
            return tuple(map(int, v.split('.')))
        
        v1 = version_tuple(version1)
        v2 = version_tuple(version2)
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        try:
            current_version = self.get_current_version()
            available_migrations = self.get_available_migrations()
            
            # Get applied migrations
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT version, description, applied_at 
                    FROM migrations 
                    ORDER BY applied_at
                """))
                
                applied_migrations = [
                    {
                        'version': row[0],
                        'description': row[1],
                        'applied_at': row[2].isoformat() if row[2] else None
                    }
                    for row in result.fetchall()
                ]
            
            # Get pending migrations
            applied_versions = {m['version'] for m in applied_migrations}
            pending_migrations = [
                m for m in available_migrations 
                if m['version'] not in applied_versions
            ]
            
            return {
                'current_version': current_version,
                'applied_migrations': applied_migrations,
                'pending_migrations': pending_migrations,
                'total_migrations': len(available_migrations),
                'applied_count': len(applied_migrations),
                'pending_count': len(pending_migrations)
            }
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {}


def main():
    """Main entry point for database migration."""
    parser = argparse.ArgumentParser(description='LERK Database Migration')
    parser.add_argument('--database-url', 
                       default=os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/lerk_db'),
                       help='Database connection URL')
    parser.add_argument('--migrate-to', 
                       help='Target version to migrate to')
    parser.add_argument('--rollback', 
                       help='Version to rollback to')
    parser.add_argument('--status', action='store_true',
                       help='Show migration status')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create database migrator
    migrator = DatabaseMigrator(database_url=args.database_url)
    
    try:
        # Connect to database
        migrator.connect()
        
        if args.status:
            # Show migration status
            status = migrator.get_migration_status()
            
            print("\n" + "="*50)
            print("DATABASE MIGRATION STATUS")
            print("="*50)
            print(f"Current version: {status.get('current_version', 'Unknown')}")
            print(f"Applied migrations: {status.get('applied_count', 0)}")
            print(f"Pending migrations: {status.get('pending_count', 0)}")
            print(f"Total migrations: {status.get('total_migrations', 0)}")
            print("="*50)
            
            if status.get('applied_migrations'):
                print("\nApplied migrations:")
                for migration in status['applied_migrations']:
                    print(f"  {migration['version']}: {migration['description']}")
            
            if status.get('pending_migrations'):
                print("\nPending migrations:")
                for migration in status['pending_migrations']:
                    print(f"  {migration['version']}: {migration['description']}")
            
        elif args.rollback:
            # Rollback to specific version
            success = migrator.rollback_migration(args.rollback)
            if success:
                print(f"Successfully rolled back to version {args.rollback}")
            else:
                print(f"Failed to rollback to version {args.rollback}")
                sys.exit(1)
                
        elif args.migrate_to:
            # Migrate to specific version
            success = migrator.migrate_to_version(args.migrate_to)
            if success:
                print(f"Successfully migrated to version {args.migrate_to}")
            else:
                print(f"Failed to migrate to version {args.migrate_to}")
                sys.exit(1)
        else:
            # Migrate to latest version
            available_migrations = migrator.get_available_migrations()
            latest_version = max(m['version'] for m in available_migrations)
            
            success = migrator.migrate_to_version(latest_version)
            if success:
                print(f"Successfully migrated to latest version {latest_version}")
            else:
                print(f"Failed to migrate to latest version {latest_version}")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
