"""
Database module for managing document and chunk metadata using SQLite.
"""

import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from .models import Document, TextChunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    SQLite database manager for storing document and chunk metadata.
    Handles all database operations for the RAG system.
    """
    
    def __init__(self, db_path: str = "data/study_assistant.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures proper connection handling and cleanup.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """
        Initialize database schema with tables for documents and chunks.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        upload_date TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        page_count INTEGER,
                        chunk_count INTEGER DEFAULT 0,
                        tags TEXT DEFAULT '[]',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create chunks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page_number INTEGER,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                    ON chunks (document_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_page_number 
                    ON chunks (page_number)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_documents_filename 
                    ON documents (filename)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def create_document(self, document: Document) -> bool:
        """
        Create a new document record in the database.
        
        Args:
            document: Document object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO documents (
                        id, filename, upload_date, file_size, 
                        page_count, chunk_count, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    document.id,
                    document.filename,
                    document.upload_date.isoformat(),
                    document.file_size,
                    document.page_count,
                    document.chunk_count,
                    json.dumps(document.tags)
                ))
                
                conn.commit()
                logger.info(f"Document {document.id} created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create document {document.id}: {str(e)}")
            return False
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Document object if found, None otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, filename, upload_date, file_size, 
                           page_count, chunk_count, tags
                    FROM documents 
                    WHERE id = ?
                """, (document_id,))
                
                row = cursor.fetchone()
                if row:
                    return Document(
                        id=row['id'],
                        filename=row['filename'],
                        upload_date=datetime.fromisoformat(row['upload_date']),
                        file_size=row['file_size'],
                        page_count=row['page_count'],
                        chunk_count=row['chunk_count'],
                        tags=json.loads(row['tags']) if row['tags'] else []
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    def list_documents(self) -> List[Document]:
        """
        List all documents in the database.
        
        Returns:
            List of Document objects
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, filename, upload_date, file_size, 
                           page_count, chunk_count, tags
                    FROM documents 
                    ORDER BY upload_date DESC
                """)
                
                documents = []
                for row in cursor.fetchall():
                    document = Document(
                        id=row['id'],
                        filename=row['filename'],
                        upload_date=datetime.fromisoformat(row['upload_date']),
                        file_size=row['file_size'],
                        page_count=row['page_count'],
                        chunk_count=row['chunk_count'],
                        tags=json.loads(row['tags']) if row['tags'] else []
                    )
                    documents.append(document)
                
                return documents
                
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return []
    
    def update_document_chunk_count(self, document_id: str, chunk_count: int) -> bool:
        """
        Update the chunk count for a document.
        
        Args:
            document_id: Document identifier
            chunk_count: New chunk count
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE documents 
                    SET chunk_count = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (chunk_count, document_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated chunk count for document {document_id}: {chunk_count}")
                    return True
                else:
                    logger.warning(f"Document {document_id} not found for chunk count update")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update chunk count for document {document_id}: {str(e)}")
            return False
    
    def update_document_tags(self, document_id: str, tags: List[str]) -> bool:
        """
        Update tags for a document.
        
        Args:
            document_id: Document identifier
            tags: List of tags to assign to the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE documents 
                    SET tags = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (json.dumps(tags), document_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated tags for document {document_id}: {tags}")
                    return True
                else:
                    logger.warning(f"Document {document_id} not found for tags update")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update tags for document {document_id}: {str(e)}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete chunks first (foreign key constraint)
                cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                chunks_deleted = cursor.rowcount
                
                # Delete document
                cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                document_deleted = cursor.rowcount
                
                conn.commit()
                
                if document_deleted > 0:
                    logger.info(f"Deleted document {document_id} and {chunks_deleted} chunks")
                    return True
                else:
                    logger.warning(f"Document {document_id} not found for deletion")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    def create_chunks(self, chunks: List[TextChunk]) -> bool:
        """
        Create multiple chunk records in the database.
        
        Args:
            chunks: List of TextChunk objects to store
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            return True
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                chunk_data = [
                    (chunk.id, chunk.document_id, chunk.content, 
                     chunk.chunk_index, chunk.page_number)
                    for chunk in chunks
                ]
                
                cursor.executemany("""
                    INSERT INTO chunks (
                        id, document_id, content, chunk_index, page_number
                    ) VALUES (?, ?, ?, ?, ?)
                """, chunk_data)
                
                conn.commit()
                logger.info(f"Created {len(chunks)} chunks successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create chunks: {str(e)}")
            return False
    
    def get_document_chunks(self, document_id: str) -> List[TextChunk]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of TextChunk objects
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, document_id, content, chunk_index, page_number
                    FROM chunks 
                    WHERE document_id = ?
                    ORDER BY chunk_index
                """, (document_id,))
                
                chunks = []
                for row in cursor.fetchall():
                    chunk = TextChunk(
                        id=row['id'],
                        document_id=row['document_id'],
                        content=row['content'],
                        chunk_index=row['chunk_index'],
                        page_number=row['page_number']
                    )
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {str(e)}")
            return []
    
    def get_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            TextChunk object if found, None otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, document_id, content, chunk_index, page_number
                    FROM chunks 
                    WHERE id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    return TextChunk(
                        id=row['id'],
                        document_id=row['document_id'],
                        content=row['content'],
                        chunk_index=row['chunk_index'],
                        page_number=row['page_number']
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {str(e)}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Count documents
                cursor.execute("SELECT COUNT(*) as count FROM documents")
                doc_count = cursor.fetchone()['count']
                
                # Count chunks
                cursor.execute("SELECT COUNT(*) as count FROM chunks")
                chunk_count = cursor.fetchone()['count']
                
                # Get database file size
                db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
                
                return {
                    'document_count': doc_count,
                    'chunk_count': chunk_count,
                    'database_size_bytes': db_size,
                    'database_path': self.db_path
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {
                'document_count': 0,
                'chunk_count': 0,
                'database_size_bytes': 0,
                'database_path': self.db_path,
                'error': str(e)
            }


# Global database instance
db_manager = DatabaseManager()


def init_database():
    """
    Initialize the database. Can be called from main application.
    """
    try:
        db_manager.init_database()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("Database initialized successfully!")