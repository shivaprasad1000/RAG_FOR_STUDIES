"""
Tests for database functionality.
"""

import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path

from app.database import DatabaseManager
from app.models import Document, TextChunk


class TestDatabaseManager:
    """Test cases for DatabaseManager class"""
    
    def setup_method(self):
        """Set up test fixtures with temporary database"""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db_manager = DatabaseManager(db_path=self.temp_db.name)
        
        # Test data
        self.test_document = Document(
            id="test-doc-123",
            filename="test_document.pdf",
            upload_date=datetime.now(),
            file_size=1024000,
            page_count=10,
            chunk_count=5,
            tags=["math", "calculus"]
        )
        
        self.test_chunks = [
            TextChunk(
                id="chunk-1",
                document_id="test-doc-123",
                content="This is the first chunk of text.",
                chunk_index=0,
                page_number=1
            ),
            TextChunk(
                id="chunk-2",
                document_id="test-doc-123",
                content="This is the second chunk of text.",
                chunk_index=1,
                page_number=2
            )
        ]
    
    def teardown_method(self):
        """Clean up temporary database"""
        try:
            os.unlink(self.temp_db.name)
        except FileNotFoundError:
            pass
    
    def test_database_initialization(self):
        """Test database initialization creates tables"""
        # Database should be initialized in setup_method
        # Check if we can query the tables (this will fail if tables don't exist)
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check documents table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            assert cursor.fetchone() is not None
            
            # Check chunks table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
            assert cursor.fetchone() is not None
    
    def test_create_document_success(self):
        """Test successful document creation"""
        success = self.db_manager.create_document(self.test_document)
        assert success is True
        
        # Verify document was created
        retrieved_doc = self.db_manager.get_document(self.test_document.id)
        assert retrieved_doc is not None
        assert retrieved_doc.id == self.test_document.id
        assert retrieved_doc.filename == self.test_document.filename
        assert retrieved_doc.file_size == self.test_document.file_size
        assert retrieved_doc.tags == self.test_document.tags
    
    def test_create_document_duplicate_id(self):
        """Test creating document with duplicate ID fails gracefully"""
        # Create document first time
        success1 = self.db_manager.create_document(self.test_document)
        assert success1 is True
        
        # Try to create same document again
        success2 = self.db_manager.create_document(self.test_document)
        assert success2 is False
    
    def test_get_document_not_found(self):
        """Test getting non-existent document returns None"""
        result = self.db_manager.get_document("non-existent-id")
        assert result is None
    
    def test_list_documents_empty(self):
        """Test listing documents when database is empty"""
        documents = self.db_manager.list_documents()
        assert documents == []
    
    def test_list_documents_with_data(self):
        """Test listing documents with data in database"""
        # Create test document
        self.db_manager.create_document(self.test_document)
        
        documents = self.db_manager.list_documents()
        assert len(documents) == 1
        assert documents[0].id == self.test_document.id
    
    def test_update_document_chunk_count(self):
        """Test updating document chunk count"""
        # Create document first
        self.db_manager.create_document(self.test_document)
        
        # Update chunk count
        success = self.db_manager.update_document_chunk_count(self.test_document.id, 10)
        assert success is True
        
        # Verify update
        updated_doc = self.db_manager.get_document(self.test_document.id)
        assert updated_doc.chunk_count == 10
    
    def test_update_document_chunk_count_not_found(self):
        """Test updating chunk count for non-existent document"""
        success = self.db_manager.update_document_chunk_count("non-existent", 5)
        assert success is False
    
    def test_delete_document_success(self):
        """Test successful document deletion"""
        # Create document and chunks
        self.db_manager.create_document(self.test_document)
        self.db_manager.create_chunks(self.test_chunks)
        
        # Delete document
        success = self.db_manager.delete_document(self.test_document.id)
        assert success is True
        
        # Verify document is gone
        result = self.db_manager.get_document(self.test_document.id)
        assert result is None
        
        # Verify chunks are also gone
        chunks = self.db_manager.get_document_chunks(self.test_document.id)
        assert chunks == []
    
    def test_delete_document_not_found(self):
        """Test deleting non-existent document"""
        success = self.db_manager.delete_document("non-existent-id")
        assert success is False
    
    def test_create_chunks_success(self):
        """Test successful chunk creation"""
        # Create parent document first
        self.db_manager.create_document(self.test_document)
        
        success = self.db_manager.create_chunks(self.test_chunks)
        assert success is True
        
        # Verify chunks were created
        retrieved_chunks = self.db_manager.get_document_chunks(self.test_document.id)
        assert len(retrieved_chunks) == 2
        assert retrieved_chunks[0].content == self.test_chunks[0].content
        assert retrieved_chunks[1].content == self.test_chunks[1].content
    
    def test_create_chunks_empty_list(self):
        """Test creating empty list of chunks"""
        success = self.db_manager.create_chunks([])
        assert success is True
    
    def test_get_document_chunks_empty(self):
        """Test getting chunks for document with no chunks"""
        # Create document without chunks
        self.db_manager.create_document(self.test_document)
        
        chunks = self.db_manager.get_document_chunks(self.test_document.id)
        assert chunks == []
    
    def test_get_document_chunks_not_found(self):
        """Test getting chunks for non-existent document"""
        chunks = self.db_manager.get_document_chunks("non-existent-id")
        assert chunks == []
    
    def test_get_chunk_success(self):
        """Test successful chunk retrieval by ID"""
        # Create document and chunks
        self.db_manager.create_document(self.test_document)
        self.db_manager.create_chunks(self.test_chunks)
        
        chunk = self.db_manager.get_chunk(self.test_chunks[0].id)
        assert chunk is not None
        assert chunk.id == self.test_chunks[0].id
        assert chunk.content == self.test_chunks[0].content
    
    def test_get_chunk_not_found(self):
        """Test getting non-existent chunk"""
        result = self.db_manager.get_chunk("non-existent-chunk")
        assert result is None
    
    def test_get_database_stats_empty(self):
        """Test database statistics with empty database"""
        stats = self.db_manager.get_database_stats()
        
        assert stats['document_count'] == 0
        assert stats['chunk_count'] == 0
        assert stats['database_size_bytes'] > 0  # SQLite creates some overhead
        assert stats['database_path'] == self.temp_db.name
    
    def test_get_database_stats_with_data(self):
        """Test database statistics with data"""
        # Add test data
        self.db_manager.create_document(self.test_document)
        self.db_manager.create_chunks(self.test_chunks)
        
        stats = self.db_manager.get_database_stats()
        
        assert stats['document_count'] == 1
        assert stats['chunk_count'] == 2
        assert stats['database_size_bytes'] > 0
    
    def test_connection_context_manager(self):
        """Test database connection context manager"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_connection_context_manager_with_error(self):
        """Test database connection context manager handles errors"""
        with pytest.raises(Exception):
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INVALID SQL QUERY")
    
    def test_foreign_key_constraint(self):
        """Test that foreign key constraints work properly"""
        # Try to create chunks without parent document
        success = self.db_manager.create_chunks(self.test_chunks)
        # This should still succeed as SQLite doesn't enforce foreign keys by default
        # But the chunks will be orphaned
        assert success is True
        
        # Verify chunks exist but have no parent document
        chunk = self.db_manager.get_chunk(self.test_chunks[0].id)
        assert chunk is not None
        
        parent_doc = self.db_manager.get_document(chunk.document_id)
        assert parent_doc is None  # No parent document exists


if __name__ == "__main__":
    pytest.main([__file__])


class TestDocumentTagging:
    """Test document tagging functionality in database."""
    
    def setup_method(self):
        """Set up test fixtures with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = DatabaseManager(db_path=self.temp_db.name)
    
    def teardown_method(self):
        """Clean up temporary database"""
        os.unlink(self.temp_db.name)
    
    def test_create_document_with_tags(self):
        """Test creating a document with tags."""
        document = Document(
            id="tagged-doc-1",
            filename="tagged_document.pdf",
            upload_date=datetime.now(),
            file_size=2048000,
            page_count=15,
            chunk_count=8,
            tags=["physics", "mechanics", "chapter-1"]
        )
        
        # Create document
        success = self.db_manager.create_document(document)
        assert success
        
        # Retrieve and verify tags
        retrieved_doc = self.db_manager.get_document("tagged-doc-1")
        assert retrieved_doc is not None
        assert retrieved_doc.tags == ["physics", "mechanics", "chapter-1"]
    
    def test_create_document_without_tags(self):
        """Test creating a document without tags."""
        document = Document(
            id="untagged-doc-1",
            filename="untagged_document.pdf",
            upload_date=datetime.now(),
            file_size=1024000,
            page_count=5,
            chunk_count=3,
            tags=[]
        )
        
        # Create document
        success = self.db_manager.create_document(document)
        assert success
        
        # Retrieve and verify empty tags
        retrieved_doc = self.db_manager.get_document("untagged-doc-1")
        assert retrieved_doc is not None
        assert retrieved_doc.tags == []
    
    def test_update_document_tags(self):
        """Test updating document tags."""
        # Create document with initial tags
        document = Document(
            id="update-tags-doc",
            filename="update_tags.pdf",
            upload_date=datetime.now(),
            file_size=1024000,
            page_count=5,
            chunk_count=3,
            tags=["old-tag"]
        )
        
        success = self.db_manager.create_document(document)
        assert success
        
        # Update tags
        new_tags = ["chemistry", "organic", "reactions"]
        success = self.db_manager.update_document_tags("update-tags-doc", new_tags)
        assert success
        
        # Verify tags were updated
        updated_doc = self.db_manager.get_document("update-tags-doc")
        assert updated_doc is not None
        assert updated_doc.tags == new_tags
    
    def test_update_tags_nonexistent_document(self):
        """Test updating tags for a document that doesn't exist."""
        success = self.db_manager.update_document_tags("nonexistent-doc", ["tag1", "tag2"])
        assert not success
    
    def test_update_tags_empty_list(self):
        """Test updating document with empty tag list."""
        # Create document with tags
        document = Document(
            id="clear-tags-doc",
            filename="clear_tags.pdf",
            upload_date=datetime.now(),
            file_size=1024000,
            page_count=5,
            chunk_count=3,
            tags=["tag1", "tag2", "tag3"]
        )
        
        success = self.db_manager.create_document(document)
        assert success
        
        # Clear tags
        success = self.db_manager.update_document_tags("clear-tags-doc", [])
        assert success
        
        # Verify tags were cleared
        updated_doc = self.db_manager.get_document("clear-tags-doc")
        assert updated_doc is not None
        assert updated_doc.tags == []
    
    def test_tags_with_special_characters(self):
        """Test tags with special characters and unicode."""
        special_tags = [
            "Math & Science",
            "Chapter 1-5",
            "Review (Final)",
            "Español",
            "数学",
            "Tag with spaces"
        ]
        
        document = Document(
            id="special-chars-doc",
            filename="special_chars.pdf",
            upload_date=datetime.now(),
            file_size=1024000,
            page_count=5,
            chunk_count=3,
            tags=special_tags
        )
        
        # Create document
        success = self.db_manager.create_document(document)
        assert success
        
        # Retrieve and verify special characters are preserved
        retrieved_doc = self.db_manager.get_document("special-chars-doc")
        assert retrieved_doc is not None
        assert retrieved_doc.tags == special_tags
    
    def test_list_documents_with_tags(self):
        """Test listing documents includes tag information."""
        # Create multiple documents with different tags
        documents = [
            Document(
                id="doc-1",
                filename="doc1.pdf",
                upload_date=datetime.now(),
                file_size=1024000,
                page_count=5,
                chunk_count=3,
                tags=["math", "algebra"]
            ),
            Document(
                id="doc-2",
                filename="doc2.pdf",
                upload_date=datetime.now(),
                file_size=2048000,
                page_count=10,
                chunk_count=6,
                tags=["science", "physics"]
            ),
            Document(
                id="doc-3",
                filename="doc3.pdf",
                upload_date=datetime.now(),
                file_size=512000,
                page_count=3,
                chunk_count=2,
                tags=[]
            )
        ]
        
        # Create all documents
        for doc in documents:
            success = self.db_manager.create_document(doc)
            assert success
        
        # List documents and verify tags
        retrieved_docs = self.db_manager.list_documents()
        assert len(retrieved_docs) == 3
        
        # Find each document and verify tags
        doc_dict = {doc.id: doc for doc in retrieved_docs}
        
        assert doc_dict["doc-1"].tags == ["math", "algebra"]
        assert doc_dict["doc-2"].tags == ["science", "physics"]
        assert doc_dict["doc-3"].tags == []
    
    def test_tags_persistence_after_database_restart(self):
        """Test that tags persist after database connection is closed and reopened."""
        # Create document with tags
        document = Document(
            id="persistence-doc",
            filename="persistence.pdf",
            upload_date=datetime.now(),
            file_size=1024000,
            page_count=5,
            chunk_count=3,
            tags=["persistent", "tag", "test"]
        )
        
        success = self.db_manager.create_document(document)
        assert success
        
        # Close current database manager
        db_path = self.db_manager.db_path
        del self.db_manager
        
        # Create new database manager with same path
        new_db_manager = DatabaseManager(db_path=db_path)
        
        # Retrieve document and verify tags persisted
        retrieved_doc = new_db_manager.get_document("persistence-doc")
        assert retrieved_doc is not None
        assert retrieved_doc.tags == ["persistent", "tag", "test"]


class TestStudyPreferencesIntegration:
    """Test study preferences integration with existing functionality."""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = DatabaseManager(db_path=self.temp_db.name)
    
    def teardown_method(self):
        """Clean up temporary database"""
        os.unlink(self.temp_db.name)
    
    def test_document_filtering_by_tags(self):
        """Test that documents can be conceptually filtered by tags (for frontend)."""
        # Create documents with various tags
        documents = [
            Document(
                id="math-doc",
                filename="mathematics.pdf",
                upload_date=datetime.now(),
                file_size=1024000,
                page_count=5,
                chunk_count=3,
                tags=["math", "calculus", "derivatives"]
            ),
            Document(
                id="physics-doc",
                filename="physics.pdf",
                upload_date=datetime.now(),
                file_size=2048000,
                page_count=10,
                chunk_count=6,
                tags=["physics", "mechanics", "forces"]
            ),
            Document(
                id="mixed-doc",
                filename="applied_math.pdf",
                upload_date=datetime.now(),
                file_size=1536000,
                page_count=8,
                chunk_count=4,
                tags=["math", "physics", "applications"]
            )
        ]
        
        # Create all documents
        for doc in documents:
            success = self.db_manager.create_document(doc)
            assert success
        
        # Retrieve all documents
        all_docs = self.db_manager.list_documents()
        assert len(all_docs) == 3
        
        # Test filtering logic (this would be done in frontend)
        def filter_by_tag(docs, tag):
            return [doc for doc in docs if tag.lower() in [t.lower() for t in doc.tags]]
        
        math_docs = filter_by_tag(all_docs, "math")
        assert len(math_docs) == 2
        assert all("math" in [t.lower() for t in doc.tags] for doc in math_docs)
        
        physics_docs = filter_by_tag(all_docs, "physics")
        assert len(physics_docs) == 2
        assert all("physics" in [t.lower() for t in doc.tags] for doc in physics_docs)
        
        calculus_docs = filter_by_tag(all_docs, "calculus")
        assert len(calculus_docs) == 1
        assert calculus_docs[0].id == "math-doc"


if __name__ == "__main__":
    pytest.main([__file__])