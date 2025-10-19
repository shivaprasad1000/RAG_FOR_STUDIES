"""
Tests for vector store functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.vector_store import VectorStore, VectorMetadata
from app.models import TextChunk, SearchResult


class TestVectorStore:
    """Test cases for VectorStore class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Use temporary directory for index storage
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / "test_index"
        
        # Initialize vector store with 3D embeddings for testing
        self.embedding_dim = 3
        self.vector_store = VectorStore(
            embedding_dimension=self.embedding_dim,
            index_path=str(self.index_path)
        )
        
        # Test data
        self.test_chunks = [
            TextChunk(
                id="chunk-1",
                document_id="doc-1",
                content="First test chunk content",
                chunk_index=0,
                page_number=1
            ),
            TextChunk(
                id="chunk-2",
                document_id="doc-1",
                content="Second test chunk content",
                chunk_index=1,
                page_number=1
            ),
            TextChunk(
                id="chunk-3",
                document_id="doc-2",
                content="Third test chunk from different document",
                chunk_index=0,
                page_number=2
            )
        ]
        
        self.test_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        
        self.document_name = "test_document.pdf"
    
    def teardown_method(self):
        """Clean up temporary directory"""
        try:
            shutil.rmtree(self.temp_dir)
        except FileNotFoundError:
            pass
    
    def test_init_creates_empty_index(self):
        """Test VectorStore initialization creates empty index"""
        assert self.vector_store.embedding_dimension == self.embedding_dim
        assert self.vector_store.index.ntotal == 0
        assert len(self.vector_store.metadata) == 0
        assert len(self.vector_store.doc_to_indices) == 0
    
    def test_init_with_custom_path(self):
        """Test VectorStore initialization with custom path"""
        custom_path = Path(self.temp_dir) / "custom_index"
        store = VectorStore(embedding_dimension=4, index_path=str(custom_path))
        
        assert store.index_path == custom_path
        assert store.metadata_path == custom_path.with_suffix('.metadata')
    
    def test_add_documents_success(self):
        """Test successful document addition"""
        chunks_subset = self.test_chunks[:2]  # First two chunks from same document
        embeddings_subset = self.test_embeddings[:2]
        
        self.vector_store.add_documents(chunks_subset, embeddings_subset, self.document_name)
        
        # Check index was updated
        assert self.vector_store.index.ntotal == 2
        assert len(self.vector_store.metadata) == 2
        assert "doc-1" in self.vector_store.doc_to_indices
        assert len(self.vector_store.doc_to_indices["doc-1"]) == 2
    
    def test_add_documents_length_mismatch(self):
        """Test adding documents with mismatched chunks and embeddings"""
        chunks_subset = self.test_chunks[:2]
        embeddings_subset = self.test_embeddings[:1]  # Different length
        
        with pytest.raises(ValueError) as exc_info:
            self.vector_store.add_documents(chunks_subset, embeddings_subset, self.document_name)
        
        assert "length mismatch" in str(exc_info.value)
    
    def test_add_documents_empty_list(self):
        """Test adding empty list of documents"""
        self.vector_store.add_documents([], np.array([]).reshape(0, 3), self.document_name)
        
        # Should remain empty
        assert self.vector_store.index.ntotal == 0
        assert len(self.vector_store.metadata) == 0
    
    def test_add_documents_multiple_calls(self):
        """Test adding documents in multiple calls"""
        # Add first document
        chunks1 = self.test_chunks[:2]
        embeddings1 = self.test_embeddings[:2]
        self.vector_store.add_documents(chunks1, embeddings1, "doc1.pdf")
        
        # Add second document
        chunks2 = self.test_chunks[2:]
        embeddings2 = self.test_embeddings[2:]
        self.vector_store.add_documents(chunks2, embeddings2, "doc2.pdf")
        
        # Check both documents are stored
        assert self.vector_store.index.ntotal == 3
        assert len(self.vector_store.metadata) == 3
        assert len(self.vector_store.doc_to_indices) == 2
    
    def test_search_empty_index(self):
        """Test searching in empty index"""
        query_embedding = np.array([0.1, 0.2, 0.3])
        results = self.vector_store.search(query_embedding, top_k=3)
        
        assert results == []
    
    def test_search_success(self):
        """Test successful similarity search"""
        # Add documents first
        self.vector_store.add_documents(self.test_chunks, self.test_embeddings, self.document_name)
        
        # Search with query similar to first embedding
        query_embedding = np.array([0.1, 0.2, 0.3])
        results = self.vector_store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result, SearchResult) for result in results)
        assert all(result.similarity_score >= 0 for result in results)
        
        # Results should be sorted by similarity (highest first)
        if len(results) > 1:
            assert results[0].similarity_score >= results[1].similarity_score
    
    def test_search_wrong_dimension(self):
        """Test search with wrong embedding dimension"""
        query_embedding = np.array([0.1, 0.2])  # 2D instead of 3D
        
        with pytest.raises(ValueError) as exc_info:
            self.vector_store.search(query_embedding, top_k=3)
        
        assert "dimension" in str(exc_info.value).lower()
    
    def test_search_top_k_larger_than_index(self):
        """Test search with top_k larger than index size"""
        # Add only one document
        self.vector_store.add_documents([self.test_chunks[0]], self.test_embeddings[:1], self.document_name)
        
        query_embedding = np.array([0.1, 0.2, 0.3])
        results = self.vector_store.search(query_embedding, top_k=10)
        
        # Should return only available results
        assert len(results) == 1
    
    def test_delete_document_success(self):
        """Test successful document deletion"""
        # Add documents separately to ensure proper document tracking
        chunks1 = self.test_chunks[:2]  # doc-1 chunks
        embeddings1 = self.test_embeddings[:2]
        self.vector_store.add_documents(chunks1, embeddings1, "doc1.pdf")
        
        chunks2 = self.test_chunks[2:]  # doc-2 chunks
        embeddings2 = self.test_embeddings[2:]
        self.vector_store.add_documents(chunks2, embeddings2, "doc2.pdf")
        
        # Delete first document
        success = self.vector_store.delete_document("doc-1")
        
        assert success is True
        assert self.vector_store.index.ntotal == 1  # Only doc-2 chunk remains
        assert "doc-1" not in self.vector_store.doc_to_indices
        assert "doc-2" in self.vector_store.doc_to_indices
    
    def test_delete_document_not_found(self):
        """Test deleting non-existent document"""
        success = self.vector_store.delete_document("non-existent-doc")
        
        assert success is False
    
    def test_delete_document_empty_indices(self):
        """Test deleting document with empty indices list"""
        # Manually add document with empty indices (edge case)
        self.vector_store.doc_to_indices["empty-doc"] = []
        
        success = self.vector_store.delete_document("empty-doc")
        
        assert success is False
    
    def test_delete_all_documents(self):
        """Test deleting all documents leaves empty index"""
        # Add documents
        self.vector_store.add_documents(self.test_chunks, self.test_embeddings, self.document_name)
        
        # Delete all documents
        self.vector_store.delete_document("doc-1")
        self.vector_store.delete_document("doc-2")
        
        assert self.vector_store.index.ntotal == 0
        assert len(self.vector_store.metadata) == 0
        assert len(self.vector_store.doc_to_indices) == 0
    
    def test_get_stats_empty(self):
        """Test statistics for empty vector store"""
        stats = self.vector_store.get_stats()
        
        assert stats['total_vectors'] == 0
        assert stats['embedding_dimension'] == self.embedding_dim
        assert stats['total_documents'] == 0
        assert stats['documents'] == {}
    
    def test_get_stats_with_data(self):
        """Test statistics with data in vector store"""
        # Add documents separately to get correct document count
        chunks1 = self.test_chunks[:2]  # doc-1 chunks
        embeddings1 = self.test_embeddings[:2]
        self.vector_store.add_documents(chunks1, embeddings1, "doc1.pdf")
        
        chunks2 = self.test_chunks[2:]  # doc-2 chunks
        embeddings2 = self.test_embeddings[2:]
        self.vector_store.add_documents(chunks2, embeddings2, "doc2.pdf")
        
        stats = self.vector_store.get_stats()
        
        assert stats['total_vectors'] == 3
        assert stats['total_documents'] == 2
        assert stats['documents']['doc-1'] == 2
        assert stats['documents']['doc-2'] == 1
    
    @patch('faiss.write_index')
    @patch('builtins.open')
    def test_save_index_success(self, mock_open, mock_write_index):
        """Test successful index saving"""
        # Add some data
        self.vector_store.add_documents([self.test_chunks[0]], self.test_embeddings[:1], self.document_name)
        
        # Save should be called during add_documents
        mock_write_index.assert_called()
        mock_open.assert_called()
    
    @patch('faiss.write_index')
    def test_save_index_failure(self, mock_write_index):
        """Test index saving failure"""
        mock_write_index.side_effect = Exception("Save failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            self.vector_store._save_index()
        
        assert "Could not save vector index" in str(exc_info.value)
    
    @patch('faiss.read_index')
    @patch('builtins.open')
    @patch.object(Path, 'exists')
    def test_load_index_success(self, mock_exists, mock_open, mock_read_index):
        """Test successful index loading"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_read_index.return_value = mock_index
        
        # Mock metadata file
        mock_metadata = {
            'metadata': [
                VectorMetadata("chunk-1", "doc-1", "test.pdf", 0, 1),
                VectorMetadata("chunk-2", "doc-1", "test.pdf", 1, 1)
            ],
            'doc_to_indices': {"doc-1": [0, 1]}
        }
        mock_open.return_value.__enter__.return_value.read.return_value = mock_metadata
        
        # Create new vector store (should load existing)
        new_store = VectorStore(embedding_dimension=3, index_path=str(self.index_path))
        
        mock_read_index.assert_called_once()
    
    @patch('faiss.read_index')
    @patch.object(Path, 'exists')
    def test_load_index_failure(self, mock_exists, mock_read_index):
        """Test index loading failure falls back to empty index"""
        mock_exists.return_value = True
        mock_read_index.side_effect = Exception("Load failed")
        
        # Should create empty index on load failure
        new_store = VectorStore(embedding_dimension=3, index_path=str(self.index_path))
        
        assert new_store.index.ntotal == 0
        assert len(new_store.metadata) == 0


    def test_vector_similarity_search_accuracy(self):
        """Test that vector similarity search returns accurate results"""
        # Create embeddings with known similarity relationships
        # Embedding 1 and 2 are similar (small angle), 3 is different
        similar_embeddings = np.array([
            [1.0, 0.1, 0.0],    # Base vector
            [0.9, 0.2, 0.1],    # Similar to base (should rank high)
            [0.0, 0.0, 1.0]     # Orthogonal to base (should rank low)
        ])
        
        # Normalize embeddings for cosine similarity
        similar_embeddings = similar_embeddings / np.linalg.norm(similar_embeddings, axis=1, keepdims=True)
        
        # Add documents with these embeddings
        self.vector_store.add_documents(self.test_chunks, similar_embeddings, self.document_name)
        
        # Query with vector similar to the first embedding
        query_embedding = np.array([0.95, 0.15, 0.05])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        results = self.vector_store.search(query_embedding, top_k=3)
        
        # Should return all 3 results, sorted by similarity
        assert len(results) == 3
        
        # First result should have highest similarity (most similar to query)
        assert results[0].similarity_score >= results[1].similarity_score
        assert results[1].similarity_score >= results[2].similarity_score
        
        # First result should be reasonably similar (cosine similarity > 0.8)
        assert results[0].similarity_score > 0.8
        
        # Last result should be less similar (orthogonal vector)
        assert results[2].similarity_score < 0.3
    
    def test_index_persistence_and_loading(self):
        """Test that vector index can be saved and loaded correctly"""
        # Add some documents to the original vector store
        original_chunks = self.test_chunks[:2]
        original_embeddings = self.test_embeddings[:2]
        self.vector_store.add_documents(original_chunks, original_embeddings, "original.pdf")
        
        # Get original stats
        original_stats = self.vector_store.get_stats()
        
        # Perform a search to get baseline results
        query_embedding = np.array([0.1, 0.2, 0.3])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        original_results = self.vector_store.search(query_embedding, top_k=2)
        
        # Create a new vector store with the same index path (should load existing data)
        new_vector_store = VectorStore(
            embedding_dimension=self.embedding_dim,
            index_path=str(self.index_path)
        )
        
        # New vector store should have loaded the same data
        new_stats = new_vector_store.get_stats()
        assert new_stats['total_vectors'] == original_stats['total_vectors']
        assert new_stats['total_documents'] == original_stats['total_documents']
        
        # Search results should be identical
        new_results = new_vector_store.search(query_embedding, top_k=2)
        assert len(new_results) == len(original_results)
        
        for orig, new in zip(original_results, new_results):
            assert orig.chunk.id == new.chunk.id
            assert abs(orig.similarity_score - new.similarity_score) < 1e-6
    
    def test_index_persistence_with_modifications(self):
        """Test index persistence after adding and deleting documents"""
        # Add initial documents
        chunks1 = self.test_chunks[:2]
        embeddings1 = self.test_embeddings[:2]
        self.vector_store.add_documents(chunks1, embeddings1, "doc1.pdf")
        
        # Add more documents
        chunks2 = self.test_chunks[2:]
        embeddings2 = self.test_embeddings[2:]
        self.vector_store.add_documents(chunks2, embeddings2, "doc2.pdf")
        
        # Delete one document
        self.vector_store.delete_document("doc-1")
        
        # Get final stats
        final_stats = self.vector_store.get_stats()
        
        # Create new vector store (should load persisted state)
        new_vector_store = VectorStore(
            embedding_dimension=self.embedding_dim,
            index_path=str(self.index_path)
        )
        
        # Should match the final state after modifications
        loaded_stats = new_vector_store.get_stats()
        assert loaded_stats['total_vectors'] == final_stats['total_vectors']
        assert loaded_stats['total_documents'] == final_stats['total_documents']
        assert "doc-1" not in loaded_stats['documents']
        assert "doc-2" in loaded_stats['documents']
    
    def test_index_loading_with_corrupted_files(self):
        """Test graceful handling of corrupted index files"""
        # Create corrupted index file
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w') as f:
            f.write("corrupted data")
        
        # Should create empty index when loading fails
        new_vector_store = VectorStore(
            embedding_dimension=self.embedding_dim,
            index_path=str(self.index_path)
        )
        
        assert new_vector_store.index.ntotal == 0
        assert len(new_vector_store.metadata) == 0
        assert len(new_vector_store.doc_to_indices) == 0


class TestVectorMetadata:
    """Test cases for VectorMetadata dataclass"""
    
    def test_vector_metadata_creation(self):
        """Test VectorMetadata creation with all fields"""
        metadata = VectorMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="test.pdf",
            chunk_index=0,
            page_number=1
        )
        
        assert metadata.chunk_id == "chunk-1"
        assert metadata.document_id == "doc-1"
        assert metadata.document_name == "test.pdf"
        assert metadata.chunk_index == 0
        assert metadata.page_number == 1
    
    def test_vector_metadata_optional_page_number(self):
        """Test VectorMetadata creation with optional page number"""
        metadata = VectorMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="test.pdf",
            chunk_index=0
        )
        
        assert metadata.page_number is None


if __name__ == "__main__":
    pytest.main([__file__])