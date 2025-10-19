"""
Vector storage module for the Study Assistant RAG system.

This module provides FAISS-based vector storage for efficient similarity search
of document embeddings.
"""

import numpy as np
import faiss
import pickle
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

from .models import TextChunk, SearchResult

logger = logging.getLogger(__name__)

@dataclass
class VectorMetadata:
    """Metadata for vectors stored in the index."""
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page_number: Optional[int] = None

class VectorStore:
    """
    FAISS-based vector store for similarity search of document embeddings.
    
    Uses IndexFlatIP (Inner Product) for simplicity and good performance
    with normalized embeddings.
    """
    
    def __init__(self, embedding_dimension: int, index_path: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings to store
            index_path: Optional path to save/load the index
        """
        self.embedding_dimension = embedding_dimension
        self.index_path = Path(index_path) if index_path else Path("data/faiss_index")
        self.metadata_path = self.index_path.with_suffix('.metadata')
        
        # Initialize FAISS index (Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dimension)
        
        # Store metadata for each vector
        self.metadata: List[VectorMetadata] = []
        
        # Document ID to vector indices mapping for efficient deletion
        self.doc_to_indices: Dict[str, List[int]] = {}
        
        # Load existing index if available
        self._load_index()    

    def _load_index(self) -> None:
        """Load existing FAISS index and metadata from disk."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data['metadata']
                    self.doc_to_indices = data['doc_to_indices']
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            else:
                logger.info("No existing index found, starting with empty index")
                
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            # Reset to empty index on load failure
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.metadata = []
            self.doc_to_indices = {}
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            # Ensure directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'doc_to_indices': self.doc_to_indices
                }, f)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise RuntimeError(f"Could not save vector index: {str(e)}")
    
    def add_documents(self, chunks: List[TextChunk], embeddings: np.ndarray, 
                     document_name: str) -> None:
        """
        Add document chunks and their embeddings to the vector store.
        
        Args:
            chunks: List of text chunks to add
            embeddings: Numpy array of embeddings (shape: [len(chunks), embedding_dim])
            document_name: Name of the source document
            
        Raises:
            ValueError: If chunks and embeddings don't match in length
            RuntimeError: If adding to index fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")
        
        if len(chunks) == 0:
            logger.warning("No chunks provided to add")
            return
        
        try:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Get starting index for new vectors
            start_idx = self.index.ntotal
            
            # Add embeddings to FAISS index
            self.index.add(normalized_embeddings.astype(np.float32))
            
            # Add metadata for each chunk
            document_id = chunks[0].document_id  # All chunks should have same document_id
            new_indices = []
            
            for i, chunk in enumerate(chunks):
                vector_idx = start_idx + i
                metadata = VectorMetadata(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_name=document_name,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number
                )
                self.metadata.append(metadata)
                new_indices.append(vector_idx)
            
            # Update document to indices mapping
            if document_id not in self.doc_to_indices:
                self.doc_to_indices[document_id] = []
            self.doc_to_indices[document_id].extend(new_indices)
            
            # Save updated index
            self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks from document '{document_name}' to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise RuntimeError(f"Could not add documents to vector store: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """
        Search for similar vectors in the store.
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects with chunks and similarity scores
            
        Raises:
            ValueError: If query embedding has wrong dimension
            RuntimeError: If search fails
        """
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(f"Query embedding dimension {len(query_embedding)} doesn't match index dimension {self.embedding_dimension}")
        
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty, returning no results")
            return []
        
        try:
            # Normalize query embedding for cosine similarity
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            query_vector = normalized_query.reshape(1, -1).astype(np.float32)
            
            # Search FAISS index
            scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Convert results to SearchResult objects
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                metadata = self.metadata[idx]
                
                # Reconstruct TextChunk from metadata
                # Note: We don't store the actual content in metadata to save space
                # In a real implementation, you'd retrieve this from the database
                chunk = TextChunk(
                    id=metadata.chunk_id,
                    document_id=metadata.document_id,
                    content="",  # Content would be retrieved from database
                    chunk_index=metadata.chunk_index,
                    page_number=metadata.page_number
                )
                
                result = SearchResult(
                    chunk=chunk,
                    similarity_score=float(score),
                    document_name=metadata.document_name
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {str(e)}")
            raise RuntimeError(f"Vector search failed: {str(e)}")
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all vectors associated with a document.
        
        Note: FAISS doesn't support efficient deletion, so we rebuild the index
        without the deleted document's vectors.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if document was found and deleted, False otherwise
            
        Raises:
            RuntimeError: If deletion fails
        """
        if document_id not in self.doc_to_indices:
            logger.warning(f"Document {document_id} not found in vector store")
            return False
        
        try:
            # Get indices to delete
            indices_to_delete = set(self.doc_to_indices[document_id])
            
            if not indices_to_delete:
                return False
            
            # Create new index without deleted vectors
            new_index = faiss.IndexFlatIP(self.embedding_dimension)
            new_metadata = []
            new_doc_to_indices = {}
            
            # Copy vectors that are not being deleted
            vectors_to_keep = []
            new_idx = 0
            
            for old_idx in range(self.index.ntotal):
                if old_idx not in indices_to_delete:
                    # Get vector from old index
                    vector = self.index.reconstruct(old_idx)
                    vectors_to_keep.append(vector)
                    
                    # Update metadata
                    old_metadata = self.metadata[old_idx]
                    new_metadata.append(old_metadata)
                    
                    # Update document mapping
                    doc_id = old_metadata.document_id
                    if doc_id not in new_doc_to_indices:
                        new_doc_to_indices[doc_id] = []
                    new_doc_to_indices[doc_id].append(new_idx)
                    
                    new_idx += 1
            
            # Add kept vectors to new index
            if vectors_to_keep:
                vectors_array = np.array(vectors_to_keep).astype(np.float32)
                new_index.add(vectors_array)
            
            # Replace old index and metadata
            self.index = new_index
            self.metadata = new_metadata
            self.doc_to_indices = new_doc_to_indices
            
            # Save updated index
            self._save_index()
            
            logger.info(f"Deleted document {document_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document from vector store: {str(e)}")
            raise RuntimeError(f"Could not delete document from vector store: {str(e)}")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dimension,
            "total_documents": len(self.doc_to_indices),
            "index_path": str(self.index_path),
            "documents": {doc_id: len(indices) for doc_id, indices in self.doc_to_indices.items()}
        }