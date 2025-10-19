"""
Embedding generation module for the Study Assistant RAG system.

This module provides functionality to convert text chunks into vector embeddings
using sentence-transformers for similarity search and retrieval.
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Handles text-to-vector conversion using sentence-transformers.
    
    Uses the 'all-MiniLM-L6-v2' model which provides good quality embeddings
    with reasonable performance for academic content.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_folder: Optional custom cache folder for model storage
        """
        self.model_name = model_name
        self.cache_folder = cache_folder or str(Path.home() / ".cache" / "sentence_transformers")
        self.model: Optional[SentenceTransformer] = None
        self._embedding_dimension: Optional[int] = None
        
    def _load_model(self) -> None:
        """Load the sentence transformer model if not already loaded."""
        if self.model is None:
            try:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_folder
                )
                # Get embedding dimension by encoding a test string
                test_embedding = self.model.encode("test")
                self._embedding_dimension = len(test_embedding)
                logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dimension}")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {str(e)}")
                raise RuntimeError(f"Could not load embedding model: {str(e)}")
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self._embedding_dimension is None:
            self._load_model()
        return self._embedding_dimension
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Convert a single text string into a vector embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            numpy array representing the text embedding
            
        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If model fails to encode text
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        
        self._load_model()
        
        try:
            # Normalize text by removing extra whitespace
            normalized_text = " ".join(text.strip().split())
            embedding = self.model.encode(normalized_text, convert_to_numpy=True)
            
            # Validate embedding dimension
            if len(embedding) != self._embedding_dimension:
                raise RuntimeError(f"Unexpected embedding dimension: {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise RuntimeError(f"Text encoding failed: {str(e)}")
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple text strings into vector embeddings efficiently.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of shape (len(texts), embedding_dimension)
            
        Raises:
            ValueError: If texts list is empty or contains invalid entries
            RuntimeError: If batch encoding fails
        """
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace only")
        
        self._load_model()
        
        try:
            # Normalize all texts
            normalized_texts = [" ".join(text.strip().split()) for text in texts]
            
            logger.info(f"Encoding batch of {len(texts)} texts")
            embeddings = self.model.encode(
                normalized_texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10  # Show progress for large batches
            )
            
            # Validate batch dimensions
            expected_shape = (len(texts), self._embedding_dimension)
            if embeddings.shape != expected_shape:
                raise RuntimeError(f"Unexpected batch shape: {embeddings.shape}, expected: {expected_shape}")
            
            logger.info(f"Successfully encoded {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode text batch: {str(e)}")
            raise RuntimeError(f"Batch encoding failed: {str(e)}")
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding has the correct dimension and properties.
        
        Args:
            embedding: Numpy array to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        try:
            # Check if it's a numpy array
            if not isinstance(embedding, np.ndarray):
                return False
            
            # Check dimension
            if len(embedding) != self.embedding_dimension:
                return False
            
            # Check for NaN or infinite values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                return False
            
            # Check if embedding is not all zeros (which would be suspicious)
            if np.allclose(embedding, 0):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        self._load_model()
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self._embedding_dimension,
            "cache_folder": self.cache_folder,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown')
        }