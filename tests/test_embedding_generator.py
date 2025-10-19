"""
Tests for embedding generation functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Use a temporary cache folder for tests
        self.temp_cache = tempfile.mkdtemp()
        self.generator = EmbeddingGenerator(cache_folder=self.temp_cache)
        
        # Test texts
        self.test_text = "This is a sample text for testing embeddings."
        self.test_texts = [
            "First test sentence for batch processing.",
            "Second test sentence with different content.",
            "Third sentence to complete the batch test."
        ]
    
    def teardown_method(self):
        """Clean up temporary cache folder"""
        try:
            shutil.rmtree(self.temp_cache)
        except FileNotFoundError:
            pass
    
    def test_init_default_model(self):
        """Test EmbeddingGenerator initialization with default model"""
        generator = EmbeddingGenerator()
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.model is None  # Model not loaded yet
    
    def test_init_custom_model(self):
        """Test EmbeddingGenerator initialization with custom model"""
        custom_model = "all-mpnet-base-v2"
        generator = EmbeddingGenerator(model_name=custom_model)
        assert generator.model_name == custom_model
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_load_model_success(self, mock_sentence_transformer):
        """Test successful model loading"""
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])  # 3-dimensional embedding
        mock_sentence_transformer.return_value = mock_model
        
        # Load model
        self.generator._load_model()
        
        assert self.generator.model is not None
        assert self.generator._embedding_dimension == 3
        mock_sentence_transformer.assert_called_once()
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_load_model_failure(self, mock_sentence_transformer):
        """Test model loading failure"""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            self.generator._load_model()
        
        assert "Could not load embedding model" in str(exc_info.value)
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_embedding_dimension_property(self, mock_sentence_transformer):
        """Test embedding dimension property"""
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])  # 4-dimensional
        mock_sentence_transformer.return_value = mock_model
        
        dimension = self.generator.embedding_dimension
        assert dimension == 4
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_encode_text_success(self, mock_sentence_transformer):
        """Test successful single text encoding"""
        # Mock the sentence transformer
        mock_model = MagicMock()
        expected_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = expected_embedding
        mock_sentence_transformer.return_value = mock_model
        
        result = self.generator.encode_text(self.test_text)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, expected_embedding)
        # Model is called twice: once for dimension detection, once for actual encoding
        assert mock_model.encode.call_count == 2
    
    def test_encode_text_empty_string(self):
        """Test encoding empty string raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            self.generator.encode_text("")
        
        assert "Text cannot be empty" in str(exc_info.value)
    
    def test_encode_text_whitespace_only(self):
        """Test encoding whitespace-only string raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            self.generator.encode_text("   \n\t   ")
        
        assert "Text cannot be empty" in str(exc_info.value)
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_encode_text_normalization(self, mock_sentence_transformer):
        """Test text normalization during encoding"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_sentence_transformer.return_value = mock_model
        
        # Text with extra whitespace
        messy_text = "  This   has    extra   spaces  \n\t  "
        self.generator.encode_text(messy_text)
        
        # Check that normalized text was passed to the model
        called_text = mock_model.encode.call_args[0][0]
        assert called_text == "This has extra spaces"
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_encode_batch_success(self, mock_sentence_transformer):
        """Test successful batch text encoding"""
        mock_model = MagicMock()
        expected_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model.encode.return_value = expected_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        result = self.generator.encode_batch(self.test_texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        assert np.array_equal(result, expected_embeddings)
    
    def test_encode_batch_empty_list(self):
        """Test batch encoding with empty list raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            self.generator.encode_batch([])
        
        assert "Text list cannot be empty" in str(exc_info.value)
    
    def test_encode_batch_with_empty_text(self):
        """Test batch encoding with empty text in list raises ValueError"""
        texts_with_empty = ["Valid text", "", "Another valid text"]
        
        with pytest.raises(ValueError) as exc_info:
            self.generator.encode_batch(texts_with_empty)
        
        assert "Text at index 1 is empty" in str(exc_info.value)
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_encode_batch_dimension_validation(self, mock_sentence_transformer):
        """Test batch encoding validates output dimensions"""
        mock_model = MagicMock()
        # Return wrong shape
        mock_model.encode.return_value = np.array([[0.1, 0.2]])  # Wrong shape
        mock_sentence_transformer.return_value = mock_model
        
        with pytest.raises(RuntimeError) as exc_info:
            self.generator.encode_batch(self.test_texts)
        
        assert "Unexpected batch shape" in str(exc_info.value)
    
    def test_validate_embedding_valid(self):
        """Test embedding validation with valid embedding"""
        # Create a valid embedding
        valid_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Mock the embedding dimension
        self.generator._embedding_dimension = 4
        
        result = self.generator.validate_embedding(valid_embedding)
        assert result is True
    
    def test_validate_embedding_wrong_type(self):
        """Test embedding validation with wrong type"""
        invalid_embedding = [0.1, 0.2, 0.3]  # List instead of numpy array
        
        result = self.generator.validate_embedding(invalid_embedding)
        assert result is False
    
    def test_validate_embedding_wrong_dimension(self):
        """Test embedding validation with wrong dimension"""
        self.generator._embedding_dimension = 4
        wrong_dim_embedding = np.array([0.1, 0.2, 0.3])  # 3D instead of 4D
        
        result = self.generator.validate_embedding(wrong_dim_embedding)
        assert result is False
    
    def test_validate_embedding_nan_values(self):
        """Test embedding validation with NaN values"""
        self.generator._embedding_dimension = 3
        nan_embedding = np.array([0.1, np.nan, 0.3])
        
        result = self.generator.validate_embedding(nan_embedding)
        assert result is False
    
    def test_validate_embedding_infinite_values(self):
        """Test embedding validation with infinite values"""
        self.generator._embedding_dimension = 3
        inf_embedding = np.array([0.1, np.inf, 0.3])
        
        result = self.generator.validate_embedding(inf_embedding)
        assert result is False
    
    def test_validate_embedding_all_zeros(self):
        """Test embedding validation with all zero values"""
        self.generator._embedding_dimension = 3
        zero_embedding = np.array([0.0, 0.0, 0.0])
        
        result = self.generator.validate_embedding(zero_embedding)
        assert result is False
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_get_model_info(self, mock_sentence_transformer):
        """Test getting model information"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.max_seq_length = 512
        mock_sentence_transformer.return_value = mock_model
        
        info = self.generator.get_model_info()
        
        assert info['model_name'] == "all-MiniLM-L6-v2"
        assert info['embedding_dimension'] == 3
        assert info['cache_folder'] == self.temp_cache
        assert info['max_sequence_length'] == 512

    @patch('app.embedding_generator.SentenceTransformer')
    def test_embedding_generation_consistency(self, mock_sentence_transformer):
        """Test that the same text produces consistent embeddings"""
        # Mock the sentence transformer to return consistent embeddings
        mock_model = MagicMock()
        consistent_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = consistent_embedding
        mock_sentence_transformer.return_value = mock_model
        
        # Generate embedding for the same text multiple times
        text = "This is a test sentence for consistency checking."
        embedding1 = self.generator.encode_text(text)
        embedding2 = self.generator.encode_text(text)
        embedding3 = self.generator.encode_text(text)
        
        # All embeddings should be identical
        assert np.array_equal(embedding1, embedding2)
        assert np.array_equal(embedding2, embedding3)
        assert np.array_equal(embedding1, consistent_embedding)
    
    @patch('app.embedding_generator.SentenceTransformer')
    def test_embedding_consistency_with_normalization(self, mock_sentence_transformer):
        """Test that text normalization produces consistent embeddings"""
        mock_model = MagicMock()
        expected_embedding = np.array([0.4, 0.5, 0.6])
        mock_model.encode.return_value = expected_embedding
        mock_sentence_transformer.return_value = mock_model
        
        # Test with different whitespace variations of the same text
        base_text = "Machine learning is fascinating"
        variations = [
            "Machine learning is fascinating",
            "  Machine   learning   is   fascinating  ",
            "\n\tMachine learning is fascinating\n",
            "Machine\nlearning\tis\nfascinating"
        ]
        
        embeddings = [self.generator.encode_text(text) for text in variations]
        
        # All normalized versions should produce the same embedding
        for embedding in embeddings:
            assert np.array_equal(embedding, expected_embedding)


if __name__ == "__main__":
    pytest.main([__file__])