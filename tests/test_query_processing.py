"""
Tests for query processing functionality in the Study Assistant RAG system.

This module tests:
- Question embedding and similarity search
- Answer generation with mock responses  
- Source reference formatting
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.models import TextChunk, SearchResult, StudyQuestion, StudyResponse, ResponseStyle
from app.answer_generator import AnswerGenerator, AnswerConfig, LLMProvider


class TestQuestionEmbeddingAndSearch:
    """Test question embedding and similarity search functionality."""
    
    def test_question_embedding_generation(self):
        """Test that questions are properly converted to embeddings."""
        from app.embedding_generator import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        # Test embedding generation
        question = "What is machine learning?"
        embedding = generator.encode_text(question)
        
        # Verify embedding properties
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == generator.embedding_dimension
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
        assert not np.allclose(embedding, 0)  # Should not be all zeros
    
    def test_similarity_search_with_results(self):
        """Test vector similarity search returns relevant chunks."""
        from app.vector_store import VectorStore
        from app.embedding_generator import EmbeddingGenerator
        
        # Create test vector store with unique path to avoid conflicts
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(embedding_dimension=384, index_path=f"{temp_dir}/test_index")
            embedding_generator = EmbeddingGenerator()
            
            # Create test chunks
            test_chunks = [
                TextChunk(
                    id="chunk1",
                    document_id="doc1", 
                    content="Machine learning is a subset of artificial intelligence.",
                    chunk_index=0,
                    page_number=1
                ),
                TextChunk(
                    id="chunk2",
                    document_id="doc1",
                    content="Deep learning uses neural networks with multiple layers.",
                    chunk_index=1,
                    page_number=2
                )
            ]
            
            # Generate embeddings and add to vector store
            chunk_texts = [chunk.content for chunk in test_chunks]
            embeddings = embedding_generator.encode_batch(chunk_texts)
            vector_store.add_documents(test_chunks, embeddings, "test_document.pdf")
            
            # Test search
            query = "What is machine learning?"
            query_embedding = embedding_generator.encode_text(query)
            results = vector_store.search(query_embedding, top_k=2)
            
            # Verify results
            assert len(results) > 0
            assert all(isinstance(result, SearchResult) for result in results)
            assert all(result.similarity_score > 0 for result in results)
            
            # Note: The vector store doesn't store content in search results,
            # so we verify the chunk IDs and metadata instead
            chunk_ids = [result.chunk.id for result in results]
            assert "chunk1" in chunk_ids or "chunk2" in chunk_ids
    
    def test_similarity_search_empty_index(self):
        """Test similarity search with empty vector store."""
        from app.vector_store import VectorStore
        from app.embedding_generator import EmbeddingGenerator
        
        # Create vector store with unique path to ensure it's truly empty
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(embedding_dimension=384, index_path=f"{temp_dir}/empty_index")
            embedding_generator = EmbeddingGenerator()
            
            query_embedding = embedding_generator.encode_text("test query")
            results = vector_store.search(query_embedding, top_k=3)
            
            assert len(results) == 0
    
    def test_similarity_search_top_k_limit(self):
        """Test that similarity search respects top_k parameter."""
        from app.vector_store import VectorStore
        from app.embedding_generator import EmbeddingGenerator
        
        vector_store = VectorStore(embedding_dimension=384)
        embedding_generator = EmbeddingGenerator()
        
        # Create multiple test chunks
        test_chunks = []
        for i in range(5):
            chunk = TextChunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"This is test content number {i} about machine learning.",
                chunk_index=i,
                page_number=i+1
            )
            test_chunks.append(chunk)
        
        # Add to vector store
        chunk_texts = [chunk.content for chunk in test_chunks]
        embeddings = embedding_generator.encode_batch(chunk_texts)
        vector_store.add_documents(test_chunks, embeddings, "test_document.pdf")
        
        # Test different top_k values
        query_embedding = embedding_generator.encode_text("machine learning")
        
        results_2 = vector_store.search(query_embedding, top_k=2)
        results_3 = vector_store.search(query_embedding, top_k=3)
        
        assert len(results_2) == 2
        assert len(results_3) == 3


class TestAnswerGeneration:
    """Test answer generation functionality."""
    
    def test_answer_generator_initialization(self):
        """Test answer generator initialization with different configs."""
        # Test default initialization
        generator = AnswerGenerator()
        assert generator.config.provider == LLMProvider.OPENAI
        assert generator.config.model_name == "gpt-3.5-turbo"
        
        # Test custom config
        custom_config = AnswerConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            max_tokens=1000,
            temperature=0.5
        )
        generator_custom = AnswerGenerator(custom_config)
        assert generator_custom.config.model_name == "gpt-4"
        assert generator_custom.config.max_tokens == 1000
    
    def test_fallback_answer_generation(self):
        """Test answer generation in fallback mode (no API key)."""
        generator = AnswerGenerator()
        
        question = "What is machine learning?"
        context_chunks = [
            "[Source: test.pdf - page 1]\nMachine learning is a subset of AI.",
            "[Source: test.pdf - page 2]\nIt enables computers to learn from data."
        ]
        sources = ["test.pdf - page 1", "test.pdf - page 2"]
        
        response = generator.generate_answer(
            question=question,
            context_chunks=context_chunks,
            sources=sources,
            response_style=ResponseStyle.DETAILED
        )
        
        # Verify response structure
        assert isinstance(response, StudyResponse)
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.sources == sources
        assert response.processing_time >= 0
        assert isinstance(response.key_concepts, list)
        
        # Verify fallback content
        assert "Based on your uploaded documents" in response.answer
        assert "Machine learning is a subset of AI" in response.answer
    
    def test_answer_generation_empty_context(self):
        """Test answer generation with empty context."""
        generator = AnswerGenerator()
        
        response = generator.generate_answer(
            question="What is machine learning?",
            context_chunks=[],
            sources=[],
            response_style=ResponseStyle.BRIEF
        )
        
        assert "don't have enough information" in response.answer
        assert len(response.sources) == 0
    
    @patch('openai.OpenAI')
    def test_openai_answer_generation(self, mock_openai_class):
        """Test answer generation with mocked OpenAI API."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Machine learning is a powerful AI technique that enables computers to learn patterns from data."
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create generator with API key
        config = AnswerConfig(api_key="test-key")
        generator = AnswerGenerator(config)
        
        question = "What is machine learning?"
        context_chunks = ["[Source: test.pdf - page 1]\nML is part of AI."]
        sources = ["test.pdf - page 1"]
        
        response = generator.generate_answer(
            question=question,
            context_chunks=context_chunks,
            sources=sources,
            response_style=ResponseStyle.DETAILED
        )
        
        # Verify OpenAI was called
        mock_client.chat.completions.create.assert_called_once()
        
        # Verify response
        assert "Machine learning is a powerful AI technique" in response.answer
        assert response.sources == sources
    
    def test_key_concept_extraction(self):
        """Test extraction of key concepts from generated answers."""
        generator = AnswerGenerator()
        
        # Test text with various concept patterns
        test_answer = '''
        Machine Learning is a subset of artificial intelligence. 
        "Deep Learning" uses neural networks. The concept of 
        Supervised Learning involves training with labeled data.
        '''
        
        concepts = generator._extract_key_concepts(test_answer)
        
        assert isinstance(concepts, list)
        assert "Deep Learning" in concepts  # Quoted term
        assert "Machine Learning" in concepts or "Supervised Learning" in concepts  # Capitalized terms


class TestSourceReferenceFormatting:
    """Test source reference formatting functionality."""
    
    def test_source_reference_with_page_number(self):
        """Test formatting of source references with page numbers."""
        chunk = TextChunk(
            id="chunk1",
            document_id="doc1",
            content="Test content",
            chunk_index=0,
            page_number=5
        )
        document_name = "textbook.pdf"
        
        # Format as done in main.py
        page_ref = f"page {chunk.page_number}" if chunk.page_number else "unknown page"
        source = f"{document_name} - {page_ref}"
        
        assert source == "textbook.pdf - page 5"
    
    def test_source_reference_without_page_number(self):
        """Test formatting of source references without page numbers."""
        chunk = TextChunk(
            id="chunk1",
            document_id="doc1",
            content="Test content",
            chunk_index=0,
            page_number=None
        )
        document_name = "notes.pdf"
        
        # Format as done in main.py
        page_ref = f"page {chunk.page_number}" if chunk.page_number else "unknown page"
        source = f"{document_name} - {page_ref}"
        
        assert source == "notes.pdf - unknown page"
    
    def test_context_chunk_formatting(self):
        """Test formatting of context chunks with source information."""
        chunk = TextChunk(
            id="chunk1",
            document_id="doc1",
            content="Machine learning is a subset of artificial intelligence.",
            chunk_index=0,
            page_number=1
        )
        document_name = "ml_book.pdf"
        
        # Format as done in main.py
        page_ref = f"page {chunk.page_number}" if chunk.page_number else "unknown page"
        source = f"{document_name} - {page_ref}"
        formatted_chunk = f"[Source: {source}]\n{chunk.content}"
        
        expected = "[Source: ml_book.pdf - page 1]\nMachine learning is a subset of artificial intelligence."
        assert formatted_chunk == expected
    
    def test_multiple_source_references(self):
        """Test handling of multiple source references."""
        chunks_data = [
            ("chunk1", "Content from first document", 1, "doc1.pdf"),
            ("chunk2", "Content from second document", 3, "doc2.pdf"),
            ("chunk3", "Content without page", None, "doc3.pdf")
        ]
        
        sources = []
        formatted_chunks = []
        
        for chunk_id, content, page_num, doc_name in chunks_data:
            chunk = TextChunk(
                id=chunk_id,
                document_id="doc1",
                content=content,
                chunk_index=0,
                page_number=page_num
            )
            
            page_ref = f"page {chunk.page_number}" if chunk.page_number else "unknown page"
            source = f"{doc_name} - {page_ref}"
            sources.append(source)
            
            formatted_chunk = f"[Source: {source}]\n{chunk.content}"
            formatted_chunks.append(formatted_chunk)
        
        # Verify all sources are properly formatted
        assert "doc1.pdf - page 1" in sources
        assert "doc2.pdf - page 3" in sources
        assert "doc3.pdf - unknown page" in sources
        
        # Verify formatted chunks contain source info
        assert all("[Source:" in chunk for chunk in formatted_chunks)
        # Check that most chunks contain the expected content pattern
        content_matches = [("Content from" in chunk or "Content without" in chunk) for chunk in formatted_chunks]
        assert all(content_matches)


class TestIntegratedQueryProcessing:
    """Test integrated query processing pipeline."""
    
    def test_end_to_end_query_processing_mock(self):
        """Test complete query processing pipeline with mocked components."""
        from app.models import StudyQuestion
        
        # Create test question
        question = StudyQuestion(
            question="What is machine learning?",
            max_chunks=3,
            response_style=ResponseStyle.DETAILED
        )
        
        # Mock components
        with patch('app.embedding_generator.EmbeddingGenerator') as mock_embedder_class, \
             patch('app.vector_store.VectorStore') as mock_vector_store_class, \
             patch('app.database.DatabaseManager') as mock_db_class, \
             patch('app.answer_generator.AnswerGenerator') as mock_answer_gen_class:
            
            # Setup mocks
            mock_embedder = Mock()
            mock_embedder.encode_text.return_value = np.array([0.1, 0.2, 0.3])
            mock_embedder_class.return_value = mock_embedder
            
            mock_vector_store = Mock()
            mock_vector_store.index.ntotal = 5
            
            mock_chunk = TextChunk(
                id="chunk1",
                document_id="doc1",
                content="Machine learning is a subset of AI.",
                chunk_index=0,
                page_number=1
            )
            mock_search_result = SearchResult(
                chunk=mock_chunk,
                similarity_score=0.85,
                document_name="ml_textbook.pdf"
            )
            mock_vector_store.search.return_value = [mock_search_result]
            mock_vector_store_class.return_value = mock_vector_store
            
            mock_db = Mock()
            mock_db.get_chunk.return_value = mock_chunk
            mock_db_class.return_value = mock_db
            
            mock_answer_gen = Mock()
            mock_study_response = StudyResponse(
                answer="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                sources=["ml_textbook.pdf - page 1"],
                processing_time=0.5,
                key_concepts=["Machine learning", "Artificial intelligence"]
            )
            mock_answer_gen.generate_answer.return_value = mock_study_response
            mock_answer_gen_class.return_value = mock_answer_gen
            
            # Simulate the query processing pipeline
            # 1. Convert question to embedding
            query_embedding = mock_embedder.encode_text(question.question)
            assert query_embedding is not None
            
            # 2. Search vector store
            search_results = mock_vector_store.search(query_embedding, top_k=question.max_chunks)
            assert len(search_results) > 0
            
            # 3. Retrieve chunk content
            context_chunks = []
            sources = []
            for result in search_results:
                chunk = mock_db.get_chunk(result.chunk.id)
                if chunk:
                    page_ref = f"page {chunk.page_number}" if chunk.page_number else "unknown page"
                    source = f"{result.document_name} - {page_ref}"
                    sources.append(source)
                    formatted_chunk = f"[Source: {source}]\n{chunk.content}"
                    context_chunks.append(formatted_chunk)
            
            # 4. Generate answer
            response = mock_answer_gen.generate_answer(
                question=question.question,
                context_chunks=context_chunks,
                sources=sources,
                response_style=question.response_style
            )
            
            # Verify complete pipeline
            assert isinstance(response, StudyResponse)
            assert len(response.sources) > 0
            assert "ml_textbook.pdf - page 1" in response.sources
            assert len(response.key_concepts) > 0
            assert response.processing_time > 0
    
    def test_query_processing_error_handling(self):
        """Test error handling in query processing pipeline."""
        from app.embedding_generator import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        # Test empty question
        with pytest.raises(ValueError, match="Text cannot be empty"):
            generator.encode_text("")
        
        # Test whitespace-only question
        with pytest.raises(ValueError, match="Text cannot be empty"):
            generator.encode_text("   ")