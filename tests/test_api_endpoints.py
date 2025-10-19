"""
Tests for FastAPI endpoints in the Study Assistant RAG system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
import json

from app.main import app
from app.database import DatabaseManager
from app.models import Document, ResponseStyle
from datetime import datetime

# Create test client
client = TestClient(app)

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        temp_db_path = tmp_file.name
    
    # Create temporary database manager
    temp_db_manager = DatabaseManager(temp_db_path)
    
    yield temp_db_manager
    
    # Cleanup
    Path(temp_db_path).unlink(missing_ok=True)

@pytest.fixture
def sample_pdf_content():
    """Create sample PDF content for testing."""
    # This is a minimal PDF content for testing
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF content) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""
    return pdf_content

class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data

class TestPDFUploadEndpoint:
    """Test PDF upload endpoint functionality."""
    
    def test_upload_valid_pdf(self, sample_pdf_content):
        """Test uploading a valid PDF file."""
        with patch('app.main.pdf_processor') as mock_processor, \
             patch('app.main.embedding_generator') as mock_embedder, \
             patch('app.main.vector_store') as mock_vector_store, \
             patch('app.main.db_manager') as mock_db:
            
            # Mock successful processing
            mock_processor.process_pdf.return_value = MagicMock(
                success=True,
                chunks=[MagicMock(id="chunk1", content="test content")],
                page_count=1
            )
            mock_embedder.encode_batch.return_value = [[0.1, 0.2, 0.3]]
            mock_db.create_document.return_value = True
            mock_db.create_chunks.return_value = True
            
            # Create test file
            files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
            
            response = client.post("/upload-pdf", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert "document_id" in data
            assert data["filename"] == "test.pdf"
            assert "chunk_count" in data
    
    def test_upload_non_pdf_file(self):
        """Test uploading a non-PDF file returns error."""
        files = {"file": ("test.txt", io.BytesIO(b"test content"), "text/plain")}
        
        response = client.post("/upload-pdf", files=files)
        
        assert response.status_code == 400
        assert "Only PDF files are allowed" in response.json()["detail"]
    
    def test_upload_oversized_file(self):
        """Test uploading a file larger than 50MB returns error."""
        # Create a large content (simulate 51MB)
        large_content = b"x" * (51 * 1024 * 1024)
        files = {"file": ("large.pdf", io.BytesIO(large_content), "application/pdf")}
        
        response = client.post("/upload-pdf", files=files)
        
        assert response.status_code == 400
        assert "File size exceeds 50MB limit" in response.json()["detail"]
    
    def test_upload_pdf_processing_failure(self, sample_pdf_content):
        """Test handling of PDF processing failures."""
        with patch('app.main.pdf_processor') as mock_processor:
            # Mock processing failure
            mock_processor.process_pdf.return_value = MagicMock(
                success=False,
                error_message="Failed to extract text"
            )
            
            files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
            
            response = client.post("/upload-pdf", files=files)
            
            assert response.status_code == 400
            assert "PDF processing failed" in response.json()["detail"]

class TestDocumentListEndpoint:
    """Test document listing endpoint."""
    
    def test_list_empty_documents(self):
        """Test listing documents when none exist."""
        with patch('app.main.db_manager') as mock_db:
            mock_db.list_documents.return_value = []
            
            response = client.get("/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert data["documents"] == []
            assert data["total_count"] == 0
    
    def test_list_documents_with_data(self):
        """Test listing documents with existing data."""
        with patch('app.main.db_manager') as mock_db:
            # Mock document data
            mock_document = Document(
                id="doc1",
                filename="test.pdf",
                upload_date=datetime(2023, 1, 1, 12, 0, 0),
                file_size=1024,
                page_count=5,
                chunk_count=10,
                tags=["study", "math"]
            )
            mock_db.list_documents.return_value = [mock_document]
            
            response = client.get("/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["documents"]) == 1
            assert data["total_count"] == 1
            
            doc = data["documents"][0]
            assert doc["id"] == "doc1"
            assert doc["filename"] == "test.pdf"
            assert doc["file_size"] == 1024
            assert doc["page_count"] == 5
            assert doc["chunk_count"] == 10
            assert doc["tags"] == ["study", "math"]
    
    def test_list_documents_database_error(self):
        """Test handling of database errors when listing documents."""
        with patch('app.main.db_manager') as mock_db:
            mock_db.list_documents.side_effect = Exception("Database error")
            
            response = client.get("/documents")
            
            assert response.status_code == 500
            assert "Failed to retrieve documents" in response.json()["detail"]

class TestDocumentDetailEndpoint:
    """Test individual document detail endpoint."""
    
    def test_get_existing_document(self):
        """Test getting details for an existing document."""
        with patch('app.main.db_manager') as mock_db:
            mock_document = Document(
                id="doc1",
                filename="test.pdf",
                upload_date=datetime(2023, 1, 1, 12, 0, 0),
                file_size=1024,
                page_count=5,
                chunk_count=10
            )
            mock_db.get_document.return_value = mock_document
            
            response = client.get("/documents/doc1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "doc1"
            assert data["filename"] == "test.pdf"
            assert data["file_size"] == 1024
    
    def test_get_nonexistent_document(self):
        """Test getting details for a non-existent document."""
        with patch('app.main.db_manager') as mock_db:
            mock_db.get_document.return_value = None
            
            response = client.get("/documents/nonexistent")
            
            assert response.status_code == 404
            assert "Document nonexistent not found" in response.json()["detail"]

class TestDocumentDeleteEndpoint:
    """Test document deletion endpoint."""
    
    def test_delete_existing_document(self):
        """Test deleting an existing document."""
        with patch('app.main.db_manager') as mock_db, \
             patch('app.main.vector_store') as mock_vector_store:
            
            mock_document = Document(
                id="doc1",
                filename="test.pdf",
                upload_date=datetime.now(),
                file_size=1024,
                page_count=5,
                chunk_count=10
            )
            mock_db.get_document.return_value = mock_document
            mock_db.delete_document.return_value = True
            mock_vector_store.delete_document.return_value = True
            
            response = client.delete("/documents/doc1")
            
            assert response.status_code == 200
            data = response.json()
            assert "deleted successfully" in data["message"]
            assert data["document_id"] == "doc1"
            assert data["filename"] == "test.pdf"
    
    def test_delete_nonexistent_document(self):
        """Test deleting a non-existent document."""
        with patch('app.main.db_manager') as mock_db:
            mock_db.get_document.return_value = None
            
            response = client.delete("/documents/nonexistent")
            
            assert response.status_code == 404
            assert "Document nonexistent not found" in response.json()["detail"]
    
    def test_delete_document_database_failure(self):
        """Test handling of database failure during deletion."""
        with patch('app.main.db_manager') as mock_db, \
             patch('app.main.vector_store') as mock_vector_store:
            
            mock_document = Document(
                id="doc1",
                filename="test.pdf",
                upload_date=datetime.now(),
                file_size=1024,
                page_count=5,
                chunk_count=10
            )
            mock_db.get_document.return_value = mock_document
            mock_db.delete_document.return_value = False  # Simulate failure
            mock_vector_store.delete_document.return_value = True
            
            response = client.delete("/documents/doc1")
            
            assert response.status_code == 500
            assert "Failed to delete document" in response.json()["detail"]

class TestAskEndpoint:
    """Test the ask question endpoint functionality."""
    
    def test_ask_question_no_documents(self):
        """Test asking a question when no documents are uploaded."""
        with patch('app.main.vector_store') as mock_vector_store:
            mock_vector_store.index.ntotal = 0  # No documents
            
            question_data = {
                "question": "What is machine learning?",
                "max_chunks": 3,
                "response_style": "detailed"
            }
            
            response = client.post("/ask", json=question_data)
            
            assert response.status_code == 404
            assert "No documents have been uploaded" in response.json()["detail"]
    
    def test_ask_question_empty_question(self):
        """Test asking an empty question."""
        question_data = {
            "question": "",
            "max_chunks": 3,
            "response_style": "detailed"
        }
        
        response = client.post("/ask", json=question_data)
        
        assert response.status_code == 400
        assert "Question cannot be empty" in response.json()["detail"]
    
    def test_ask_question_with_results(self):
        """Test asking a question with successful search results."""
        from app.models import TextChunk, SearchResult, StudyResponse
        
        with patch('app.main.vector_store') as mock_vector_store, \
             patch('app.main.embedding_generator') as mock_embedder, \
             patch('app.main.db_manager') as mock_db, \
             patch('app.main.answer_generator') as mock_answer_gen:
            
            # Mock vector store has documents
            mock_vector_store.index.ntotal = 5
            
            # Mock embedding generation
            mock_embedder.encode_text.return_value = [0.1, 0.2, 0.3]
            
            # Mock search results
            mock_chunk = TextChunk(
                id="chunk1",
                document_id="doc1",
                content="Machine learning is a subset of artificial intelligence.",
                chunk_index=0,
                page_number=1
            )
            mock_search_result = SearchResult(
                chunk=mock_chunk,
                similarity_score=0.85,
                document_name="ml_textbook.pdf"
            )
            mock_vector_store.search.return_value = [mock_search_result]
            
            # Mock database chunk retrieval
            mock_db.get_chunk.return_value = mock_chunk
            
            # Mock answer generation
            mock_study_response = StudyResponse(
                answer="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.",
                sources=["ml_textbook.pdf - page 1"],
                processing_time=0.5,
                key_concepts=["Machine learning", "Artificial intelligence", "Data"]
            )
            mock_answer_gen.generate_answer.return_value = mock_study_response
            
            question_data = {
                "question": "What is machine learning?",
                "max_chunks": 3,
                "response_style": "detailed"
            }
            
            response = client.post("/ask", json=question_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "processing_time" in data
            assert "key_concepts" in data
            assert len(data["sources"]) > 0
            assert "ml_textbook.pdf - page 1" in data["sources"]
            assert len(data["key_concepts"]) > 0
    
    def test_ask_question_no_results(self):
        """Test asking a question with no search results."""
        with patch('app.main.vector_store') as mock_vector_store, \
             patch('app.main.embedding_generator') as mock_embedder:
            
            # Mock vector store has documents but no results
            mock_vector_store.index.ntotal = 5
            mock_embedder.encode_text.return_value = [0.1, 0.2, 0.3]
            mock_vector_store.search.return_value = []
            
            question_data = {
                "question": "What is quantum computing?",
                "max_chunks": 3,
                "response_style": "detailed"
            }
            
            response = client.post("/ask", json=question_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "couldn't find any relevant information" in data["answer"]
            assert len(data["sources"]) == 0


class TestDocumentTagging:
    """Test document tagging functionality."""
    
    def test_update_document_tags_success(self, temp_db):
        """Test successful document tag update."""
        with patch('app.main.db_manager', temp_db):
            # First create a document
            document = Document(
                id="test-doc-1",
                filename="test.pdf",
                upload_date=datetime.now(),
                file_size=1024,
                page_count=5,
                chunk_count=10,
                tags=[]
            )
            temp_db.create_document(document)
            
            # Update tags
            new_tags = ["Math", "Algebra", "Chapter 1"]
            response = client.put(
                "/documents/test-doc-1/tags",
                json=new_tags
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Tags updated successfully for 'test.pdf'"
            assert data["document_id"] == "test-doc-1"
            assert data["tags"] == new_tags
            
            # Verify tags were saved
            updated_doc = temp_db.get_document("test-doc-1")
            assert updated_doc.tags == new_tags
    
    def test_update_document_tags_not_found(self, temp_db):
        """Test updating tags for non-existent document."""
        with patch('app.main.db_manager', temp_db):
            response = client.put(
                "/documents/non-existent/tags",
                json=["Math"]
            )
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_update_document_tags_empty_list(self, temp_db):
        """Test updating document with empty tag list."""
        with patch('app.main.db_manager', temp_db):
            # Create a document with existing tags
            document = Document(
                id="test-doc-2",
                filename="test2.pdf",
                upload_date=datetime.now(),
                file_size=1024,
                page_count=5,
                chunk_count=10,
                tags=["Old Tag"]
            )
            temp_db.create_document(document)
            
            # Clear tags
            response = client.put(
                "/documents/test-doc-2/tags",
                json=[]
            )
            
            assert response.status_code == 200
            
            # Verify tags were cleared
            updated_doc = temp_db.get_document("test-doc-2")
            assert updated_doc.tags == []
    
    def test_update_document_tags_special_characters(self, temp_db):
        """Test updating tags with special characters."""
        with patch('app.main.db_manager', temp_db):
            document = Document(
                id="test-doc-3",
                filename="test3.pdf",
                upload_date=datetime.now(),
                file_size=1024,
                page_count=5,
                chunk_count=10,
                tags=[]
            )
            temp_db.create_document(document)
            
            # Tags with special characters
            special_tags = ["Math & Science", "Chapter 1-5", "Review (Final)"]
            response = client.put(
                "/documents/test-doc-3/tags",
                json=special_tags
            )
            
            assert response.status_code == 200
            
            # Verify special characters are preserved
            updated_doc = temp_db.get_document("test-doc-3")
            assert updated_doc.tags == special_tags


class TestStudyPreferences:
    """Test study preferences functionality in queries."""
    
    @patch('app.main.vector_store')
    @patch('app.main.embedding_generator')
    @patch('app.main.answer_generator')
    def test_ask_question_with_response_style(self, mock_answer_gen, mock_embedding_gen, mock_vector_store, temp_db):
        """Test asking questions with different response styles."""
        with patch('app.main.db_manager', temp_db):
            # Mock vector store to have documents
            mock_vector_store.index.ntotal = 1
            
            # Mock embedding generation
            mock_embedding_gen.encode_text.return_value = [0.1, 0.2, 0.3]
            
            # Mock search results
            from app.models import TextChunk, SearchResult
            mock_chunk = TextChunk(
                id="chunk-1",
                document_id="doc-1",
                content="Test content about math",
                chunk_index=0,
                page_number=1
            )
            mock_search_result = SearchResult(
                chunk=mock_chunk,
                similarity_score=0.9,
                document_name="test.pdf"
            )
            mock_vector_store.search.return_value = [mock_search_result]
            
            # Add chunk to database
            temp_db.create_chunks([mock_chunk])
            
            # Mock answer generation
            from app.models import StudyResponse
            mock_response = StudyResponse(
                answer="Test answer with **key concepts**",
                sources=["test.pdf - page 1"],
                processing_time=1.0,
                key_concepts=["key concepts"]
            )
            mock_answer_gen.generate_answer.return_value = mock_response
            
            # Test different response styles
            for style in ["brief", "detailed", "comprehensive"]:
                response = client.post(
                    "/ask",
                    json={
                        "question": "What is math?",
                        "response_style": style,
                        "max_chunks": 3
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "sources" in data
                assert "key_concepts" in data
                
                # Verify the response style was passed to answer generator
                mock_answer_gen.generate_answer.assert_called()
                call_args = mock_answer_gen.generate_answer.call_args
                assert call_args[1]["response_style"] == ResponseStyle(style)
    
    @patch('app.main.vector_store')
    @patch('app.main.embedding_generator')
    @patch('app.main.answer_generator')
    def test_ask_question_with_max_chunks(self, mock_answer_gen, mock_embedding_gen, mock_vector_store, temp_db):
        """Test asking questions with different max_chunks values."""
        with patch('app.main.db_manager', temp_db):
            # Mock vector store to have documents
            mock_vector_store.index.ntotal = 1
            
            # Mock embedding generation
            mock_embedding_gen.encode_text.return_value = [0.1, 0.2, 0.3]
            
            # Mock search results
            from app.models import TextChunk, SearchResult, StudyResponse
            mock_chunks = []
            mock_search_results = []
            
            for i in range(5):
                chunk = TextChunk(
                    id=f"chunk-{i}",
                    document_id="doc-1",
                    content=f"Test content {i}",
                    chunk_index=i,
                    page_number=i+1
                )
                mock_chunks.append(chunk)
                mock_search_results.append(SearchResult(
                    chunk=chunk,
                    similarity_score=0.9 - i*0.1,
                    document_name="test.pdf"
                ))
            
            # Add chunks to database
            temp_db.create_chunks(mock_chunks)
            
            # Mock answer generation
            mock_response = StudyResponse(
                answer="Test answer",
                sources=["test.pdf - page 1"],
                processing_time=1.0,
                key_concepts=[]
            )
            mock_answer_gen.generate_answer.return_value = mock_response
            
            # Test different max_chunks values
            for max_chunks in [2, 3, 4, 5]:
                mock_vector_store.search.return_value = mock_search_results[:max_chunks]
                
                response = client.post(
                    "/ask",
                    json={
                        "question": "What is math?",
                        "response_style": "detailed",
                        "max_chunks": max_chunks
                    }
                )
                
                assert response.status_code == 200
                
                # Verify the correct number of chunks was requested
                mock_vector_store.search.assert_called_with(
                    mock_embedding_gen.encode_text.return_value,
                    top_k=max_chunks
                )
    
    def test_ask_question_invalid_response_style(self):
        """Test asking question with invalid response style."""
        response = client.post(
            "/ask",
            json={
                "question": "What is math?",
                "response_style": "invalid_style",
                "max_chunks": 3
            }
        )
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_ask_question_invalid_max_chunks(self):
        """Test asking question with invalid max_chunks value."""
        # Test with max_chunks too high
        response = client.post(
            "/ask",
            json={
                "question": "What is math?",
                "response_style": "detailed",
                "max_chunks": 15  # Above the limit of 10
            }
        )
        
        assert response.status_code == 422
        
        # Test with max_chunks too low
        response = client.post(
            "/ask",
            json={
                "question": "What is math?",
                "response_style": "detailed",
                "max_chunks": 0
            }
        )
        
        assert response.status_code == 422


class TestAnswerFormatting:
    """Test enhanced answer formatting functionality."""
    
    def test_answer_formatting_with_key_concepts(self):
        """Test that answers include properly formatted key concepts."""
        from app.answer_generator import AnswerGenerator
        
        # Create answer generator without OpenAI (fallback mode)
        generator = AnswerGenerator()
        
        # Test with sample context
        context_chunks = [
            "[Source: test.pdf - page 1]\nPhotosynthesis is the process by which plants convert sunlight into energy."
        ]
        sources = ["test.pdf - page 1"]
        
        response = generator.generate_answer(
            question="What is photosynthesis?",
            context_chunks=context_chunks,
            sources=sources,
            response_style=ResponseStyle.DETAILED
        )
        
        assert response.answer is not None
        assert len(response.sources) > 0
        assert response.processing_time > 0
        
        # Check that the answer contains formatted elements
        assert "**" in response.answer  # Should have bold formatting
    
    def test_key_concept_extraction(self):
        """Test key concept extraction from formatted answers."""
        from app.answer_generator import AnswerGenerator
        
        generator = AnswerGenerator()
        
        # Test answer with various key concept patterns
        test_answer = """
        **🔑 Photosynthesis** is a crucial process. 
        **📖 Definition:** *Cellular respiration is the breakdown of glucose.*
        The "Calvin cycle" is important.
        Chlorophyll is a green pigment found in plants.
        """
        
        concepts = generator._extract_key_concepts(test_answer)
        
        assert len(concepts) > 0
        # Should extract concepts from different patterns
        expected_concepts = ["Photosynthesis", "Calvin cycle", "Chlorophyll"]
        for concept in expected_concepts:
            assert any(concept in extracted for extracted in concepts)
    
    def test_basic_formatting_fallback(self):
        """Test basic formatting when LLM is not available."""
        from app.answer_generator import AnswerGenerator
        
        generator = AnswerGenerator()
        
        test_text = "Photosynthesis is the process by which plants make food. 1. Light reactions occur first. 2. Dark reactions follow."
        
        formatted = generator._add_basic_formatting(test_text)
        
        # Should format definitions and lists
        assert "**🔑 Photosynthesis**" in formatted
        assert "• **1.**" in formatted
        assert "• **2.**" in formatted


if __name__ == "__main__":
    pytest.main([__file__])