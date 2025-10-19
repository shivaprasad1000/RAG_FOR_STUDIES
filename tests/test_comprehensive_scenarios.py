"""
Comprehensive Test Scenarios for Study Assistant RAG System

This module tests various real-world scenarios including:
- Different PDF types and sizes
- Concurrent usage scenarios
- Edge cases and error conditions
- Performance under load
- Data integrity and consistency
- Recovery from failures
"""

import pytest
import tempfile
import time
import os
import shutil
import threading
import concurrent.futures
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from datetime import datetime

from app.main import app
from app.database import DatabaseManager
from app.pdf_processor import PDFProcessor
from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore
from app.answer_generator import AnswerGenerator
from app.models import Document, TextChunk, StudyQuestion, StudyResponse, ResponseStyle
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

class TestComprehensiveScenarios:
    """Comprehensive test scenarios for real-world usage patterns."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment with temporary database and clean state."""
        # Create temporary database
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()
        
        # Create temporary directories
        self.temp_uploads_dir = tempfile.mkdtemp(prefix="test_uploads_")
        self.temp_data_dir = tempfile.mkdtemp(prefix="test_data_")
        
        # Patch the global components to use test instances
        self.test_db_manager = DatabaseManager(self.temp_db_path)
        self.test_vector_store = VectorStore(embedding_dimension=384, index_path=os.path.join(self.temp_data_dir, "test_index"))
        
        # Create test directories if they don't exist
        os.makedirs(self.temp_uploads_dir, exist_ok=True)
        os.makedirs(self.temp_data_dir, exist_ok=True)
        
    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary files and directories
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
        
        if os.path.exists(self.temp_uploads_dir):
            shutil.rmtree(self.temp_uploads_dir, ignore_errors=True)
        
        if os.path.exists(self.temp_data_dir):
            shutil.rmtree(self.temp_data_dir, ignore_errors=True)
    
    def create_test_pdf_content(self, size_kb=1):
        """Create a test PDF with specified size in KB."""
        base_content = b"""%PDF-1.4
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
>>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000074 00000 n 
0000000120 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
179
%%EOF"""
        
        # Pad to desired size
        padding_needed = max(0, (size_kb * 1024) - len(base_content))
        padding = b"%" + b"x" * padding_needed + b"\n"
        
        return base_content + padding
    
    def test_various_pdf_sizes(self):
        """Test handling of PDFs with different sizes."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            # Test different PDF sizes
            test_sizes = [
                (1, "small.pdf", "Small document content."),
                (100, "medium.pdf", "Medium sized document with more content. " * 50),
                (1000, "large.pdf", "Large document with extensive content. " * 500)
            ]
            
            uploaded_docs = []
            
            for size_kb, filename, content in test_sizes:
                # Mock PDF processing for each size
                mock_chunks = [
                    TextChunk(
                        id=f"{filename}-chunk-1",
                        document_id=f"{filename}-doc-id",
                        content=content[:800],
                        chunk_index=0,
                        page_number=1
                    )
                ]
                
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.chunks = mock_chunks
                mock_result.page_count = max(1, size_kb // 100)  # Simulate more pages for larger files
                mock_pdf_processor.process_pdf.return_value = mock_result
                
                # Upload PDF
                pdf_content = self.create_test_pdf_content(size_kb)
                files = {"file": (filename, pdf_content, "application/pdf")}
                
                start_time = time.time()
                response = client.post("/upload-pdf", files=files)
                upload_time = time.time() - start_time
                
                assert response.status_code == 200
                data = response.json()
                uploaded_docs.append(data["document_id"])
                
                # Verify upload details
                assert data["filename"] == filename
                assert data["page_count"] == max(1, size_kb // 100)
                
                # Upload time should be reasonable (allowing for processing)
                assert upload_time < 30.0, f"Upload of {size_kb}KB file took {upload_time:.2f}s"
            
            # Verify all documents are listed
            list_response = client.get("/documents")
            assert list_response.status_code == 200
            documents = list_response.json()["documents"]
            assert len(documents) == len(test_sizes)
            
            # Clean up
            for doc_id in uploaded_docs:
                client.delete(f"/documents/{doc_id}")
    
    def test_concurrent_uploads_and_queries(self):
        """Test system behavior under concurrent load."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            # Mock successful PDF processing
            def mock_process_pdf(file_path, doc_id):
                content = f"Concurrent test document {doc_id}. This contains test content for concurrent processing."
                chunks = [
                    TextChunk(
                        id=f"{doc_id}-chunk-1",
                        document_id=doc_id,
                        content=content,
                        chunk_index=0,
                        page_number=1
                    )
                ]
                
                result = MagicMock()
                result.success = True
                result.chunks = chunks
                result.page_count = 1
                return result
            
            mock_pdf_processor.process_pdf.side_effect = mock_process_pdf
            
            # Upload multiple documents concurrently
            def upload_document(doc_num):
                filename = f"concurrent_doc_{doc_num}.pdf"
                pdf_content = self.create_test_pdf_content(10)  # 10KB files
                files = {"file": (filename, pdf_content, "application/pdf")}
                
                response = client.post("/upload-pdf", files=files)
                return response.status_code == 200, response.json() if response.status_code == 200 else None
            
            # Test concurrent uploads
            num_concurrent_uploads = 5
            upload_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_uploads) as executor:
                futures = [executor.submit(upload_document, i) for i in range(num_concurrent_uploads)]
                upload_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # All uploads should succeed
            successful_uploads = [result for success, result in upload_results if success]
            assert len(successful_uploads) == num_concurrent_uploads
            
            # Test concurrent queries
            def query_documents(query_num):
                question_data = {
                    "question": f"What is document {query_num} about?",
                    "max_chunks": 3,
                    "response_style": "brief"
                }
                
                response = client.post("/ask", json=question_data)
                return response.status_code == 200, response.json() if response.status_code == 200 else None
            
            num_concurrent_queries = 10
            query_results = []
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_queries) as executor:
                futures = [executor.submit(query_documents, i) for i in range(num_concurrent_queries)]
                query_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            total_query_time = time.time() - start_time
            
            # All queries should succeed
            successful_queries = [result for success, result in query_results if success]
            assert len(successful_queries) == num_concurrent_queries
            
            # Average query time should be reasonable
            avg_query_time = total_query_time / num_concurrent_queries
            assert avg_query_time < 5.0, f"Average query time {avg_query_time:.2f}s too slow under load"
            
            # Clean up
            for success, result in upload_results:
                if success and result:
                    client.delete(f"/documents/{result['document_id']}")
    
    def test_edge_cases_and_error_recovery(self):
        """Test various edge cases and error recovery scenarios."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store):
            
            # Test 1: Empty PDF content
            empty_pdf = b""
            files = {"file": ("empty.pdf", empty_pdf, "application/pdf")}
            response = client.post("/upload-pdf", files=files)
            assert response.status_code == 400
            
            # Test 2: Corrupted PDF
            corrupted_pdf = b"This is not a valid PDF content"
            files = {"file": ("corrupted.pdf", corrupted_pdf, "application/pdf")}
            response = client.post("/upload-pdf", files=files)
            assert response.status_code == 400
            
            # Test 3: Very long question
            long_question = "What is " + "very " * 1000 + "long question?"
            question_data = {
                "question": long_question,
                "max_chunks": 3,
                "response_style": "brief"
            }
            response = client.post("/ask", json=question_data)
            # Should handle gracefully (either succeed or return appropriate error)
            assert response.status_code in [200, 400, 413]  # OK, Bad Request, or Payload Too Large
            
            # Test 4: Invalid response style
            question_data = {
                "question": "What is machine learning?",
                "max_chunks": 3,
                "response_style": "invalid_style"
            }
            response = client.post("/ask", json=question_data)
            assert response.status_code == 400
            
            # Test 5: Invalid max_chunks values
            for invalid_chunks in [-1, 0, 100]:
                question_data = {
                    "question": "What is machine learning?",
                    "max_chunks": invalid_chunks,
                    "response_style": "brief"
                }
                response = client.post("/ask", json=question_data)
                assert response.status_code == 400
            
            # Test 6: Special characters in questions
            special_questions = [
                "What is 机器学习?",  # Chinese characters
                "¿Qué es el aprendizaje automático?",  # Spanish with accents
                "What about symbols: @#$%^&*()?",  # Special symbols
                "Question with\nnewlines\tand\ttabs",  # Whitespace characters
            ]
            
            for question in special_questions:
                question_data = {
                    "question": question,
                    "max_chunks": 3,
                    "response_style": "brief"
                }
                response = client.post("/ask", json=question_data)
                # Should handle gracefully
                assert response.status_code in [200, 400, 404]  # OK, Bad Request, or Not Found (no docs)
    
    def test_data_consistency_and_integrity(self):
        """Test data consistency across operations."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            # Mock successful PDF processing
            test_content = "Data consistency test document with specific content for verification."
            mock_chunks = [
                TextChunk(
                    id="consistency-chunk-1",
                    document_id="consistency-doc-id",
                    content=test_content,
                    chunk_index=0,
                    page_number=1
                )
            ]
            
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.chunks = mock_chunks
            mock_result.page_count = 1
            mock_pdf_processor.process_pdf.return_value = mock_result
            
            # Upload document
            pdf_content = self.create_test_pdf_content(50)
            files = {"file": ("consistency_test.pdf", pdf_content, "application/pdf")}
            upload_response = client.post("/upload-pdf", files=files)
            assert upload_response.status_code == 200
            
            doc_id = upload_response.json()["document_id"]
            original_upload_data = upload_response.json()
            
            # Verify document appears in listing
            list_response = client.get("/documents")
            assert list_response.status_code == 200
            documents = list_response.json()["documents"]
            
            found_doc = None
            for doc in documents:
                if doc["id"] == doc_id:
                    found_doc = doc
                    break
            
            assert found_doc is not None
            assert found_doc["filename"] == original_upload_data["filename"]
            assert found_doc["chunk_count"] == original_upload_data["chunk_count"]
            
            # Verify individual document retrieval
            doc_response = client.get(f"/documents/{doc_id}")
            assert doc_response.status_code == 200
            individual_doc = doc_response.json()
            
            assert individual_doc["id"] == doc_id
            assert individual_doc["filename"] == original_upload_data["filename"]
            
            # Test querying returns consistent results
            question_data = {
                "question": "What is this document about?",
                "max_chunks": 3,
                "response_style": "detailed"
            }
            
            # Query multiple times to ensure consistency
            query_results = []
            for _ in range(3):
                response = client.post("/ask", json=question_data)
                assert response.status_code == 200
                query_results.append(response.json())
            
            # Results should be consistent (same sources, similar processing times)
            first_result = query_results[0]
            for result in query_results[1:]:
                assert result["sources"] == first_result["sources"]
                assert len(result["answer"]) > 0
                # Processing times may vary slightly but should be in similar range
                assert abs(result["processing_time"] - first_result["processing_time"]) < 2.0
            
            # Delete document and verify cleanup
            delete_response = client.delete(f"/documents/{doc_id}")
            assert delete_response.status_code == 200
            
            # Verify document no longer appears in listing
            list_response_after = client.get("/documents")
            assert list_response_after.status_code == 200
            documents_after = list_response_after.json()["documents"]
            
            for doc in documents_after:
                assert doc["id"] != doc_id
            
            # Verify individual document retrieval fails
            doc_response_after = client.get(f"/documents/{doc_id}")
            assert doc_response_after.status_code == 404
    
    def test_performance_under_load(self):
        """Test system performance under various load conditions."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            # Mock successful PDF processing
            def mock_process_pdf(file_path, doc_id):
                content = f"Performance test document {doc_id}. " * 100  # Longer content
                chunks = []
                
                # Create multiple chunks for more realistic scenario
                for i in range(5):
                    chunk_content = content[i*200:(i+1)*200] if i*200 < len(content) else content[-200:]
                    chunks.append(
                        TextChunk(
                            id=f"{doc_id}-chunk-{i}",
                            document_id=doc_id,
                            content=chunk_content,
                            chunk_index=i,
                            page_number=i//2 + 1
                        )
                    )
                
                result = MagicMock()
                result.success = True
                result.chunks = chunks
                result.page_count = 3
                return result
            
            mock_pdf_processor.process_pdf.side_effect = mock_process_pdf
            
            # Upload multiple documents to create a realistic knowledge base
            num_docs = 10
            uploaded_docs = []
            
            upload_start = time.time()
            for i in range(num_docs):
                filename = f"perf_doc_{i}.pdf"
                pdf_content = self.create_test_pdf_content(100)  # 100KB each
                files = {"file": (filename, pdf_content, "application/pdf")}
                
                response = client.post("/upload-pdf", files=files)
                assert response.status_code == 200
                uploaded_docs.append(response.json()["document_id"])
            
            total_upload_time = time.time() - upload_start
            avg_upload_time = total_upload_time / num_docs
            
            # Average upload time should be reasonable
            assert avg_upload_time < 10.0, f"Average upload time {avg_upload_time:.2f}s too slow"
            
            # Test query performance with larger knowledge base
            test_questions = [
                "What are the main topics covered?",
                "Explain the key concepts.",
                "What is the document about?",
                "Summarize the content.",
                "What are the important points?"
            ]
            
            query_times = []
            for question in test_questions:
                question_data = {
                    "question": question,
                    "max_chunks": 5,
                    "response_style": "detailed"
                }
                
                start_time = time.time()
                response = client.post("/ask", json=question_data)
                query_time = time.time() - start_time
                
                assert response.status_code == 200
                query_times.append(query_time)
                
                # Individual query should meet performance requirements
                assert query_time < 15.0, f"Query '{question}' took {query_time:.2f}s, too slow"
            
            # Average query time should be good
            avg_query_time = sum(query_times) / len(query_times)
            assert avg_query_time < 10.0, f"Average query time {avg_query_time:.2f}s too slow"
            
            # Test document listing performance
            list_start = time.time()
            list_response = client.get("/documents")
            list_time = time.time() - list_start
            
            assert list_response.status_code == 200
            assert len(list_response.json()["documents"]) == num_docs
            assert list_time < 2.0, f"Document listing took {list_time:.2f}s, too slow"
            
            # Clean up
            for doc_id in uploaded_docs:
                client.delete(f"/documents/{doc_id}")
    
    def test_memory_and_resource_usage(self):
        """Test memory usage and resource management."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            # Mock processing for large documents
            def mock_process_large_pdf(file_path, doc_id):
                # Simulate a document with many chunks
                chunks = []
                for i in range(50):  # 50 chunks per document
                    content = f"Large document chunk {i}. " + "Content " * 100
                    chunks.append(
                        TextChunk(
                            id=f"{doc_id}-chunk-{i}",
                            document_id=doc_id,
                            content=content,
                            chunk_index=i,
                            page_number=i//5 + 1
                        )
                    )
                
                result = MagicMock()
                result.success = True
                result.chunks = chunks
                result.page_count = 10
                return result
            
            mock_pdf_processor.process_pdf.side_effect = mock_process_large_pdf
            
            # Upload several large documents
            large_docs = []
            for i in range(5):
                filename = f"large_doc_{i}.pdf"
                pdf_content = self.create_test_pdf_content(2000)  # 2MB each
                files = {"file": (filename, pdf_content, "application/pdf")}
                
                response = client.post("/upload-pdf", files=files)
                assert response.status_code == 200
                large_docs.append(response.json()["document_id"])
            
            # Test that system still responds after loading large documents
            question_data = {
                "question": "What are these documents about?",
                "max_chunks": 10,
                "response_style": "comprehensive"
            }
            
            response = client.post("/ask", json=question_data)
            assert response.status_code == 200
            
            # Test document deletion to verify cleanup
            for doc_id in large_docs:
                delete_response = client.delete(f"/documents/{doc_id}")
                assert delete_response.status_code == 200
            
            # Verify all documents are cleaned up
            list_response = client.get("/documents")
            assert list_response.status_code == 200
            assert len(list_response.json()["documents"]) == 0
    
    def test_system_recovery_scenarios(self):
        """Test system behavior in recovery scenarios."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            # Test 1: Recovery from PDF processing failure
            def mock_failing_pdf_processor(file_path, doc_id):
                result = MagicMock()
                result.success = False
                result.error = "PDF processing failed"
                return result
            
            mock_pdf_processor.process_pdf.side_effect = mock_failing_pdf_processor
            
            pdf_content = self.create_test_pdf_content(10)
            files = {"file": ("failing_doc.pdf", pdf_content, "application/pdf")}
            response = client.post("/upload-pdf", files=files)
            
            # Should handle failure gracefully
            assert response.status_code == 400
            assert "processing failed" in response.json()["detail"].lower()
            
            # Test 2: Recovery after successful processing
            def mock_successful_pdf_processor(file_path, doc_id):
                content = "Recovery test document content."
                chunks = [
                    TextChunk(
                        id=f"{doc_id}-chunk-1",
                        document_id=doc_id,
                        content=content,
                        chunk_index=0,
                        page_number=1
                    )
                ]
                
                result = MagicMock()
                result.success = True
                result.chunks = chunks
                result.page_count = 1
                return result
            
            mock_pdf_processor.process_pdf.side_effect = mock_successful_pdf_processor
            
            # Now upload should succeed
            files = {"file": ("recovery_doc.pdf", pdf_content, "application/pdf")}
            response = client.post("/upload-pdf", files=files)
            assert response.status_code == 200
            
            doc_id = response.json()["document_id"]
            
            # System should be fully functional
            question_data = {
                "question": "What is this document about?",
                "max_chunks": 3,
                "response_style": "brief"
            }
            
            ask_response = client.post("/ask", json=question_data)
            assert ask_response.status_code == 200
            
            # Clean up
            client.delete(f"/documents/{doc_id}")
    
    def test_cross_platform_compatibility(self):
        """Test features that should work across different platforms."""
        # Test file handling with different path separators and encodings
        
        # Test filenames with various characters
        test_filenames = [
            "simple_doc.pdf",
            "doc_with_spaces.pdf",
            "doc-with-dashes.pdf",
            "doc_with_numbers_123.pdf",
            "UPPERCASE_DOC.pdf",
        ]
        
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            # Mock successful processing for all files
            def mock_process_pdf(file_path, doc_id):
                content = f"Cross-platform test content for {file_path}."
                chunks = [
                    TextChunk(
                        id=f"{doc_id}-chunk-1",
                        document_id=doc_id,
                        content=content,
                        chunk_index=0,
                        page_number=1
                    )
                ]
                
                result = MagicMock()
                result.success = True
                result.chunks = chunks
                result.page_count = 1
                return result
            
            mock_pdf_processor.process_pdf.side_effect = mock_process_pdf
            
            uploaded_docs = []
            
            for filename in test_filenames:
                pdf_content = self.create_test_pdf_content(10)
                files = {"file": (filename, pdf_content, "application/pdf")}
                
                response = client.post("/upload-pdf", files=files)
                assert response.status_code == 200
                
                data = response.json()
                assert data["filename"] == filename
                uploaded_docs.append(data["document_id"])
            
            # Verify all documents are accessible
            list_response = client.get("/documents")
            assert list_response.status_code == 200
            documents = list_response.json()["documents"]
            assert len(documents) == len(test_filenames)
            
            # Test querying works with all documents
            question_data = {
                "question": "What are these documents about?",
                "max_chunks": 5,
                "response_style": "brief"
            }
            
            response = client.post("/ask", json=question_data)
            assert response.status_code == 200
            
            # Clean up
            for doc_id in uploaded_docs:
                client.delete(f"/documents/{doc_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])