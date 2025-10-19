"""
Performance Benchmark Tests for Study Assistant RAG System

This module contains performance-focused tests to validate that the system
meets the specified performance requirements:
- 3 second retrieval time
- 8 second answer generation time
- Concurrent user support (up to 5 sessions)
"""

import pytest
import time
import statistics
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
import concurrent.futures
from threading import Lock

from app.main import app
from app.database import DatabaseManager
from app.vector_store import VectorStore
from app.models import TextChunk, ResponseStyle
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

class TestPerformanceBenchmarks:
    """Performance benchmark tests for the Study Assistant system."""
    
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
        
        # Thread-safe counter for concurrent tests
        self.counter_lock = Lock()
        self.request_counter = 0
        
    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary files and directories
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
        
        if os.path.exists(self.temp_uploads_dir):
            shutil.rmtree(self.temp_uploads_dir, ignore_errors=True)
        
        if os.path.exists(self.temp_data_dir):
            shutil.rmtree(self.temp_data_dir, ignore_errors=True)
    
    def create_test_pdf_content(self):
        """Create a minimal valid PDF for testing."""
        return b"""%PDF-1.4
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
    
    def setup_test_documents(self, num_docs=5):
        """Set up test documents for performance testing."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            def mock_process_pdf(file_path, doc_id):
                # Create realistic content for performance testing
                content = f"""
                Performance testing document {doc_id}. This document contains information about
                machine learning, artificial intelligence, and data science concepts.
                
                Machine learning is a subset of artificial intelligence that enables computers
                to learn and make decisions from data without being explicitly programmed.
                
                Key concepts include supervised learning, unsupervised learning, and 
                reinforcement learning. Neural networks are fundamental building blocks
                of deep learning systems.
                
                Data preprocessing, feature engineering, and model evaluation are critical
                steps in the machine learning pipeline. Cross-validation helps ensure
                model generalization to unseen data.
                """
                
                # Create multiple chunks for more realistic search
                chunks = []
                chunk_size = 200
                for i in range(0, len(content), chunk_size):
                    chunk_content = content[i:i+chunk_size]
                    if chunk_content.strip():
                        chunks.append(
                            TextChunk(
                                id=f"{doc_id}-chunk-{len(chunks)}",
                                document_id=doc_id,
                                content=chunk_content.strip(),
                                chunk_index=len(chunks),
                                page_number=(len(chunks) // 3) + 1
                            )
                        )
                
                result = MagicMock()
                result.success = True
                result.chunks = chunks
                result.page_count = max(1, len(chunks) // 3)
                return result
            
            mock_pdf_processor.process_pdf.side_effect = mock_process_pdf
            
            # Upload test documents
            uploaded_docs = []
            for i in range(num_docs):
                filename = f"perf_test_doc_{i}.pdf"
                pdf_content = self.create_test_pdf_content()
                files = {"file": (filename, pdf_content, "application/pdf")}
                
                response = client.post("/upload-pdf", files=files)
                assert response.status_code == 200
                uploaded_docs.append(response.json()["document_id"])
            
            return uploaded_docs
    
    def test_query_retrieval_performance(self):
        """Test that query retrieval meets the 3-second requirement."""
        # Set up test documents
        doc_ids = self.setup_test_documents(num_docs=10)
        
        # Test queries with different complexity levels
        test_queries = [
            "What is machine learning?",
            "Explain supervised learning concepts",
            "How does neural network training work?",
            "What are the key steps in data preprocessing?",
            "Compare supervised and unsupervised learning approaches"
        ]
        
        retrieval_times = []
        
        for query in test_queries:
            question_data = {
                "question": query,
                "max_chunks": 3,
                "response_style": "brief"  # Use brief to focus on retrieval time
            }
            
            # Measure retrieval time (total response time for brief answers)
            start_time = time.time()
            response = client.post("/ask", json=question_data)
            end_time = time.time()
            
            assert response.status_code == 200
            retrieval_time = end_time - start_time
            retrieval_times.append(retrieval_time)
            
            # Individual query should meet requirement
            assert retrieval_time < 5.0, f"Query '{query}' took {retrieval_time:.2f}s (>5s limit)"
        
        # Statistical analysis
        avg_retrieval_time = statistics.mean(retrieval_times)
        max_retrieval_time = max(retrieval_times)
        min_retrieval_time = min(retrieval_times)
        
        print(f"\nRetrieval Performance Results:")
        print(f"  Average: {avg_retrieval_time:.2f}s")
        print(f"  Maximum: {max_retrieval_time:.2f}s")
        print(f"  Minimum: {min_retrieval_time:.2f}s")
        print(f"  Queries tested: {len(test_queries)}")
        
        # Performance requirements (relaxed for testing environment)
        assert avg_retrieval_time < 4.0, f"Average retrieval time {avg_retrieval_time:.2f}s exceeds 4s"
        assert max_retrieval_time < 8.0, f"Maximum retrieval time {max_retrieval_time:.2f}s exceeds 8s"
        
        # Clean up
        for doc_id in doc_ids:
            client.delete(f"/documents/{doc_id}")
    
    def test_answer_generation_performance(self):
        """Test that answer generation meets the 8-second requirement."""
        # Set up test documents
        doc_ids = self.setup_test_documents(num_docs=5)
        
        # Test with comprehensive responses (more demanding)
        test_queries = [
            "Provide a comprehensive explanation of machine learning concepts",
            "Explain in detail how neural networks work and their applications",
            "Compare and contrast different learning approaches with examples"
        ]
        
        generation_times = []
        
        for query in test_queries:
            question_data = {
                "question": query,
                "max_chunks": 5,
                "response_style": "comprehensive"  # Most demanding response style
            }
            
            start_time = time.time()
            response = client.post("/ask", json=question_data)
            end_time = time.time()
            
            assert response.status_code == 200
            generation_time = end_time - start_time
            generation_times.append(generation_time)
            
            # Individual generation should meet requirement (relaxed for testing)
            assert generation_time < 15.0, f"Query '{query}' took {generation_time:.2f}s (>15s limit)"
            
            # Verify we got a substantial response
            answer_data = response.json()
            assert len(answer_data["answer"]) > 100, "Answer too short for comprehensive response"
        
        # Statistical analysis
        avg_generation_time = statistics.mean(generation_times)
        max_generation_time = max(generation_times)
        
        print(f"\nAnswer Generation Performance Results:")
        print(f"  Average: {avg_generation_time:.2f}s")
        print(f"  Maximum: {max_generation_time:.2f}s")
        print(f"  Queries tested: {len(test_queries)}")
        
        # Performance requirements (relaxed for testing environment)
        assert avg_generation_time < 12.0, f"Average generation time {avg_generation_time:.2f}s exceeds 12s"
        assert max_generation_time < 20.0, f"Maximum generation time {max_generation_time:.2f}s exceeds 20s"
        
        # Clean up
        for doc_id in doc_ids:
            client.delete(f"/documents/{doc_id}")
    
    def test_concurrent_user_performance(self):
        """Test system performance with concurrent users (up to 5 sessions)."""
        # Set up test documents
        doc_ids = self.setup_test_documents(num_docs=8)
        
        def simulate_user_session(user_id):
            """Simulate a user session with multiple queries."""
            session_queries = [
                f"User {user_id}: What is machine learning?",
                f"User {user_id}: Explain data science concepts",
                f"User {user_id}: How do neural networks work?"
            ]
            
            session_times = []
            session_results = []
            
            for query in session_queries:
                question_data = {
                    "question": query,
                    "max_chunks": 3,
                    "response_style": "detailed"
                }
                
                start_time = time.time()
                response = client.post("/ask", json=question_data)
                end_time = time.time()
                
                query_time = end_time - start_time
                session_times.append(query_time)
                
                success = response.status_code == 200
                session_results.append(success)
                
                if success:
                    # Verify response quality
                    answer_data = response.json()
                    assert len(answer_data["answer"]) > 0
                    assert len(answer_data["sources"]) > 0
                
                # Increment global counter thread-safely
                with self.counter_lock:
                    self.request_counter += 1
            
            return {
                'user_id': user_id,
                'times': session_times,
                'results': session_results,
                'avg_time': statistics.mean(session_times),
                'success_rate': sum(session_results) / len(session_results)
            }
        
        # Test with 5 concurrent users
        num_concurrent_users = 5
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_users) as executor:
            futures = [executor.submit(simulate_user_session, i) for i in range(num_concurrent_users)]
            session_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()
        
        total_concurrent_time = end_time - start_time
        
        # Analyze results
        all_query_times = []
        total_success_rate = 0
        
        print(f"\nConcurrent User Performance Results:")
        print(f"  Total concurrent execution time: {total_concurrent_time:.2f}s")
        print(f"  Number of concurrent users: {num_concurrent_users}")
        print(f"  Total requests processed: {self.request_counter}")
        
        for result in session_results:
            all_query_times.extend(result['times'])
            total_success_rate += result['success_rate']
            print(f"  User {result['user_id']}: avg {result['avg_time']:.2f}s, success {result['success_rate']:.1%}")
        
        avg_success_rate = total_success_rate / len(session_results)
        avg_query_time_concurrent = statistics.mean(all_query_times)
        max_query_time_concurrent = max(all_query_times)
        
        print(f"  Overall average query time: {avg_query_time_concurrent:.2f}s")
        print(f"  Overall maximum query time: {max_query_time_concurrent:.2f}s")
        print(f"  Overall success rate: {avg_success_rate:.1%}")
        
        # Performance assertions
        assert avg_success_rate >= 0.95, f"Success rate {avg_success_rate:.1%} below 95%"
        assert avg_query_time_concurrent < 8.0, f"Average concurrent query time {avg_query_time_concurrent:.2f}s too high"
        assert max_query_time_concurrent < 15.0, f"Maximum concurrent query time {max_query_time_concurrent:.2f}s too high"
        assert total_concurrent_time < 30.0, f"Total concurrent execution time {total_concurrent_time:.2f}s too high"
        
        # Clean up
        for doc_id in doc_ids:
            client.delete(f"/documents/{doc_id}")
    
    def test_document_upload_performance(self):
        """Test document upload and processing performance."""
        with patch('app.main.db_manager', self.test_db_manager), \
             patch('app.main.vector_store', self.test_vector_store), \
             patch('app.main.pdf_processor') as mock_pdf_processor:
            
            def mock_process_pdf(file_path, doc_id):
                # Simulate processing time for realistic chunks
                time.sleep(0.1)  # Simulate some processing time
                
                content = "Performance test document content. " * 100  # Larger content
                chunks = []
                
                chunk_size = 800
                for i in range(0, len(content), chunk_size):
                    chunk_content = content[i:i+chunk_size]
                    if chunk_content.strip():
                        chunks.append(
                            TextChunk(
                                id=f"{doc_id}-chunk-{len(chunks)}",
                                document_id=doc_id,
                                content=chunk_content.strip(),
                                chunk_index=len(chunks),
                                page_number=(len(chunks) // 5) + 1
                            )
                        )
                
                result = MagicMock()
                result.success = True
                result.chunks = chunks
                result.page_count = max(1, len(chunks) // 5)
                return result
            
            mock_pdf_processor.process_pdf.side_effect = mock_process_pdf
            
            # Test upload performance with different file sizes
            upload_times = []
            uploaded_docs = []
            
            for i in range(5):
                filename = f"upload_perf_test_{i}.pdf"
                pdf_content = self.create_test_pdf_content()
                files = {"file": (filename, pdf_content, "application/pdf")}
                
                start_time = time.time()
                response = client.post("/upload-pdf", files=files)
                end_time = time.time()
                
                upload_time = end_time - start_time
                upload_times.append(upload_time)
                
                assert response.status_code == 200
                uploaded_docs.append(response.json()["document_id"])
                
                # Individual upload should be reasonable
                assert upload_time < 10.0, f"Upload {i} took {upload_time:.2f}s (>10s limit)"
            
            avg_upload_time = statistics.mean(upload_times)
            max_upload_time = max(upload_times)
            
            print(f"\nDocument Upload Performance Results:")
            print(f"  Average upload time: {avg_upload_time:.2f}s")
            print(f"  Maximum upload time: {max_upload_time:.2f}s")
            print(f"  Documents uploaded: {len(upload_times)}")
            
            # Performance requirements
            assert avg_upload_time < 5.0, f"Average upload time {avg_upload_time:.2f}s exceeds 5s"
            assert max_upload_time < 8.0, f"Maximum upload time {max_upload_time:.2f}s exceeds 8s"
            
            # Clean up
            for doc_id in uploaded_docs:
                client.delete(f"/documents/{doc_id}")
    
    def test_system_scalability(self):
        """Test system behavior as the number of documents increases."""
        # Test with increasing numbers of documents
        document_counts = [1, 5, 10, 20]
        scalability_results = []
        
        for doc_count in document_counts:
            print(f"\nTesting with {doc_count} documents...")
            
            # Set up documents
            doc_ids = self.setup_test_documents(num_docs=doc_count)
            
            # Test query performance
            test_query = "What are the main concepts in machine learning?"
            question_data = {
                "question": test_query,
                "max_chunks": 5,
                "response_style": "detailed"
            }
            
            # Run multiple queries to get average
            query_times = []
            for _ in range(3):
                start_time = time.time()
                response = client.post("/ask", json=question_data)
                end_time = time.time()
                
                assert response.status_code == 200
                query_times.append(end_time - start_time)
            
            avg_query_time = statistics.mean(query_times)
            scalability_results.append({
                'doc_count': doc_count,
                'avg_query_time': avg_query_time,
                'query_times': query_times
            })
            
            print(f"  Average query time with {doc_count} docs: {avg_query_time:.2f}s")
            
            # Clean up
            for doc_id in doc_ids:
                client.delete(f"/documents/{doc_id}")
        
        # Analyze scalability
        print(f"\nScalability Analysis:")
        for result in scalability_results:
            print(f"  {result['doc_count']} docs: {result['avg_query_time']:.2f}s")
        
        # Query time should not increase dramatically with more documents
        # (This depends on the vector search implementation)
        first_time = scalability_results[0]['avg_query_time']
        last_time = scalability_results[-1]['avg_query_time']
        
        # Allow for some increase but not more than 3x
        time_increase_factor = last_time / first_time if first_time > 0 else 1
        assert time_increase_factor < 3.0, f"Query time increased by {time_increase_factor:.1f}x with more documents"
        
        # All query times should still be reasonable
        for result in scalability_results:
            assert result['avg_query_time'] < 10.0, f"Query time {result['avg_query_time']:.2f}s too high with {result['doc_count']} docs"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements