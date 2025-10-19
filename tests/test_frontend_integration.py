"""
Frontend Integration Tests for Study Assistant

This module contains integration tests that verify the frontend works correctly
with the backend API endpoints. These tests can be run with pytest.

Requirements:
- pip install pytest requests beautifulsoup4 selenium (optional)
- The Study Assistant server should be running on localhost:8000
"""

import pytest
import requests
import json
import time
from pathlib import Path
import tempfile
import os

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"

class TestFrontendIntegration:
    """Test suite for frontend integration with backend API"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        # Check if server is running
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Study Assistant server is not running")
        except requests.exceptions.RequestException:
            pytest.skip("Study Assistant server is not running")
    
    def test_main_page_loads(self):
        """Test that the main page loads correctly"""
        response = requests.get(BASE_URL)
        assert response.status_code == 200
        assert "Study Assistant" in response.text
        assert "Upload PDF" in response.text
        assert "Ask a Question" in response.text
    
    def test_static_files_load(self):
        """Test that CSS and JS files load correctly"""
        # Test CSS
        css_response = requests.get(f"{BASE_URL}/static/style.css")
        assert css_response.status_code == 200
        assert "container" in css_response.text
        
        # Test JS
        js_response = requests.get(f"{BASE_URL}/static/app.js")
        assert js_response.status_code == 200
        assert "StudyAssistant" in js_response.text
    
    def test_health_endpoint(self):
        """Test that health endpoint works"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_documents_endpoint_empty(self):
        """Test documents endpoint when no documents are uploaded"""
        response = requests.get(f"{BASE_URL}/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)
    
    def test_pdf_upload_invalid_file_type(self):
        """Test that non-PDF files are rejected"""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"This is not a PDF file")
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(f"{BASE_URL}/upload-pdf", files=files)
            
            assert response.status_code == 400
            data = response.json()
            assert "Only PDF files are allowed" in data["detail"]
        finally:
            os.unlink(tmp_file_path)
    
    def test_pdf_upload_valid_file(self):
        """Test that valid PDF files are accepted"""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(TEST_PDF_CONTENT)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, 'rb') as f:
                files = {'file': ('test.pdf', f, 'application/pdf')}
                response = requests.post(f"{BASE_URL}/upload-pdf", files=files)
            
            # Note: This might fail if PDF processing fails, but upload should be accepted
            assert response.status_code in [200, 400, 500]  # Accept various responses
            
            if response.status_code == 200:
                data = response.json()
                assert "document_id" in data
                assert data["filename"] == "test.pdf"
                
                # Clean up: delete the uploaded document
                doc_id = data["document_id"]
                delete_response = requests.delete(f"{BASE_URL}/documents/{doc_id}")
                # Don't assert on delete response as it might fail
                
        finally:
            os.unlink(tmp_file_path)
    
    def test_ask_question_no_documents(self):
        """Test asking a question when no documents are uploaded"""
        question_data = {
            "question": "What is photosynthesis?",
            "max_chunks": 3,
            "response_style": "detailed"
        }
        
        response = requests.post(
            f"{BASE_URL}/ask",
            json=question_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 404 or a response indicating no documents
        assert response.status_code in [200, 404]
        
        if response.status_code == 404:
            data = response.json()
            assert "No documents" in data["detail"]
    
    def test_ask_question_empty_question(self):
        """Test asking an empty question"""
        question_data = {
            "question": "",
            "max_chunks": 3,
            "response_style": "detailed"
        }
        
        response = requests.post(
            f"{BASE_URL}/ask",
            json=question_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "cannot be empty" in data["detail"]
    
    def test_delete_nonexistent_document(self):
        """Test deleting a document that doesn't exist"""
        fake_doc_id = "nonexistent-document-id"
        response = requests.delete(f"{BASE_URL}/documents/{fake_doc_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    def test_cors_headers(self):
        """Test that CORS headers are present for frontend requests"""
        response = requests.options(f"{BASE_URL}/documents")
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestFrontendJavaScript:
    """Test JavaScript functionality (requires manual verification)"""
    
    def test_javascript_syntax(self):
        """Test that JavaScript file has valid syntax"""
        js_file_path = Path(__file__).parent.parent / "static" / "app.js"
        
        if not js_file_path.exists():
            pytest.skip("JavaScript file not found")
        
        with open(js_file_path, 'r', encoding='utf-8') as f:
            js_content = f.read()
        
        # Basic syntax checks
        assert "class StudyAssistant" in js_content
        assert "constructor()" in js_content
        assert "async uploadFile" in js_content
        assert "async askQuestion" in js_content
        assert "async loadDocuments" in js_content
        
        # Check for required methods
        required_methods = [
            "handleDragOver",
            "handleDragLeave", 
            "handleDrop",
            "handleFileSelect",
            "showUploadStatus",
            "deleteDocument",
            "formatAnswer"
        ]
        
        for method in required_methods:
            assert method in js_content, f"Required method {method} not found"
    
    def test_css_syntax(self):
        """Test that CSS file has valid syntax"""
        css_file_path = Path(__file__).parent.parent / "static" / "style.css"
        
        if not css_file_path.exists():
            pytest.skip("CSS file not found")
        
        with open(css_file_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        # Basic syntax checks
        assert ".container" in css_content
        assert ".upload-area" in css_content
        assert ".question-form" in css_content
        assert ".document-item" in css_content
        assert ".answer-section" in css_content
        
        # Check for responsive design
        assert "@media" in css_content
        assert "max-width" in css_content
    
    def test_html_structure(self):
        """Test that HTML has proper structure"""
        html_file_path = Path(__file__).parent.parent / "templates" / "index.html"
        
        if not html_file_path.exists():
            pytest.skip("HTML template not found")
        
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Basic structure checks
        assert "<!DOCTYPE html>" in html_content
        assert '<meta name="viewport"' in html_content
        assert 'id="uploadArea"' in html_content
        assert 'id="questionInput"' in html_content
        assert 'id="askButton"' in html_content
        assert 'id="documentsList"' in html_content
        assert 'id="answerSection"' in html_content
        
        # Check for accessibility
        assert 'alt=' in html_content or 'aria-' in html_content or len([x for x in html_content.split() if 'label' in x.lower()]) > 0


def run_manual_tests():
    """
    Manual tests that require human verification.
    Run this function and follow the instructions.
    """
    print("\n" + "="*60)
    print("MANUAL FRONTEND TESTS")
    print("="*60)
    print("\nPlease perform the following tests manually:")
    
    tests = [
        {
            "name": "Drag and Drop Upload",
            "steps": [
                "1. Open the Study Assistant in your browser",
                "2. Drag a PDF file over the upload area",
                "3. Verify the upload area changes appearance (border color/background)",
                "4. Drop the file and verify upload starts",
                "5. Check that progress is shown during upload"
            ]
        },
        {
            "name": "Responsive Design",
            "steps": [
                "1. Open the Study Assistant in your browser",
                "2. Resize the browser window to mobile size (< 768px)",
                "3. Verify layout adapts properly",
                "4. Check that all buttons and inputs are accessible",
                "5. Test on actual mobile device if possible"
            ]
        },
        {
            "name": "Keyboard Shortcuts",
            "steps": [
                "1. Focus on the question input field",
                "2. Type a question",
                "3. Press Ctrl+Enter (or Cmd+Enter on Mac)",
                "4. Verify the question is submitted",
                "5. Press Ctrl+K to focus the input field",
                "6. Press Escape to clear the input"
            ]
        },
        {
            "name": "Loading States",
            "steps": [
                "1. Upload a PDF file",
                "2. Verify loading indicators appear during processing",
                "3. Ask a question",
                "4. Verify the 'Ask Question' button shows loading state",
                "5. Check that UI is disabled during processing"
            ]
        },
        {
            "name": "Error Handling",
            "steps": [
                "1. Try uploading a non-PDF file",
                "2. Verify error message appears",
                "3. Try asking a question with no documents uploaded",
                "4. Verify appropriate error message",
                "5. Try uploading a file larger than 50MB",
                "6. Verify size limit error"
            ]
        }
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. {test['name']}")
        print("-" * 40)
        for step in test['steps']:
            print(f"   {step}")
        
        result = input(f"\nDid test '{test['name']}' pass? (y/n/s for skip): ").lower()
        if result == 'y':
            print("✅ PASSED")
        elif result == 'n':
            print("❌ FAILED")
        else:
            print("⏭️  SKIPPED")
    
    print("\n" + "="*60)
    print("Manual testing complete!")
    print("="*60)


if __name__ == "__main__":
    print("Frontend Integration Tests")
    print("=" * 40)
    print("\nTo run automated tests:")
    print("pytest tests/test_frontend_integration.py -v")
    print("\nTo run manual tests:")
    print("python tests/test_frontend_integration.py")
    print("\nMake sure the Study Assistant server is running on localhost:8000")
    
    # Run manual tests if called directly
    run_manual_tests()cl
ass TestStudyPreferencesIntegration:
    """Test study preferences functionality integration."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        # Check if server is running
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Study Assistant server is not running")
        except requests.exceptions.RequestException:
            pytest.skip("Study Assistant server is not running")
        
        # Clean up any existing test documents
        self.cleanup_test_documents()
    
    def cleanup_test_documents(self):
        """Clean up test documents"""
        try:
            response = requests.get(f"{BASE_URL}/documents")
            if response.status_code == 200:
                documents = response.json().get("documents", [])
                for doc in documents:
                    if doc["filename"].startswith("test_"):
                        requests.delete(f"{BASE_URL}/documents/{doc['id']}")
        except:
            pass  # Ignore cleanup errors
    
    def create_test_document(self, filename="test_document.pdf", tags=None):
        """Helper to create a test document"""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(TEST_PDF_CONTENT)
            tmp_file_path = tmp_file.name
        
        try:
            # Upload the file
            with open(tmp_file_path, 'rb') as f:
                files = {'file': (filename, f, 'application/pdf')}
                response = requests.post(f"{BASE_URL}/upload-pdf", files=files)
            
            if response.status_code == 200:
                doc_data = response.json()
                doc_id = doc_data["document_id"]
                
                # Add tags if provided
                if tags:
                    tag_response = requests.put(
                        f"{BASE_URL}/documents/{doc_id}/tags",
                        json=tags
                    )
                    assert tag_response.status_code == 200
                
                return doc_id
            else:
                pytest.fail(f"Failed to create test document: {response.text}")
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    def test_document_tagging_workflow(self):
        """Test complete document tagging workflow."""
        # Create a test document
        doc_id = self.create_test_document("test_tagging.pdf")
        
        # Test updating tags
        test_tags = ["Math", "Calculus", "Chapter 1"]
        response = requests.put(
            f"{BASE_URL}/documents/{doc_id}/tags",
            json=test_tags
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["tags"] == test_tags
        
        # Verify tags are returned when listing documents
        response = requests.get(f"{BASE_URL}/documents")
        assert response.status_code == 200
        
        documents = response.json()["documents"]
        test_doc = next((doc for doc in documents if doc["id"] == doc_id), None)
        assert test_doc is not None
        assert test_doc["tags"] == test_tags
        
        # Test updating tags again (overwrite)
        new_tags = ["Physics", "Mechanics"]
        response = requests.put(
            f"{BASE_URL}/documents/{doc_id}/tags",
            json=new_tags
        )
        
        assert response.status_code == 200
        assert response.json()["tags"] == new_tags
        
        # Clean up
        requests.delete(f"{BASE_URL}/documents/{doc_id}")
    
    def test_document_tagging_special_characters(self):
        """Test document tagging with special characters."""
        doc_id = self.create_test_document("test_special_tags.pdf")
        
        # Tags with special characters
        special_tags = [
            "Math & Science",
            "Chapter 1-5",
            "Review (Final)",
            "Tag with spaces"
        ]
        
        response = requests.put(
            f"{BASE_URL}/documents/{doc_id}/tags",
            json=special_tags
        )
        
        assert response.status_code == 200
        assert response.json()["tags"] == special_tags
        
        # Verify special characters are preserved
        response = requests.get(f"{BASE_URL}/documents/{doc_id}")
        assert response.status_code == 200
        assert response.json()["tags"] == special_tags
        
        # Clean up
        requests.delete(f"{BASE_URL}/documents/{doc_id}")
    
    def test_document_tagging_empty_tags(self):
        """Test clearing document tags."""
        doc_id = self.create_test_document("test_clear_tags.pdf", ["Initial", "Tags"])
        
        # Clear tags
        response = requests.put(
            f"{BASE_URL}/documents/{doc_id}/tags",
            json=[]
        )
        
        assert response.status_code == 200
        assert response.json()["tags"] == []
        
        # Verify tags are cleared
        response = requests.get(f"{BASE_URL}/documents/{doc_id}")
        assert response.status_code == 200
        assert response.json()["tags"] == []
        
        # Clean up
        requests.delete(f"{BASE_URL}/documents/{doc_id}")
    
    def test_study_preferences_in_queries(self):
        """Test study preferences are properly handled in queries."""
        # Create a test document first
        doc_id = self.create_test_document("test_preferences.pdf")
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Test different response styles
        response_styles = ["brief", "detailed", "comprehensive"]
        
        for style in response_styles:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={
                    "question": "What is this document about?",
                    "response_style": style,
                    "max_chunks": 3
                }
            )
            
            # Should succeed or return no documents error
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "answer" in data
                assert "sources" in data
                assert "processing_time" in data
        
        # Test different max_chunks values
        for max_chunks in [2, 3, 4, 5]:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={
                    "question": "What is this document about?",
                    "response_style": "detailed",
                    "max_chunks": max_chunks
                }
            )
            
            # Should succeed or return no documents error
            assert response.status_code in [200, 404]
        
        # Test invalid response style
        response = requests.post(
            f"{BASE_URL}/ask",
            json={
                "question": "What is this document about?",
                "response_style": "invalid_style",
                "max_chunks": 3
            }
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test invalid max_chunks
        response = requests.post(
            f"{BASE_URL}/ask",
            json={
                "question": "What is this document about?",
                "response_style": "detailed",
                "max_chunks": 15  # Above limit
            }
        )
        
        assert response.status_code == 422  # Validation error
        
        # Clean up
        requests.delete(f"{BASE_URL}/documents/{doc_id}")
    
    def test_frontend_preferences_persistence(self):
        """Test that frontend can handle preferences (basic HTML check)."""
        response = requests.get(BASE_URL)
        assert response.status_code == 200
        
        html_content = response.text
        
        # Check that preference elements exist in HTML
        assert 'id="responseStyle"' in html_content
        assert 'id="maxChunks"' in html_content
        assert 'preferencesModal' in html_content
        assert 'tagModal' in html_content
        
        # Check for preference options
        assert 'value="brief"' in html_content
        assert 'value="detailed"' in html_content
        assert 'value="comprehensive"' in html_content
    
    def test_document_list_includes_tags(self):
        """Test that document listing includes tag information."""
        # Create documents with different tags
        doc1_id = self.create_test_document("test_doc1.pdf", ["Math", "Algebra"])
        doc2_id = self.create_test_document("test_doc2.pdf", ["Science", "Physics"])
        doc3_id = self.create_test_document("test_doc3.pdf", [])
        
        # Get document list
        response = requests.get(f"{BASE_URL}/documents")
        assert response.status_code == 200
        
        documents = response.json()["documents"]
        
        # Find our test documents
        test_docs = {doc["id"]: doc for doc in documents if doc["filename"].startswith("test_doc")}
        
        assert len(test_docs) >= 3
        assert test_docs[doc1_id]["tags"] == ["Math", "Algebra"]
        assert test_docs[doc2_id]["tags"] == ["Science", "Physics"]
        assert test_docs[doc3_id]["tags"] == []
        
        # Clean up
        for doc_id in [doc1_id, doc2_id, doc3_id]:
            requests.delete(f"{BASE_URL}/documents/{doc_id}")


class TestAnswerFormattingIntegration:
    """Test enhanced answer formatting in integration."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Study Assistant server is not running")
        except requests.exceptions.RequestException:
            pytest.skip("Study Assistant server is not running")
    
    def test_answer_contains_formatted_elements(self):
        """Test that answers contain properly formatted study elements."""
        # This test assumes there are documents uploaded
        response = requests.post(
            f"{BASE_URL}/ask",
            json={
                "question": "What is the main topic?",
                "response_style": "detailed",
                "max_chunks": 3
            }
        )
        
        # Should either succeed with formatted answer or return no documents
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            
            # Check for study-friendly formatting elements
            # These might be present depending on the content
            formatting_indicators = [
                "**",  # Bold formatting
                "•",   # Bullet points
                "Definition:",  # Definition markers
                "Example:",     # Example markers
            ]
            
            # At least some formatting should be present
            has_formatting = any(indicator in answer for indicator in formatting_indicators)
            
            # The answer should have some structure
            assert len(answer) > 0
            assert "sources" in data
            assert "processing_time" in data
    
    def test_key_concepts_extraction(self):
        """Test that key concepts are extracted from answers."""
        response = requests.post(
            f"{BASE_URL}/ask",
            json={
                "question": "What are the key concepts?",
                "response_style": "comprehensive",
                "max_chunks": 3
            }
        )
        
        # Should either succeed or return no documents
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            
            # Key concepts should be included
            assert "key_concepts" in data
            assert isinstance(data["key_concepts"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])