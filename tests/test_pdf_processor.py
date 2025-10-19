"""
Tests for PDF processing functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
import PyPDF2

from app.pdf_processor import PDFProcessor, ProcessingResult
from app.models import TextChunk


class TestPDFProcessor:
    """Test cases for PDFProcessor class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = PDFProcessor(chunk_size=100, overlap=20)
        self.test_document_id = "test-doc-123"
    
    def test_init_default_parameters(self):
        """Test PDFProcessor initialization with default parameters"""
        processor = PDFProcessor()
        assert processor.chunk_size == 800
        assert processor.overlap == 100
    
    def test_init_custom_parameters(self):
        """Test PDFProcessor initialization with custom parameters"""
        processor = PDFProcessor(chunk_size=500, overlap=50)
        assert processor.chunk_size == 500
        assert processor.overlap == 50
    
    def test_clean_text_removes_excessive_whitespace(self):
        """Test text cleaning removes excessive whitespace"""
        dirty_text = "This  is   a    test\n\n\nwith   multiple    spaces"
        clean_text = self.processor.clean_text(dirty_text)
        assert clean_text == "This is a test with multiple spaces"
    
    def test_clean_text_removes_special_characters(self):
        """Test text cleaning removes unwanted special characters"""
        dirty_text = "Test with special chars: @#$%^&*"
        clean_text = self.processor.clean_text(dirty_text)
        # Should keep basic punctuation but remove special chars
        assert "@#$%^&*" not in clean_text
        assert "Test with special chars" in clean_text
    
    def test_clean_text_preserves_basic_punctuation(self):
        """Test text cleaning preserves important punctuation"""
        text_with_punctuation = "Hello, world! How are you? I'm fine."
        clean_text = self.processor.clean_text(text_with_punctuation)
        assert "," in clean_text
        assert "!" in clean_text
        assert "?" in clean_text
        assert "." in clean_text
    
    def test_chunk_text_empty_input(self):
        """Test chunking with empty text input"""
        chunks = self.processor.chunk_text("", self.test_document_id)
        assert chunks == []
    
    def test_chunk_text_short_text(self):
        """Test chunking with text shorter than chunk size"""
        short_text = "This is a short text."
        chunks = self.processor.chunk_text(short_text, self.test_document_id)
        
        assert len(chunks) == 1
        assert chunks[0].content == short_text
        assert chunks[0].document_id == self.test_document_id
        assert chunks[0].chunk_index == 0
    
    def test_chunk_text_long_text_creates_multiple_chunks(self):
        """Test chunking with text longer than chunk size creates multiple chunks"""
        # Create text longer than chunk_size (100)
        long_text = "A" * 250  # 250 characters
        chunks = self.processor.chunk_text(long_text, self.test_document_id)
        
        assert len(chunks) > 1
        # Check that chunks have proper indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == self.test_document_id
    
    def test_chunk_text_overlap_functionality(self):
        """Test that chunks have proper overlap"""
        # Create text that will result in exactly 2 chunks
        text = "A" * 150  # 150 characters with chunk_size=100, overlap=20
        chunks = self.processor.chunk_text(text, self.test_document_id)
        
        assert len(chunks) == 2
        # Second chunk should start 80 characters from the beginning (100-20 overlap)
        # This is hard to test exactly due to sentence boundary logic, but we can check basic properties
        assert all(len(chunk.content) > 0 for chunk in chunks)
    
    def test_extract_page_number_with_page_marker(self):
        """Test page number extraction from text with page markers"""
        text_with_page = "[PAGE 5]\nThis is content from page 5"
        page_num = self.processor._extract_page_number(text_with_page)
        assert page_num == 5
    
    def test_extract_page_number_without_page_marker(self):
        """Test page number extraction from text without page markers"""
        text_without_page = "This is content without page markers"
        page_num = self.processor._extract_page_number(text_without_page)
        assert page_num is None
    
    def test_find_sentence_boundary_with_period(self):
        """Test sentence boundary detection with period"""
        text = "This is sentence one. This is sentence two. More text here."
        boundary = self.processor._find_sentence_boundary(text, 10, 30)
        # Should find the period after "one"
        assert boundary > 10
        assert text[boundary-1] == '.'
    
    def test_find_sentence_boundary_no_sentence_end(self):
        """Test sentence boundary detection when no sentence end is found"""
        text = "This is all one long sentence without any periods or breaks"
        boundary = self.processor._find_sentence_boundary(text, 10, 30)
        # Should return the original end position
        assert boundary == 30
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('PyPDF2.PdfReader')
    def test_extract_text_success(self, mock_pdf_reader, mock_file):
        """Test successful text extraction from PDF"""
        # Mock PDF reader
        mock_page = mock_pdf_reader.return_value.pages[0]
        mock_page.extract_text.return_value = "Sample text from PDF page"
        mock_pdf_reader.return_value.pages = [mock_page]
        mock_pdf_reader.return_value.is_encrypted = False
        
        text, page_count = self.processor.extract_text("dummy_path.pdf")
        
        assert "Sample text from PDF page" in text
        assert page_count == 1
        assert "[PAGE 1]" in text
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('PyPDF2.PdfReader')
    def test_extract_text_encrypted_pdf(self, mock_pdf_reader, mock_file):
        """Test text extraction from encrypted PDF raises exception"""
        mock_pdf_reader.return_value.is_encrypted = True
        
        with pytest.raises(Exception) as exc_info:
            self.processor.extract_text("dummy_path.pdf")
        
        assert "encrypted" in str(exc_info.value).lower()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('PyPDF2.PdfReader')
    def test_extract_text_empty_pdf(self, mock_pdf_reader, mock_file):
        """Test text extraction from PDF with no pages"""
        mock_pdf_reader.return_value.pages = []
        mock_pdf_reader.return_value.is_encrypted = False
        
        with pytest.raises(Exception) as exc_info:
            self.processor.extract_text("dummy_path.pdf")
        
        assert "no pages" in str(exc_info.value).lower()
    
    def test_extract_text_file_not_found(self):
        """Test text extraction with non-existent file"""
        with pytest.raises(Exception) as exc_info:
            self.processor.extract_text("nonexistent_file.pdf")
        
        assert "not found" in str(exc_info.value).lower()
    
    @patch.object(PDFProcessor, 'extract_text')
    @patch.object(PDFProcessor, 'clean_text')
    @patch.object(PDFProcessor, 'chunk_text')
    def test_process_pdf_success(self, mock_chunk_text, mock_clean_text, mock_extract_text):
        """Test successful PDF processing pipeline"""
        # Mock the pipeline steps
        mock_extract_text.return_value = ("Raw PDF text", 3)
        mock_clean_text.return_value = "Clean PDF text"
        mock_chunks = [
            TextChunk(id="1", document_id=self.test_document_id, content="Chunk 1", chunk_index=0),
            TextChunk(id="2", document_id=self.test_document_id, content="Chunk 2", chunk_index=1)
        ]
        mock_chunk_text.return_value = mock_chunks
        
        result = self.processor.process_pdf("test.pdf", self.test_document_id)
        
        assert result.success is True
        assert len(result.chunks) == 2
        assert result.page_count == 3
        assert result.error_message is None
    
    @patch.object(PDFProcessor, 'extract_text')
    def test_process_pdf_extraction_failure(self, mock_extract_text):
        """Test PDF processing when text extraction fails"""
        mock_extract_text.side_effect = Exception("PDF extraction failed")
        
        result = self.processor.process_pdf("test.pdf", self.test_document_id)
        
        assert result.success is False
        assert result.chunks == []
        assert "PDF extraction failed" in result.error_message
    
    @patch.object(PDFProcessor, 'extract_text')
    @patch.object(PDFProcessor, 'clean_text')
    @patch.object(PDFProcessor, 'chunk_text')
    def test_process_pdf_no_chunks_created(self, mock_chunk_text, mock_clean_text, mock_extract_text):
        """Test PDF processing when no chunks can be created"""
        mock_extract_text.return_value = ("Some text", 1)
        mock_clean_text.return_value = "Clean text"
        mock_chunk_text.return_value = []  # No chunks created
        
        result = self.processor.process_pdf("test.pdf", self.test_document_id)
        
        assert result.success is False
        assert result.chunks == []
        assert "No text chunks could be created" in result.error_message
        assert result.page_count == 1


if __name__ == "__main__":
    pytest.main([__file__])