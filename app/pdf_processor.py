"""
PDF processing module for extracting and chunking text from PDF documents.
"""

import PyPDF2
import re
import uuid
from typing import List, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

from .models import TextChunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of PDF processing operation"""
    success: bool
    chunks: List[TextChunk]
    error_message: Optional[str] = None
    page_count: Optional[int] = None

class PDFProcessor:
    """
    PDF processor for extracting text and creating chunks for the RAG system.
    Designed specifically for study materials processing.
    """
    
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        """
        Initialize PDF processor with chunking parameters.
        
        Args:
            chunk_size: Target size for text chunks in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def extract_text(self, pdf_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, page_count)
            
        Raises:
            Exception: If PDF cannot be read or processed
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise Exception("PDF is encrypted and cannot be processed")
                
                page_count = len(pdf_reader.pages)
                if page_count == 0:
                    raise Exception("PDF contains no pages")
                
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            # Add page marker for reference
                            text_content.append(f"[PAGE {page_num}]\n{page_text}")
                        else:
                            logger.warning(f"Page {page_num} contains no extractable text")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                        continue
                
                if not text_content:
                    raise Exception("No text could be extracted from PDF")
                
                full_text = "\n\n".join(text_content)
                logger.info(f"Successfully extracted text from {page_count} pages")
                
                return full_text, page_count
                
        except FileNotFoundError:
            raise Exception(f"PDF file not found: {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text suitable for chunking
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        # Keep basic punctuation and page markers
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\/\n]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, document_id: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Cleaned text to chunk
            document_id: ID of the parent document
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(text):
            # Calculate end position for this chunk
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only create non-empty chunks
                # Extract page number if available
                page_number = self._extract_page_number(chunk_text)
                
                # Remove page markers from chunk content for cleaner text
                clean_chunk_text = re.sub(r'\[PAGE \d+\]\s*', '', chunk_text)
                
                chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=clean_chunk_text,
                    chunk_index=chunk_index,
                    page_number=page_number
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position for next chunk with overlap
            start = end - self.overlap
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        logger.info(f"Created {len(chunks)} chunks from document {document_id}")
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find the best sentence boundary within the given range.
        
        Args:
            text: Text to search
            start: Start position to search from
            end: End position to search to
            
        Returns:
            Position of sentence boundary, or end if none found
        """
        # Look for sentence endings (., !, ?)
        for i in range(end - 1, start - 1, -1):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1].isspace():
                return i + 1
        
        # If no sentence boundary found, look for paragraph breaks
        for i in range(end - 1, start - 1, -1):
            if text[i] == '\n':
                return i + 1
        
        # If no good boundary found, return original end
        return end
    
    def _extract_page_number(self, chunk_text: str) -> Optional[int]:
        """
        Extract page number from chunk text if available.
        
        Args:
            chunk_text: Text chunk that may contain page markers
            
        Returns:
            Page number if found, None otherwise
        """
        page_match = re.search(r'\[PAGE (\d+)\]', chunk_text)
        if page_match:
            return int(page_match.group(1))
        return None
    
    def process_pdf(self, pdf_path: str, document_id: str) -> ProcessingResult:
        """
        Complete PDF processing pipeline: extract, clean, and chunk text.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Unique identifier for the document
            
        Returns:
            ProcessingResult with success status and chunks or error message
        """
        try:
            # Extract text from PDF
            raw_text, page_count = self.extract_text(pdf_path)
            
            # Clean the extracted text
            clean_text = self.clean_text(raw_text)
            
            # Create chunks
            chunks = self.chunk_text(clean_text, document_id)
            
            if not chunks:
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    error_message="No text chunks could be created from PDF",
                    page_count=page_count
                )
            
            logger.info(f"Successfully processed PDF: {len(chunks)} chunks created from {page_count} pages")
            
            return ProcessingResult(
                success=True,
                chunks=chunks,
                page_count=page_count
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                chunks=[],
                error_message=str(e)
            )