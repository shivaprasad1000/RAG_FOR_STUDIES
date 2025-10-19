from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class ResponseStyle(str, Enum):
    """Response style options for study answers"""
    BRIEF = "brief"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

class Document(BaseModel):
    """Document model for uploaded PDFs"""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    upload_date: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    file_size: int = Field(..., description="File size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages in PDF")
    chunk_count: int = Field(0, description="Number of text chunks created")
    tags: List[str] = Field(default_factory=list, description="Subject/topic tags")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TextChunk(BaseModel):
    """Text chunk model for processed document content"""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Text content of the chunk")
    chunk_index: int = Field(..., description="Index of chunk within document")
    page_number: Optional[int] = Field(None, description="Source page number")
    
class StudyQuestion(BaseModel):
    """Study question model for user queries"""
    question: str = Field(..., description="The study question to ask")
    max_chunks: int = Field(3, description="Maximum number of chunks to retrieve", ge=1, le=10)
    response_style: ResponseStyle = Field(ResponseStyle.DETAILED, description="Desired response style")
    
class StudyResponse(BaseModel):
    """Study response model for generated answers"""
    answer: str = Field(..., description="Generated answer to the study question")
    sources: List[str] = Field(..., description="Source references (filename - page X)")
    processing_time: float = Field(..., description="Time taken to process the question")
    key_concepts: List[str] = Field(default_factory=list, description="Highlighted key concepts")

class SearchResult(BaseModel):
    """Search result model for vector similarity search"""
    chunk: TextChunk = Field(..., description="Retrieved text chunk")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    document_name: str = Field(..., description="Source document filename")

class ErrorResponse(BaseModel):
    """Error response model"""
    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }