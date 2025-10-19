from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import os
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from .models import Document, TextChunk, StudyQuestion, StudyResponse
from .database import db_manager
from .pdf_processor import PDFProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .answer_generator import AnswerGenerator

# Create FastAPI app
app = FastAPI(
    title="Study Assistant RAG System",
    description="A RAG system for students to upload PDFs and ask study questions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Create upload directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Set up logging
logger = logging.getLogger(__name__)

# Initialize components
pdf_processor = PDFProcessor()
embedding_generator = EmbeddingGenerator()
vector_store = VectorStore(embedding_dimension=384)  # all-MiniLM-L6-v2 dimension
answer_generator = AnswerGenerator()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Study Assistant RAG System is running"}

@app.get("/debug", response_class=HTMLResponse)
async def debug_page():
    """Debug page for frontend testing"""
    with open("debug_frontend.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF endpoint with full processing pipeline:
    extract text → chunk → generate embeddings → store in FAISS
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Check file size (50MB limit)
    content = await file.read()
    file_size = len(content)
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
    
    # Reset file pointer
    await file.seek(0)
    
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        upload_path = Path("uploads") / f"{document_id}_{file.filename}"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF: extract text and create chunks
        processing_result = pdf_processor.process_pdf(str(upload_path), document_id)
        
        if not processing_result.success:
            # Clean up uploaded file
            upload_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400, 
                detail=f"PDF processing failed: {processing_result.error_message}"
            )
        
        chunks = processing_result.chunks
        page_count = processing_result.page_count
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedding_generator.encode_batch(chunk_texts)
        
        # Create document record
        document = Document(
            id=document_id,
            filename=file.filename,
            upload_date=datetime.now(),
            file_size=file_size,
            page_count=page_count,
            chunk_count=len(chunks)
        )
        
        # Save document metadata to database
        if not db_manager.create_document(document):
            # Clean up uploaded file
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Failed to save document metadata")
        
        # Save chunks to database
        if not db_manager.create_chunks(chunks):
            # Clean up: delete document and uploaded file
            db_manager.delete_document(document_id)
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Failed to save document chunks")
        
        # Add embeddings to vector store
        vector_store.add_documents(chunks, embeddings, file.filename)
        
        # Clean up uploaded file (we don't need to keep it after processing)
        upload_path.unlink(missing_ok=True)
        
        return {
            "message": f"PDF {file.filename} uploaded and processed successfully",
            "document_id": document_id,
            "filename": file.filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "file_size": file_size
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up uploaded file on any error
        if 'upload_path' in locals():
            upload_path.unlink(missing_ok=True)
        
        # Clean up database records if they were created
        if 'document_id' in locals():
            db_manager.delete_document(document_id)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during PDF processing: {str(e)}"
        )

@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents with metadata.
    Returns document details including filename, upload_date, and chunk_count.
    """
    try:
        documents = db_manager.list_documents()
        
        # Convert documents to response format
        document_list = []
        for doc in documents:
            document_list.append({
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date.isoformat(),
                "file_size": doc.file_size,
                "page_count": doc.page_count,
                "chunk_count": doc.chunk_count,
                "tags": doc.tags
            })
        
        return {
            "documents": document_list,
            "total_count": len(document_list)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """
    Get details for a specific document.
    """
    try:
        document = db_manager.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        return {
            "id": document.id,
            "filename": document.filename,
            "upload_date": document.upload_date.isoformat(),
            "file_size": document.file_size,
            "page_count": document.page_count,
            "chunk_count": document.chunk_count,
            "tags": document.tags
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document: {str(e)}"
        )

@app.put("/documents/{doc_id}/tags")
async def update_document_tags(doc_id: str, tags: List[str]):
    """
    Update tags for a specific document.
    """
    try:
        # Check if document exists
        document = db_manager.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Update tags
        success = db_manager.update_document_tags(doc_id, tags)
        if not success:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to update tags for document {doc_id}"
            )
        
        return {
            "message": f"Tags updated successfully for '{document.filename}'",
            "document_id": doc_id,
            "tags": tags
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update document tags: {str(e)}"
        )

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and all its associated chunks from both database and vector store.
    """
    try:
        # Check if document exists
        document = db_manager.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Delete from vector store first
        vector_deleted = vector_store.delete_document(doc_id)
        if not vector_deleted:
            # Log warning but continue with database deletion
            print(f"Warning: Document {doc_id} not found in vector store")
        
        # Delete from database (this will cascade delete chunks due to foreign key)
        db_deleted = db_manager.delete_document(doc_id)
        if not db_deleted:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to delete document {doc_id} from database"
            )
        
        return {
            "message": f"Document '{document.filename}' deleted successfully",
            "document_id": doc_id,
            "filename": document.filename
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@app.post("/ask")
async def ask_question(question: StudyQuestion):
    """
    Process a study question and return relevant information from uploaded documents.
    
    This endpoint:
    1. Converts the user question to embedding using the same model as documents
    2. Searches FAISS index for top similar chunks
    3. Retrieves chunk content from database
    4. Formats retrieved chunks with source information
    """
    import time
    start_time = time.time()
    
    try:
        # Validate question
        if not question.question or not question.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if vector store has any documents
        if vector_store.index.ntotal == 0:
            raise HTTPException(
                status_code=404, 
                detail="No documents have been uploaded yet. Please upload PDF documents first."
            )
        
        # Convert user question to embedding using same model as documents
        query_embedding = embedding_generator.encode_text(question.question.strip())
        
        # Search FAISS index for top similar chunks
        search_results = vector_store.search(
            query_embedding, 
            top_k=min(question.max_chunks, 10)  # Limit to reasonable number
        )
        
        if not search_results:
            return StudyResponse(
                answer="I couldn't find any relevant information in your uploaded documents for this question.",
                sources=[],
                processing_time=time.time() - start_time,
                key_concepts=[]
            )
        
        # Retrieve chunk content from database and format with source information
        context_chunks = []
        sources = []
        
        for result in search_results:
            # Get full chunk content from database
            chunk = db_manager.get_chunk(result.chunk.id)
            if chunk:
                # Format source reference
                page_ref = f"page {chunk.page_number}" if chunk.page_number else "unknown page"
                source = f"{result.document_name} - {page_ref}"
                sources.append(source)
                
                # Format chunk with source info for context
                formatted_chunk = f"[Source: {source}]\n{chunk.content}"
                context_chunks.append(formatted_chunk)
        
        if not context_chunks:
            return StudyResponse(
                answer="I found some relevant documents, but couldn't retrieve the content. Please try again.",
                sources=[],
                processing_time=time.time() - start_time,
                key_concepts=[]
            )
        
        # Generate study-focused answer using LLM
        study_response = answer_generator.generate_answer(
            question=question.question,
            context_chunks=context_chunks,
            sources=sources,
            response_style=question.response_style
        )
        
        # Update processing time to include the full pipeline
        study_response.processing_time = time.time() - start_time
        
        return study_response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Failed to process question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process your question: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)