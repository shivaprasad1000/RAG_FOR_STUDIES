#!/usr/bin/env python3
"""
Study Assistant RAG System
Simple startup script for development
"""

import uvicorn
from app.main import app

if __name__ == "__main__":
    print("🚀 Starting Study Assistant RAG System...")
    print("📚 Access the application at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )