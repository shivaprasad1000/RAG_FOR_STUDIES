# Study Assistant RAG System

A complete Retrieval-Augmented Generation (RAG) system designed specifically for students to upload PDF textbooks and notes, then query them using natural language for exam preparation and studying.

## ✨ Features

- 📚 **PDF Upload & Processing**: Upload textbooks, notes, and study materials in PDF format
- 🔍 **Natural Language Queries**: Ask questions in plain English about your documents
- 💡 **Study-Focused Answers**: Get contextual answers with highlighted key concepts
- 📖 **Source References**: Every answer includes page numbers and document sources
- 🏷️ **Document Organization**: Tag and categorize your study materials by subject
- ⚙️ **Customizable Responses**: Choose between brief, detailed, or comprehensive answer styles
- ⚡ **Fast Local Processing**: Works offline with local embeddings (no API costs required)
- 🤖 **Optional AI Enhancement**: Integrate with OpenAI for enhanced answer generation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- At least 2GB RAM for embedding models
- 1GB disk space for dependencies and data

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd study-assistant

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python run.py
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Start Using

1. **Upload PDFs**: Drag and drop PDF files or use the upload button
2. **Wait for Processing**: Documents are automatically processed and indexed
3. **Ask Questions**: Type natural language questions about your materials
4. **Get Answers**: Receive study-focused responses with source references

## 📁 Project Structure

```
study-assistant/
├── app/                    # Main application code
│   ├── __init__.py
│   ├── main.py            # FastAPI application and endpoints
│   ├── models.py          # Pydantic data models
│   ├── database.py        # SQLite database operations
│   ├── pdf_processor.py   # PDF text extraction and chunking
│   ├── embedding_generator.py  # Text-to-vector embeddings
│   ├── vector_store.py    # FAISS vector similarity search
│   └── answer_generator.py # Answer generation and formatting
├── static/                # Frontend assets
│   ├── style.css          # Responsive CSS styling
│   └── app.js             # Interactive JavaScript
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── tests/                 # Comprehensive test suite
│   ├── test_*.py          # Unit and integration tests
│   └── test_end_to_end_integration.py  # Full workflow tests
├── data/                  # Generated data storage
│   ├── study_assistant.db # SQLite database
│   ├── faiss_index        # Vector index files
│   └── faiss_index.metadata
├── uploads/               # Temporary PDF upload storage
├── requirements.txt       # Python dependencies
├── run.py                # Development server launcher
└── README.md             # This documentation
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **PDF Processing**: PyPDF2 for text extraction
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2 model)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Database**: SQLite for metadata and document management
- **Answer Generation**: OpenAI GPT (optional) or local fallback
- **Frontend**: Responsive HTML, CSS, JavaScript
- **Testing**: pytest with comprehensive test coverage

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Enhanced AI answers (requires OpenAI account)
export OPENAI_API_KEY="your-api-key-here"

# Optional: Custom model settings
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export CHUNK_SIZE="800"
export CHUNK_OVERLAP="100"
```

### Study Preferences

The system supports customizable study preferences:

- **Response Style**: Brief, Detailed, or Comprehensive
- **Max Chunks**: Number of document sections to use (2-10)
- **Answer Format**: Bullet points, definitions, examples

## 📚 Usage Guide

### Uploading Documents

1. **Supported Formats**: PDF files only (up to 50MB each)
2. **Processing**: Automatic text extraction, chunking, and indexing
3. **Organization**: Add tags like "Math", "Chapter 1", "Final Exam"

### Asking Questions

**Good Question Examples**:
- "What is photosynthesis and how does it work?"
- "Explain the key differences between supervised and unsupervised learning"
- "What are the main causes of World War I?"
- "How do you solve quadratic equations?"

**Tips for Better Results**:
- Be specific about what you want to know
- Use keywords from your documents
- Ask follow-up questions for clarification

### Managing Documents

- **View All Documents**: See upload date, size, and tags
- **Delete Documents**: Remove documents and all associated data
- **Tag Documents**: Organize by subject, chapter, or topic
- **Search by Tags**: Filter documents by categories

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/test_end_to_end_integration.py  # Full workflow
python -m pytest tests/test_api_endpoints.py          # API testing
python -m pytest tests/test_pdf_processor.py          # PDF processing

# Run with verbose output
python -m pytest -v

# Run with coverage report
python -m pytest --cov=app
```

## 🚀 Deployment

### Local Development

```bash
# Development server with auto-reload
python run.py

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or use the provided script
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔍 API Documentation

The system provides a complete REST API:

### Document Management
- `POST /upload-pdf` - Upload and process PDF documents
- `GET /documents` - List all uploaded documents
- `GET /documents/{id}` - Get specific document details
- `DELETE /documents/{id}` - Delete document and associated data
- `PUT /documents/{id}/tags` - Update document tags

### Question Answering
- `POST /ask` - Ask questions about uploaded documents

### System
- `GET /health` - System health check
- `GET /` - Web interface
- `GET /docs` - Interactive API documentation

Visit http://localhost:8000/docs for interactive API documentation.

## 🎯 Performance

The system is designed to meet specific performance requirements:

- **Document Upload**: Processes most PDFs in under 30 seconds
- **Query Response**: Returns relevant content within 3 seconds
- **Answer Generation**: Completes responses within 8 seconds
- **Concurrent Users**: Supports up to 5 simultaneous study sessions
- **Accuracy**: Provides relevant answers for 85%+ of document-related questions

## 🐛 Troubleshooting

### Common Issues

**PDF Upload Fails**
- Ensure file is a valid PDF (not scanned images)
- Check file size is under 50MB
- Verify PDF is not password-protected

**No Answers Found**
- Upload relevant documents first
- Try rephrasing your question
- Use keywords that appear in your documents

**Slow Performance**
- Ensure adequate RAM (2GB+ recommended)
- Close other applications to free memory
- Consider using fewer documents for faster search

**Installation Issues**
- Update pip: `pip install --upgrade pip`
- Install build tools if needed
- Check Python version compatibility (3.8+)

### Getting Help

1. Check the logs in the terminal for error messages
2. Verify all dependencies are installed correctly
3. Test with the provided sample documents
4. Run the test suite to identify issues

## 🤝 Contributing

This is an educational project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning about RAG systems, FastAPI, and AI applications.

## 🙏 Acknowledgments

- **sentence-transformers** for local embeddings
- **FAISS** for efficient vector search
- **FastAPI** for the web framework
- **OpenAI** for enhanced answer generation (optional)

---

**Happy Studying! 📚✨**