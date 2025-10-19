# Study Assistant RAG System - Troubleshooting Guide

This guide helps resolve common issues when setting up and using the Study Assistant RAG System.

## 🚀 Quick Diagnostics

### System Health Check

1. **Check if server is running**:
   ```bash
   curl http://localhost:8000/health
   ```
   Expected response: `{"status": "healthy", "message": "..."}`

2. **Check API documentation**:
   Visit: http://localhost:8000/docs

3. **Check logs**:
   Look at the terminal where you started the server for error messages.

## 🔧 Installation Issues

### Python Version Problems

**Error**: `Python 3.8 or higher is required`

**Solution**:
```bash
# Check your Python version
python --version

# If too old, install Python 3.8+ from python.org
# Or use pyenv to manage multiple Python versions
```

### Virtual Environment Issues

**Error**: `'python' is not recognized` or `command not found`

**Solutions**:
```bash
# On Windows, try:
py -m venv .venv
.venv\Scripts\activate

# On macOS/Linux, try:
python3 -m venv .venv
source .venv/bin/activate

# If still failing, install virtualenv:
pip install virtualenv
virtualenv .venv
```

### Dependency Installation Failures

**Error**: `Failed building wheel for [package]`

**Solutions**:
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install build tools (Windows)
# Download Microsoft C++ Build Tools

# Install build tools (macOS)
xcode-select --install

# Install build tools (Ubuntu/Debian)
sudo apt-get install build-essential python3-dev

# Try installing problematic packages individually
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install faiss-cpu
```

### FAISS Installation Issues

**Error**: `No module named 'faiss'` or FAISS import errors

**Solutions**:
```bash
# Try CPU version first
pip uninstall faiss-gpu faiss-cpu
pip install faiss-cpu

# If still failing, try conda
conda install -c conda-forge faiss-cpu

# For Apple Silicon Macs
pip install faiss-cpu --no-cache-dir
```

## 📄 PDF Processing Issues

### PDF Upload Failures

**Error**: `Only PDF files are allowed`

**Causes & Solutions**:
- **File not actually PDF**: Ensure file has `.pdf` extension and is a real PDF
- **Corrupted PDF**: Try opening the PDF in a PDF reader first
- **Scanned PDF**: The system works best with text-based PDFs, not scanned images

**Error**: `File size exceeds 50MB limit`

**Solutions**:
- Compress the PDF using online tools or PDF software
- Split large PDFs into smaller sections
- Increase the limit in `app/main.py` if needed (line with `50 * 1024 * 1024`)

### Text Extraction Problems

**Error**: `No text could be extracted from PDF`

**Causes & Solutions**:
- **Scanned/Image PDF**: Use OCR software to convert to text-based PDF
- **Password-protected PDF**: Remove password protection first
- **Corrupted PDF**: Try re-downloading or re-creating the PDF
- **Complex formatting**: Some PDFs with complex layouts may not extract well

**Error**: `PDF processing failed`

**Solutions**:
```bash
# Check PDF manually
python -c "
import PyPDF2
with open('your_file.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(f'Pages: {len(reader.pages)}')
    print(f'Text: {reader.pages[0].extract_text()[:100]}')
"

# Try alternative PDF processing
pip install pdfplumber
# (You'd need to modify the code to use pdfplumber instead)
```

## 🔍 Query and Answer Issues

### No Answers Found

**Error**: `No documents have been uploaded yet`

**Solutions**:
- Upload PDF documents first
- Check that upload was successful (documents appear in the list)
- Verify documents were processed (check for chunk_count > 0)

**Error**: `I couldn't find any relevant information`

**Solutions**:
- **Rephrase your question**: Use keywords that appear in your documents
- **Be more specific**: Instead of "What is this about?", ask "What is photosynthesis?"
- **Check document content**: Ensure your documents contain information about your question
- **Try different questions**: Test with questions you know are answered in your documents

### Poor Answer Quality

**Issue**: Answers are too brief or not helpful

**Solutions**:
- **Change response style**: Try "comprehensive" instead of "brief"
- **Increase max_chunks**: Use 5 instead of 3 to get more context
- **Add OpenAI API key**: Set `OPENAI_API_KEY` environment variable for better answers
- **Upload more relevant documents**: More context usually means better answers

**Issue**: Answers don't match the question

**Solutions**:
- **Check document relevance**: Ensure your PDFs contain information about your question
- **Use specific keywords**: Include exact terms from your documents
- **Try simpler questions**: Start with basic questions and build up complexity

## 🚀 Performance Issues

### Slow Upload Processing

**Issue**: PDF uploads take too long

**Solutions**:
- **Check file size**: Smaller files process faster
- **Close other applications**: Free up RAM and CPU
- **Check system resources**: Ensure adequate memory (2GB+ recommended)
- **Restart the application**: Sometimes helps with memory issues

### Slow Query Responses

**Issue**: Questions take too long to answer

**Solutions**:
- **Reduce max_chunks**: Use 2-3 instead of 5-10
- **Use brief response style**: Faster than comprehensive
- **Check system resources**: Ensure adequate CPU and memory
- **Restart the application**: Clear any memory leaks

### Memory Issues

**Error**: `Out of memory` or system becomes unresponsive

**Solutions**:
```bash
# Check memory usage
# On Windows: Task Manager
# On macOS: Activity Monitor  
# On Linux: htop or free -h

# Reduce memory usage:
# 1. Process fewer documents at once
# 2. Use smaller chunk sizes
# 3. Restart the application periodically
# 4. Close other applications
```

## 🌐 Server and Network Issues

### Server Won't Start

**Error**: `Address already in use` or `Port 8000 is already in use`

**Solutions**:
```bash
# Find what's using port 8000
# On Windows:
netstat -ano | findstr :8000

# On macOS/Linux:
lsof -i :8000

# Kill the process or use a different port
python run.py --port 8001
```

**Error**: `Permission denied` when binding to port

**Solutions**:
- Use a port above 1024 (like 8000, 8080)
- Run with administrator/sudo privileges (not recommended)
- Check firewall settings

### Can't Access Web Interface

**Issue**: Browser shows "This site can't be reached"

**Solutions**:
- **Check server is running**: Look for startup messages in terminal
- **Try different URL**: http://127.0.0.1:8000 instead of localhost
- **Check firewall**: Ensure port 8000 is not blocked
- **Try different browser**: Rule out browser-specific issues

## 🧪 Testing Issues

### Tests Failing

**Error**: Various test failures

**Solutions**:
```bash
# Run tests with more verbose output
python -m pytest -v -s

# Run specific test file
python -m pytest tests/test_pdf_processor.py -v

# Skip failing tests temporarily
python -m pytest -k "not test_failing_function"

# Check test dependencies
pip install pytest pytest-cov
```

## 🐳 Docker Issues

### Docker Build Failures

**Error**: Docker build fails

**Solutions**:
```bash
# Clean Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t study-assistant .

# Check Docker version
docker --version

# Ensure Docker is running
docker info
```

### Container Won't Start

**Error**: Container exits immediately

**Solutions**:
```bash
# Check container logs
docker logs study-assistant

# Run interactively for debugging
docker run -it study-assistant /bin/bash

# Check port conflicts
docker ps -a
```

## 🔑 Environment Variables

### OpenAI API Issues

**Issue**: Enhanced answers not working

**Solutions**:
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Set API key (replace with your key)
export OPENAI_API_KEY="sk-your-key-here"

# On Windows:
set OPENAI_API_KEY=sk-your-key-here

# Test API key
python -c "
import openai
openai.api_key = 'your-key'
print('API key is valid')
"
```

## 📊 Database Issues

### SQLite Database Problems

**Error**: Database locked or corrupted

**Solutions**:
```bash
# Stop the application first
# Delete the database file (you'll lose data)
rm data/study_assistant.db

# Or backup and restore
cp data/study_assistant.db data/study_assistant.db.backup

# Check database integrity
sqlite3 data/study_assistant.db "PRAGMA integrity_check;"
```

### Vector Store Issues

**Error**: FAISS index problems

**Solutions**:
```bash
# Delete vector index files (you'll need to re-upload documents)
rm data/faiss_index*

# Check index files exist
ls -la data/

# Restart application to rebuild index
```

## 🆘 Getting More Help

### Collecting Debug Information

When reporting issues, include:

1. **System Information**:
   ```bash
   python --version
   pip list | grep -E "(fastapi|sentence|faiss|torch)"
   ```

2. **Error Messages**: Full error text from terminal

3. **Steps to Reproduce**: Exact steps that cause the issue

4. **Log Output**: Any relevant log messages

### Common Log Locations

- **Application logs**: Terminal where you started the server
- **System logs**: 
  - Windows: Event Viewer
  - macOS: Console app
  - Linux: `/var/log/` or `journalctl`

### Reset Everything

If all else fails, complete reset:

```bash
# Stop the application
# Delete all data (you'll lose uploaded documents)
rm -rf data/ uploads/ .venv/

# Start fresh
python deploy.py --dev
```

## 📞 Support Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check README.md for setup instructions
- **API Docs**: http://localhost:8000/docs for API reference
- **Test Suite**: Run tests to verify functionality

---

**Remember**: Most issues are related to environment setup or file permissions. Start with the basics and work your way up to more complex solutions.