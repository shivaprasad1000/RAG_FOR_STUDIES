#!/bin/bash
# Study Assistant RAG System - Setup Script for Unix/Linux/macOS

set -e  # Exit on any error

echo "🎓 Study Assistant RAG System - Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python 3.8+ is available
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

print_success "Python version: $PYTHON_VERSION"

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating directories..."
mkdir -p data uploads static templates

# Run tests
print_status "Running tests..."
if python -m pytest tests/ -v --tb=short; then
    print_success "All tests passed"
else
    print_warning "Some tests failed, but setup continues"
fi

# Success message
echo ""
print_success "Setup complete!"
echo ""
echo "🚀 To start the Study Assistant:"
echo "   1. Activate the virtual environment: source .venv/bin/activate"
echo "   2. Start the server: python run.py"
echo ""
echo "🌐 The application will be available at:"
echo "   - Web Interface: http://localhost:8000"
echo "   - API Documentation: http://localhost:8000/docs"
echo ""
echo "📚 For more information, see README.md"
echo "🐛 For troubleshooting, see TROUBLESHOOTING.md"