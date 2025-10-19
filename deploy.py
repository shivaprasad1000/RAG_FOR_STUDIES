#!/usr/bin/env python3
"""
Study Assistant RAG System - Deployment Script

This script helps deploy the Study Assistant system for production use.
It handles environment setup, dependency installation, and server startup.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def setup_environment():
    """Set up the Python virtual environment."""
    print("\n📦 Setting up Python environment...")
    
    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("   Creating virtual environment...")
        if not run_command("python -m venv .venv", "Creating virtual environment"):
            return False
    else:
        print("   Virtual environment already exists")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
    
    print("   Installing dependencies...")
    install_cmd = f"{pip_cmd} install -r requirements.txt"
    if not run_command(install_cmd, "Installing Python dependencies"):
        return False
    
    print("✅ Environment setup complete")
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["data", "uploads", "static", "templates"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True

def check_dependencies():
    """Check if all required dependencies are available."""
    print("\n🔍 Checking dependencies...")
    
    try:
        # Test imports
        import fastapi
        import uvicorn
        import sentence_transformers
        import faiss
        import PyPDF2
        
        print("   ✅ FastAPI")
        print("   ✅ Uvicorn")
        print("   ✅ Sentence Transformers")
        print("   ✅ FAISS")
        print("   ✅ PyPDF2")
        
        return True
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        return False

def run_tests():
    """Run the test suite to verify everything works."""
    print("\n🧪 Running tests...")
    
    if os.name == 'nt':  # Windows
        python_cmd = ".venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = ".venv/bin/python"
    
    test_cmd = f"{python_cmd} -m pytest tests/ -v --tb=short"
    if run_command(test_cmd, "Running test suite"):
        print("✅ All tests passed")
        return True
    else:
        print("⚠️  Some tests failed - check output above")
        return False

def start_server(host="0.0.0.0", port=8000, workers=1, reload=False):
    """Start the FastAPI server."""
    print(f"\n🚀 Starting Study Assistant server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Workers: {workers}")
    print(f"   Reload: {reload}")
    
    if os.name == 'nt':  # Windows
        python_cmd = ".venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = ".venv/bin/python"
    
    if reload:
        # Development mode with auto-reload
        server_cmd = f"{python_cmd} -m uvicorn app.main:app --host {host} --port {port} --reload"
    else:
        # Production mode
        server_cmd = f"{python_cmd} -m uvicorn app.main:app --host {host} --port {port} --workers {workers}"
    
    print(f"\n🌐 Server will be available at:")
    print(f"   Web Interface: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"   API Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    print(f"   Health Check: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health")
    print(f"\n   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the server
    os.system(server_cmd)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Study Assistant RAG System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--dev", action="store_true", help="Run in development mode with auto-reload")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--setup-only", action="store_true", help="Only setup environment, don't start server")
    
    args = parser.parse_args()
    
    print("🎓 Study Assistant RAG System - Deployment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        print("   Try running: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests (unless skipped)
    if not args.skip_tests:
        if not run_tests():
            print("⚠️  Tests failed, but continuing with deployment")
            print("   You may want to investigate test failures")
    
    print("\n✅ Setup complete!")
    
    # Start server (unless setup-only)
    if not args.setup_only:
        start_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.dev
        )
    else:
        print(f"\n🎯 To start the server manually, run:")
        if args.dev:
            print(f"   python run.py")
        else:
            print(f"   uvicorn app.main:app --host {args.host} --port {args.port}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Deployment interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)