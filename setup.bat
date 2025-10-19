@echo off
REM Study Assistant RAG System - Setup Script for Windows

echo 🎓 Study Assistant RAG System - Setup
echo ======================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8 or higher.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python version: %PYTHON_VERSION%

REM Create virtual environment
echo 🔄 Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo ✅ Virtual environment created
) else (
    echo ⚠️  Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 🔄 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 🔄 Creating directories...
if not exist "data" mkdir data
if not exist "uploads" mkdir uploads
if not exist "static" mkdir static
if not exist "templates" mkdir templates

REM Run tests
echo 🔄 Running tests...
python -m pytest tests/ -v --tb=short
if %errorlevel% equ 0 (
    echo ✅ All tests passed
) else (
    echo ⚠️  Some tests failed, but setup continues
)

REM Success message
echo.
echo ✅ Setup complete!
echo.
echo 🚀 To start the Study Assistant:
echo    1. Activate the virtual environment: .venv\Scripts\activate.bat
echo    2. Start the server: python run.py
echo.
echo 🌐 The application will be available at:
echo    - Web Interface: http://localhost:8000
echo    - API Documentation: http://localhost:8000/docs
echo.
echo 📚 For more information, see README.md
echo 🐛 For troubleshooting, see TROUBLESHOOTING.md

pause