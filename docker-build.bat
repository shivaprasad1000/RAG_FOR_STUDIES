@echo off
REM Study Assistant RAG System - Docker Build Script for Windows

echo 🐳 Building Study Assistant Docker Image
echo ==========================================

REM Build the Docker image
docker build -t study-assistant-rag .

if %errorlevel% equ 0 (
    echo ✅ Docker image built successfully!
    echo.
    echo 🚀 To run the application:
    echo    docker-compose up
    echo.
    echo 🔧 To run with custom environment:
    echo    1. Copy .env.example to .env
    echo    2. Add your GEMINI_API_KEY or OPENAI_API_KEY
    echo    3. Run: docker-compose up
) else (
    echo ❌ Docker build failed!
    echo Check the error messages above.
)

pause