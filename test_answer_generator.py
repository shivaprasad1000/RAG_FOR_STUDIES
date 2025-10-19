#!/usr/bin/env python3
"""
Quick test of the AnswerGenerator functionality.
"""

from app.answer_generator import AnswerGenerator
from app.models import ResponseStyle

def test_answer_generator():
    """Test the answer generator in fallback mode."""
    
    # Initialize answer generator (will use fallback mode without API key)
    generator = AnswerGenerator()
    
    print(f"Answer generator available: {generator.is_available()}")
    print(f"Config info: {generator.get_config_info()}")
    
    # Test question and context
    question = "What is machine learning?"
    context_chunks = [
        "[Source: ml_textbook.pdf - page 1]\nMachine learning is a subset of artificial intelligence (AI) that enables computers to learn and make decisions from data without being explicitly programmed.",
        "[Source: ml_textbook.pdf - page 2]\nThere are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning."
    ]
    sources = ["ml_textbook.pdf - page 1", "ml_textbook.pdf - page 2"]
    
    # Generate answer
    response = generator.generate_answer(
        question=question,
        context_chunks=context_chunks,
        sources=sources,
        response_style=ResponseStyle.DETAILED
    )
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {response.answer}")
    print(f"Sources: {response.sources}")
    print(f"Key concepts: {response.key_concepts}")
    print(f"Processing time: {response.processing_time:.2f}s")

if __name__ == "__main__":
    test_answer_generator()