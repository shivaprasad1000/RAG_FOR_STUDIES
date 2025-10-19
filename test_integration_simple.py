"""
Simple integration test for embedding and vector functionality.
"""

import numpy as np
from app.embedding_generator import EmbeddingGenerator
from app.vector_store import VectorStore
from app.models import TextChunk

def test_basic_functionality():
    """Test basic embedding and vector store functionality"""
    try:
        # Test embedding generator
        print("Testing EmbeddingGenerator...")
        generator = EmbeddingGenerator()
        
        # Test single text encoding
        test_text = "This is a test sentence for embedding generation."
        embedding = generator.encode_text(test_text)
        print(f"Generated embedding shape: {embedding.shape}")
        print(f"Embedding dimension: {generator.embedding_dimension}")
        
        # Test batch encoding
        test_texts = [
            "First test sentence.",
            "Second test sentence with different content.",
            "Third sentence for batch testing."
        ]
        batch_embeddings = generator.encode_batch(test_texts)
        print(f"Batch embeddings shape: {batch_embeddings.shape}")
        
        # Test vector store
        print("\nTesting VectorStore...")
        vector_store = VectorStore(embedding_dimension=generator.embedding_dimension)
        
        # Create test chunks
        test_chunks = [
            TextChunk(
                id="chunk-1",
                document_id="doc-1",
                content=test_texts[0],
                chunk_index=0,
                page_number=1
            ),
            TextChunk(
                id="chunk-2",
                document_id="doc-1",
                content=test_texts[1],
                chunk_index=1,
                page_number=1
            ),
            TextChunk(
                id="chunk-3",
                document_id="doc-1",
                content=test_texts[2],
                chunk_index=2,
                page_number=2
            )
        ]
        
        # Add documents to vector store
        vector_store.add_documents(test_chunks, batch_embeddings, "test_document.pdf")
        print(f"Added {len(test_chunks)} chunks to vector store")
        
        # Test search
        query_embedding = generator.encode_text("Test sentence for searching")
        results = vector_store.search(query_embedding, top_k=2)
        print(f"Search returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"Result {i+1}: similarity={result.similarity_score:.3f}, document={result.document_name}")
        
        # Test stats
        stats = vector_store.get_stats()
        print(f"\nVector store stats: {stats}")
        
        print("\n✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()