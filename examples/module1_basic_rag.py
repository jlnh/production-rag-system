#!/usr/bin/env python3
"""
Module 1 Example - Basic RAG Implementation

This example demonstrates the core RAG functionality:
- Document processing and chunking
- Embedding generation
- Vector storage
- Simple query processing

Run this example:
    python examples/module1_basic_rag.py

Prerequisites:
    - OpenAI API key set as OPENAI_API_KEY environment variable
    - Vector store configured (Pinecone, ChromaDB, etc.)
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, environment variables can still be set manually
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_system.core import DocumentProcessor, EmbeddingGenerator, VectorStore, RAGQueryProcessor


def main():
    """Run the basic RAG example."""
    print("🚀 Basic RAG System Example")
    print("=" * 50)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize components
    print("📝 Initializing RAG components...")

    # 1. Document Processor
    doc_processor = DocumentProcessor(chunk_size=500, overlap=50)
    print("✅ Document processor initialized")

    # 2. Embedding Generator
    embedding_generator = EmbeddingGenerator(
        model="text-embedding-3-small",
        batch_size=10
    )
    print("✅ Embedding generator initialized")

    # 3. Vector Store (using ChromaDB for this example)
    vector_store = VectorStore(
        backend="chroma",
        index_name="basic-rag-example"
    )
    print("✅ Vector store initialized")

    # 4. RAG Query Processor
    query_processor = RAGQueryProcessor(
        embedding_generator=embedding_generator,
        vector_store=vector_store
    )
    print("✅ Query processor initialized")

    # Sample documents
    print("\n📚 Processing sample documents...")
    sample_documents = [
        {
            'content': """
            Machine learning is a subset of artificial intelligence that focuses on the development of algorithms
            and statistical models that enable computer systems to improve their performance on a specific task
            through experience. Unlike traditional programming where explicit instructions are provided, machine
            learning systems learn patterns from data to make predictions or decisions.
            """,
            'metadata': {'source': 'ml_basics.txt', 'chunk_id': 0}
        },
        {
            'content': """
            Natural language processing (NLP) is a field of AI that focuses on the interaction between computers
            and human language. It involves developing algorithms and models that can understand, interpret, and
            generate human language in a valuable way. NLP combines computational linguistics with machine learning
            to process and analyze large amounts of natural language data.
            """,
            'metadata': {'source': 'nlp_intro.txt', 'chunk_id': 0}
        },
        {
            'content': """
            Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers
            to model and understand complex patterns in data. These networks are inspired by the structure and
            function of the human brain. Deep learning has achieved remarkable success in areas such as image
            recognition, speech processing, and natural language understanding.
            """,
            'metadata': {'source': 'deep_learning.txt', 'chunk_id': 0}
        },
        {
            'content': """
            Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with
            text generation. It works by first retrieving relevant documents from a knowledge base, then using
            that retrieved information to generate more accurate and contextually relevant responses. RAG systems
            are particularly useful for question-answering tasks where up-to-date information is crucial.
            """,
            'metadata': {'source': 'rag_overview.txt', 'chunk_id': 0}
        }
    ]

    # Generate embeddings
    print("🔢 Generating embeddings...")
    documents_with_embeddings = embedding_generator.generate_embeddings(sample_documents)
    print(f"✅ Generated embeddings for {len(documents_with_embeddings)} documents")

    # Store in vector database
    print("💾 Storing documents in vector database...")
    vector_store.store_chunks(documents_with_embeddings)
    print("✅ Documents stored successfully")

    # Example queries
    print("\n❓ Running example queries...")
    test_queries = [
        "What is machine learning?",
        "How does natural language processing work?",
        "What is the difference between machine learning and deep learning?",
        "What is RAG and how does it work?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 60)

        try:
            # Process query
            result = query_processor.query(
                question=query,
                top_k=2,
                model="gpt-4.1"
            )

            # Display results
            print(f"💡 Answer: {result['answer']}")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            print(f"⏱️  Processing time: {result['metadata']['query_time']:.2f}s")

            print("\n📄 Sources:")
            for j, source in enumerate(result['sources'], 1):
                print(f"  {j}. Score: {source['score']:.3f} | {source['source_file']}")
                print(f"     Preview: {source['preview'][:100]}...")

        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")

    print("\n🎉 Basic RAG example completed!")
    print("\nNext steps:")
    print("- Try module2_hybrid_search.py for advanced retrieval")
    print("- Check module3_production_api.py for production deployment")


if __name__ == "__main__":
    main()