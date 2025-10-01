#!/usr/bin/env python3
"""
Module 2 Example - Hybrid Search and Advanced Retrieval

This example demonstrates advanced retrieval techniques:
- Hybrid search (vector + keyword)
- Smart query analysis
- Re-ranking with cross-encoders
- Evaluation metrics

Run this example:
    python examples/module2_hybrid_search.py

Prerequisites:
    - OpenAI API key set as OPENAI_API_KEY environment variable
    - sentence-transformers installed for re-ranking
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

from rag_system.core import EmbeddingGenerator, VectorStore
from rag_system.retrieval import HybridRetriever, SmartRetriever, ReRanker
from rag_system.evaluation import RetrievalEvaluator


def main():
    """Run the hybrid search example."""
    print("🔍 Hybrid Search & Advanced Retrieval Example")
    print("=" * 60)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Sample documents for search
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Natural language processing (NLP) helps computers understand and interpret human language effectively.",
        "Deep learning uses neural networks with multiple layers to recognize complex patterns in data.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation for better AI responses.",
        "Vector databases store high-dimensional embeddings for efficient similarity search in AI applications.",
        "Transformer models revolutionized NLP with their attention mechanism and parallel processing capabilities.",
        "Fine-tuning allows pre-trained models to adapt to specific tasks with minimal additional training data.",
        "Prompt engineering involves crafting effective prompts to get better responses from language models.",
        "Semantic search uses meaning and context rather than exact keyword matching to find relevant information.",
        "Cross-encoder models provide more accurate relevance scoring by processing query-document pairs together."
    ]

    # Initialize components
    print("🔧 Initializing retrieval components...")

    # Embedding generator
    embedding_generator = EmbeddingGenerator(model="text-embedding-3-small")

    # Vector store
    vector_store = VectorStore(backend="chroma", index_name="hybrid-search-example")

    # Create document objects for vector store
    doc_objects = []
    for i, doc in enumerate(documents):
        doc_obj = {
            'content': doc,
            'metadata': {'source': f'doc_{i}.txt', 'chunk_id': i}
        }
        doc_objects.append(doc_obj)

    # Generate embeddings and store
    print("📊 Processing documents...")
    docs_with_embeddings = embedding_generator.generate_embeddings(doc_objects)
    vector_store.store_chunks(docs_with_embeddings)

    # 1. Basic Hybrid Retriever
    print("\n🔄 Testing Hybrid Retriever...")
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        documents=documents,
        vector_weight=0.6,
        keyword_weight=0.4
    )

    test_queries = [
        "machine learning algorithms",  # Should favor keyword search
        "How do neural networks work?",  # Should favor vector search
        "transformer attention mechanism",  # Hybrid approach
        '"deep learning"'  # Exact match (quotes)
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = hybrid_retriever.retrieve(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.3f} | {result['content'][:80]}...")

    # 2. Smart Retriever with Query Analysis
    print("\n🧠 Testing Smart Retriever...")
    smart_retriever = SmartRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        documents=documents
    )

    smart_queries = [
        "error code 404",  # Should detect exact match pattern
        "What is machine learning?",  # Should detect conceptual query
        "how to implement transformer",  # Should detect technical query
        "api documentation setup"  # Should detect technical query
    ]

    for query in smart_queries:
        print(f"\n🎯 Query: {query}")

        # Get strategy explanation
        strategy = smart_retriever.get_query_strategy(query)
        print(f"   Strategy: {strategy['recommended_strategy']} "
              f"(Vector: {strategy['vector_weight']:.1f}, Keyword: {strategy['keyword_weight']:.1f})")

        # Get results
        results = smart_retriever.retrieve(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result['score']:.3f} | {result['content'][:60]}...")

    # 3. Re-ranking with Cross-Encoder
    print("\n🎯 Testing Re-ranking...")
    try:
        reranker = ReRanker(model_name="BAAI/bge-reranker-base")

        query = "deep learning neural networks"
        print(f"Query: {query}")

        # Get initial results
        initial_results = hybrid_retriever.retrieve(query, top_k=6)
        print(f"\n📊 Initial results (top 3):")
        for i, result in enumerate(initial_results[:3], 1):
            print(f"  {i}. Score: {result['score']:.3f} | {result['content'][:60]}...")

        # Re-rank results
        reranked_results = reranker.rerank(query, initial_results, top_k=3)
        print(f"\n🔄 Re-ranked results:")
        for i, result in enumerate(reranked_results, 1):
            original_score = result.get('original_score', 0)
            rerank_score = result.get('rerank_score', 0)
            print(f"  {i}. Original: {original_score:.3f}, Re-rank: {rerank_score:.3f}")
            print(f"     {result['content'][:60]}...")

    except ImportError:
        print("⚠️  Skipping re-ranking (sentence-transformers not available)")
    except Exception as e:
        print(f"⚠️  Re-ranking error: {str(e)}")

    # 4. Evaluation Example
    print("\n📈 Testing Evaluation...")

    # Create simple test queries with expected documents
    evaluator = RetrievalEvaluator()

    # Add test queries (in practice, these would come from a test set)
    evaluator.add_test_query(
        "machine learning artificial intelligence",
        ["doc_0.txt"]  # Document about ML
    )

    evaluator.add_test_query(
        "natural language processing NLP",
        ["doc_1.txt"]  # Document about NLP
    )

    evaluator.add_test_query(
        "neural networks deep learning",
        ["doc_2.txt"]  # Document about deep learning
    )

    try:
        # Evaluate the hybrid retriever
        evaluation_results = evaluator.evaluate_system(hybrid_retriever, top_k_values=[1, 3, 5])

        print("📊 Evaluation Results:")
        print(f"   Precision@3: {evaluation_results['avg_precision@3']:.3f}")
        print(f"   Recall@3: {evaluation_results['avg_recall@3']:.3f}")
        print(f"   MRR: {evaluation_results['avg_mrr']:.3f}")
        print(f"   Queries evaluated: {evaluation_results['num_queries']}")

    except Exception as e:
        print(f"⚠️  Evaluation error: {str(e)}")

    # 5. Query Strategy Explanation
    print("\n🔍 Query Strategy Analysis...")
    analysis_queries = [
        "machine learning",
        '"exact phrase search"',
        "How does attention mechanism work in transformers?",
        "bug fix error 500"
    ]

    for query in analysis_queries:
        print(f"\n📝 Query: {query}")
        explanation = smart_retriever.explain_decision(query)
        print("   " + explanation.replace("\n", "\n   "))

    print("\n🎉 Hybrid search example completed!")
    print("\nKey takeaways:")
    print("- Hybrid search improves retrieval quality by combining vector and keyword search")
    print("- Smart retrievers adapt strategy based on query characteristics")
    print("- Re-ranking provides more accurate relevance scoring")
    print("- Evaluation helps measure and improve retrieval performance")


if __name__ == "__main__":
    main()