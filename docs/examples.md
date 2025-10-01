# Examples and Usage Guide

This guide provides comprehensive examples for using the RAG Production System across all modules.

## Quick Reference

| Example | Description | File |
|---------|-------------|------|
| Basic RAG | Core document processing and querying | `examples/module1_basic_rag.py` |
| Hybrid Search | Advanced retrieval with vector + keyword search | `examples/module2_hybrid_search.py` |
| Production API | Full production server with monitoring | `examples/module3_production_api.py` |

## Module 1: Core RAG Implementation

### Basic Document Processing

```python
from rag_system.core import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    chunk_size=500,
    overlap=50,
    min_chunk_size=100
)

# Process different file types
pdf_chunks = processor.process_file("document.pdf")
docx_chunks = processor.process_file("document.docx")
txt_chunks = processor.process_file("document.txt")

# Process from text directly
text_chunks = processor.create_chunks(
    "Your long text content here...",
    metadata={"source": "direct_input"}
)

print(f"Created {len(text_chunks)} chunks")
for chunk in text_chunks[:2]:
    print(f"Chunk: {chunk['content'][:100]}...")
    print(f"Metadata: {chunk['metadata']}")
```

### Embedding Generation

```python
from rag_system.core import EmbeddingGenerator

# Initialize embedding generator
embedding_gen = EmbeddingGenerator(
    model="text-embedding-3-small",
    batch_size=100
)

# Generate embeddings for documents
documents = [
    {"content": "Machine learning is...", "metadata": {"source": "doc1.txt"}},
    {"content": "Natural language processing...", "metadata": {"source": "doc2.txt"}},
]

docs_with_embeddings = embedding_gen.generate_embeddings(documents)

for doc in docs_with_embeddings:
    print(f"Document: {doc['content'][:50]}...")
    print(f"Embedding shape: {len(doc['embedding'])}")
    print(f"Metadata: {doc['metadata']}")
```

### Vector Store Operations

```python
from rag_system.core import VectorStore

# Initialize vector store (Pinecone example)
vector_store = VectorStore(
    backend="pinecone",
    index_name="my-documents"
)

# Store documents with embeddings
vector_store.store_chunks(docs_with_embeddings)

# Search for similar documents
query_embedding = embedding_gen.generate_query_embedding("What is machine learning?")
results = vector_store.search(
    query_embedding=query_embedding,
    top_k=5,
    filter_metadata={"source": "doc1.txt"}  # Optional filtering
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Source: {result['metadata']['source']}")
```

### Complete RAG Query

```python
from rag_system.core import RAGQueryProcessor

# Initialize RAG processor
rag_processor = RAGQueryProcessor(
    embedding_generator=embedding_gen,
    vector_store=vector_store,
    llm_model="gpt-4.1",
    temperature=0.1
)

# Process a query
result = rag_processor.query(
    question="What is machine learning and how does it work?",
    top_k=3,
    context_length=8000
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Processing time: {result['metadata']['query_time']:.2f}s")

print("\nSources:")
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['source_file']} (score: {source['score']:.3f})")
    print(f"     {source['preview'][:100]}...")
```

## Module 2: Advanced Retrieval

### Hybrid Search

```python
from rag_system.retrieval import HybridRetriever

# Initialize hybrid retriever
hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    embedding_generator=embedding_gen,
    documents=document_texts,  # List of document texts for BM25
    vector_weight=0.6,
    keyword_weight=0.4
)

# Search with hybrid approach
results = hybrid_retriever.retrieve(
    query="machine learning algorithms",
    top_k=5
)

for result in results:
    print(f"Hybrid Score: {result['score']:.3f}")
    print(f"Vector Score: {result.get('vector_score', 'N/A')}")
    print(f"Keyword Score: {result.get('keyword_score', 'N/A')}")
    print(f"Content: {result['content'][:100]}...")
```

### Smart Query Analysis

```python
from rag_system.retrieval import SmartRetriever

# Initialize smart retriever
smart_retriever = SmartRetriever(
    vector_store=vector_store,
    embedding_generator=embedding_gen,
    documents=document_texts
)

# Analyze different query types
queries = [
    "machine learning",                    # Conceptual query
    '"exact phrase search"',              # Exact match
    "error code 404",                     # Technical query
    "How does neural networks work?"       # Question format
]

for query in queries:
    # Get query analysis
    strategy = smart_retriever.get_query_strategy(query)
    print(f"\nQuery: {query}")
    print(f"Strategy: {strategy['recommended_strategy']}")
    print(f"Vector weight: {strategy['vector_weight']:.2f}")
    print(f"Keyword weight: {strategy['keyword_weight']:.2f}")
    print(f"Reasoning: {strategy['reasoning']}")

    # Get results with adaptive strategy
    results = smart_retriever.retrieve(query, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result['score']:.3f} | {result['content'][:60]}...")
```

### Re-ranking with Cross-Encoders

```python
from rag_system.retrieval import ReRanker

# Initialize re-ranker
reranker = ReRanker(model_name="BAAI/bge-reranker-base")

query = "deep learning neural networks"

# Get initial results from hybrid search
initial_results = hybrid_retriever.retrieve(query, top_k=10)
print(f"Initial results: {len(initial_results)}")

# Re-rank for better relevance
reranked_results = reranker.rerank(
    query=query,
    documents=initial_results,
    top_k=5
)

print(f"\nRe-ranked results:")
for i, result in enumerate(reranked_results, 1):
    original_score = result.get('original_score', 0)
    rerank_score = result.get('rerank_score', 0)
    print(f"  {i}. Original: {original_score:.3f}, Re-rank: {rerank_score:.3f}")
    print(f"     {result['content'][:80]}...")
```

### Evaluation and Metrics

```python
from rag_system.evaluation import RetrievalEvaluator

# Initialize evaluator
evaluator = RetrievalEvaluator()

# Add test queries with expected relevant documents
test_queries = [
    ("machine learning basics", ["doc1.txt", "doc3.txt"]),
    ("natural language processing", ["doc2.txt"]),
    ("deep learning concepts", ["doc3.txt", "doc4.txt"])
]

for query, relevant_docs in test_queries:
    evaluator.add_test_query(query, relevant_docs)

# Evaluate different retrieval systems
systems = {
    "hybrid": hybrid_retriever,
    "smart": smart_retriever
}

for name, system in systems.items():
    results = evaluator.evaluate_system(system, top_k_values=[1, 3, 5])

    print(f"\n{name.title()} Retriever Results:")
    print(f"  Precision@3: {results['avg_precision@3']:.3f}")
    print(f"  Recall@3: {results['avg_recall@3']:.3f}")
    print(f"  MRR: {results['avg_mrr']:.3f}")
    print(f"  Queries: {results['num_queries']}")
```

### A/B Testing Framework

```python
from rag_system.evaluation import ABTestFramework

# Initialize A/B testing
ab_test = ABTestFramework()

# Define test variants
ab_test.add_variant("control", hybrid_retriever)
ab_test.add_variant("treatment", smart_retriever)

# Simulate user queries
test_queries = [
    "machine learning algorithms",
    "natural language processing",
    "deep learning concepts"
]

for query in test_queries:
    # Assign user to variant
    user_id = f"user_{hash(query) % 1000}"
    variant = ab_test.assign_user(user_id)

    # Get results from assigned variant
    results = ab_test.get_results(variant, query, top_k=3)

    # Record user interaction (simulate)
    ab_test.record_interaction(
        user_id=user_id,
        variant=variant,
        query=query,
        results=results,
        clicked_position=0,  # User clicked first result
        satisfied=True       # User was satisfied
    )

# Analyze results
analysis = ab_test.analyze_results()
print(f"\nA/B Test Results:")
print(f"Control CTR: {analysis['control']['ctr']:.3f}")
print(f"Treatment CTR: {analysis['treatment']['ctr']:.3f}")
print(f"Statistical significance: {analysis['significant']}")
```

## Module 3: Production API

### Production Service Setup

```python
from rag_system.api.dependencies import ProductionRAGService
import asyncio

async def setup_production_service():
    # Initialize production service
    service = ProductionRAGService()

    # Initialize all components
    await service.initialize()

    # Run health check
    health = await service.health_check()
    print(f"Service health: {health['healthy']}")

    # Process queries with full production features
    result = await service.process_query(
        question="What is machine learning?",
        top_k=3,
        user_id="user_123"
    )

    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Cached: {result['metadata']['from_cache']}")

    # Cleanup
    await service.cleanup()

# Run the async function
asyncio.run(setup_production_service())
```

### Rate Limiting

```python
from rag_system.infrastructure import RateLimiter

# Initialize rate limiter
rate_limiter = RateLimiter(
    max_requests=100,
    window_seconds=3600  # 1 hour
)

# Check rate limits
user_id = "user_123"
for i in range(105):  # Test beyond limit
    allowed = rate_limiter.allow_request(user_id)
    if not allowed:
        stats = rate_limiter.get_user_stats(user_id)
        print(f"Rate limit exceeded: {stats['current_requests']}/{stats['max_requests']}")
        break
    print(f"Request {i+1}: Allowed")

# Check remaining quota
remaining = rate_limiter.get_remaining_quota(user_id)
print(f"Remaining quota: {remaining}")
```

### Caching System

```python
from rag_system.infrastructure import QueryCache

# Initialize cache
cache = QueryCache(default_ttl=3600)

# Cache query results
query = "What is machine learning?"
cache_key = cache.get_cache_key(query)

# Simulate expensive query
result = {
    "answer": "Machine learning is a subset of AI...",
    "confidence": 0.85,
    "sources": []
}

# Cache the result
cache.cache_result(cache_key, result, ttl=1800)

# Retrieve from cache
cached_result = cache.get_cached_result(cache_key)
if cached_result:
    print("Cache hit!")
    print(f"Cached answer: {cached_result['answer'][:50]}...")
else:
    print("Cache miss")

# Cache statistics
stats = cache.get_cache_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit ratio: {stats['hit_ratio']:.2%}")
```

### Monitoring and Metrics

```python
from rag_system.infrastructure.monitoring import performance_monitor, get_system_metrics

# Record performance metrics
performance_monitor.record_measurement("query_duration", 2.5)
performance_monitor.record_measurement("embedding_duration", 0.8)
performance_monitor.record_measurement("retrieval_duration", 1.2)

# Get performance summary
summary = performance_monitor.get_performance_summary()
for metric, stats in summary.items():
    if isinstance(stats, dict) and 'avg' in stats:
        print(f"{metric}:")
        print(f"  Average: {stats['avg']:.2f}s")
        print(f"  P95: {stats['p95']:.2f}s")
        print(f"  Count: {stats['count']}")

# Get system health metrics
system_metrics = get_system_metrics()
print(f"\nSystem Health:")
print(f"  CPU Usage: {system_metrics['cpu']['usage']:.1%}")
print(f"  Memory Usage: {system_metrics['memory']['usage']:.1%}")
print(f"  Disk Usage: {system_metrics['disk']['usage']:.1%}")
```

### Quality Gates

```python
from rag_system.evaluation import QualityGate

# Initialize quality gate
quality_gate = QualityGate()

# Add test queries for validation
quality_gate.evaluator.add_test_query(
    "machine learning algorithms",
    ["doc_1", "doc_2"]  # Expected relevant documents
)

# Run comprehensive quality tests
health_result = quality_gate.run_health_tests(service)
performance_result = quality_gate.run_performance_tests(service)
retrieval_result = quality_gate.run_retrieval_tests(hybrid_retriever)

print(f"Health Tests: {'✅ PASSED' if health_result.passed else '❌ FAILED'}")
print(f"Performance Tests: {'✅ PASSED' if performance_result.passed else '❌ FAILED'}")
print(f"Retrieval Tests: {'✅ PASSED' if retrieval_result.passed else '❌ FAILED'}")

# Overall quality gate
overall_result = quality_gate.run_all_tests(hybrid_retriever, service)
if overall_result.passed:
    print("🎉 Quality gate passed - ready for deployment!")
else:
    print("⚠️ Quality gate failed - deployment blocked")
    for issue in overall_result.details.get('issues', []):
        print(f"  - {issue}")
```

## Complete End-to-End Example

```python
import asyncio
from rag_system.core import DocumentProcessor, EmbeddingGenerator, VectorStore, RAGQueryProcessor
from rag_system.retrieval import HybridRetriever
from rag_system.infrastructure import QueryCache, RateLimiter
from rag_system.evaluation import QualityGate

async def complete_rag_workflow():
    """Complete end-to-end RAG workflow example."""

    print("🚀 Starting complete RAG workflow...")

    # 1. Document Processing
    print("📄 Processing documents...")
    processor = DocumentProcessor(chunk_size=500, overlap=50)

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence...",
        "Natural language processing enables computers to understand human language...",
        "Deep learning uses neural networks with multiple layers..."
    ]

    all_chunks = []
    for i, doc in enumerate(documents):
        chunks = processor.create_chunks(doc, metadata={"source": f"doc_{i}.txt"})
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks")

    # 2. Embedding Generation
    print("🔢 Generating embeddings...")
    embedding_gen = EmbeddingGenerator(model="text-embedding-3-small")
    chunks_with_embeddings = embedding_gen.generate_embeddings(all_chunks)

    # 3. Vector Storage
    print("💾 Storing in vector database...")
    vector_store = VectorStore(backend="chroma", index_name="complete-example")
    vector_store.store_chunks(chunks_with_embeddings)

    # 4. Setup Advanced Retrieval
    print("🔍 Setting up hybrid retrieval...")
    document_texts = [chunk['content'] for chunk in all_chunks]
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_gen,
        documents=document_texts,
        vector_weight=0.6,
        keyword_weight=0.4
    )

    # 5. Setup Production Infrastructure
    print("🏭 Setting up production infrastructure...")
    cache = QueryCache(default_ttl=1800)
    rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

    # 6. Setup Quality Gate
    print("🛡️ Setting up quality gate...")
    quality_gate = QualityGate()
    quality_gate.evaluator.add_test_query(
        "machine learning", ["doc_0.txt"]
    )

    # 7. Process Queries with Full Pipeline
    print("❓ Processing queries...")
    queries = [
        "What is machine learning?",
        "How does natural language processing work?",
        "Explain deep learning concepts"
    ]

    user_id = "demo_user"

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")

        # Check rate limit
        if not rate_limiter.allow_request(user_id):
            print("🚫 Rate limit exceeded")
            continue

        # Check cache
        cache_key = cache.get_cache_key(query)
        cached_result = cache.get_cached_result(cache_key)

        if cached_result:
            print("💾 Retrieved from cache")
            result = cached_result
        else:
            # Process with hybrid retrieval
            retrieval_results = hybrid_retriever.retrieve(query, top_k=3)

            # Simulate LLM processing
            result = {
                "answer": f"Based on the retrieved documents, {query.lower()} refers to...",
                "confidence": 0.85,
                "sources": [
                    {
                        "content": r['content'][:100] + "...",
                        "score": r['score'],
                        "source": r.get('metadata', {}).get('source', 'unknown')
                    }
                    for r in retrieval_results
                ]
            }

            # Cache the result
            cache.cache_result(cache_key, result, ttl=1800)

        # Display results
        print(f"💡 Answer: {result['answer'][:100]}...")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"📚 Sources: {len(result['sources'])} documents")

    # 8. Run Quality Gate
    print("\n🛡️ Running quality gate...")
    quality_result = quality_gate.run_retrieval_tests(hybrid_retriever)

    if quality_result.passed:
        print("✅ Quality gate passed!")
    else:
        print("❌ Quality gate failed")

    # 9. Display Statistics
    print(f"\n📊 Session Statistics:")
    cache_stats = cache.get_cache_stats()
    rate_stats = rate_limiter.get_user_stats(user_id)

    print(f"  Cache hits: {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Rate limit usage: {rate_stats['current_requests']}/{rate_stats['max_requests']}")

    print("\n🎉 Complete RAG workflow finished!")

# Run the complete example
if __name__ == "__main__":
    asyncio.run(complete_rag_workflow())
```

## Running the Examples

### Command Line Usage

```bash
# Basic RAG example
python examples/module1_basic_rag.py

# Hybrid search example
python examples/module2_hybrid_search.py

# Production API example
python examples/module3_production_api.py

# Run with custom configuration
OPENAI_API_KEY=your-key python examples/module1_basic_rag.py

# Run in dry-run mode (no API calls)
python examples/module1_basic_rag.py --dry-run
```

### Jupyter Notebooks

Interactive examples are available as Jupyter notebooks:

```bash
# Start Jupyter
jupyter lab

# Open notebooks
# - examples/notebooks/01_document_processing.ipynb
# - examples/notebooks/02_retrieval_optimization.ipynb
# - examples/notebooks/03_evaluation_metrics.ipynb
```

## Best Practices

1. **Start with Module 1** to understand core concepts
2. **Use hybrid search** for better retrieval quality
3. **Implement caching** for production performance
4. **Set up monitoring** to track system health
5. **Use quality gates** before deployment
6. **Test with real data** to validate performance

## Next Steps

- **[Installation Guide](installation.md)** - Set up your environment
- **[Configuration Guide](configuration.md)** - Configure for your use case
- **[Deployment Guide](deployment.md)** - Deploy to production
- **[Architecture Guide](architecture.md)** - Understand system design