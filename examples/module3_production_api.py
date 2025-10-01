#!/usr/bin/env python3
"""
Module 3 Example - Production API Server

This example demonstrates production deployment:
- FastAPI server with monitoring
- Rate limiting and caching
- Health checks and metrics
- Quality gates

Run this example:
    python examples/module3_production_api.py

Or run the server directly:
    uvicorn rag_system.api.main:app --reload

Prerequisites:
    - Redis server running (for caching and rate limiting)
    - OpenAI API key set as OPENAI_API_KEY environment variable
"""

import os
import sys
import asyncio
import time
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

import httpx
from rag_system.api.dependencies import ProductionRAGService
from rag_system.infrastructure import RateLimiter, QueryCache, setup_prometheus_metrics
from rag_system.evaluation import QualityGate
from rag_system.core import EmbeddingGenerator, VectorStore
from rag_system.retrieval import HybridRetriever


async def test_production_service():
    """Test the production RAG service."""
    print("🏭 Production RAG Service Example")
    print("=" * 50)

    # 1. Initialize Production Service
    print("🔧 Initializing production service...")
    rag_service = ProductionRAGService()

    try:
        await rag_service.initialize()
        print("✅ Production service initialized")
    except Exception as e:
        print(f"❌ Failed to initialize service: {str(e)}")
        print("Make sure Redis and API keys are configured")
        return

    # 2. Health Check
    print("\n🏥 Running health checks...")
    health_result = await rag_service.health_check()

    if health_result['healthy']:
        print("✅ All services healthy")
        for service_name, status in health_result['services'].items():
            health_emoji = "✅" if status['healthy'] else "❌"
            print(f"   {health_emoji} {service_name}: {'OK' if status['healthy'] else 'FAILED'}")
    else:
        print("⚠️  Some services are unhealthy")

    # 3. Test Rate Limiting
    print("\n⚡ Testing rate limiting...")
    rate_limiter = RateLimiter(max_requests=5, window_seconds=10)

    user_id = "test_user_123"
    for i in range(7):
        allowed = rate_limiter.allow_request(user_id)
        status = "✅ ALLOWED" if allowed else "🚫 BLOCKED"
        print(f"   Request {i+1}: {status}")

    # Get rate limit stats
    stats = rate_limiter.get_user_stats(user_id)
    print(f"   Rate limit stats: {stats['current_requests']}/{stats['max_requests']} requests used")

    # 4. Test Caching
    print("\n💾 Testing query caching...")
    cache = QueryCache(default_ttl=60)

    try:
        # Test cache operations
        cache_key = cache.get_cache_key("What is machine learning?")
        print(f"   Cache key generated: {cache_key[:20]}...")

        # Simulate caching a result
        sample_result = {
            'answer': 'Machine learning is a subset of AI...',
            'confidence': 0.85,
            'sources': []
        }

        cache.cache_result(cache_key, sample_result, ttl=60)
        print("   ✅ Result cached")

        # Retrieve from cache
        cached_result = cache.get_cached_result(cache_key)
        if cached_result:
            print("   ✅ Cache hit - result retrieved")
        else:
            print("   ❌ Cache miss")

    except Exception as e:
        print(f"   ⚠️  Cache test failed: {str(e)}")

    # 5. Test Query Processing
    print("\n❓ Testing query processing...")
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning concepts"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")

        try:
            start_time = time.time()
            result = await rag_service.process_query(
                question=query,
                top_k=3,
                user_id=f"test_user_{i}"
            )

            processing_time = time.time() - start_time
            print(f"   ⏱️  Processing time: {processing_time:.2f}s")
            print(f"   🎯 Confidence: {result['confidence']:.2f}")
            print(f"   📄 Sources: {len(result['sources'])} documents")
            print(f"   💡 Answer preview: {result['answer'][:100]}...")

        except Exception as e:
            print(f"   ❌ Query failed: {str(e)}")

    # 6. Monitoring Example
    print("\n📊 Testing monitoring...")
    setup_prometheus_metrics()

    try:
        from rag_system.infrastructure.monitoring import performance_monitor, get_system_metrics

        # Record some sample measurements
        performance_monitor.record_measurement("query_duration", 1.5)
        performance_monitor.record_measurement("query_duration", 2.1)
        performance_monitor.record_measurement("embedding_duration", 0.8)

        # Get performance summary
        summary = performance_monitor.get_performance_summary()
        print("   📈 Performance metrics:")
        for metric, stats in summary.items():
            if isinstance(stats, dict) and 'avg' in stats:
                print(f"      {metric}: avg={stats['avg']:.2f}s, p95={stats['p95']:.2f}s")

        # Get system metrics
        system_metrics = get_system_metrics()
        print(f"   🖥️  System health: {'✅ Healthy' if system_metrics['health']['healthy'] else '⚠️  Issues detected'}")

    except Exception as e:
        print(f"   ⚠️  Monitoring test failed: {str(e)}")

    # 7. Quality Gate Example
    print("\n🛡️  Testing quality gates...")

    try:
        # Initialize components for quality testing
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore(backend="chroma", index_name="quality-test")
        retriever = HybridRetriever(vector_store, embedding_generator, [])

        # Create quality gate
        quality_gate = QualityGate()

        # Add a simple test query
        quality_gate.evaluator.add_test_query(
            "machine learning algorithms",
            ["doc_1", "doc_2"]  # Expected relevant documents
        )

        print("   🔍 Running quality tests...")

        # Run health tests only (retrieval tests need actual data)
        health_result = quality_gate.run_health_tests(rag_service)
        performance_result = quality_gate.run_performance_tests(rag_service)

        print(f"   🏥 Health tests: {'✅ PASSED' if health_result.passed else '❌ FAILED'}")
        print(f"   ⚡ Performance tests: {'✅ PASSED' if performance_result.passed else '❌ FAILED'}")

        if performance_result.passed:
            details = performance_result.details
            avg_time = details.get('avg_response_time', 0)
            success_rate = details.get('success_rate', 0)
            print(f"      Average response time: {avg_time:.2f}s")
            print(f"      Success rate: {success_rate:.1%}")

    except Exception as e:
        print(f"   ⚠️  Quality gate test failed: {str(e)}")

    # 8. Cleanup
    print("\n🧹 Cleaning up...")
    await rag_service.cleanup()
    print("✅ Cleanup completed")

    print("\n🎉 Production API example completed!")
    print("\nNext steps:")
    print("- Start the full API server: uvicorn rag_system.api.main:app --reload")
    print("- Visit http://localhost:8000/docs for API documentation")
    print("- Check http://localhost:8000/health for health status")
    print("- Monitor metrics at http://localhost:8000/metrics")


async def test_api_client():
    """Test the API using HTTP client."""
    print("\n🌐 Testing API Client...")

    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        try:
            # Test health endpoint
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("   ✅ Health check passed")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")

            # Test metrics endpoint
            response = await client.get(f"{base_url}/metrics")
            if response.status_code == 200:
                print("   ✅ Metrics endpoint accessible")
            else:
                print(f"   ⚠️  Metrics endpoint: {response.status_code}")

        except httpx.ConnectError:
            print("   ⚠️  API server not running. Start with:")
            print("      uvicorn rag_system.api.main:app --reload")


def main():
    """Run the production API example."""
    print("Starting production service tests...")

    # Test the service
    asyncio.run(test_production_service())

    # Test API client (if server is running)
    asyncio.run(test_api_client())

    print("\n📚 Documentation:")
    print("- API docs: http://localhost:8000/docs")
    print("- Health: http://localhost:8000/health")
    print("- Metrics: http://localhost:8000/metrics")


if __name__ == "__main__":
    main()