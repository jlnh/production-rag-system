"""
Test Configuration - Shared test fixtures and setup

This module provides common test fixtures and configuration for the test suite.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

# Import test modules
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_system.core import DocumentProcessor, EmbeddingGenerator, VectorStore
from rag_system.retrieval import HybridRetriever
from rag_system.evaluation import RetrievalEvaluator


@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Natural language processing helps computers understand and interpret human language effectively.",
        "Deep learning uses neural networks with multiple layers to recognize complex patterns in data.",
        "Retrieval-Augmented Generation combines information retrieval with text generation for better responses.",
        "Vector databases store high-dimensional embeddings for efficient similarity search applications.",
    ]


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Sample document chunks with metadata."""
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "ml_doc.txt", "chunk_id": 0},
            "embedding": [0.1] * 1536,  # Mock embedding
        },
        {
            "content": "Natural language processing helps computers understand language.",
            "metadata": {"source": "nlp_doc.txt", "chunk_id": 0},
            "embedding": [0.2] * 1536,  # Mock embedding
        },
        {
            "content": "Deep learning uses neural networks for pattern recognition.",
            "metadata": {"source": "dl_doc.txt", "chunk_id": 0},
            "embedding": [0.3] * 1536,  # Mock embedding
        },
    ]


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator for testing."""
    mock = Mock(spec=EmbeddingGenerator)
    mock.embed_query.return_value = [0.1] * 1536
    mock.generate_embeddings.return_value = []
    mock.validate_embeddings.return_value = True
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock = Mock(spec=VectorStore)
    mock.store_chunks.return_value = None
    mock.search.return_value = [
        {
            "id": "doc_1",
            "score": 0.95,
            "content": "Machine learning is a subset of AI.",
            "metadata": {"source": "ml_doc.txt"},
        }
    ]
    mock.ping.return_value = True
    return mock


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock = Mock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    mock.delete.return_value = 1
    mock.ping.return_value = True
    mock.incr.return_value = 1
    mock.expire.return_value = True
    return mock


@pytest.fixture
def document_processor():
    """Document processor instance for testing."""
    return DocumentProcessor(chunk_size=100, overlap=20)


@pytest.fixture
def test_queries() -> Dict[str, List[str]]:
    """Test queries with expected relevant documents."""
    return {
        "machine learning artificial intelligence": ["ml_doc.txt"],
        "natural language processing NLP": ["nlp_doc.txt"],
        "deep learning neural networks": ["dl_doc.txt"],
        "vector search similarity": ["vector_doc.txt"],
    }


@pytest.fixture
def retrieval_evaluator(test_queries):
    """Retrieval evaluator with test data."""
    evaluator = RetrievalEvaluator()
    for query, expected_docs in test_queries.items():
        evaluator.add_test_query(query, expected_docs)
    return evaluator


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a sample AI response."
    mock_response.data = [Mock()]
    mock_response.data[0].embedding = [0.1] * 1536
    return mock_response


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_mock_rag_service():
    """Mock RAG service for async testing."""
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.process_query = AsyncMock()
    mock.health_check = AsyncMock()
    mock.cleanup = AsyncMock()

    # Set return values
    mock.health_check.return_value = {
        "healthy": True,
        "services": {"vector_store": {"healthy": True}, "embedding_generator": {"healthy": True}},
    }

    mock.process_query.return_value = {
        "answer": "This is a test answer.",
        "confidence": 0.85,
        "sources": [],
        "metadata": {"query_time": 1.2},
    }

    return mock


class AsyncMock(MagicMock):
    """Helper class for async mocking."""

    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow
