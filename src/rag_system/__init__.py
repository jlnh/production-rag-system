"""
RAG Production System - A comprehensive production-ready RAG implementation

This package provides all the components needed to build, deploy, and maintain
a production-grade Retrieval-Augmented Generation system.

Module 1: Core Implementation
- Document processing and chunking
- Embedding generation
- Vector storage and retrieval
- Basic RAG query processing

Module 2: Advanced Retrieval
- Hybrid search (vector + keyword)
- Smart query analysis
- Re-ranking with cross-encoders
- Evaluation and A/B testing

Module 3: Production Infrastructure
- FastAPI production server
- Rate limiting and caching
- Monitoring and logging
- Quality gates and CI/CD

License: MIT
"""

__version__ = "1.0.0"
__author__ = "RAG Course"
__email__ = "contact@ragcourse.com"

# Core exports
from .core.document_processor import DocumentProcessor
from .core.embedding_generator import EmbeddingGenerator
from .core.vector_store import VectorStore, VectorStoreBase
from .core.query import RAGQueryProcessor, rag_query

# Retrieval exports
from .retrieval.base import BaseRetriever, VectorRetriever, KeywordRetriever, SearchResult
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.smart_retriever import SmartRetriever
from .retrieval.reranker import ReRanker

# Evaluation exports
from .evaluation.retrieval_evaluator import RetrievalEvaluator
from .evaluation.ab_testing import ABTestFramework, ABTestConfig, UserInteraction
from .evaluation.quality_gate import QualityGate, QualityThreshold, QualityGateResult

# Infrastructure exports
from .infrastructure.rate_limiter import RateLimiter
from .infrastructure.cache import QueryCache
from .infrastructure.monitoring import (
    setup_prometheus_metrics,
    query_duration_tracker,
    performance_monitor,
)

# Configuration exports
from .config import RAGConfig, ConfigManager, get_config, get_config_manager

# Utility exports
from .utils.helpers import (
    generate_hash,
    chunk_list,
    clean_text,
    Timer,
    format_duration,
    retry_with_backoff,
)

__all__ = [
    # Core
    "DocumentProcessor",
    "EmbeddingGenerator",
    "VectorStore",
    "VectorStoreBase",
    "RAGQueryProcessor",
    "rag_query",
    # Retrieval
    "BaseRetriever",
    "VectorRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "SmartRetriever",
    "ReRanker",
    "SearchResult",
    # Evaluation
    "RetrievalEvaluator",
    "ABTestFramework",
    "ABTestConfig",
    "UserInteraction",
    "QualityGate",
    "QualityThreshold",
    "QualityGateResult",
    # Infrastructure
    "RateLimiter",
    "QueryCache",
    "setup_prometheus_metrics",
    "query_duration_tracker",
    "performance_monitor",
    # Configuration
    "RAGConfig",
    "ConfigManager",
    "get_config",
    "get_config_manager",
    # Utilities
    "generate_hash",
    "chunk_list",
    "clean_text",
    "Timer",
    "format_duration",
    "retry_with_backoff",
]
