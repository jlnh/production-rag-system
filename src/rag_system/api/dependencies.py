"""
API Dependencies - Dependency injection for FastAPI

Part of the RAG Production System course implementation.
Module 3: Production Engineering & Deployment

License: MIT
"""

from fastapi import HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
import logging
import os
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional, environment variables can still be set manually
    pass

from ..core.query import RAGQueryProcessor
from ..core.embedding_generator import EmbeddingGenerator
from ..core.vector_store import VectorStore
from ..infrastructure.rate_limiter import RateLimiter
from ..infrastructure.cache import QueryCache
from ..infrastructure.monitoring import query_duration_tracker

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global instances (initialized lazily)
_rag_service = None
_rate_limiter = None
_query_cache = None


class ProductionRAGService:
    """
    Production RAG service with comprehensive logging and error handling.
    """

    def __init__(self):
        """Initialize the production RAG service."""
        self.embedding_generator = None
        self.vector_store = None
        self.query_processor = None
        self.initialized = False

    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return

        try:
            logger.info("Initializing RAG service components...")

            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            )

            # Initialize vector store
            vector_backend = os.getenv("VECTOR_STORE_BACKEND", "pinecone")
            vector_config = {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "environment": os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"),
                "index_name": os.getenv("PINECONE_INDEX_NAME", "rag-documents"),
                "dimension": int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            }

            self.vector_store = VectorStore(backend=vector_backend, **vector_config)

            # Initialize query processor
            self.query_processor = RAGQueryProcessor(
                embedding_generator=self.embedding_generator, vector_store=self.vector_store
            )

            self.initialized = True
            logger.info("RAG service initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise

    async def process_query(
        self,
        question: str,
        top_k: int = 5,
        user_id: str = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a query with comprehensive logging and error handling.

        Args:
            question: User's question
            top_k: Number of documents to retrieve
            user_id: User identifier for logging
            user_context: Additional user context

        Returns:
            Dictionary containing answer, sources, and metadata

        Raises:
            ValueError: For invalid input
            Exception: For processing errors
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()
        query_id = f"{user_id}_{int(time.time())}" if user_id else f"anon_{int(time.time())}"

        try:
            logger.info(
                "Query started",
                extra={
                    "query_id": query_id,
                    "user_id": user_id,
                    "question_length": len(question),
                    "top_k": top_k,
                },
            )

            # Validate input
            if not question.strip():
                raise ValueError("Question cannot be empty")

            if top_k < 1 or top_k > 20:
                raise ValueError("top_k must be between 1 and 20")

            # Process query
            with query_duration_tracker():
                result = self.query_processor.query(
                    question=question,
                    top_k=top_k,
                    model=os.getenv("LLM_MODEL", "gpt-4.1"),
                    max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "8000")),
                )

            # Add query tracking metadata
            result["metadata"].update(
                {
                    "query_id": query_id,
                    "user_id": user_id,
                    "processed_at": time.time(),
                    "version": "1.0",
                }
            )

            processing_time = time.time() - start_time

            logger.info(
                "Query completed successfully",
                extra={
                    "query_id": query_id,
                    "user_id": user_id,
                    "processing_time": processing_time,
                    "num_sources": len(result.get("sources", [])),
                    "confidence": result.get("confidence", 0.0),
                },
            )

            return result

        except ValueError as e:
            logger.warning(
                f"Invalid query: {str(e)}", extra={"query_id": query_id, "user_id": user_id}
            )
            raise

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Query processing failed",
                extra={
                    "query_id": query_id,
                    "user_id": user_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time": processing_time,
                },
                exc_info=True,
            )
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of all services.

        Returns:
            Dictionary with health status and service details
        """
        try:
            if not self.initialized:
                await self.initialize()

            services = {}

            # Check vector store
            try:
                vector_healthy = self.vector_store.ping()
                services["vector_store"] = {
                    "healthy": vector_healthy,
                    "backend": self.vector_store.backend,
                }
            except Exception as e:
                services["vector_store"] = {"healthy": False, "error": str(e)}

            # Check embedding service (simple test)
            try:
                test_embedding = self.embedding_generator.embed_query("test")
                services["embedding_generator"] = {
                    "healthy": len(test_embedding) > 0,
                    "model": self.embedding_generator.model,
                }
            except Exception as e:
                services["embedding_generator"] = {"healthy": False, "error": str(e)}

            # Overall health
            all_healthy = all(service.get("healthy", False) for service in services.values())

            return {"healthy": all_healthy, "services": services, "timestamp": time.time()}

        except Exception as e:
            return {"healthy": False, "error": str(e), "timestamp": time.time()}

    async def is_ready(self) -> bool:
        """Check if service is ready to receive traffic."""
        return self.initialized

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up RAG service resources")
        # Add cleanup logic here if needed
        self.initialized = False

    async def store_feedback(
        self,
        query_id: str,
        user_id: str,
        satisfaction_score: int,
        feedback_text: Optional[str] = None,
        helpful: Optional[bool] = None,
    ):
        """Store user feedback for analytics."""
        logger.info(
            "Feedback received",
            extra={
                "query_id": query_id,
                "user_id": user_id,
                "satisfaction_score": satisfaction_score,
                "helpful": helpful,
            },
        )
        # TODO: Implement feedback storage

    async def queue_document_processing(
        self, file_content: bytes, filename: str, user_id: str
    ) -> str:
        """Queue document for processing."""
        document_id = f"doc_{user_id}_{int(time.time())}"
        logger.info(f"Document processing queued: {document_id}")
        # TODO: Implement document processing queue
        return document_id

    async def get_document_status(self, document_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get document processing status."""
        # TODO: Implement document status tracking
        return {"document_id": document_id, "status": "completed", "processed_at": time.time()}

    async def get_query_history(
        self, user_id: str, limit: int, offset: int
    ) -> List[Dict[str, Any]]:
        """Get user's query history."""
        # TODO: Implement query history storage and retrieval
        return []

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        # TODO: Implement system statistics
        return {"total_queries": 0, "active_users": 0, "average_response_time": 0.0}


def get_rag_service() -> ProductionRAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = ProductionRAGService()
    return _rag_service


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_requests=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "100")),
            window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600")),
        )
    return _rate_limiter


def get_query_cache() -> QueryCache:
    """Get or create query cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
    return _query_cache


def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Validate API key from Authorization header.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is invalid
    """
    api_key = credentials.credentials
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")

    if not valid_api_keys or api_key not in valid_api_keys:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=401, detail="Invalid API key", headers={"WWW-Authenticate": "Bearer"}
        )

    return api_key


def get_user_id(request: Request, x_user_id: Optional[str] = Header(None)) -> str:
    """
    Extract user ID from request headers or generate anonymous ID.

    Args:
        request: FastAPI request object
        x_user_id: Optional user ID from X-User-ID header

    Returns:
        User identifier
    """
    if x_user_id:
        return x_user_id

    # Generate anonymous user ID based on request characteristics
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")

    # Simple hash-based anonymous ID (not for security, just for tracking)
    import hashlib

    anonymous_id = hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()[:16]

    return f"anon_{anonymous_id}"


def get_request_id(request: Request) -> str:
    """
    Get or generate request ID for tracing.

    Args:
        request: FastAPI request object

    Returns:
        Request identifier
    """
    # Check if request ID already exists
    if hasattr(request.state, "request_id"):
        return request.state.request_id

    # Generate new request ID
    import uuid

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    return request_id
