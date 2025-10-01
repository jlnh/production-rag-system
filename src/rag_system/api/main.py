"""
Production FastAPI Application - Main entry point for RAG API

Part of the RAG Production System course implementation.
Module 3: Production Engineering & Deployment

License: MIT
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
import logging
from contextlib import asynccontextmanager

from .routes import router
from .dependencies import get_rag_service, RateLimiter
from ..infrastructure.monitoring import setup_prometheus_metrics, REQUEST_COUNT, REQUEST_DURATION
from ..infrastructure.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    logger.info("Starting RAG Production API")
    setup_logging()
    setup_prometheus_metrics(app)

    # Initialize services
    rag_service = get_rag_service()
    await rag_service.initialize()

    logger.info("RAG API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down RAG API")
    await rag_service.cleanup()
    logger.info("RAG API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Production RAG API",
    description="Production-ready Retrieval-Augmented Generation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Monitoring middleware
@app.middleware("http")
async def add_monitoring(request: Request, call_next):
    """Add monitoring and metrics to all requests."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Record metrics
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()

    # Add timing header
    response.headers["X-Process-Time"] = str(duration)

    return response


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancer and monitoring.

    Returns:
        Dictionary with service health status
    """
    try:
        # Check all dependencies
        rag_service = get_rag_service()
        health_status = await rag_service.health_check()

        if health_status["healthy"]:
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "1.0.0",
                "services": health_status["services"]
            }
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Service unhealthy: {health_status['error']}"
            )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


# Readiness check for Kubernetes
@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check for Kubernetes deployments.

    Returns:
        Dictionary indicating if service is ready to receive traffic
    """
    try:
        rag_service = get_rag_service()
        is_ready = await rag_service.is_ready()

        if is_ready:
            return {"status": "ready", "timestamp": time.time()}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")

    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")


# Liveness check for Kubernetes
@app.get("/live", tags=["Health"])
async def liveness_check():
    """
    Liveness check for Kubernetes deployments.

    Returns:
        Simple response indicating service is alive
    """
    return {"status": "alive", "timestamp": time.time()}


# Metrics endpoint for Prometheus
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics
    """
    try:
        from prometheus_client import generate_latest
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


# Include API routes
app.include_router(router, prefix="/api/v1")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Args:
        request: The request that caused the exception
        exc: The exception that was raised

    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


# Rate limiting exception handler
@app.exception_handler(HTTPException)
async def rate_limit_exception_handler(request: Request, exc: HTTPException):
    """
    Custom handler for rate limiting exceptions.

    Args:
        request: The request that was rate limited
        exc: The HTTP exception

    Returns:
        JSON error response with rate limit information
    """
    if exc.status_code == 429:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": 60
            },
            headers={"Retry-After": "60"}
        )

    # Re-raise other HTTP exceptions
    raise exc


if __name__ == "__main__":
    import uvicorn

    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )