"""
Production API Components - FastAPI server and endpoints

This module provides production-ready API components including:
- FastAPI application with monitoring and security
- REST endpoints for querying and management
- Dependency injection and service management

License: MIT
"""

from .main import app
from .dependencies import ProductionRAGService, get_rag_service

__all__ = [
    "app",
    "ProductionRAGService",
    "get_rag_service",
]
