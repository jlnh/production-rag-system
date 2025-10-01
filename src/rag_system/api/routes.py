"""
API Routes - HTTP endpoints for RAG system

Part of the RAG Production System course implementation.
Module 3: Production Engineering & Deployment

License: MIT
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import time
import logging

from .dependencies import (
    get_rag_service,
    get_rate_limiter,
    get_user_id,
    get_query_cache,
    validate_api_key
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="Optional user context")

    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()


class Source(BaseModel):
    """Source document model."""
    id: str
    score: float
    preview: str
    source_file: Optional[str] = None
    chunk_id: Optional[int] = None


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[Source]
    confidence: float = Field(ge=0.0, le=1.0)
    response_time: float
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    query_id: str
    satisfaction_score: int = Field(ge=1, le=5, description="Satisfaction score (1-5)")
    feedback_text: Optional[str] = Field(default=None, max_length=500)
    helpful: Optional[bool] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    request_id: Optional[str] = None


# Query endpoint
@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_user_id),
    rag_service = Depends(get_rag_service),
    rate_limiter = Depends(get_rate_limiter),
    query_cache = Depends(get_query_cache),
    api_key: str = Depends(validate_api_key)
):
    """
    Process a RAG query and return an answer with sources.

    Args:
        request: Query request containing question and parameters
        background_tasks: FastAPI background tasks
        user_id: Authenticated user ID
        rag_service: RAG service instance
        rate_limiter: Rate limiter instance
        query_cache: Query cache instance
        api_key: Validated API key

    Returns:
        QueryResponse with answer, sources, and metadata

    Raises:
        HTTPException: For rate limiting, validation, or processing errors
    """
    start_time = time.time()

    # Rate limiting check
    if not rate_limiter.allow_request(user_id):
        logger.warning(f"Rate limit exceeded for user {user_id}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

    try:
        # Check cache first
        cache_key = query_cache.get_cache_key(
            request.question,
            filters={
                'top_k': request.top_k,
                'include_sources': request.include_sources
            },
            user_id=user_id
        )

        cached_result = query_cache.get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Cache hit for user {user_id}")
            cached_result['metadata']['cached'] = True
            cached_result['metadata']['cache_key'] = cache_key
            return QueryResponse(**cached_result)

        # Process query
        logger.info(f"Processing query for user {user_id}: {request.question[:100]}...")

        result = await rag_service.process_query(
            question=request.question,
            top_k=request.top_k,
            user_id=user_id,
            user_context=request.user_context
        )

        # Format response
        sources = []
        if request.include_sources:
            for source_data in result.get('sources', []):
                source = Source(
                    id=source_data['id'],
                    score=source_data['score'],
                    preview=source_data['preview'],
                    source_file=source_data.get('source_file'),
                    chunk_id=source_data.get('chunk_id')
                )
                sources.append(source)

        # Calculate response time
        response_time = time.time() - start_time

        # Create response
        response = QueryResponse(
            answer=result['answer'],
            sources=sources,
            confidence=result['confidence'],
            response_time=response_time,
            metadata={
                **result['metadata'],
                'user_id': user_id,
                'cached': False,
                'api_version': '1.0'
            }
        )

        # Cache result for future requests
        background_tasks.add_task(
            query_cache.cache_result,
            cache_key,
            response.dict(),
            ttl=3600  # 1 hour
        )

        # Log successful query
        logger.info(
            f"Query processed successfully for user {user_id} in {response_time:.2f}s"
        )

        return response

    except ValueError as e:
        logger.warning(f"Invalid query from user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Query processing failed for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again."
        )


# Feedback endpoint
@router.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_user_id),
    rag_service = Depends(get_rag_service),
    api_key: str = Depends(validate_api_key)
):
    """
    Submit user feedback for a query.

    Args:
        feedback: Feedback data
        background_tasks: FastAPI background tasks
        user_id: Authenticated user ID
        rag_service: RAG service instance
        api_key: Validated API key

    Returns:
        Success message

    Raises:
        HTTPException: For validation or processing errors
    """
    try:
        # Store feedback asynchronously
        background_tasks.add_task(
            rag_service.store_feedback,
            query_id=feedback.query_id,
            user_id=user_id,
            satisfaction_score=feedback.satisfaction_score,
            feedback_text=feedback.feedback_text,
            helpful=feedback.helpful
        )

        logger.info(f"Feedback submitted by user {user_id} for query {feedback.query_id}")

        return {
            "message": "Feedback submitted successfully",
            "query_id": feedback.query_id
        }

    except Exception as e:
        logger.error(f"Failed to submit feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit feedback. Please try again."
        )


# Document upload endpoint
@router.post("/documents/upload", tags=["Documents"])
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_user_id),
    rag_service = Depends(get_rag_service),
    api_key: str = Depends(validate_api_key)
):
    """
    Upload and process a document for the RAG system.

    Args:
        request: HTTP request with file upload
        background_tasks: FastAPI background tasks
        user_id: Authenticated user ID
        rag_service: RAG service instance
        api_key: Validated API key

    Returns:
        Upload status and document ID

    Raises:
        HTTPException: For validation or processing errors
    """
    try:
        # Get uploaded file
        form = await request.form()
        file = form.get("file")

        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate file
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        allowed_types = ['.pdf', '.docx', '.txt', '.md']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_types):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )

        # Process document asynchronously
        document_id = await rag_service.queue_document_processing(
            file_content=await file.read(),
            filename=file.filename,
            user_id=user_id
        )

        logger.info(f"Document upload queued for user {user_id}: {file.filename}")

        return {
            "message": "Document uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "status": "processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload document. Please try again."
        )


# Document status endpoint
@router.get("/documents/{document_id}/status", tags=["Documents"])
async def get_document_status(
    document_id: str,
    user_id: str = Depends(get_user_id),
    rag_service = Depends(get_rag_service),
    api_key: str = Depends(validate_api_key)
):
    """
    Get the processing status of a document.

    Args:
        document_id: Document identifier
        user_id: Authenticated user ID
        rag_service: RAG service instance
        api_key: Validated API key

    Returns:
        Document processing status

    Raises:
        HTTPException: If document not found or access denied
    """
    try:
        status = await rag_service.get_document_status(document_id, user_id)

        if not status:
            raise HTTPException(status_code=404, detail="Document not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document status"
        )


# User query history endpoint
@router.get("/history", tags=["History"])
async def get_query_history(
    limit: int = 10,
    offset: int = 0,
    user_id: str = Depends(get_user_id),
    rag_service = Depends(get_rag_service),
    api_key: str = Depends(validate_api_key)
):
    """
    Get user's query history.

    Args:
        limit: Maximum number of queries to return
        offset: Number of queries to skip
        user_id: Authenticated user ID
        rag_service: RAG service instance
        api_key: Validated API key

    Returns:
        List of historical queries

    Raises:
        HTTPException: For validation or access errors
    """
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")

    try:
        history = await rag_service.get_query_history(
            user_id=user_id,
            limit=limit,
            offset=offset
        )

        return {
            "queries": history,
            "limit": limit,
            "offset": offset,
            "total": len(history)
        }

    except Exception as e:
        logger.error(f"Failed to get query history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve query history"
        )


# System statistics endpoint (admin only)
@router.get("/admin/stats", tags=["Admin"])
async def get_system_stats(
    rag_service = Depends(get_rag_service),
    api_key: str = Depends(validate_api_key)
):
    """
    Get system statistics (admin endpoint).

    Args:
        rag_service: RAG service instance
        api_key: Validated API key (must be admin key)

    Returns:
        System statistics

    Raises:
        HTTPException: For authorization or processing errors
    """
    # TODO: Add admin role validation
    # if not is_admin(api_key):
    #     raise HTTPException(status_code=403, detail="Admin access required")

    try:
        stats = await rag_service.get_system_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get system stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system statistics"
        )