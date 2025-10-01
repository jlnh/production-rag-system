"""
Base Retrieval Classes - Abstract interfaces for retrieval systems

Part of the RAG Production System course implementation.
Module 2: Advanced Retrieval Techniques

License: MIT
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers.

    Defines the common interface that all retrieval systems must implement.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional retrieval parameters

        Returns:
            List of retrieved documents with scores and metadata
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retrieval system.

        Args:
            documents: List of documents to add
        """
        pass

    @abstractmethod
    def remove_documents(self, document_ids: List[str]) -> None:
        """
        Remove documents from the retrieval system.

        Args:
            document_ids: List of document IDs to remove
        """
        pass

    def validate_query(self, query: str) -> bool:
        """
        Validate query input.

        Args:
            query: Query string to validate

        Returns:
            True if query is valid

        Raises:
            ValueError: If query is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > 10000:  # Reasonable limit
            raise ValueError("Query is too long")

        return True

    def format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format retrieval results with consistent structure.

        Args:
            results: Raw retrieval results

        Returns:
            Formatted results with standard fields
        """
        formatted = []

        for result in results:
            formatted_result = {
                'id': result.get('id', ''),
                'content': result.get('content', ''),
                'score': result.get('score', 0.0),
                'metadata': result.get('metadata', {}),
                'retrieval_method': getattr(self, 'method_name', 'unknown')
            }
            formatted.append(formatted_result)

        return formatted


class VectorRetriever(BaseRetriever):
    """
    Vector-based retrieval using semantic embeddings.
    """

    def __init__(self, vector_store, embedding_generator):
        """
        Initialize vector retriever.

        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.method_name = "vector"

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents using vector similarity search.

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents
        """
        self.validate_query(query)

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.embed_query(query)

            # Search vector store
            results = self.vector_store.search(query_embedding, top_k=top_k)

            return self.format_results(results)

        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to vector store."""
        # Generate embeddings for new documents
        documents_with_embeddings = self.embedding_generator.generate_embeddings(documents)

        # Store in vector store
        self.vector_store.store_chunks(documents_with_embeddings)

    def remove_documents(self, document_ids: List[str]) -> None:
        """Remove documents from vector store."""
        self.vector_store.delete(document_ids)


class KeywordRetriever(BaseRetriever):
    """
    Keyword-based retrieval using BM25 or similar algorithms.
    """

    def __init__(self, documents: List[str] = None):
        """
        Initialize keyword retriever.

        Args:
            documents: Initial document collection
        """
        self.method_name = "keyword"
        self._documents = documents or []
        self._bm25_index = None
        self._document_map = {}

        if documents:
            self._build_index()

    def _build_index(self):
        """Build BM25 index from documents."""
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize documents
            tokenized_docs = [doc.split() for doc in self._documents]
            self._bm25_index = BM25Okapi(tokenized_docs)

            # Create document mapping
            for i, doc in enumerate(self._documents):
                self._document_map[i] = doc

        except ImportError:
            raise ImportError("rank_bm25 is required for keyword retrieval. Install with: pip install rank_bm25")

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 keyword matching.

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents
        """
        self.validate_query(query)

        if not self._bm25_index:
            logger.warning("No BM25 index available for keyword search")
            return []

        try:
            # Tokenize query
            query_tokens = query.split()

            # Get BM25 scores
            scores = self._bm25_index.get_scores(query_tokens)

            # Get top-k results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    result = {
                        'id': f"doc_{idx}",
                        'content': self._document_map[idx],
                        'score': scores[idx],
                        'metadata': {'document_index': idx}
                    }
                    results.append(result)

            return self.format_results(results)

        except Exception as e:
            logger.error(f"Keyword retrieval failed: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to keyword index."""
        new_docs = [doc.get('content', '') for doc in documents]
        self._documents.extend(new_docs)
        self._build_index()

    def remove_documents(self, document_ids: List[str]) -> None:
        """Remove documents from keyword index."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated index management
        logger.warning("Document removal not fully implemented for keyword retriever")


class SearchResult:
    """
    Standardized search result representation.
    """

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
        retrieval_method: str = "unknown"
    ):
        """
        Initialize search result.

        Args:
            id: Unique identifier for the document
            content: Document content
            score: Relevance score
            metadata: Additional metadata
            retrieval_method: Method used for retrieval
        """
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.retrieval_method = retrieval_method

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'content': self.content,
            'score': self.score,
            'metadata': self.metadata,
            'retrieval_method': self.retrieval_method
        }

    def __repr__(self) -> str:
        """String representation of search result."""
        return f"SearchResult(id='{self.id}', score={self.score:.3f}, method='{self.retrieval_method}')"