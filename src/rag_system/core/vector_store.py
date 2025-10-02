"""
Vector Store - Storage and retrieval for document embeddings

Part of the RAG Production System course implementation.
Module 1: RAG Architecture & Core Implementation

License: MIT
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from abc import ABC, abstractmethod

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional, environment variables can still be set manually
    pass

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks with embeddings in the vector store."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using query embedding."""
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by their IDs."""
        pass

    @abstractmethod
    def ping(self) -> bool:
        """Check if the vector store is healthy."""
        pass


class VectorStore(VectorStoreBase):
    """
    Production vector store implementation.

    Supports multiple backends including Pinecone, Weaviate, and ChromaDB.
    """

    def __init__(self, backend: str = "pinecone", **kwargs: Any) -> None:
        """
        Initialize the vector store.

        Args:
            backend: Vector store backend ('pinecone', 'weaviate', 'chroma')
            **kwargs: Backend-specific configuration
        """
        self.backend = backend
        self.config = kwargs
        self._client: Any = None
        self._index: Any = None

    @property
    def client(self) -> Any:
        """Lazy initialization of vector store client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def index(self) -> Any:
        """Get or create the vector index."""
        if self._index is None:
            self._index = self._get_or_create_index()
        return self._index

    def _create_client(self) -> Any:
        """Create the appropriate vector store client."""
        if self.backend == "pinecone":
            return self._create_pinecone_client()
        elif self.backend == "weaviate":
            return self._create_weaviate_client()
        elif self.backend == "chroma":
            return self._create_chroma_client()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _create_pinecone_client(self) -> Any:
        """Create Pinecone client."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.config.get("api_key"))
            return pc

        except ImportError:
            raise ImportError("Pinecone library required. Install with: pip install pinecone")

    def _create_weaviate_client(self) -> Any:
        """Create Weaviate client."""
        try:
            import weaviate

            return weaviate.Client(
                url=self.config.get("url", "http://localhost:8080"),
                auth_client_secret=self.config.get("auth_config"),
            )

        except ImportError:
            raise ImportError(
                "Weaviate library required. Install with: pip install weaviate-client"
            )

    def _create_chroma_client(self) -> Any:
        """Create ChromaDB client."""
        try:
            import chromadb

            return chromadb.Client()

        except ImportError:
            raise ImportError("ChromaDB library required. Install with: pip install chromadb")

    def _get_or_create_index(self) -> Any:
        """Get or create the vector index."""
        index_name = self.config.get("index_name", "rag-documents")

        if self.backend == "pinecone":
            return self._get_pinecone_index(index_name)
        elif self.backend == "weaviate":
            return self._get_weaviate_collection(index_name)
        elif self.backend == "chroma":
            return self._get_chroma_collection(index_name)
        return None

    def _get_pinecone_index(self, index_name: str) -> Any:
        """Get or create Pinecone index."""
        from pinecone import ServerlessSpec

        existing_indexes = [idx.name for idx in self.client.list_indexes()]

        if index_name not in existing_indexes:
            self.client.create_index(
                name=index_name,
                dimension=self.config.get("dimension", 1536),
                metric=self.config.get("metric", "cosine"),
                spec=ServerlessSpec(
                    cloud=self.config.get("cloud", "aws"),
                    region=self.config.get("region", "us-east-1"),
                ),
            )
            logger.info(f"Created Pinecone index: {index_name}")

        return self.client.Index(index_name)

    def _get_weaviate_collection(self, collection_name: str) -> Any:
        """Get or create Weaviate collection."""
        # Implementation would depend on specific Weaviate setup
        return collection_name

    def _get_chroma_collection(self, collection_name: str) -> Any:
        """Get or create ChromaDB collection."""
        return self.client.get_or_create_collection(name=collection_name)

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Store chunks with embeddings in the vector store.

        Args:
            chunks: List of chunks with embeddings and metadata

        Raises:
            ValueError: If chunks don't have required fields
        """
        if not chunks:
            logger.warning("No chunks provided for storage")
            return

        # Validate chunks
        for i, chunk in enumerate(chunks):
            if "embedding" not in chunk:
                raise ValueError(f"Chunk {i} missing embedding")
            if "metadata" not in chunk:
                raise ValueError(f"Chunk {i} missing metadata")

        try:
            if self.backend == "pinecone":
                self._store_pinecone(chunks)
            elif self.backend == "chroma":
                self._store_chroma(chunks)
            else:
                raise NotImplementedError(f"Storage not implemented for {self.backend}")

            logger.info(f"Successfully stored {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise

    def _store_pinecone(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks in Pinecone."""
        vectors = []

        for chunk in chunks:
            chunk_id = f"{chunk['metadata']['source']}#{chunk['metadata']['chunk_id']}"
            vector = {
                "id": chunk_id,
                "values": chunk["embedding"],
                "metadata": {**chunk["metadata"], "content": chunk["content"]},
            }
            vectors.append(vector)

        # Batch upsert for better performance
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)

    def _store_chroma(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks in ChromaDB."""
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"{chunk['metadata']['source']}#{chunk['metadata']['chunk_id']}"
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["content"])
            metadatas.append(chunk["metadata"])

        self.index.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding.

        Args:
            query_embedding: Query vector to search with
            top_k: Number of top results to return

        Returns:
            List of matching chunks with similarity scores

        Raises:
            ValueError: If query_embedding is invalid
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        try:
            if self.backend == "pinecone":
                return self._search_pinecone(query_embedding, top_k)
            elif self.backend == "chroma":
                return self._search_chroma(query_embedding, top_k)
            else:
                raise NotImplementedError(f"Search not implemented for {self.backend}")

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    def _search_pinecone(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search Pinecone index."""
        response = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        results = []
        for match in response.matches:
            result = {
                "id": match.id,
                "score": match.score,
                "content": match.metadata.get("content", ""),
                "metadata": match.metadata,
            }
            results.append(result)

        return results

    def _search_chroma(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search ChromaDB collection."""
        response = self.index.query(query_embeddings=[query_embedding], n_results=top_k)

        results = []
        for i in range(len(response["ids"][0])):
            result = {
                "id": response["ids"][0][i],
                "score": 1 - response["distances"][0][i],  # Convert distance to similarity
                "content": response["documents"][0][i],
                "metadata": response["metadatas"][0][i],
            }
            results.append(result)

        return results

    def delete(self, chunk_ids: List[str]) -> None:
        """
        Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not chunk_ids:
            return

        try:
            if self.backend == "pinecone":
                self.index.delete(ids=chunk_ids)
            elif self.backend == "chroma":
                self.index.delete(ids=chunk_ids)

            logger.info(f"Deleted {len(chunk_ids)} chunks")

        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}")
            raise

    def ping(self) -> bool:
        """
        Check if the vector store is healthy.

        Returns:
            True if the store is accessible and responsive
        """
        try:
            if self.backend == "pinecone":
                # Try to get index stats
                self.index.describe_index_stats()
                return True
            elif self.backend == "chroma":
                # Try to list collections
                self.client.list_collections()
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return False
