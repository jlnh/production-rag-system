"""
Embedding Generator - Generate embeddings for RAG system chunks

Part of the RAG Production System course implementation.
Module 1: RAG Architecture & Core Implementation

License: MIT
"""

from typing import List, Dict, Any, Optional
import logging
import time
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional, environment variables can still be set manually
    pass

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using OpenAI's embedding models.

    Supports batch processing and error handling for production use.
    """

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        """
        Initialize the embedding generator.

        Args:
            model: OpenAI embedding model to use
            batch_size: Number of texts to process in each batch
        """
        self.model = model
        self.batch_size = batch_size
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI()
            except ImportError:
                raise ImportError("OpenAI library is required. Install with: pip install openai")
        return self._client

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of chunk dictionaries with 'content' field

        Returns:
            Updated chunks with 'embedding' field added

        Raises:
            ValueError: If chunks are empty or malformed
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return chunks

        # Validate chunk format
        for i, chunk in enumerate(chunks):
            if "content" not in chunk:
                raise ValueError(f"Chunk {i} missing 'content' field")

        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()

        try:
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]
                self._process_batch(batch)

                # Log progress for large sets
                if len(chunks) > 100:
                    progress = min(i + self.batch_size, len(chunks))
                    logger.info(f"Processed {progress}/{len(chunks)} chunks")

            total_time = time.time() - start_time
            logger.info(f"Embedding generation completed in {total_time:.2f} seconds")

            return chunks

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Process a single batch of chunks.

        Args:
            batch: List of chunk dictionaries to process
        """
        texts = [chunk["content"] for chunk in batch]

        try:
            response = self.client.embeddings.create(input=texts, model=self.model)

            # Add embeddings to chunks
            for chunk, embedding_data in zip(batch, response.data):
                chunk["embedding"] = embedding_data.embedding

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Add None embeddings to maintain consistency
            for chunk in batch:
                chunk["embedding"] = None
            raise

    async def generate_embeddings_async(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings asynchronously for better performance.

        Args:
            chunks: List of chunk dictionaries with 'content' field

        Returns:
            Updated chunks with 'embedding' field added
        """
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings asynchronously for {len(chunks)} chunks")

        # Split into batches
        batches = [chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]

        # Process batches concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self._process_batch, batch) for batch in batches
            ]

            await asyncio.gather(*tasks)

        return chunks

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector

        Raises:
            ValueError: If query is empty
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            response = self.client.embeddings.create(input=[query], model=self.model)

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error embedding query '{query}': {str(e)}")
            raise

    def validate_embeddings(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Validate that all chunks have valid embeddings.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            True if all chunks have valid embeddings
        """
        for i, chunk in enumerate(chunks):
            if "embedding" not in chunk:
                logger.error(f"Chunk {i} missing embedding")
                return False

            if chunk["embedding"] is None:
                logger.error(f"Chunk {i} has null embedding")
                return False

            if not isinstance(chunk["embedding"], list):
                logger.error(f"Chunk {i} has invalid embedding type")
                return False

        logger.info(f"All {len(chunks)} chunks have valid embeddings")
        return True
