"""
Re-ranker - Improve retrieval results with cross-encoder models

Part of the RAG Production System course implementation.
Module 2: Advanced Retrieval Techniques

License: MIT
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class ReRanker:
    """
    Re-ranking system using cross-encoder models to improve retrieval quality.

    Uses pre-trained cross-encoder models to score query-document pairs
    and re-rank initial retrieval results for better relevance.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Initialize the re-ranker.

        Args:
            model_name: Name of the cross-encoder model to use
            max_length: Maximum sequence length for the model
            batch_size: Batch size for processing multiple pairs
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy initialization of the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded re-ranking model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for re-ranking. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {str(e)}")
                raise

        return self._model

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on query-document relevance.

        Args:
            query: Query string
            documents: List of document dictionaries with 'content' field
            top_k: Number of top documents to return (None = return all)

        Returns:
            Re-ranked list of documents with updated scores

        Raises:
            ValueError: If query is empty or documents are invalid
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            logger.warning("No documents provided for re-ranking")
            return []

        # Validate document format
        for i, doc in enumerate(documents):
            if "content" not in doc:
                raise ValueError(f"Document {i} missing 'content' field")

        start_time = time.time()

        try:
            # If we have fewer documents than requested, return all
            if top_k is None or len(documents) <= top_k:
                if len(documents) <= 10:  # Skip re-ranking for small sets
                    logger.info(f"Skipping re-ranking for {len(documents)} documents (too few)")
                    return documents

            logger.info(f"Re-ranking {len(documents)} documents for query: {query[:50]}...")

            # Prepare query-document pairs
            pairs = self._prepare_pairs(query, documents)

            # Get relevance scores in batches
            scores = self._score_pairs(pairs)

            # Combine documents with new scores
            scored_documents = []
            for doc, score in zip(documents, scores):
                reranked_doc = doc.copy()
                reranked_doc["rerank_score"] = float(score)
                reranked_doc["original_score"] = doc.get("score", 0.0)
                scored_documents.append(reranked_doc)

            # Sort by re-ranking score
            ranked_documents = sorted(
                scored_documents, key=lambda x: x["rerank_score"], reverse=True
            )

            # Return top-k if specified
            if top_k is not None:
                ranked_documents = ranked_documents[:top_k]

            total_time = time.time() - start_time
            logger.info(
                f"Re-ranking completed in {total_time:.2f}s, returned {len(ranked_documents)} documents"
            )

            return ranked_documents

        except Exception as e:
            logger.error(f"Re-ranking failed: {str(e)}")
            # Return original documents on failure
            return documents[:top_k] if top_k else documents

    def _prepare_pairs(self, query: str, documents: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Prepare query-document pairs for the cross-encoder.

        Args:
            query: Query string
            documents: List of document dictionaries

        Returns:
            List of (query, document_content) tuples
        """
        pairs = []

        for doc in documents:
            content = doc.get("content", "").strip()

            # Truncate content if too long
            if len(content) > self.max_length:
                # Try to truncate at sentence boundaries
                sentences = content.split(". ")
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence) > self.max_length - 100:  # Leave some buffer
                        break
                    truncated += sentence + ". "

                content = truncated.strip() if truncated else content[: self.max_length]

            pairs.append((query, content))

        return pairs

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs using the cross-encoder model.

        Args:
            pairs: List of (query, document) tuples

        Returns:
            List of relevance scores
        """
        if not pairs:
            return []

        try:
            # Process in batches for better performance
            all_scores = []

            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                batch_scores = self.model.predict(batch)

                # Convert to list if it's a numpy array
                if hasattr(batch_scores, "tolist"):
                    batch_scores = batch_scores.tolist()
                elif not isinstance(batch_scores, list):
                    batch_scores = [float(score) for score in batch_scores]

                all_scores.extend(batch_scores)

                # Log progress for large batches
                if len(pairs) > 100:
                    progress = min(i + self.batch_size, len(pairs))
                    logger.debug(f"Scored {progress}/{len(pairs)} pairs")

            return all_scores

        except Exception as e:
            logger.error(f"Error scoring pairs: {str(e)}")
            # Return neutral scores on failure
            return [0.5] * len(pairs)

    def rerank_with_fusion(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        alpha: float = 0.7,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank with fusion of original and re-ranking scores.

        Args:
            query: Query string
            documents: List of document dictionaries
            alpha: Weight for re-ranking score (1-alpha for original score)
            top_k: Number of top documents to return

        Returns:
            Re-ranked documents with fused scores
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")

        # First get re-ranking scores
        reranked_docs = self.rerank(query, documents, top_k=None)

        # Normalize both score types for fair fusion
        original_scores = [
            doc.get("original_score", doc.get("score", 0.0)) for doc in reranked_docs
        ]
        rerank_scores = [doc["rerank_score"] for doc in reranked_docs]

        norm_original = self._normalize_scores(original_scores)
        norm_rerank = self._normalize_scores(rerank_scores)

        # Compute fused scores
        for i, doc in enumerate(reranked_docs):
            fused_score = alpha * norm_rerank[i] + (1 - alpha) * norm_original[i]
            doc["fused_score"] = fused_score
            doc["score"] = fused_score  # Update main score

        # Sort by fused score
        fused_ranked = sorted(reranked_docs, key=lambda x: x["fused_score"], reverse=True)

        # Return top-k if specified
        return fused_ranked[:top_k] if top_k else fused_ranked

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: List of scores to normalize

        Returns:
            List of normalized scores
        """
        if not scores or len(set(scores)) <= 1:
            return [1.0] * len(scores)

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            return [1.0] * len(scores)

        return [(score - min_score) / score_range for score in scores]

    def benchmark_model(
        self, test_queries: List[str], test_documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Benchmark the re-ranking model performance.

        Args:
            test_queries: List of test queries
            test_documents: List of test documents

        Returns:
            Dictionary with performance metrics
        """
        if not test_queries or not test_documents:
            return {}

        logger.info(
            f"Benchmarking re-ranker with {len(test_queries)} queries and {len(test_documents)} documents"
        )

        total_time = 0
        total_pairs = 0

        for query in test_queries:
            start_time = time.time()
            self.rerank(query, test_documents, top_k=5)
            total_time += time.time() - start_time
            total_pairs += len(test_documents)

        avg_time_per_query = total_time / len(test_queries)
        avg_time_per_pair = total_time / total_pairs

        return {
            "avg_time_per_query": avg_time_per_query,
            "avg_time_per_pair": avg_time_per_pair,
            "total_queries": len(test_queries),
            "total_documents": len(test_documents),
            "model_name": self.model_name,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "is_loaded": self._model is not None,
        }
