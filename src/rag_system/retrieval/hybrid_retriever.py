"""
Hybrid Retriever - Combines vector and keyword search for better results

Part of the RAG Production System course implementation.
Module 2: Advanced Retrieval Techniques

License: MIT
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from .base import BaseRetriever, VectorRetriever, KeywordRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval system combining vector and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from multiple
    retrieval methods for improved accuracy and coverage.
    """

    def __init__(
        self,
        vector_store,
        embedding_generator,
        documents: List[str] = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            documents: Document collection for keyword search
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
            rrf_k: RRF parameter (typically 60)
        """
        self.vector_retriever = VectorRetriever(vector_store, embedding_generator)
        self.keyword_retriever = KeywordRetriever(documents) if documents else None
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.method_name = "hybrid"

        # Validate weights
        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {vector_weight + keyword_weight}")

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Query string
            top_k: Number of final documents to retrieve
            **kwargs: Additional parameters including:
                - vector_weight_override: Override default vector weight
                - keyword_weight_override: Override default keyword weight
                - fusion_method: 'rrf' or 'weighted' (default: 'rrf')

        Returns:
            List of retrieved documents with combined scores
        """
        self.validate_query(query)

        # Get weight overrides if provided
        vector_weight = kwargs.get("vector_weight_override", self.vector_weight)
        keyword_weight = kwargs.get("keyword_weight_override", self.keyword_weight)
        fusion_method = kwargs.get("fusion_method", "rrf")

        try:
            # Retrieve more candidates for better fusion
            candidate_k = min(top_k * 4, 20)

            # Get results from both methods
            vector_results = self._vector_search(query, candidate_k)
            keyword_results = (
                self._keyword_search(query, candidate_k) if self.keyword_retriever else []
            )

            logger.info(f"Vector search returned {len(vector_results)} results")
            logger.info(f"Keyword search returned {len(keyword_results)} results")

            # Combine results using specified fusion method
            if fusion_method == "rrf":
                combined_results = self._reciprocal_rank_fusion(vector_results, keyword_results)
            else:
                combined_results = self._weighted_fusion(
                    vector_results, keyword_results, vector_weight, keyword_weight
                )

            # Return top-k results
            final_results = combined_results[:top_k]

            logger.info(f"Hybrid search returned {len(final_results)} final results")
            return self.format_results(final_results)

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            # Fallback to vector search only
            logger.info("Falling back to vector search only")
            return self.vector_retriever.retrieve(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform vector search.

        Args:
            query: Query string
            top_k: Number of results to retrieve

        Returns:
            List of vector search results
        """
        try:
            return self.vector_retriever.retrieve(query, top_k)
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform keyword search.

        Args:
            query: Query string
            top_k: Number of results to retrieve

        Returns:
            List of keyword search results
        """
        if not self.keyword_retriever:
            return []

        try:
            return self.keyword_retriever.retrieve(query, top_k)
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []

    def _reciprocal_rank_fusion(
        self, vector_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF formula: score = 1 / (k + rank)
        where k is typically 60 and rank starts from 1.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search

        Returns:
            Combined and ranked results
        """
        combined_scores = {}
        all_documents = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_score = 1 / (self.rrf_k + rank + 1)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
            all_documents[doc_id] = result

        # Process keyword results
        for rank, result in enumerate(keyword_results):
            doc_id = result["id"]
            rrf_score = 1 / (self.rrf_k + rank + 1)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

            # Prefer vector result if we have both
            if doc_id not in all_documents:
                all_documents[doc_id] = result

        # Sort by combined score
        ranked_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        combined_results = []
        for doc_id, score in ranked_ids:
            if doc_id in all_documents:
                result = all_documents[doc_id].copy()
                result["score"] = score
                result["fusion_method"] = "rrf"
                combined_results.append(result)

        return combined_results

    def _weighted_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        vector_weight: float,
        keyword_weight: float,
    ) -> List[Dict[str, Any]]:
        """
        Combine results using weighted score fusion.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            vector_weight: Weight for vector scores
            keyword_weight: Weight for keyword scores

        Returns:
            Combined and ranked results
        """
        combined_scores = {}
        all_documents = {}

        # Normalize scores for each method
        vector_scores = self._normalize_scores(vector_results)
        keyword_scores = self._normalize_scores(keyword_results)

        # Process vector results
        for result, normalized_score in zip(vector_results, vector_scores):
            doc_id = result["id"]
            weighted_score = normalized_score * vector_weight
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weighted_score
            all_documents[doc_id] = result

        # Process keyword results
        for result, normalized_score in zip(keyword_results, keyword_scores):
            doc_id = result["id"]
            weighted_score = normalized_score * keyword_weight
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weighted_score

            if doc_id not in all_documents:
                all_documents[doc_id] = result

        # Sort by combined score
        ranked_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        combined_results = []
        for doc_id, score in ranked_ids:
            if doc_id in all_documents:
                result = all_documents[doc_id].copy()
                result["score"] = score
                result["fusion_method"] = "weighted"
                combined_results.append(result)

        return combined_results

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            results: List of search results

        Returns:
            List of normalized scores
        """
        if not results:
            return []

        scores = [result.get("score", 0.0) for result in results]

        if len(set(scores)) == 1:  # All scores are the same
            return [1.0] * len(scores)

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            return [1.0] * len(scores)

        normalized = [(score - min_score) / score_range for score in scores]
        return normalized

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to both vector and keyword retrievers.

        Args:
            documents: List of documents to add
        """
        # Add to vector retriever
        self.vector_retriever.add_documents(documents)

        # Add to keyword retriever if available
        if self.keyword_retriever:
            self.keyword_retriever.add_documents(documents)

    def remove_documents(self, document_ids: List[str]) -> None:
        """
        Remove documents from both retrievers.

        Args:
            document_ids: List of document IDs to remove
        """
        self.vector_retriever.remove_documents(document_ids)

        if self.keyword_retriever:
            self.keyword_retriever.remove_documents(document_ids)

    def update_weights(self, vector_weight: float, keyword_weight: float) -> None:
        """
        Update the fusion weights.

        Args:
            vector_weight: New weight for vector search
            keyword_weight: New weight for keyword search
        """
        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

        logger.info(f"Updated weights: vector={vector_weight}, keyword={keyword_weight}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.

        Returns:
            Dictionary with retrieval statistics
        """
        return {
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "rrf_k": self.rrf_k,
            "has_keyword_retriever": self.keyword_retriever is not None,
            "method": "hybrid",
        }
