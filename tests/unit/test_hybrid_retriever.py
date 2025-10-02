"""
Unit Tests for Hybrid Retriever

Tests the hybrid retrieval functionality including:
- Vector and keyword search combination
- Reciprocal Rank Fusion (RRF)
- Weighted fusion methods
- Error handling
"""

import pytest
from unittest.mock import Mock, patch

from rag_system.retrieval import HybridRetriever


class TestHybridRetriever:
    """Test cases for HybridRetriever class."""

    @pytest.fixture
    def hybrid_retriever(self, mock_vector_store, mock_embedding_generator):
        """Create a hybrid retriever for testing."""
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand language.",
            "Deep learning uses neural networks for pattern recognition.",
        ]

        return HybridRetriever(
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            documents=documents,
            vector_weight=0.6,
            keyword_weight=0.4,
        )

    def test_init_default_weights(self, mock_vector_store, mock_embedding_generator):
        """Test initialization with default weights."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store, embedding_generator=mock_embedding_generator
        )

        assert retriever.vector_weight == 0.6
        assert retriever.keyword_weight == 0.4
        assert retriever.rrf_k == 60

    def test_init_custom_weights(self, mock_vector_store, mock_embedding_generator):
        """Test initialization with custom weights."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            vector_weight=0.7,
            keyword_weight=0.3,
            rrf_k=50,
        )

        assert retriever.vector_weight == 0.7
        assert retriever.keyword_weight == 0.3
        assert retriever.rrf_k == 50

    def test_retrieve_basic(self, hybrid_retriever):
        """Test basic retrieval functionality."""
        query = "machine learning algorithms"

        # Mock vector search results
        vector_results = [
            {"id": "doc_1", "score": 0.9, "content": "ML content"},
            {"id": "doc_2", "score": 0.8, "content": "AI content"},
        ]

        # Mock keyword search results
        keyword_results = [
            {"id": "doc_1", "score": 2.5, "content": "ML content"},
            {"id": "doc_3", "score": 1.8, "content": "Algorithm content"},
        ]

        with patch.object(hybrid_retriever, "_vector_search", return_value=vector_results):
            with patch.object(hybrid_retriever, "_keyword_search", return_value=keyword_results):
                results = hybrid_retriever.retrieve(query, top_k=3)

        assert len(results) <= 3
        assert all("id" in result for result in results)
        assert all("score" in result for result in results)
        assert all("content" in result for result in results)

    def test_retrieve_empty_query(self, hybrid_retriever):
        """Test retrieval with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid_retriever.retrieve("")

    def test_vector_search(self, hybrid_retriever, mock_embedding_generator):
        """Test vector search functionality."""
        query = "test query"
        mock_embedding_generator.embed_query.return_value = [0.1] * 1536

        with patch.object(hybrid_retriever.vector_store, "search") as mock_search:
            mock_search.return_value = [{"id": "doc_1", "score": 0.9, "content": "Test content"}]

            results = hybrid_retriever._vector_search(query, top_k=5)

            mock_embedding_generator.embed_query.assert_called_once_with(query)
            mock_search.assert_called_once()
            assert len(results) == 1

    def test_keyword_search_no_retriever(self, mock_vector_store, mock_embedding_generator):
        """Test keyword search when no keyword retriever available."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            documents=None,  # No documents for keyword search
        )

        results = retriever._keyword_search("test query", top_k=5)
        assert results == []

    def test_keyword_search_with_documents(self, hybrid_retriever):
        """Test keyword search with documents."""
        query = "machine learning"

        # Mock the BM25 functionality
        with patch.object(hybrid_retriever.keyword_retriever, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = [
                {"id": "doc_0", "score": 2.5, "content": "Machine learning content"}
            ]

            results = hybrid_retriever._keyword_search(query, top_k=5)

            mock_retrieve.assert_called_once_with(query, 5)
            assert len(results) == 1

    def test_reciprocal_rank_fusion(self, hybrid_retriever):
        """Test RRF scoring algorithm."""
        vector_results = [
            {"id": "doc_1", "score": 0.9, "content": "Content 1"},
            {"id": "doc_2", "score": 0.8, "content": "Content 2"},
            {"id": "doc_3", "score": 0.7, "content": "Content 3"},
        ]

        keyword_results = [
            {"id": "doc_3", "score": 2.5, "content": "Content 3"},
            {"id": "doc_1", "score": 2.0, "content": "Content 1"},
            {"id": "doc_4", "score": 1.5, "content": "Content 4"},
        ]

        combined = hybrid_retriever._reciprocal_rank_fusion(vector_results, keyword_results)

        # doc_1 and doc_3 should have higher scores (appear in both lists)
        assert len(combined) == 4  # 4 unique documents
        assert combined[0]["id"] in ["doc_1", "doc_3"]  # Top result should be from both lists

        # Check that fusion_method is set
        assert all(result.get("fusion_method") == "rrf" for result in combined)

    def test_weighted_fusion(self, hybrid_retriever):
        """Test weighted score fusion."""
        vector_results = [
            {"id": "doc_1", "score": 0.9, "content": "Content 1"},
            {"id": "doc_2", "score": 0.7, "content": "Content 2"},
        ]

        keyword_results = [
            {"id": "doc_1", "score": 2.0, "content": "Content 1"},
            {"id": "doc_3", "score": 1.5, "content": "Content 3"},
        ]

        combined = hybrid_retriever._weighted_fusion(vector_results, keyword_results, 0.6, 0.4)

        assert len(combined) == 3  # 3 unique documents
        assert all(result.get("fusion_method") == "weighted" for result in combined)

        # doc_1 should have highest score (appears in both with good scores)
        assert combined[0]["id"] == "doc_1"

    def test_normalize_scores(self, hybrid_retriever):
        """Test score normalization."""
        results = [{"score": 0.9}, {"score": 0.5}, {"score": 0.1}]

        normalized = hybrid_retriever._normalize_scores(results)

        assert len(normalized) == 3
        assert max(normalized) == 1.0  # Max should be 1.0
        assert min(normalized) == 0.0  # Min should be 0.0
        assert 0.0 <= normalized[1] <= 1.0  # Middle value normalized

    def test_normalize_scores_equal(self, hybrid_retriever):
        """Test score normalization with equal scores."""
        results = [{"score": 0.8}, {"score": 0.8}, {"score": 0.8}]

        normalized = hybrid_retriever._normalize_scores(results)

        assert all(score == 1.0 for score in normalized)

    def test_normalize_scores_empty(self, hybrid_retriever):
        """Test score normalization with empty results."""
        normalized = hybrid_retriever._normalize_scores([])
        assert normalized == []

    def test_update_weights(self, hybrid_retriever):
        """Test updating fusion weights."""
        hybrid_retriever.update_weights(0.8, 0.2)

        assert hybrid_retriever.vector_weight == 0.8
        assert hybrid_retriever.keyword_weight == 0.2

    def test_update_weights_invalid_sum(self, hybrid_retriever):
        """Test updating weights with invalid sum."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            hybrid_retriever.update_weights(0.8, 0.3)

    def test_get_stats(self, hybrid_retriever):
        """Test getting retriever statistics."""
        stats = hybrid_retriever.get_stats()

        assert "vector_weight" in stats
        assert "keyword_weight" in stats
        assert "rrf_k" in stats
        assert "has_keyword_retriever" in stats
        assert "method" in stats

        assert stats["vector_weight"] == 0.6
        assert stats["keyword_weight"] == 0.4
        assert stats["method"] == "hybrid"

    def test_retrieve_with_weight_override(self, hybrid_retriever):
        """Test retrieval with weight overrides."""
        query = "test query"

        vector_results = [{"id": "doc_1", "score": 0.9, "content": "Content 1"}]
        keyword_results = [{"id": "doc_2", "score": 2.0, "content": "Content 2"}]

        with patch.object(hybrid_retriever, "_vector_search", return_value=vector_results):
            with patch.object(hybrid_retriever, "_keyword_search", return_value=keyword_results):
                with patch.object(hybrid_retriever, "_weighted_fusion") as mock_fusion:
                    mock_fusion.return_value = vector_results

                    hybrid_retriever.retrieve(
                        query,
                        top_k=5,
                        vector_weight_override=0.9,
                        keyword_weight_override=0.1,
                        fusion_method="weighted",
                    )

                    # Check that weighted fusion was called with overridden weights
                    mock_fusion.assert_called_once()
                    args = mock_fusion.call_args[0]
                    kwargs = mock_fusion.call_args[1] if mock_fusion.call_args[1] else {}

                    # Should be called with overridden weights
                    if len(args) >= 4:
                        assert args[2] == 0.9  # vector_weight
                        assert args[3] == 0.1  # keyword_weight

    def test_retrieve_fallback_on_error(self, hybrid_retriever, mock_embedding_generator):
        """Test fallback to vector search on error."""
        query = "test query"

        # Mock vector search to work
        mock_embedding_generator.embed_query.return_value = [0.1] * 1536
        vector_results = [{"id": "doc_1", "score": 0.9, "content": "Content 1"}]

        with patch.object(hybrid_retriever.vector_store, "search", return_value=vector_results):
            with patch.object(
                hybrid_retriever, "_keyword_search", side_effect=Exception("Keyword error")
            ):
                # Should fallback to vector search
                results = hybrid_retriever.retrieve(query, top_k=5)

                assert len(results) > 0
                assert results[0]["id"] == "doc_1"

    def test_add_documents(self, hybrid_retriever):
        """Test adding documents to both retrievers."""
        new_documents = [{"content": "New document content", "metadata": {"source": "new.txt"}}]

        with patch.object(hybrid_retriever.vector_retriever, "add_documents") as mock_vector_add:
            with patch.object(
                hybrid_retriever.keyword_retriever, "add_documents"
            ) as mock_keyword_add:
                hybrid_retriever.add_documents(new_documents)

                mock_vector_add.assert_called_once_with(new_documents)
                mock_keyword_add.assert_called_once_with(new_documents)

    def test_remove_documents(self, hybrid_retriever):
        """Test removing documents from both retrievers."""
        doc_ids = ["doc_1", "doc_2"]

        with patch.object(
            hybrid_retriever.vector_retriever, "remove_documents"
        ) as mock_vector_remove:
            with patch.object(
                hybrid_retriever.keyword_retriever, "remove_documents"
            ) as mock_keyword_remove:
                hybrid_retriever.remove_documents(doc_ids)

                mock_vector_remove.assert_called_once_with(doc_ids)
                mock_keyword_remove.assert_called_once_with(doc_ids)
