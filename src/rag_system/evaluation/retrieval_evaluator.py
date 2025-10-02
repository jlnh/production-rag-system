"""
Retrieval Evaluator - Measure and evaluate retrieval system performance

Part of the RAG Production System course implementation.
Module 2: Advanced Retrieval Techniques

License: MIT
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import json
import time
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Comprehensive evaluation system for retrieval performance.

    Provides metrics including precision, recall, MRR, NDCG,
    and other standard information retrieval evaluation measures.
    """

    def __init__(self, test_queries_path: Optional[str] = None):
        """
        Initialize the retrieval evaluator.

        Args:
            test_queries_path: Path to test queries file (JSON format)
        """
        self.test_queries_path = test_queries_path
        self._test_queries = None

    @property
    def test_queries(self) -> Dict[str, List[str]]:
        """Load test queries lazily."""
        if self._test_queries is None:
            self._test_queries = self.load_test_queries()
        return self._test_queries

    def load_test_queries(self) -> Dict[str, List[str]]:
        """
        Load test queries from file or create default set.

        Returns:
            Dictionary mapping queries to lists of relevant document IDs

        Format:
        {
            "query_text": ["doc_id_1", "doc_id_2", ...],
            ...
        }
        """
        if self.test_queries_path and Path(self.test_queries_path).exists():
            try:
                with open(self.test_queries_path, "r", encoding="utf-8") as f:
                    queries = json.load(f)
                    logger.info(f"Loaded {len(queries)} test queries from {self.test_queries_path}")
                    return queries
            except Exception as e:
                logger.error(f"Error loading test queries: {str(e)}")

        # Return empty dict if no test queries available
        logger.warning("No test queries available. Use add_test_query() to add queries.")
        return {}

    def add_test_query(self, query: str, relevant_doc_ids: List[str]) -> None:
        """
        Add a test query with known relevant documents.

        Args:
            query: Query string
            relevant_doc_ids: List of relevant document IDs
        """
        if self._test_queries is None:
            self._test_queries = {}

        self._test_queries[query] = relevant_doc_ids
        logger.info(f"Added test query: '{query}' with {len(relevant_doc_ids)} relevant docs")

    def save_test_queries(self, output_path: str) -> None:
        """
        Save test queries to file.

        Args:
            output_path: Path to save test queries JSON file
        """
        if not self.test_queries:
            logger.warning("No test queries to save")
            return

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.test_queries, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.test_queries)} test queries to {output_path}")
        except Exception as e:
            logger.error(f"Error saving test queries: {str(e)}")

    def evaluate_system(
        self,
        retriever,
        top_k_values: List[int] = [1, 3, 5, 10],
        queries: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval system performance.

        Args:
            retriever: Retrieval system to evaluate
            top_k_values: List of k values for precision@k and recall@k
            queries: Custom queries to use (defaults to self.test_queries)

        Returns:
            Dictionary containing evaluation metrics
        """
        evaluation_queries = queries or self.test_queries

        if not evaluation_queries:
            raise ValueError("No test queries available for evaluation")

        logger.info(f"Evaluating retrieval system with {len(evaluation_queries)} queries")

        start_time = time.time()
        results = []

        for query, expected_docs in evaluation_queries.items():
            try:
                # Retrieve documents
                retrieved = retriever.retrieve(query, top_k=max(top_k_values))
                retrieved_ids = [doc.get("id", "") for doc in retrieved]

                # Calculate metrics for this query
                query_metrics = self._calculate_query_metrics(
                    retrieved_ids, expected_docs, top_k_values
                )
                query_metrics["query"] = query
                query_metrics["num_expected"] = len(expected_docs)
                query_metrics["num_retrieved"] = len(retrieved_ids)

                results.append(query_metrics)

            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {str(e)}")
                continue

        evaluation_time = time.time() - start_time

        if not results:
            raise ValueError("No queries could be evaluated successfully")

        # Aggregate results
        aggregated_metrics = self._aggregate_metrics(results, top_k_values)
        aggregated_metrics["evaluation_time"] = evaluation_time
        aggregated_metrics["num_queries"] = len(results)
        aggregated_metrics["individual_results"] = results

        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        return aggregated_metrics

    def _calculate_query_metrics(
        self, retrieved_ids: List[str], expected_ids: List[str], top_k_values: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a single query.

        Args:
            retrieved_ids: List of retrieved document IDs
            expected_ids: List of expected relevant document IDs
            top_k_values: List of k values for precision@k and recall@k

        Returns:
            Dictionary containing metrics for this query
        """
        expected_set = set(expected_ids)
        metrics = {}

        # Calculate precision@k and recall@k for each k value
        for k in top_k_values:
            retrieved_at_k = set(retrieved_ids[:k])
            relevant_at_k = retrieved_at_k & expected_set

            # Precision@k
            precision_k = len(relevant_at_k) / k if k > 0 else 0.0
            metrics[f"precision@{k}"] = precision_k

            # Recall@k
            recall_k = len(relevant_at_k) / len(expected_set) if expected_set else 0.0
            metrics[f"recall@{k}"] = recall_k

            # F1@k
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_k = 0.0
            metrics[f"f1@{k}"] = f1_k

        # Mean Reciprocal Rank (MRR)
        mrr = self._calculate_mrr(retrieved_ids, expected_set)
        metrics["mrr"] = mrr

        # Normalized Discounted Cumulative Gain (NDCG)
        # Using simple binary relevance (relevant=1, not relevant=0)
        for k in top_k_values:
            ndcg_k = self._calculate_ndcg(retrieved_ids[:k], expected_set)
            metrics[f"ndcg@{k}"] = ndcg_k

        return metrics

    def _calculate_mrr(self, retrieved_ids: List[str], expected_set: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank for a single query.

        Args:
            retrieved_ids: List of retrieved document IDs
            expected_set: Set of expected relevant document IDs

        Returns:
            MRR value for this query
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in expected_set:
                return 1.0 / rank
        return 0.0

    def _calculate_ndcg(self, retrieved_ids: List[str], expected_set: Set[str]) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        Args:
            retrieved_ids: List of retrieved document IDs
            expected_set: Set of expected relevant document IDs

        Returns:
            NDCG value
        """
        if not retrieved_ids or not expected_set:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            relevance = 1.0 if doc_id in expected_set else 0.0
            dcg += relevance / (math.log2(rank + 1))

        # Calculate IDCG (ideal DCG)
        ideal_relevances = [1.0] * min(len(retrieved_ids), len(expected_set))
        idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_relevances))

        return dcg / idcg if idcg > 0 else 0.0

    def _aggregate_metrics(
        self, results: List[Dict[str, Any]], top_k_values: List[int]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across all queries.

        Args:
            results: List of per-query metric dictionaries
            top_k_values: List of k values

        Returns:
            Dictionary containing aggregated metrics
        """
        aggregated = {}

        # Aggregate each metric
        metric_names = (
            [f"precision@{k}" for k in top_k_values]
            + [f"recall@{k}" for k in top_k_values]
            + [f"f1@{k}" for k in top_k_values]
            + [f"ndcg@{k}" for k in top_k_values]
            + ["mrr"]
        )

        for metric in metric_names:
            values = [result[metric] for result in results if metric in result]
            if values:
                aggregated[f"avg_{metric}"] = statistics.mean(values)
                aggregated[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
                aggregated[f"min_{metric}"] = min(values)
                aggregated[f"max_{metric}"] = max(values)

        # Additional aggregate metrics
        total_expected = sum(result["num_expected"] for result in results)
        total_retrieved = sum(result["num_retrieved"] for result in results)

        aggregated["total_expected_docs"] = total_expected
        aggregated["total_retrieved_docs"] = total_retrieved
        aggregated["avg_expected_per_query"] = total_expected / len(results)
        aggregated["avg_retrieved_per_query"] = total_retrieved / len(results)

        return aggregated

    def compare_systems(
        self,
        system_a,
        system_b,
        system_a_name: str = "System A",
        system_b_name: str = "System B",
        top_k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        Compare two retrieval systems.

        Args:
            system_a: First retrieval system
            system_b: Second retrieval system
            system_a_name: Name for first system
            system_b_name: Name for second system
            top_k_values: List of k values for evaluation

        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing {system_a_name} vs {system_b_name}")

        # Evaluate both systems
        results_a = self.evaluate_system(system_a, top_k_values)
        results_b = self.evaluate_system(system_b, top_k_values)

        # Calculate improvements
        comparison = {
            "system_a": {"name": system_a_name, "metrics": results_a},
            "system_b": {"name": system_b_name, "metrics": results_b},
            "improvements": {},
        }

        # Compare key metrics
        key_metrics = [f"avg_precision@{k}" for k in top_k_values] + ["avg_mrr"]

        for metric in key_metrics:
            if metric in results_a and metric in results_b:
                value_a = results_a[metric]
                value_b = results_b[metric]

                improvement = ((value_b - value_a) / value_a * 100) if value_a > 0 else 0
                comparison["improvements"][metric] = {
                    "absolute": value_b - value_a,
                    "relative_percent": improvement,
                    "winner": system_b_name if value_b > value_a else system_a_name,
                }

        # Overall winner
        precision_5_improvement = (
            comparison["improvements"].get("avg_precision@5", {}).get("relative_percent", 0)
        )
        mrr_improvement = comparison["improvements"].get("avg_mrr", {}).get("relative_percent", 0)

        overall_improvement = (precision_5_improvement + mrr_improvement) / 2
        comparison["overall_winner"] = system_b_name if overall_improvement > 0 else system_a_name
        comparison["overall_improvement_percent"] = overall_improvement

        return comparison

    def create_evaluation_report(
        self, evaluation_results: Dict[str, Any], output_path: str
    ) -> None:
        """
        Create a human-readable evaluation report.

        Args:
            evaluation_results: Results from evaluate_system()
            output_path: Path to save the report
        """
        report_lines = [
            "# Retrieval System Evaluation Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total queries evaluated: {evaluation_results['num_queries']}",
            f"- Evaluation time: {evaluation_results['evaluation_time']:.2f} seconds",
            f"- Average expected docs per query: {evaluation_results['avg_expected_per_query']:.1f}",
            f"- Average retrieved docs per query: {evaluation_results['avg_retrieved_per_query']:.1f}",
            "",
            "## Key Metrics",
        ]

        # Add key metrics
        key_metrics = ["avg_precision@5", "avg_recall@5", "avg_f1@5", "avg_mrr", "avg_ndcg@5"]
        for metric in key_metrics:
            if metric in evaluation_results:
                value = evaluation_results[metric]
                report_lines.append(f"- {metric}: {value:.3f}")

        # Add detailed metrics table
        report_lines.extend(
            [
                "",
                "## Detailed Metrics",
                "",
                "| Metric | Mean | Std | Min | Max |",
                "|--------|------|-----|-----|-----|",
            ]
        )

        for metric in sorted(evaluation_results.keys()):
            if metric.startswith("avg_") and not metric.startswith("avg_expected"):
                base_metric = metric[4:]  # Remove 'avg_' prefix
                mean_val = evaluation_results[metric]
                std_val = evaluation_results.get(f"std_{base_metric}", 0)
                min_val = evaluation_results.get(f"min_{base_metric}", mean_val)
                max_val = evaluation_results.get(f"max_{base_metric}", mean_val)

                report_lines.append(
                    f"| {base_metric} | {mean_val:.3f} | {std_val:.3f} | {min_val:.3f} | {max_val:.3f} |"
                )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            logger.info(f"Evaluation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation report: {str(e)}")


# Import math for NDCG calculation
import math
