"""
Quality Gate - Automated quality testing for RAG system deployments

Part of the RAG Production System course implementation.
Module 3: Production Engineering & Deployment

License: MIT
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from .retrieval_evaluator import RetrievalEvaluator

logger = logging.getLogger(__name__)


@dataclass
class QualityThreshold:
    """Quality threshold configuration."""
    metric_name: str
    min_value: float
    max_value: Optional[float] = None
    critical: bool = True  # If True, failure blocks deployment


@dataclass
class QualityTestResult:
    """Result of a quality test."""
    test_name: str
    passed: bool
    score: float
    threshold: QualityThreshold
    details: Dict[str, Any]
    execution_time: float


@dataclass
class QualityGateResult:
    """Overall quality gate result."""
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    critical_failures: int
    results: List[QualityTestResult]
    execution_time: float
    metadata: Dict[str, Any]


class QualityGate:
    """
    Automated quality gate for RAG system deployments.

    Runs comprehensive tests to ensure system quality before deployment.
    """

    def __init__(
        self,
        test_queries_path: Optional[str] = None,
        thresholds_config_path: Optional[str] = None
    ):
        """
        Initialize the quality gate.

        Args:
            test_queries_path: Path to test queries file
            thresholds_config_path: Path to quality thresholds configuration
        """
        self.test_queries_path = test_queries_path
        self.thresholds_config_path = thresholds_config_path
        self.evaluator = RetrievalEvaluator(test_queries_path)
        self._thresholds = self._load_thresholds()

    def _load_thresholds(self) -> Dict[str, QualityThreshold]:
        """Load quality thresholds from configuration."""
        if self.thresholds_config_path and Path(self.thresholds_config_path).exists():
            try:
                with open(self.thresholds_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                thresholds = {}
                for threshold_config in config.get('thresholds', []):
                    threshold = QualityThreshold(**threshold_config)
                    thresholds[threshold.metric_name] = threshold

                logger.info(f"Loaded {len(thresholds)} quality thresholds from {self.thresholds_config_path}")
                return thresholds

            except Exception as e:
                logger.error(f"Error loading thresholds: {str(e)}")

        # Default thresholds
        return self._get_default_thresholds()

    def _get_default_thresholds(self) -> Dict[str, QualityThreshold]:
        """Get default quality thresholds."""
        return {
            'avg_precision@5': QualityThreshold(
                metric_name='avg_precision@5',
                min_value=0.7,
                critical=True
            ),
            'avg_recall@5': QualityThreshold(
                metric_name='avg_recall@5',
                min_value=0.6,
                critical=True
            ),
            'avg_mrr': QualityThreshold(
                metric_name='avg_mrr',
                min_value=0.75,
                critical=True
            ),
            'avg_response_time': QualityThreshold(
                metric_name='avg_response_time',
                min_value=0.0,
                max_value=3.0,  # Max 3 seconds
                critical=False
            ),
            'system_availability': QualityThreshold(
                metric_name='system_availability',
                min_value=0.99,
                critical=True
            )
        }

    def run_retrieval_tests(self, retriever) -> QualityTestResult:
        """
        Run retrieval quality tests.

        Args:
            retriever: Retrieval system to test

        Returns:
            Quality test result
        """
        start_time = time.time()

        try:
            logger.info("Running retrieval quality tests...")

            # Run evaluation
            evaluation_results = self.evaluator.evaluate_system(retriever)

            # Check thresholds
            passed_metrics = 0
            total_metrics = 0
            failed_details = {}

            for metric_name, threshold in self._thresholds.items():
                if metric_name in evaluation_results:
                    total_metrics += 1
                    value = evaluation_results[metric_name]

                    # Check threshold
                    if self._check_threshold(value, threshold):
                        passed_metrics += 1
                    else:
                        failed_details[metric_name] = {
                            'value': value,
                            'threshold': asdict(threshold),
                            'passed': False
                        }

            execution_time = time.time() - start_time
            overall_score = passed_metrics / total_metrics if total_metrics > 0 else 0
            test_passed = len(failed_details) == 0

            result = QualityTestResult(
                test_name="retrieval_quality",
                passed=test_passed,
                score=overall_score,
                threshold=QualityThreshold("overall", 1.0),  # All must pass
                details={
                    'evaluation_results': evaluation_results,
                    'passed_metrics': passed_metrics,
                    'total_metrics': total_metrics,
                    'failed_metrics': failed_details
                },
                execution_time=execution_time
            )

            logger.info(f"Retrieval tests completed: {passed_metrics}/{total_metrics} metrics passed")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Retrieval tests failed: {str(e)}")

            return QualityTestResult(
                test_name="retrieval_quality",
                passed=False,
                score=0.0,
                threshold=QualityThreshold("overall", 1.0),
                details={'error': str(e)},
                execution_time=execution_time
            )

    async def run_performance_tests(self, rag_service) -> QualityTestResult:
        """
        Run performance tests.

        Args:
            rag_service: RAG service to test

        Returns:
            Quality test result
        """
        start_time = time.time()

        try:
            logger.info("Running performance tests...")

            # Test queries for performance
            test_queries = [
                "What is machine learning?",
                "How does natural language processing work?",
                "Explain deep learning algorithms",
                "What are the benefits of cloud computing?",
                "How to implement REST APIs?"
            ]

            response_times = []
            successful_queries = 0

            for query in test_queries:
                query_start = time.time()
                try:
                    result = await rag_service.process_query(query, top_k=5)
                    response_time = time.time() - query_start
                    response_times.append(response_time)
                    successful_queries += 1
                except Exception as e:
                    logger.warning(f"Performance test query failed: {str(e)}")

            # Calculate metrics
            avg_response_time = sum(response_times) / len(response_times) if response_times else float('inf')
            success_rate = successful_queries / len(test_queries)

            # Check thresholds
            response_time_threshold = self._thresholds.get('avg_response_time')
            response_time_passed = True

            if response_time_threshold:
                response_time_passed = self._check_threshold(avg_response_time, response_time_threshold)

            success_rate_threshold = self._thresholds.get('system_availability')
            success_rate_passed = True

            if success_rate_threshold:
                success_rate_passed = self._check_threshold(success_rate, success_rate_threshold)

            test_passed = response_time_passed and success_rate_passed
            execution_time = time.time() - start_time

            result = QualityTestResult(
                test_name="performance",
                passed=test_passed,
                score=min(success_rate, 1.0 if response_time_passed else 0.5),
                threshold=QualityThreshold("performance", 0.8),
                details={
                    'avg_response_time': avg_response_time,
                    'response_times': response_times,
                    'success_rate': success_rate,
                    'successful_queries': successful_queries,
                    'total_queries': len(test_queries),
                    'response_time_passed': response_time_passed,
                    'success_rate_passed': success_rate_passed
                },
                execution_time=execution_time
            )

            logger.info(f"Performance tests completed: avg response time {avg_response_time:.2f}s, success rate {success_rate:.2%}")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Performance tests failed: {str(e)}")

            return QualityTestResult(
                test_name="performance",
                passed=False,
                score=0.0,
                threshold=QualityThreshold("performance", 0.8),
                details={'error': str(e)},
                execution_time=execution_time
            )

    async def run_health_tests(self, rag_service) -> QualityTestResult:
        """
        Run system health tests.

        Args:
            rag_service: RAG service to test

        Returns:
            Quality test result
        """
        start_time = time.time()

        try:
            logger.info("Running health tests...")

            # Check system health
            health_result = await rag_service.health_check()

            # Verify all components are healthy
            overall_healthy = health_result.get('healthy', False)
            services = health_result.get('services', {})

            unhealthy_services = []
            for service_name, service_status in services.items():
                if not service_status.get('healthy', False):
                    unhealthy_services.append(service_name)

            test_passed = overall_healthy and len(unhealthy_services) == 0
            score = 1.0 if test_passed else 0.0

            execution_time = time.time() - start_time

            result = QualityTestResult(
                test_name="health",
                passed=test_passed,
                score=score,
                threshold=QualityThreshold("health", 1.0),
                details={
                    'overall_healthy': overall_healthy,
                    'services': services,
                    'unhealthy_services': unhealthy_services,
                    'health_check_response': health_result
                },
                execution_time=execution_time
            )

            logger.info(f"Health tests completed: {'PASSED' if test_passed else 'FAILED'}")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Health tests failed: {str(e)}")

            return QualityTestResult(
                test_name="health",
                passed=False,
                score=0.0,
                threshold=QualityThreshold("health", 1.0),
                details={'error': str(e)},
                execution_time=execution_time
            )

    def run_all_tests(self, retriever, rag_service) -> QualityGateResult:
        """
        Run all quality tests.

        Args:
            retriever: Retrieval system to test
            rag_service: RAG service to test

        Returns:
            Overall quality gate result
        """
        start_time = time.time()

        logger.info("Starting quality gate execution...")

        results = []

        # Run all test suites
        test_suites = [
            ("retrieval", lambda: self.run_retrieval_tests(retriever)),
            ("performance", lambda: self.run_performance_tests(rag_service)),
            ("health", lambda: self.run_health_tests(rag_service))
        ]

        for test_name, test_func in test_suites:
            try:
                result = test_func()
                results.append(result)

                status = "PASSED" if result.passed else "FAILED"
                logger.info(f"{test_name.title()} tests {status} (score: {result.score:.3f})")

            except Exception as e:
                logger.error(f"Error running {test_name} tests: {str(e)}")
                # Create failed result
                failed_result = QualityTestResult(
                    test_name=test_name,
                    passed=False,
                    score=0.0,
                    threshold=QualityThreshold(test_name, 1.0),
                    details={'error': str(e)},
                    execution_time=0.0
                )
                results.append(failed_result)

        # Calculate overall results
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = len(results) - passed_tests

        # Count critical failures
        critical_failures = sum(
            1 for r in results
            if not r.passed and r.threshold.critical
        )

        # Quality gate passes if all critical tests pass
        overall_passed = critical_failures == 0

        execution_time = time.time() - start_time

        quality_result = QualityGateResult(
            passed=overall_passed,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            critical_failures=critical_failures,
            results=results,
            execution_time=execution_time,
            metadata={
                'thresholds_used': len(self._thresholds),
                'test_queries_available': len(self.evaluator.test_queries),
                'timestamp': time.time()
            }
        )

        # Log final result
        status = "PASSED" if overall_passed else "FAILED"
        logger.info(
            f"Quality gate {status}: {passed_tests}/{len(results)} tests passed, "
            f"{critical_failures} critical failures in {execution_time:.2f}s"
        )

        return quality_result

    def _check_threshold(self, value: float, threshold: QualityThreshold) -> bool:
        """
        Check if a value meets the threshold criteria.

        Args:
            value: Value to check
            threshold: Threshold configuration

        Returns:
            True if threshold is met
        """
        # Check minimum value
        if value < threshold.min_value:
            return False

        # Check maximum value if specified
        if threshold.max_value is not None and value > threshold.max_value:
            return False

        return True

    def save_results(self, result: QualityGateResult, output_path: str) -> None:
        """
        Save quality gate results to file.

        Args:
            result: Quality gate result
            output_path: Path to save results
        """
        try:
            # Convert to dictionary for JSON serialization
            result_dict = asdict(result)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, default=str)

            logger.info(f"Quality gate results saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving quality gate results: {str(e)}")

    def create_report(self, result: QualityGateResult, output_path: str) -> None:
        """
        Create a human-readable quality gate report.

        Args:
            result: Quality gate result
            output_path: Path to save report
        """
        try:
            report_lines = [
                "# Quality Gate Report",
                f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"## Overall Result: {'✅ PASSED' if result.passed else '❌ FAILED'}",
                "",
                f"- Total tests: {result.total_tests}",
                f"- Passed tests: {result.passed_tests}",
                f"- Failed tests: {result.failed_tests}",
                f"- Critical failures: {result.critical_failures}",
                f"- Execution time: {result.execution_time:.2f} seconds",
                "",
                "## Test Results",
                ""
            ]

            for test_result in result.results:
                status = "✅ PASSED" if test_result.passed else "❌ FAILED"
                report_lines.extend([
                    f"### {test_result.test_name.title()} Tests - {status}",
                    f"- Score: {test_result.score:.3f}",
                    f"- Execution time: {test_result.execution_time:.2f} seconds",
                    f"- Critical: {'Yes' if test_result.threshold.critical else 'No'}",
                    ""
                ])

                # Add details if test failed
                if not test_result.passed:
                    report_lines.extend([
                        "**Failure Details:**",
                        f"```json",
                        json.dumps(test_result.details, indent=2, default=str),
                        "```",
                        ""
                    ])

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))

            logger.info(f"Quality gate report saved to {output_path}")

        except Exception as e:
            logger.error(f"Error creating quality gate report: {str(e)}")

    def update_thresholds(self, new_thresholds: Dict[str, QualityThreshold]) -> None:
        """
        Update quality thresholds.

        Args:
            new_thresholds: New threshold configurations
        """
        self._thresholds.update(new_thresholds)
        logger.info(f"Updated {len(new_thresholds)} quality thresholds")

    def get_threshold_summary(self) -> Dict[str, Any]:
        """
        Get summary of current thresholds.

        Returns:
            Dictionary with threshold information
        """
        return {
            'total_thresholds': len(self._thresholds),
            'critical_thresholds': sum(1 for t in self._thresholds.values() if t.critical),
            'thresholds': {name: asdict(threshold) for name, threshold in self._thresholds.items()}
        }