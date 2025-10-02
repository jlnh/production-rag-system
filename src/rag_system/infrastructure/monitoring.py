"""
Monitoring - Prometheus metrics and performance tracking

Part of the RAG Production System course implementation.
Module 3: Production Engineering & Deployment

License: MIT
"""

from typing import Dict, Any, Optional, List
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Prometheus metrics (initialized lazily)
_metrics_initialized = False
REQUEST_COUNT = None
REQUEST_DURATION = None
QUERY_DURATION = None
EMBEDDING_DURATION = None
VECTOR_SEARCH_DURATION = None
LLM_GENERATION_DURATION = None
CACHE_HITS = None
CACHE_MISSES = None
ERROR_COUNT = None


def setup_prometheus_metrics(app=None):
    """
    Initialize Prometheus metrics for the RAG system.

    Args:
        app: FastAPI application instance (optional)
    """
    global _metrics_initialized
    global REQUEST_COUNT, REQUEST_DURATION, QUERY_DURATION
    global EMBEDDING_DURATION, VECTOR_SEARCH_DURATION, LLM_GENERATION_DURATION
    global CACHE_HITS, CACHE_MISSES, ERROR_COUNT

    if _metrics_initialized:
        return

    try:
        from prometheus_client import Counter, Histogram, Gauge, generate_latest

        # HTTP request metrics
        REQUEST_COUNT = Counter(
            "rag_http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
        )

        REQUEST_DURATION = Histogram(
            "rag_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
        )

        # RAG-specific metrics
        QUERY_DURATION = Histogram(
            "rag_query_duration_seconds", "RAG query processing duration in seconds", ["query_type"]
        )

        EMBEDDING_DURATION = Histogram(
            "rag_embedding_duration_seconds", "Embedding generation duration in seconds"
        )

        VECTOR_SEARCH_DURATION = Histogram(
            "rag_vector_search_duration_seconds", "Vector search duration in seconds"
        )

        LLM_GENERATION_DURATION = Histogram(
            "rag_llm_generation_duration_seconds", "LLM generation duration in seconds", ["model"]
        )

        # Cache metrics
        CACHE_HITS = Counter("rag_cache_hits_total", "Total cache hits")

        CACHE_MISSES = Counter("rag_cache_misses_total", "Total cache misses")

        # Error metrics
        ERROR_COUNT = Counter("rag_errors_total", "Total errors", ["error_type", "component"])

        # System metrics
        ACTIVE_USERS = Gauge("rag_active_users", "Number of active users")

        VECTOR_STORE_SIZE = Gauge(
            "rag_vector_store_documents", "Number of documents in vector store"
        )

        _metrics_initialized = True
        logger.info("Prometheus metrics initialized")

        # Add metrics endpoint to FastAPI app if provided
        if app:
            from fastapi.responses import Response

            @app.get("/metrics")
            async def get_metrics():
                return Response(content=generate_latest(), media_type="text/plain")

    except ImportError:
        logger.warning("prometheus_client not available. Metrics disabled.")
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {str(e)}")


@contextmanager
def query_duration_tracker(query_type: str = "default"):
    """
    Context manager to track query duration.

    Args:
        query_type: Type of query for labeling

    Usage:
        with query_duration_tracker("hybrid"):
            # Process query
            pass
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if QUERY_DURATION:
            QUERY_DURATION.labels(query_type=query_type).observe(duration)


@contextmanager
def embedding_duration_tracker():
    """Context manager to track embedding generation duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if EMBEDDING_DURATION:
            EMBEDDING_DURATION.observe(duration)


@contextmanager
def vector_search_duration_tracker():
    """Context manager to track vector search duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if VECTOR_SEARCH_DURATION:
            VECTOR_SEARCH_DURATION.observe(duration)


@contextmanager
def llm_generation_duration_tracker(model: str = "unknown"):
    """
    Context manager to track LLM generation duration.

    Args:
        model: Model name for labeling
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if LLM_GENERATION_DURATION:
            LLM_GENERATION_DURATION.labels(model=model).observe(duration)


def record_cache_hit():
    """Record a cache hit."""
    if CACHE_HITS:
        CACHE_HITS.inc()


def record_cache_miss():
    """Record a cache miss."""
    if CACHE_MISSES:
        CACHE_MISSES.inc()


def record_error(error_type: str, component: str):
    """
    Record an error occurrence.

    Args:
        error_type: Type of error (e.g., 'validation', 'timeout', 'api_error')
        component: Component where error occurred (e.g., 'embedding', 'vector_store')
    """
    if ERROR_COUNT:
        ERROR_COUNT.labels(error_type=error_type, component=component).inc()


class PerformanceMonitor:
    """
    Performance monitoring and alerting system for RAG components.
    """

    def __init__(self):
        """Initialize the performance monitor."""
        self.alert_thresholds = {
            "query_duration": 5.0,  # seconds
            "embedding_duration": 2.0,  # seconds
            "vector_search_duration": 1.0,  # seconds
            "llm_generation_duration": 10.0,  # seconds
            "error_rate": 0.05,  # 5%
            "cache_hit_rate": 0.6,  # 60%
        }
        self.measurement_window = 300  # 5 minutes
        self._measurements = []

    def record_measurement(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """
        Record a performance measurement.

        Args:
            metric_name: Name of the metric
            value: Measured value
            labels: Optional labels for the measurement
        """
        measurement = {
            "timestamp": time.time(),
            "metric": metric_name,
            "value": value,
            "labels": labels or {},
        }

        self._measurements.append(measurement)

        # Clean old measurements
        cutoff_time = time.time() - self.measurement_window
        self._measurements = [m for m in self._measurements if m["timestamp"] > cutoff_time]

        # Check for alerts
        self._check_alerts(metric_name, value)

    def _check_alerts(self, metric_name: str, value: float):
        """
        Check if measurement triggers any alerts.

        Args:
            metric_name: Name of the metric
            value: Measured value
        """
        threshold = self.alert_thresholds.get(metric_name)

        if threshold is None:
            return

        if value > threshold:
            logger.warning(
                f"Performance alert: {metric_name} = {value:.3f} exceeds threshold {threshold:.3f}"
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for the current measurement window.

        Returns:
            Dictionary with performance statistics
        """
        if not self._measurements:
            return {"error": "No measurements available"}

        # Group measurements by metric
        metrics: Dict[str, List[float]] = {}
        for measurement in self._measurements:
            metric_name = measurement["metric"]
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(measurement["value"])

        # Calculate statistics for each metric
        summary: Dict[str, Any] = {}
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": self._percentile(values, 0.95),
                    "p99": self._percentile(values, 0.99),
                }

        summary["measurement_window_seconds"] = self.measurement_window
        summary["total_measurements"] = len(self._measurements)

        return summary

    def _percentile(self, values: list, percentile: float) -> float:
        """
        Calculate percentile of a list of values.

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0.0 to 1.0)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        index = min(index, len(sorted_values) - 1)

        return sorted_values[index]

    def set_alert_threshold(self, metric_name: str, threshold: float):
        """
        Set alert threshold for a metric.

        Args:
            metric_name: Name of the metric
            threshold: Alert threshold value
        """
        self.alert_thresholds[metric_name] = threshold
        logger.info(f"Alert threshold set: {metric_name} = {threshold}")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status based on recent measurements.

        Returns:
            Dictionary with health status
        """
        summary = self.get_performance_summary()

        if "error" in summary:
            return {"healthy": True, "reason": "No measurements available"}

        issues = []

        # Check each metric against thresholds
        for metric_name, stats in summary.items():
            if metric_name.endswith("_seconds") and isinstance(stats, dict):
                threshold = self.alert_thresholds.get(metric_name)
                if threshold and stats.get("p95", 0) > threshold:
                    issues.append(
                        f"{metric_name} P95 ({stats['p95']:.3f}s) exceeds threshold ({threshold}s)"
                    )

        healthy = len(issues) == 0

        return {"healthy": healthy, "issues": issues, "summary": summary}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_system_metrics() -> Dict[str, Any]:
    """
    Get comprehensive system metrics.

    Returns:
        Dictionary with system-wide metrics
    """
    try:
        metrics = {
            "timestamp": time.time(),
            "performance": performance_monitor.get_performance_summary(),
            "health": performance_monitor.get_health_status(),
        }

        # Add Prometheus metrics if available
        if _metrics_initialized:
            try:
                from prometheus_client import REGISTRY

                # Get metric families
                metric_families = REGISTRY.collect()
                prometheus_metrics = {}

                for family in metric_families:
                    if family.name.startswith("rag_"):
                        prometheus_metrics[family.name] = {
                            "type": family.type,
                            "help": family.documentation,
                            "samples": len(family.samples) if hasattr(family, "samples") else 0,
                        }

                metrics["prometheus_metrics"] = prometheus_metrics

            except Exception as e:
                logger.error(f"Error collecting Prometheus metrics: {str(e)}")

        return metrics

    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return {"error": str(e), "timestamp": time.time()}
