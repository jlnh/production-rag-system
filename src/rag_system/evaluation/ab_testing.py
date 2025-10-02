"""
A/B Testing Framework - Test different retrieval strategies with real users

Part of the RAG Production System course implementation.
Module 2: Advanced Retrieval Techniques

License: MIT
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import json
import hashlib
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ABTestStatus(Enum):
    """Status of an A/B test."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""

    test_id: str
    name: str
    description: str
    traffic_split: float = 0.5  # Percentage for variant B (0.0 = all A, 1.0 = all B)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_sample_size: int = 100
    significance_level: float = 0.05
    status: ABTestStatus = ABTestStatus.DRAFT


@dataclass
class UserInteraction:
    """Record of a user interaction with the system."""

    user_id: str
    query: str
    variant: str  # 'A' or 'B'
    satisfaction_score: Optional[float] = None  # 1-5 scale
    click_through: bool = False
    timestamp: datetime = None
    response_time: float = 0.0
    num_results: int = 0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ABTestFramework:
    """
    A/B testing framework for comparing retrieval strategies.

    Supports user assignment, result tracking, and statistical analysis
    of different retrieval approaches.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the A/B testing framework.

        Args:
            storage_path: Path to store test data (JSON format)
        """
        self.storage_path = storage_path
        self.user_assignments: Dict[str, str] = {}  # user_id -> variant
        self.test_configs: Dict[str, ABTestConfig] = {}
        self.interactions: List[UserInteraction] = []
        self.results_cache: Dict[str, Dict[str, Any]] = {}

        # Load existing data if available
        if storage_path:
            self._load_data()

    def create_test(
        self,
        test_id: str,
        name: str,
        description: str,
        traffic_split: float = 0.5,
        duration_days: int = 7,
        min_sample_size: int = 100,
    ) -> ABTestConfig:
        """
        Create a new A/B test configuration.

        Args:
            test_id: Unique identifier for the test
            name: Human-readable name
            description: Description of what's being tested
            traffic_split: Percentage of traffic for variant B
            duration_days: How long to run the test
            min_sample_size: Minimum interactions needed per variant

        Returns:
            Created test configuration

        Raises:
            ValueError: If test_id already exists or parameters are invalid
        """
        if test_id in self.test_configs:
            raise ValueError(f"Test {test_id} already exists")

        if not (0.0 <= traffic_split <= 1.0):
            raise ValueError("Traffic split must be between 0.0 and 1.0")

        if min_sample_size < 10:
            raise ValueError("Minimum sample size must be at least 10")

        end_date = datetime.now() + timedelta(days=duration_days)

        config = ABTestConfig(
            test_id=test_id,
            name=name,
            description=description,
            traffic_split=traffic_split,
            start_date=None,  # Set when test starts
            end_date=end_date,
            min_sample_size=min_sample_size,
        )

        self.test_configs[test_id] = config
        self._save_data()

        logger.info(f"Created A/B test '{test_id}': {name}")
        return config

    def start_test(self, test_id: str) -> None:
        """
        Start an A/B test.

        Args:
            test_id: ID of the test to start

        Raises:
            ValueError: If test doesn't exist or can't be started
        """
        if test_id not in self.test_configs:
            raise ValueError(f"Test {test_id} not found")

        config = self.test_configs[test_id]

        if config.status != ABTestStatus.DRAFT:
            raise ValueError(f"Test {test_id} cannot be started from status {config.status}")

        config.start_date = datetime.now()
        config.status = ABTestStatus.RUNNING

        self._save_data()
        logger.info(f"Started A/B test '{test_id}'")

    def get_variant(self, user_id: str, test_id: str) -> str:
        """
        Get the variant assignment for a user.

        Uses consistent hashing to ensure users always get the same variant.

        Args:
            user_id: Unique user identifier
            test_id: Test identifier

        Returns:
            Variant assignment ('A' or 'B')

        Raises:
            ValueError: If test doesn't exist or isn't running
        """
        if test_id not in self.test_configs:
            raise ValueError(f"Test {test_id} not found")

        config = self.test_configs[test_id]

        if config.status != ABTestStatus.RUNNING:
            return "A"  # Default to control group for non-running tests

        # Use consistent hashing for assignment
        assignment_key = f"{test_id}:{user_id}"

        if assignment_key not in self.user_assignments:
            # Create deterministic assignment based on user_id and test_id
            hash_value = int(hashlib.md5(assignment_key.encode()).hexdigest(), 16)
            assignment_ratio = (hash_value % 10000) / 10000.0

            variant = "B" if assignment_ratio < config.traffic_split else "A"
            self.user_assignments[assignment_key] = variant

        return self.user_assignments[assignment_key]

    def record_interaction(
        self,
        user_id: str,
        test_id: str,
        query: str,
        satisfaction_score: Optional[float] = None,
        click_through: bool = False,
        response_time: float = 0.0,
        num_results: int = 0,
    ) -> None:
        """
        Record a user interaction with the system.

        Args:
            user_id: User identifier
            test_id: Test identifier
            query: User's query
            satisfaction_score: User satisfaction (1-5 scale)
            click_through: Whether user clicked on results
            response_time: Response time in seconds
            num_results: Number of results returned
        """
        if test_id not in self.test_configs:
            logger.warning(f"Recording interaction for unknown test {test_id}")
            return

        variant = self.get_variant(user_id, test_id)

        interaction = UserInteraction(
            user_id=user_id,
            query=query,
            variant=variant,
            satisfaction_score=satisfaction_score,
            click_through=click_through,
            response_time=response_time,
            num_results=num_results,
        )

        self.interactions.append(interaction)

        # Clear results cache for this test since we have new data
        if test_id in self.results_cache:
            del self.results_cache[test_id]

        # Periodically save data
        if len(self.interactions) % 10 == 0:
            self._save_data()

        logger.debug(f"Recorded interaction for user {user_id}, variant {variant}")

    def get_test_results(self, test_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get current results for an A/B test.

        Args:
            test_id: Test identifier
            force_refresh: Force recalculation of results

        Returns:
            Dictionary containing test results and statistics

        Raises:
            ValueError: If test doesn't exist
        """
        if test_id not in self.test_configs:
            raise ValueError(f"Test {test_id} not found")

        # Use cached results if available and not forcing refresh
        if test_id in self.results_cache and not force_refresh:
            return self.results_cache[test_id]

        config = self.test_configs[test_id]

        # Filter interactions for this test
        test_interactions = [
            interaction
            for interaction in self.interactions
            if self.get_variant(interaction.user_id, test_id) == interaction.variant
        ]

        # Split by variant
        variant_a_interactions = [i for i in test_interactions if i.variant == "A"]
        variant_b_interactions = [i for i in test_interactions if i.variant == "B"]

        # Calculate metrics for each variant
        variant_a_metrics = self._calculate_variant_metrics(variant_a_interactions)
        variant_b_metrics = self._calculate_variant_metrics(variant_b_interactions)

        # Statistical significance testing
        significance_results = self._calculate_significance(
            variant_a_interactions, variant_b_interactions
        )

        results = {
            "test_config": asdict(config),
            "variant_a": {
                "name": "Control (A)",
                "interactions": len(variant_a_interactions),
                "metrics": variant_a_metrics,
            },
            "variant_b": {
                "name": "Treatment (B)",
                "interactions": len(variant_b_interactions),
                "metrics": variant_b_metrics,
            },
            "comparison": {
                "satisfaction_improvement": self._calculate_improvement(
                    variant_a_metrics.get("avg_satisfaction", 0),
                    variant_b_metrics.get("avg_satisfaction", 0),
                ),
                "ctr_improvement": self._calculate_improvement(
                    variant_a_metrics.get("click_through_rate", 0),
                    variant_b_metrics.get("click_through_rate", 0),
                ),
                "response_time_improvement": self._calculate_improvement(
                    variant_a_metrics.get("avg_response_time", 0),
                    variant_b_metrics.get("avg_response_time", 0),
                    lower_is_better=True,
                ),
                "statistical_significance": significance_results,
            },
            "recommendation": self._get_recommendation(
                variant_a_metrics, variant_b_metrics, significance_results, config
            ),
            "generated_at": datetime.now().isoformat(),
        }

        # Cache results
        self.results_cache[test_id] = results

        return results

    def _calculate_variant_metrics(self, interactions: List[UserInteraction]) -> Dict[str, float]:
        """Calculate metrics for a single variant."""
        if not interactions:
            return {}

        # Satisfaction metrics
        satisfaction_scores = [
            i.satisfaction_score for i in interactions if i.satisfaction_score is not None
        ]
        avg_satisfaction = statistics.mean(satisfaction_scores) if satisfaction_scores else 0.0

        # Click-through rate
        clicks = sum(1 for i in interactions if i.click_through)
        ctr = clicks / len(interactions)

        # Response time metrics
        response_times = [i.response_time for i in interactions if i.response_time > 0]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0

        # Results metrics
        result_counts = [i.num_results for i in interactions if i.num_results > 0]
        avg_results = statistics.mean(result_counts) if result_counts else 0.0

        return {
            "avg_satisfaction": avg_satisfaction,
            "satisfaction_count": len(satisfaction_scores),
            "click_through_rate": ctr,
            "avg_response_time": avg_response_time,
            "avg_results_returned": avg_results,
            "total_interactions": len(interactions),
        }

    def _calculate_improvement(
        self, baseline: float, treatment: float, lower_is_better: bool = False
    ) -> Dict[str, float]:
        """Calculate improvement metrics between variants."""
        if baseline == 0:
            return {"absolute": treatment, "relative_percent": 0.0}

        absolute_diff = treatment - baseline
        relative_percent = (absolute_diff / baseline) * 100

        if lower_is_better:
            absolute_diff = -absolute_diff
            relative_percent = -relative_percent

        return {
            "absolute": absolute_diff,
            "relative_percent": relative_percent,
            "is_improvement": relative_percent > 0,
        }

    def _calculate_significance(
        self, variant_a: List[UserInteraction], variant_b: List[UserInteraction]
    ) -> Dict[str, Any]:
        """Calculate statistical significance using t-test for satisfaction scores."""
        # Get satisfaction scores for both variants
        scores_a = [i.satisfaction_score for i in variant_a if i.satisfaction_score is not None]
        scores_b = [i.satisfaction_score for i in variant_b if i.satisfaction_score is not None]

        if len(scores_a) < 10 or len(scores_b) < 10:
            return {
                "is_significant": False,
                "p_value": None,
                "reason": "Insufficient sample size for significance testing",
            }

        try:
            from scipy import stats

            # Perform Welch's t-test (unequal variances)
            t_stat, p_value = stats.ttest_ind(scores_b, scores_a, equal_var=False)

            is_significant = p_value < 0.05

            return {
                "is_significant": is_significant,
                "p_value": p_value,
                "t_statistic": t_stat,
                "method": "Welch t-test",
                "alpha": 0.05,
            }

        except ImportError:
            # Fallback to simple comparison if scipy not available
            mean_a = statistics.mean(scores_a)
            mean_b = statistics.mean(scores_b)

            # Simple effect size calculation
            pooled_std = statistics.stdev(scores_a + scores_b)
            effect_size = abs(mean_b - mean_a) / pooled_std if pooled_std > 0 else 0

            return {
                "is_significant": effect_size > 0.5,  # Simple heuristic
                "p_value": None,
                "effect_size": effect_size,
                "method": "Effect size heuristic (scipy not available)",
            }

    def _get_recommendation(
        self,
        variant_a_metrics: Dict[str, float],
        variant_b_metrics: Dict[str, float],
        significance: Dict[str, Any],
        config: ABTestConfig,
    ) -> Dict[str, Any]:
        """Generate recommendation based on test results."""

        # Check if we have enough data
        total_interactions_a = variant_a_metrics.get("total_interactions", 0)
        total_interactions_b = variant_b_metrics.get("total_interactions", 0)

        if (
            total_interactions_a < config.min_sample_size
            or total_interactions_b < config.min_sample_size
        ):
            return {
                "decision": "continue_test",
                "reason": "Insufficient sample size",
                "confidence": "low",
            }

        # Check statistical significance
        if not significance.get("is_significant", False):
            return {
                "decision": "no_clear_winner",
                "reason": "No statistically significant difference found",
                "confidence": "medium",
            }

        # Compare key metrics
        satisfaction_a = variant_a_metrics.get("avg_satisfaction", 0)
        satisfaction_b = variant_b_metrics.get("avg_satisfaction", 0)

        ctr_a = variant_a_metrics.get("click_through_rate", 0)
        ctr_b = variant_b_metrics.get("click_through_rate", 0)

        # Make recommendation based on overall performance
        if satisfaction_b > satisfaction_a and ctr_b >= ctr_a:
            return {
                "decision": "implement_variant_b",
                "reason": "Variant B shows significant improvement in satisfaction",
                "confidence": "high",
            }
        elif satisfaction_a > satisfaction_b and ctr_a >= ctr_b:
            return {
                "decision": "keep_variant_a",
                "reason": "Variant A performs better overall",
                "confidence": "high",
            }
        else:
            return {
                "decision": "mixed_results",
                "reason": "Results are mixed - further analysis needed",
                "confidence": "low",
            }

    def _save_data(self) -> None:
        """Save test data to storage."""
        if not self.storage_path:
            return

        try:
            data = {
                "user_assignments": self.user_assignments,
                "test_configs": {
                    test_id: asdict(config) for test_id, config in self.test_configs.items()
                },
                "interactions": [asdict(interaction) for interaction in self.interactions],
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving A/B test data: {str(e)}")

    def _load_data(self) -> None:
        """Load test data from storage."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.user_assignments = data.get("user_assignments", {})

            # Load test configs
            for test_id, config_data in data.get("test_configs", {}).items():
                # Convert datetime strings back to datetime objects
                if config_data.get("start_date"):
                    config_data["start_date"] = datetime.fromisoformat(config_data["start_date"])
                if config_data.get("end_date"):
                    config_data["end_date"] = datetime.fromisoformat(config_data["end_date"])

                config_data["status"] = ABTestStatus(config_data["status"])
                self.test_configs[test_id] = ABTestConfig(**config_data)

            # Load interactions
            for interaction_data in data.get("interactions", []):
                interaction_data["timestamp"] = datetime.fromisoformat(
                    interaction_data["timestamp"]
                )
                self.interactions.append(UserInteraction(**interaction_data))

            logger.info(
                f"Loaded A/B test data: {len(self.test_configs)} tests, {len(self.interactions)} interactions"
            )

        except FileNotFoundError:
            logger.info("No existing A/B test data found")
        except Exception as e:
            logger.error(f"Error loading A/B test data: {str(e)}")

    def stop_test(self, test_id: str) -> Dict[str, Any]:
        """
        Stop a running test and get final results.

        Args:
            test_id: Test identifier

        Returns:
            Final test results

        Raises:
            ValueError: If test doesn't exist or isn't running
        """
        if test_id not in self.test_configs:
            raise ValueError(f"Test {test_id} not found")

        config = self.test_configs[test_id]

        if config.status != ABTestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")

        config.status = ABTestStatus.COMPLETED
        self._save_data()

        final_results = self.get_test_results(test_id, force_refresh=True)
        logger.info(f"Stopped A/B test '{test_id}'")

        return final_results
