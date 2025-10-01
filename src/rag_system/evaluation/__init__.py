"""
Evaluation and Testing Components - Quality assessment and A/B testing

This module provides comprehensive evaluation tools including:
- Retrieval quality metrics (precision, recall, MRR, NDCG)
- A/B testing framework for comparing strategies
- Quality gates for deployment validation

License: MIT
"""

from .retrieval_evaluator import RetrievalEvaluator
from .ab_testing import ABTestFramework, ABTestConfig, UserInteraction, ABTestStatus
from .quality_gate import QualityGate, QualityThreshold, QualityTestResult, QualityGateResult

__all__ = [
    "RetrievalEvaluator",
    "ABTestFramework",
    "ABTestConfig",
    "UserInteraction",
    "ABTestStatus",
    "QualityGate",
    "QualityThreshold",
    "QualityTestResult",
    "QualityGateResult",
]