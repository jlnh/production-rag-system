"""
Smart Retriever - Adaptive retrieval based on query analysis

Part of the RAG Production System course implementation.
Module 2: Advanced Retrieval Techniques

License: MIT
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from .hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class SmartRetriever(HybridRetriever):
    """
    Intelligent retrieval system that adapts strategy based on query analysis.

    Analyzes query characteristics to determine optimal search strategy
    and dynamically adjusts weights for vector vs keyword search.
    """

    def __init__(self, vector_store, embedding_generator, documents: List[str] = None, **kwargs):
        """
        Initialize smart retriever.

        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            documents: Document collection for keyword search
            **kwargs: Additional parameters for parent class
        """
        super().__init__(vector_store, embedding_generator, documents, **kwargs)
        self.method_name = "smart"

        # Query pattern rules
        self.exact_match_patterns = [
            r'"[^"]+"',  # Quoted strings
            r"\b[A-Z]+\d+\b",  # Error codes like ERR404
            r"\b\w+\(\)",  # Function calls
            r"#\w+",  # Hash tags or IDs
            r"@\w+",  # Mentions or handles
        ]

        self.conceptual_indicators = [
            "how to",
            "what is",
            "explain",
            "why does",
            "when should",
            "difference between",
            "compare",
            "overview of",
            "introduction to",
        ]

        self.technical_indicators = [
            "error",
            "bug",
            "issue",
            "problem",
            "fix",
            "debug",
            "configure",
            "setup",
            "install",
            "deploy",
            "implement",
        ]

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents using adaptive smart search.

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents with optimized strategy
        """
        self.validate_query(query)

        try:
            # Analyze query to determine optimal strategy
            query_analysis = self.analyze_query(query)

            # Get adaptive weights based on analysis
            vector_weight, keyword_weight = self._get_adaptive_weights(query_analysis)

            logger.info(
                f"Smart retrieval for query type '{query_analysis['type']}': "
                f"vector={vector_weight:.2f}, keyword={keyword_weight:.2f}"
            )

            # Use hybrid retrieval with adaptive weights
            results = super().retrieve(
                query,
                top_k=top_k,
                vector_weight_override=vector_weight,
                keyword_weight_override=keyword_weight,
                **kwargs,
            )

            # Enhance results with query analysis metadata
            for result in results:
                result["metadata"]["query_analysis"] = query_analysis

            return results

        except Exception as e:
            logger.error(f"Smart retrieval failed: {str(e)}")
            # Fallback to standard hybrid search
            return super().retrieve(query, top_k, **kwargs)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine its characteristics and optimal retrieval strategy.

        Args:
            query: Query string to analyze

        Returns:
            Dictionary containing query analysis results
        """
        query_lower = query.lower().strip()

        analysis = {
            "original_query": query,
            "length": len(query.split()),
            "has_quotes": '"' in query,
            "has_exact_patterns": False,
            "has_conceptual_indicators": False,
            "has_technical_indicators": False,
            "complexity": "simple",
            "type": "general",
            "confidence": 0.5,
        }

        # Check for exact match patterns
        for pattern in self.exact_match_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                analysis["has_exact_patterns"] = True
                break

        # Check for conceptual indicators
        for indicator in self.conceptual_indicators:
            if indicator in query_lower:
                analysis["has_conceptual_indicators"] = True
                break

        # Check for technical indicators
        for indicator in self.technical_indicators:
            if indicator in query_lower:
                analysis["has_technical_indicators"] = True
                break

        # Determine query complexity
        if analysis["length"] > 10:
            analysis["complexity"] = "complex"
        elif analysis["length"] > 5:
            analysis["complexity"] = "medium"

        # Determine query type and confidence
        analysis["type"], analysis["confidence"] = self._classify_query(analysis)

        return analysis

    def _classify_query(self, analysis: Dict[str, Any]) -> Tuple[str, float]:
        """
        Classify query type based on analysis.

        Args:
            analysis: Query analysis dictionary

        Returns:
            Tuple of (query_type, confidence_score)
        """
        # Exact match queries - favor keyword search
        if analysis["has_quotes"] or analysis["has_exact_patterns"]:
            return "exact_match", 0.9

        # Conceptual queries - favor vector search
        if analysis["has_conceptual_indicators"]:
            return "conceptual", 0.8

        # Technical queries - balanced approach
        if analysis["has_technical_indicators"]:
            return "technical", 0.7

        # Complex queries - favor vector search for semantic understanding
        if analysis["complexity"] == "complex":
            return "complex_semantic", 0.6

        # Simple factual queries - slight preference for keyword
        if analysis["complexity"] == "simple" and analysis["length"] <= 3:
            return "simple_factual", 0.6

        # Default case
        return "general", 0.5

    def _get_adaptive_weights(self, query_analysis: Dict[str, Any]) -> Tuple[float, float]:
        """
        Determine optimal weights based on query analysis.

        Args:
            query_analysis: Query analysis results

        Returns:
            Tuple of (vector_weight, keyword_weight)
        """
        query_type = query_analysis["type"]
        confidence = query_analysis["confidence"]

        # Base weights for different query types
        weight_map = {
            "exact_match": (0.2, 0.8),  # Strong preference for keyword
            "conceptual": (0.85, 0.15),  # Strong preference for vector
            "technical": (0.5, 0.5),  # Balanced approach
            "complex_semantic": (0.75, 0.25),  # Prefer vector for complexity
            "simple_factual": (0.4, 0.6),  # Slight preference for keyword
            "general": (0.6, 0.4),  # Default hybrid weights
        }

        base_vector, base_keyword = weight_map.get(query_type, (0.6, 0.4))

        # Adjust weights based on confidence
        # Higher confidence = more extreme weights
        adjustment_factor = confidence * 0.3  # Max adjustment of 30%

        if base_vector > base_keyword:
            # Already favoring vector, increase the preference
            vector_weight = min(0.95, base_vector + adjustment_factor)
            keyword_weight = 1.0 - vector_weight
        else:
            # Already favoring keyword, increase the preference
            keyword_weight = min(0.95, base_keyword + adjustment_factor)
            vector_weight = 1.0 - keyword_weight

        return vector_weight, keyword_weight

    def get_query_strategy(self, query: str) -> Dict[str, Any]:
        """
        Get the retrieval strategy that would be used for a query without executing it.

        Args:
            query: Query string to analyze

        Returns:
            Dictionary containing strategy information
        """
        analysis = self.analyze_query(query)
        vector_weight, keyword_weight = self._get_adaptive_weights(analysis)

        return {
            "query_analysis": analysis,
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            "recommended_strategy": analysis["type"],
            "confidence": analysis["confidence"],
        }

    def explain_decision(self, query: str) -> str:
        """
        Provide a human-readable explanation of the retrieval strategy for a query.

        Args:
            query: Query string

        Returns:
            Explanation string
        """
        strategy = self.get_query_strategy(query)
        analysis = strategy["query_analysis"]

        explanation_parts = [
            f"Query: '{query}'",
            f"Type: {analysis['type']}",
            f"Complexity: {analysis['complexity']} ({analysis['length']} words)",
        ]

        # Add specific characteristics
        characteristics = []
        if analysis["has_quotes"]:
            characteristics.append("contains quoted text")
        if analysis["has_exact_patterns"]:
            characteristics.append("has exact match patterns")
        if analysis["has_conceptual_indicators"]:
            characteristics.append("is conceptual")
        if analysis["has_technical_indicators"]:
            characteristics.append("is technical")

        if characteristics:
            explanation_parts.append(f"Characteristics: {', '.join(characteristics)}")

        # Add strategy
        vector_pct = int(strategy["vector_weight"] * 100)
        keyword_pct = int(strategy["keyword_weight"] * 100)
        explanation_parts.append(
            f"Strategy: {vector_pct}% vector search, {keyword_pct}% keyword search"
        )

        # Add reasoning
        reasoning_map = {
            "exact_match": "Exact patterns detected, favoring keyword search for precise matching",
            "conceptual": "Conceptual question detected, favoring vector search for semantic understanding",
            "technical": "Technical query detected, using balanced approach",
            "complex_semantic": "Complex query detected, favoring vector search for nuanced understanding",
            "simple_factual": "Simple factual query, slight preference for keyword search",
            "general": "General query, using default balanced approach",
        }

        reasoning = reasoning_map.get(analysis["type"], "Using standard hybrid approach")
        explanation_parts.append(f"Reasoning: {reasoning}")

        return "\n".join(explanation_parts)

    def optimize_for_domain(self, domain: str) -> None:
        """
        Optimize retrieval strategy for specific domains.

        Args:
            domain: Domain type ('technical', 'legal', 'medical', 'general')
        """
        domain_configs = {
            "technical": {
                "vector_weight": 0.4,
                "keyword_weight": 0.6,
                "additional_technical_indicators": ["api", "sdk", "library", "framework"],
            },
            "legal": {
                "vector_weight": 0.5,
                "keyword_weight": 0.5,
                "additional_technical_indicators": ["statute", "regulation", "case", "precedent"],
            },
            "medical": {
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "additional_technical_indicators": [
                    "symptom",
                    "diagnosis",
                    "treatment",
                    "medication",
                ],
            },
            "general": {
                "vector_weight": 0.6,
                "keyword_weight": 0.4,
                "additional_technical_indicators": [],
            },
        }

        if domain in domain_configs:
            config = domain_configs[domain]
            self.vector_weight = float(config["vector_weight"])
            self.keyword_weight = float(config["keyword_weight"])

            # Add domain-specific indicators
            additional_indicators = config.get("additional_technical_indicators", [])
            if isinstance(additional_indicators, list):
                self.technical_indicators.extend(additional_indicators)

            logger.info(
                f"Optimized for {domain} domain: vector={self.vector_weight}, keyword={self.keyword_weight}"
            )
        else:
            logger.warning(
                f"Unknown domain: {domain}. Available: technical, legal, medical, general"
            )
