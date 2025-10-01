"""
Query Cache - Intelligent caching for RAG query results

Part of the RAG Production System course implementation.
Module 3: Production Engineering & Deployment

License: MIT
"""

from typing import Optional, Dict, Any, List
import json
import hashlib
import time
import logging
import os

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Intelligent caching system for RAG query results.

    Provides Redis-based caching with smart TTL, cache invalidation,
    and user-aware cache keys for optimal performance.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        max_cache_size: int = 10000
    ):
        """
        Initialize the query cache.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            max_cache_size: Maximum number of cached items
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self._redis_client = None

    @property
    def redis_client(self):
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            try:
                import redis
                self._redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    health_check_interval=30
                )
                # Test connection
                self._redis_client.ping()
                logger.info("Connected to Redis for query caching")
            except ImportError:
                raise ImportError("redis library is required for caching. Install with: pip install redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise

        return self._redis_client

    def get_cache_key(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Generate a cache key for a query.

        Args:
            query: The query string
            filters: Additional filters/parameters
            user_id: User identifier for personalized caching

        Returns:
            Cache key string
        """
        # Normalize query
        normalized_query = query.lower().strip()

        # Create key components
        key_parts = [normalized_query]

        # Add filters if present
        if filters:
            # Sort filters for consistent key generation
            sorted_filters = sorted(filters.items())
            filter_str = json.dumps(sorted_filters, sort_keys=True)
            key_parts.append(filter_str)

        # Add user context (for personalized results)
        if user_id:
            # Hash user_id for privacy
            user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]
            key_parts.append(f"user:{user_hash}")

        # Create final key
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        return f"query_cache:{key_hash}"

    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for a query.

        Args:
            cache_key: Cache key to lookup

        Returns:
            Cached result dictionary or None if not found
        """
        try:
            cached_data = self.redis_client.get(cache_key)

            if cached_data is None:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None

            # Parse cached data
            result = json.loads(cached_data)

            # Add cache metadata
            result['metadata'] = result.get('metadata', {})
            result['metadata']['cached'] = True
            result['metadata']['cache_hit_time'] = time.time()

            logger.debug(f"Cache hit for key: {cache_key}")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in cache for key {cache_key}: {str(e)}")
            # Remove corrupted cache entry
            self.redis_client.delete(cache_key)
            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def cache_result(
        self,
        cache_key: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a query result.

        Args:
            cache_key: Cache key
            result: Result dictionary to cache
            ttl: Time-to-live in seconds (None for default)

        Returns:
            True if caching was successful
        """
        try:
            # Determine TTL based on result characteristics
            cache_ttl = ttl or self._calculate_smart_ttl(result)

            # Prepare data for caching
            cache_data = result.copy()

            # Remove or modify sensitive data
            cache_data = self._sanitize_for_cache(cache_data)

            # Add cache metadata
            cache_data['cached_at'] = time.time()
            cache_data['cache_ttl'] = cache_ttl

            # Serialize and store
            serialized_data = json.dumps(cache_data, default=str)

            # Check if we need to evict old entries
            self._ensure_cache_size_limit()

            # Store in Redis
            self.redis_client.setex(cache_key, cache_ttl, serialized_data)

            logger.debug(f"Cached result with TTL {cache_ttl}s for key: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")
            return False

    def _calculate_smart_ttl(self, result: Dict[str, Any]) -> int:
        """
        Calculate intelligent TTL based on result characteristics.

        Args:
            result: Query result dictionary

        Returns:
            TTL in seconds
        """
        base_ttl = self.default_ttl

        # Longer TTL for expensive queries (high retrieval time)
        retrieval_time = result.get('metadata', {}).get('query_time', 0)
        if retrieval_time > 2.0:
            base_ttl = base_ttl * 2  # Cache expensive queries longer

        # Shorter TTL for low-confidence results
        confidence = result.get('confidence', 1.0)
        if confidence < 0.7:
            base_ttl = base_ttl // 2  # Cache uncertain results for less time

        # Longer TTL for queries with many sources (likely stable)
        num_sources = len(result.get('sources', []))
        if num_sources >= 5:
            base_ttl = int(base_ttl * 1.5)

        return min(base_ttl, 7200)  # Maximum 2 hours

    def _sanitize_for_cache(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove or sanitize sensitive data before caching.

        Args:
            result: Original result dictionary

        Returns:
            Sanitized result dictionary
        """
        sanitized = result.copy()

        # Remove potentially sensitive metadata
        metadata = sanitized.get('metadata', {})
        sensitive_keys = ['user_id', 'api_key', 'internal_ids']

        for key in sensitive_keys:
            metadata.pop(key, None)

        # Truncate very long answers for cache efficiency
        if 'answer' in sanitized and len(sanitized['answer']) > 5000:
            sanitized['answer'] = sanitized['answer'][:5000] + "... [truncated for cache]"

        return sanitized

    def _ensure_cache_size_limit(self) -> None:
        """Ensure cache doesn't exceed size limits by evicting old entries."""
        try:
            # Get approximate cache size
            cache_keys = self.redis_client.keys("query_cache:*")
            cache_size = len(cache_keys)

            if cache_size >= self.max_cache_size:
                # Evict 10% of oldest entries
                evict_count = self.max_cache_size // 10

                # Get keys with TTL information
                keys_with_ttl = []
                for key in cache_keys[:evict_count * 2]:  # Sample more than needed
                    ttl = self.redis_client.ttl(key)
                    keys_with_ttl.append((key, ttl))

                # Sort by TTL (ascending) to evict entries expiring soon
                keys_with_ttl.sort(key=lambda x: x[1])

                # Evict the keys
                keys_to_evict = [key for key, _ in keys_with_ttl[:evict_count]]
                if keys_to_evict:
                    self.redis_client.delete(*keys_to_evict)
                    logger.info(f"Evicted {len(keys_to_evict)} cache entries to maintain size limit")

        except Exception as e:
            logger.error(f"Error managing cache size: {str(e)}")

    def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cached entries.

        Args:
            pattern: Redis key pattern to match (None for all cache entries)

        Returns:
            Number of keys deleted
        """
        try:
            if pattern is None:
                pattern = "query_cache:*"

            keys = self.redis_client.keys(pattern)

            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")
                return deleted_count
            else:
                logger.info(f"No cache entries found matching pattern: {pattern}")
                return 0

        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get cache keys
            cache_keys = self.redis_client.keys("query_cache:*")
            total_entries = len(cache_keys)

            # Sample some keys for detailed stats
            sample_size = min(100, total_entries)
            sampled_keys = cache_keys[:sample_size]

            total_memory = 0
            ttl_distribution = []

            for key in sampled_keys:
                # Get memory usage (approximate)
                try:
                    value_size = len(self.redis_client.get(key) or "")
                    total_memory += value_size

                    # Get TTL
                    ttl = self.redis_client.ttl(key)
                    if ttl > 0:
                        ttl_distribution.append(ttl)

                except Exception:
                    continue

            # Calculate statistics
            avg_memory_per_entry = total_memory / sample_size if sample_size > 0 else 0
            estimated_total_memory = avg_memory_per_entry * total_entries

            avg_ttl = sum(ttl_distribution) / len(ttl_distribution) if ttl_distribution else 0

            return {
                'total_entries': total_entries,
                'max_cache_size': self.max_cache_size,
                'cache_utilization': (total_entries / self.max_cache_size) * 100,
                'estimated_memory_bytes': estimated_total_memory,
                'avg_memory_per_entry': avg_memory_per_entry,
                'avg_ttl_seconds': avg_ttl,
                'default_ttl': self.default_ttl,
                'sampled_entries': sample_size
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {
                'error': str(e),
                'max_cache_size': self.max_cache_size,
                'default_ttl': self.default_ttl
            }

    def warm_cache(self, queries: List[str], user_id: Optional[str] = None) -> int:
        """
        Pre-populate cache with common queries.

        Args:
            queries: List of queries to warm up
            user_id: Optional user ID for personalized warming

        Returns:
            Number of queries successfully warmed
        """
        logger.info(f"Warming cache with {len(queries)} queries")
        warmed_count = 0

        # This would typically integrate with your RAG service
        # For now, we'll just create cache keys to show the pattern
        for query in queries:
            try:
                cache_key = self.get_cache_key(query, user_id=user_id)

                # Check if already cached
                if self.get_cached_result(cache_key) is None:
                    # Would call RAG service here to get result and cache it
                    # For demonstration, we'll just log the intent
                    logger.debug(f"Would warm cache for query: {query[:50]}...")
                    warmed_count += 1

            except Exception as e:
                logger.error(f"Error warming cache for query '{query}': {str(e)}")

        logger.info(f"Cache warming completed: {warmed_count} queries processed")
        return warmed_count

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the caching system.

        Returns:
            Dictionary with health status
        """
        try:
            # Test Redis connection
            start_time = time.time()
            self.redis_client.ping()
            response_time = time.time() - start_time

            # Get basic stats
            stats = self.get_cache_stats()

            return {
                'healthy': True,
                'redis_connected': True,
                'response_time_ms': response_time * 1000,
                'cache_entries': stats.get('total_entries', 0),
                'cache_utilization_percent': stats.get('cache_utilization', 0),
                'default_ttl': self.default_ttl
            }

        except Exception as e:
            return {
                'healthy': False,
                'redis_connected': False,
                'error': str(e),
                'default_ttl': self.default_ttl
            }