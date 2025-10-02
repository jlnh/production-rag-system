"""
Rate Limiter - Redis-based rate limiting for API protection

Part of the RAG Production System course implementation.
Module 3: Production Engineering & Deployment

License: MIT
"""

from typing import Optional
import time
import logging
import os

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Redis-based rate limiter with sliding window implementation.

    Provides protection against API abuse while allowing legitimate usage.
    """

    def __init__(
        self, max_requests: int = 100, window_seconds: int = 3600, redis_url: Optional[str] = None
    ):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
            redis_url: Redis connection URL
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis_client = None

    @property
    def redis_client(self):
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            try:
                import redis

                self._redis_client = redis.from_url(
                    self.redis_url, decode_responses=True, health_check_interval=30
                )
                # Test connection
                self._redis_client.ping()
                logger.info("Connected to Redis for rate limiting")
            except ImportError:
                raise ImportError(
                    "redis library is required for rate limiting. Install with: pip install redis"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise

        return self._redis_client

    def allow_request(self, user_id: str) -> bool:
        """
        Check if a request should be allowed for the given user.

        Uses a sliding window algorithm with Redis for distributed rate limiting.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if request is allowed, False if rate limited

        Raises:
            Exception: If Redis operation fails
        """
        try:
            return self._sliding_window_check(user_id)
        except Exception as e:
            logger.error(f"Rate limiting check failed for user {user_id}: {str(e)}")
            # Fail open - allow request if rate limiting is broken
            return True

    def _sliding_window_check(self, user_id: str) -> bool:
        """
        Implement sliding window rate limiting using Redis.

        Args:
            user_id: User identifier

        Returns:
            True if request is allowed
        """
        now = time.time()
        window_start = now - self.window_seconds
        key = f"rate_limit:{user_id}"

        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests in window
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(now): now})

        # Set expiration
        pipe.expire(key, self.window_seconds)

        results = pipe.execute()

        # Get count of requests in current window (after cleanup, before adding new)
        current_count = results[1]

        # Check if we're within limits
        if current_count >= self.max_requests:
            # Remove the request we just added since it's rejected
            self.redis_client.zrem(key, str(now))
            logger.warning(
                f"Rate limit exceeded for user {user_id}: {current_count}/{self.max_requests}"
            )
            return False

        logger.debug(f"Request allowed for user {user_id}: {current_count + 1}/{self.max_requests}")
        return True

    def get_request_count(self, user_id: str) -> int:
        """
        Get current request count for a user in the current window.

        Args:
            user_id: User identifier

        Returns:
            Number of requests in current window
        """
        try:
            now = time.time()
            window_start = now - self.window_seconds
            key = f"rate_limit:{user_id}"

            # Clean up expired entries and get count
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = pipe.execute()

            return results[1]

        except Exception as e:
            logger.error(f"Failed to get request count for user {user_id}: {str(e)}")
            return 0

    def get_time_until_reset(self, user_id: str) -> int:
        """
        Get seconds until rate limit resets for a user.

        Args:
            user_id: User identifier

        Returns:
            Seconds until the oldest request expires
        """
        try:
            key = f"rate_limit:{user_id}"

            # Get the oldest request timestamp
            oldest_requests = self.redis_client.zrange(key, 0, 0, withscores=True)

            if not oldest_requests:
                return 0

            oldest_timestamp = oldest_requests[0][1]
            reset_time = oldest_timestamp + self.window_seconds
            time_until_reset = max(0, int(reset_time - time.time()))

            return time_until_reset

        except Exception as e:
            logger.error(f"Failed to get reset time for user {user_id}: {str(e)}")
            return self.window_seconds

    def reset_user_limit(self, user_id: str) -> bool:
        """
        Reset rate limit for a specific user (admin function).

        Args:
            user_id: User identifier

        Returns:
            True if reset was successful
        """
        try:
            key = f"rate_limit:{user_id}"
            self.redis_client.delete(key)
            logger.info(f"Rate limit reset for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to reset rate limit for user {user_id}: {str(e)}")
            return False

    def get_user_stats(self, user_id: str) -> dict:
        """
        Get detailed rate limiting statistics for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with rate limiting statistics
        """
        try:
            current_count = self.get_request_count(user_id)
            time_until_reset = self.get_time_until_reset(user_id)

            return {
                "user_id": user_id,
                "current_requests": current_count,
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
                "time_until_reset": time_until_reset,
                "requests_remaining": max(0, self.max_requests - current_count),
                "percentage_used": min(100, (current_count / self.max_requests) * 100),
            }

        except Exception as e:
            logger.error(f"Failed to get stats for user {user_id}: {str(e)}")
            return {"user_id": user_id, "error": str(e)}

    def update_limits(self, max_requests: int, window_seconds: int) -> None:
        """
        Update rate limiting parameters.

        Args:
            max_requests: New maximum requests limit
            window_seconds: New time window in seconds
        """
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("Rate limit parameters must be positive")

        old_max = self.max_requests
        old_window = self.window_seconds

        self.max_requests = max_requests
        self.window_seconds = window_seconds

        logger.info(
            f"Rate limit updated: {old_max}/{old_window}s -> {max_requests}/{window_seconds}s"
        )

    def health_check(self) -> dict:
        """
        Check the health of the rate limiting system.

        Returns:
            Dictionary with health status
        """
        try:
            # Test Redis connection
            start_time = time.time()
            self.redis_client.ping()
            response_time = time.time() - start_time

            return {
                "healthy": True,
                "redis_connected": True,
                "response_time_ms": response_time * 1000,
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
            }

        except Exception as e:
            return {
                "healthy": False,
                "redis_connected": False,
                "error": str(e),
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
            }

    def get_global_stats(self) -> dict:
        """
        Get global rate limiting statistics across all users.

        Returns:
            Dictionary with global statistics
        """
        try:
            # Get all rate limit keys
            pattern = "rate_limit:*"
            keys = self.redis_client.keys(pattern)

            total_users = len(keys)
            total_requests = 0

            # Sample some keys to get statistics
            sample_size = min(100, total_users)  # Don't sample too many for performance
            sampled_keys = keys[:sample_size] if keys else []

            for key in sampled_keys:
                count = self.redis_client.zcard(key)
                total_requests += count

            avg_requests_per_user = total_requests / sample_size if sample_size > 0 else 0

            return {
                "total_active_users": total_users,
                "sampled_users": sample_size,
                "total_requests_sampled": total_requests,
                "avg_requests_per_user": avg_requests_per_user,
                "max_requests_limit": self.max_requests,
                "window_seconds": self.window_seconds,
            }

        except Exception as e:
            logger.error(f"Failed to get global rate limiting stats: {str(e)}")
            return {
                "error": str(e),
                "max_requests_limit": self.max_requests,
                "window_seconds": self.window_seconds,
            }
