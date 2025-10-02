"""
Utility Helper Functions - Common utilities for the RAG system

Part of the RAG Production System course implementation.

License: MIT
"""

import hashlib
import json
import time
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def generate_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash for a given text.

    Args:
        text: Text to hash
        algorithm: Hash algorithm to use (md5, sha1, sha256)

    Returns:
        Hexadecimal hash string

    Raises:
        ValueError: If algorithm is not supported
    """
    algorithms = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256}

    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    hash_func = algorithms[algorithm]
    return hash_func(text.encode("utf-8")).hexdigest()


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Raises:
        ValueError: If chunk_size is less than 1
    """
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1")

    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(""", "'").replace(""", "'")

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.

    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keywords
    """
    if not text:
        return []

    # Clean text and convert to lowercase
    cleaned_text = clean_text(text.lower())

    # Split into words
    words = re.findall(r"\b[a-zA-Z]{3,}\b", cleaned_text)

    # Filter common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
    }

    words = [word for word in words if word not in stop_words]

    # Count word frequencies
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency and return top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if email is valid
    """
    if not email:
        return False

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed_file"

    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"

    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = 255 - len(ext)
        sanitized = name[:max_name_length] + ext

    return sanitized


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0

    while size_bytes >= 1024 and size_index < len(size_names) - 1:
        size_bytes /= 1024.0
        size_index += 1

    return f"{size_bytes:.1f} {size_names[size_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,),
):
    """
    Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Backoff multiplier
        exceptions: Tuple of exceptions to catch

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries:
                raise e

            delay = min(base_delay * (backoff_factor**attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


def parse_time_string(time_str: str) -> timedelta:
    """
    Parse time string to timedelta.

    Args:
        time_str: Time string (e.g., "1h", "30m", "45s", "1h30m")

    Returns:
        timedelta object

    Raises:
        ValueError: If time string format is invalid
    """
    if not time_str:
        raise ValueError("Time string cannot be empty")

    # Pattern to match time components
    pattern = r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
    match = re.match(pattern, time_str.strip())

    if not match:
        raise ValueError(f"Invalid time format: {time_str}")

    hours, minutes, seconds = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0

    if hours == 0 and minutes == 0 and seconds == 0:
        raise ValueError(f"No time components found in: {time_str}")

    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON with fallback.

    Args:
        obj: Object to serialize
        default: Default JSON string if serialization fails

    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return default


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return suffix[:max_length]

    return text[: max_length - len(suffix)] + suffix


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)

    Raises:
        ValueError: If vectors have different lengths
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    if not vec1 or not vec2:
        return 0.0

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate magnitudes
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    # Avoid division by zero
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and monitoring.

    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    import psutil

    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            },
            "disk": {"total": disk.total, "free": disk.free, "percent": disk.percent},
            "timestamp": datetime.now().isoformat(),
        }
    except ImportError:
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat(),
            "note": "psutil not available for detailed system info",
        }


class Timer:
    """Simple timer context manager for measuring execution time."""

    def __init__(self, name: str = "Timer"):
        """
        Initialize timer.

        Args:
            name: Timer name for logging
        """
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log duration."""
        self.end_time = time.time()
        duration = self.elapsed_time
        logger.debug(f"{self.name} completed in {format_duration(duration)}")

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0

        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time


def create_unique_id(prefix: str = "") -> str:
    """
    Create a unique identifier.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique identifier string
    """
    import uuid

    unique_id = str(uuid.uuid4())

    if prefix:
        return f"{prefix}_{unique_id}"

    return unique_id
