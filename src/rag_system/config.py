"""
Configuration Management - Centralized configuration for RAG system

Part of the RAG Production System course implementation.

License: MIT
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional, environment variables can still be set manually
    pass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0
    dimension: int = 1536


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    backend: str = "pinecone"
    index_name: str = "rag-documents"
    metric: str = "cosine"
    dimension: int = 1536

    # Pinecone specific
    api_key: Optional[str] = None
    environment: str = "us-west1-gcp"

    # Weaviate specific
    url: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None

    # ChromaDB specific
    persist_directory: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for language model."""

    model: str = "gpt-4.1"
    max_tokens: int = 1000
    temperature: float = 0.1
    timeout: float = 60.0
    max_retries: int = 3


@dataclass
class CacheConfig:
    """Configuration for caching."""

    redis_url: str = "redis://localhost:6379"
    default_ttl: int = 3600
    max_cache_size: int = 10000
    enabled: bool = True


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_requests: int = 100
    window_seconds: int = 3600
    enabled: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format_type: str = "structured"
    log_file: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Configuration for security."""

    api_keys: List[str] = field(default_factory=list)
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    trusted_hosts: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10485760  # 10MB


@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""

    prometheus_enabled: bool = True
    metrics_path: str = "/metrics"
    health_check_interval: int = 30
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"query_duration": 5.0, "error_rate": 0.05, "cache_hit_rate": 0.6}
    )


@dataclass
class RAGConfig:
    """Main RAG system configuration."""

    # Environment
    environment: str = "development"
    debug: bool = True

    # Component configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Data paths
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    models_dir: str = "./models"


class ConfigManager:
    """
    Configuration manager for loading and validating configuration.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self._config: Optional[RAGConfig] = None

    def load_config(self) -> RAGConfig:
        """
        Load configuration from environment variables and files.

        Returns:
            RAGConfig instance
        """
        if self._config is not None:
            return self._config

        # Start with default configuration
        config = RAGConfig()

        # Load from file if specified
        if self.config_path and Path(self.config_path).exists():
            config = self._load_from_file(config, self.config_path)

        # Override with environment variables
        config = self._load_from_env(config)

        # Validate configuration
        self._validate_config(config)

        self._config = config
        logger.info(f"Configuration loaded for environment: {config.environment}")

        return config

    def _load_from_file(self, config: RAGConfig, file_path: str) -> RAGConfig:
        """Load configuration from YAML file."""
        try:
            import yaml

            with open(file_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)

            # Update config with file values
            self._update_config_from_dict(config, file_config)
            logger.info(f"Configuration loaded from file: {file_path}")

        except ImportError:
            logger.warning("PyYAML not available. Skipping file configuration.")
        except Exception as e:
            logger.error(f"Error loading configuration from file: {str(e)}")

        return config

    def _load_from_env(self, config: RAGConfig) -> RAGConfig:
        """Load configuration from environment variables."""

        # Environment
        config.environment = os.getenv("ENVIRONMENT", config.environment)
        config.debug = os.getenv("DEBUG", str(config.debug)).lower() == "true"

        # API
        config.api_host = os.getenv("API_HOST", config.api_host)
        config.api_port = int(os.getenv("API_PORT", str(config.api_port)))
        config.api_workers = int(os.getenv("API_WORKERS", str(config.api_workers)))

        # Embedding
        config.embedding.model = os.getenv("EMBEDDING_MODEL", config.embedding.model)
        config.embedding.batch_size = int(
            os.getenv("EMBEDDING_BATCH_SIZE", str(config.embedding.batch_size))
        )

        # Vector Store
        config.vector_store.backend = os.getenv("VECTOR_STORE_BACKEND", config.vector_store.backend)
        config.vector_store.index_name = os.getenv(
            "VECTOR_STORE_INDEX_NAME", config.vector_store.index_name
        )
        config.vector_store.api_key = os.getenv("PINECONE_API_KEY")
        config.vector_store.environment = os.getenv(
            "PINECONE_ENVIRONMENT", config.vector_store.environment
        )
        config.vector_store.url = os.getenv("WEAVIATE_URL")

        # LLM
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
        config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(config.llm.max_tokens)))
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE", str(config.llm.temperature)))

        # Cache
        config.cache.redis_url = os.getenv("REDIS_URL", config.cache.redis_url)
        config.cache.default_ttl = int(
            os.getenv("CACHE_DEFAULT_TTL", str(config.cache.default_ttl))
        )
        config.cache.enabled = (
            os.getenv("CACHE_ENABLED", str(config.cache.enabled)).lower() == "true"
        )

        # Rate Limiting
        config.rate_limit.max_requests = int(
            os.getenv("RATE_LIMIT_MAX_REQUESTS", str(config.rate_limit.max_requests))
        )
        config.rate_limit.window_seconds = int(
            os.getenv("RATE_LIMIT_WINDOW_SECONDS", str(config.rate_limit.window_seconds))
        )
        config.rate_limit.enabled = (
            os.getenv("RATE_LIMIT_ENABLED", str(config.rate_limit.enabled)).lower() == "true"
        )

        # Logging
        config.logging.level = os.getenv("LOG_LEVEL", config.logging.level)
        config.logging.format_type = os.getenv("LOG_FORMAT", config.logging.format_type)
        config.logging.log_file = os.getenv("LOG_FILE")

        # Security
        api_keys_env = os.getenv("API_KEYS")
        if api_keys_env:
            config.security.api_keys = [key.strip() for key in api_keys_env.split(",")]

        cors_origins_env = os.getenv("CORS_ORIGINS")
        if cors_origins_env:
            config.security.cors_origins = [
                origin.strip() for origin in cors_origins_env.split(",")
            ]

        # Monitoring
        config.monitoring.prometheus_enabled = (
            os.getenv("PROMETHEUS_ENABLED", str(config.monitoring.prometheus_enabled)).lower()
            == "true"
        )

        # Data paths
        config.data_dir = os.getenv("DATA_DIR", config.data_dir)
        config.logs_dir = os.getenv("LOGS_DIR", config.logs_dir)
        config.models_dir = os.getenv("MODELS_DIR", config.models_dir)

        return config

    def _update_config_from_dict(self, config: RAGConfig, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_config, dict):
                section_obj = getattr(config, section_name)
                for key, value in section_config.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
            elif hasattr(config, section_name):
                setattr(config, section_name, section_config)

    def _validate_config(self, config: RAGConfig) -> None:
        """Validate configuration values."""
        errors = []

        # Validate API configuration
        if config.api_port < 1 or config.api_port > 65535:
            errors.append("API port must be between 1 and 65535")

        if config.api_workers < 1:
            errors.append("API workers must be at least 1")

        # Validate embedding configuration
        if config.embedding.batch_size < 1:
            errors.append("Embedding batch size must be at least 1")

        if config.embedding.dimension < 1:
            errors.append("Embedding dimension must be at least 1")

        # Validate vector store configuration
        if config.vector_store.backend not in ["pinecone", "weaviate", "chroma"]:
            errors.append("Vector store backend must be one of: pinecone, weaviate, chroma")

        if config.vector_store.backend == "pinecone" and not config.vector_store.api_key:
            errors.append("Pinecone API key is required when using Pinecone backend")

        # Validate LLM configuration
        if config.llm.max_tokens < 1:
            errors.append("LLM max tokens must be at least 1")

        if not (0.0 <= config.llm.temperature <= 2.0):
            errors.append("LLM temperature must be between 0.0 and 2.0")

        # Validate rate limiting
        if config.rate_limit.max_requests < 1:
            errors.append("Rate limit max requests must be at least 1")

        if config.rate_limit.window_seconds < 1:
            errors.append("Rate limit window must be at least 1 second")

        # Validate logging
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.logging.level.upper() not in valid_log_levels:
            errors.append(f"Log level must be one of: {', '.join(valid_log_levels)}")

        # Raise exception if there are validation errors
        if errors:
            error_message = "Configuration validation errors:\n" + "\n".join(
                f"- {error}" for error in errors
            )
            raise ValueError(error_message)

    def get_config(self) -> RAGConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config

    def reload_config(self) -> RAGConfig:
        """Reload configuration from sources."""
        self._config = None
        return self.load_config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config = self.get_config()

        def dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {
                    field_name: dataclass_to_dict(getattr(obj, field_name))
                    for field_name in obj.__dataclass_fields__
                }
            else:
                return obj

        return dataclass_to_dict(config)


# Global configuration manager instance
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> RAGConfig:
    """Get the current RAG configuration."""
    return get_config_manager().get_config()
