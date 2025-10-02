"""
Logging Configuration - Structured logging setup for production

Part of the RAG Production System course implementation.
Module 3: Production Engineering & Deployment

License: MIT
"""

import logging
import logging.config
import sys
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    format_type: str = "structured",
    log_file: Optional[str] = None
) -> None:
    """
    Setup comprehensive logging configuration for the RAG system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('structured', 'simple', 'json')
        log_file: Optional log file path
    """
    # Get configuration from environment
    log_level = os.getenv("LOG_LEVEL", level).upper()
    log_format = os.getenv("LOG_FORMAT", format_type).lower()
    log_file_path = os.getenv("LOG_FILE", log_file)

    # Configure based on environment
    if os.getenv("ENVIRONMENT", "development") == "production":
        setup_production_logging(log_level, log_file_path)
    else:
        setup_development_logging(log_level, log_format)


def setup_production_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup production logging with structured JSON format.

    Args:
        level: Logging level
        log_file: Optional log file path
    """
    try:
        import structlog

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Setup stdlib logging
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    'format': '%(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': level,
                    'formatter': 'json',
                    'stream': sys.stdout
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console'],
                    'level': level,
                    'propagate': False
                }
            }
        }

        # Add file handler if log file specified
        if log_file:
            logging_config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': level,
                'formatter': 'json',
                'filename': log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
            logging_config['loggers']['']['handlers'].append('file')

        logging.config.dictConfig(logging_config)

    except ImportError:
        # Fallback to standard logging if structlog not available
        setup_standard_logging(level, "json", log_file)

    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("Production logging configured", extra={
        'level': level,
        'format': 'structured_json',
        'file_logging': log_file is not None
    })


def setup_development_logging(level: str = "DEBUG", format_type: str = "simple") -> None:
    """
    Setup development logging with readable format.

    Args:
        level: Logging level
        format_type: Format type ('simple', 'detailed', 'json')
    """
    if format_type == "json":
        setup_standard_logging(level, "json")
    elif format_type == "detailed":
        setup_standard_logging(level, "detailed")
    else:
        setup_standard_logging(level, "simple")

    # Test logging
    logger = logging.getLogger(__name__)
    logger.info(f"Development logging configured: level={level}, format={format_type}")


def setup_standard_logging(
    level: str = "INFO",
    format_type: str = "simple",
    log_file: Optional[str] = None
) -> None:
    """
    Setup standard Python logging.

    Args:
        level: Logging level
        format_type: Format type
        log_file: Optional log file path
    """
    formatters = {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        },
        'json': {
            '()': 'rag_system.infrastructure.logging_config.JSONFormatter'
        }
    }

    handlers = {
        'console': {
            'class': 'logging.StreamHandler',
            'level': level,
            'formatter': format_type,
            'stream': sys.stdout
        }
    }

    # Add file handler if specified
    if log_file:
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': format_type,
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': formatters,
        'handlers': handlers,
        'root': {
            'level': level,
            'handlers': list(handlers.keys())
        }
    }

    logging.config.dictConfig(logging_config)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        # Build log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id

        if hasattr(record, 'query_id'):
            log_entry['query_id'] = record.query_id

        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add any extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text',
                          'stack_info', 'getMessage']:
                if not key.startswith('_'):
                    log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class RequestIDFilter(logging.Filter):
    """
    Logging filter to add request ID to log records.
    """

    def __init__(self, request_id: str):
        """
        Initialize the filter.

        Args:
            request_id: Request ID to add to records
        """
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request ID to log record.

        Args:
            record: Log record to modify

        Returns:
            True to allow the record through
        """
        record.request_id = self.request_id
        return True


class UserContextFilter(logging.Filter):
    """
    Logging filter to add user context to log records.
    """

    def __init__(self, user_id: str, session_id: Optional[str] = None):
        """
        Initialize the filter.

        Args:
            user_id: User ID to add to records
            session_id: Optional session ID
        """
        super().__init__()
        self.user_id = user_id
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add user context to log record.

        Args:
            record: Log record to modify

        Returns:
            True to allow the record through
        """
        record.user_id = self.user_id
        if self.session_id:
            record.session_id = self.session_id
        return True


def get_logger_with_context(
    name: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs
) -> logging.Logger:
    """
    Get a logger with additional context.

    Args:
        name: Logger name
        user_id: Optional user ID
        request_id: Optional request ID
        **kwargs: Additional context fields

    Returns:
        Logger with context filters applied
    """
    logger = logging.getLogger(name)

    # Add context filters
    if user_id:
        logger.addFilter(UserContextFilter(user_id))

    if request_id:
        logger.addFilter(RequestIDFilter(request_id))

    # Add any additional context as extra fields
    if kwargs:
        original_makeRecord = logger.makeRecord

        def makeRecord(name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
            extra = extra or {}
            extra.update(kwargs)
            return original_makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)

        logger.makeRecord = makeRecord

    return logger


def configure_external_loggers() -> None:
    """
    Configure logging levels for external libraries.
    """
    # Suppress noisy external loggers
    external_loggers = {
        'urllib3.connectionpool': 'WARNING',
        'requests.packages.urllib3': 'WARNING',
        'botocore': 'WARNING',
        'boto3': 'WARNING',
        'openai': 'WARNING',
        'httpx': 'WARNING',
        'httpcore': 'WARNING'
    }

    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level))


def setup_audit_logging(audit_log_file: str = "audit.log") -> logging.Logger:
    """
    Setup separate audit logging for security and compliance.

    Args:
        audit_log_file: Path to audit log file

    Returns:
        Audit logger instance
    """
    audit_logger = logging.getLogger('audit')
    audit_logger.setLevel(logging.INFO)

    # Create file handler for audit logs
    handler = logging.handlers.RotatingFileHandler(
        audit_log_file,
        maxBytes=10485760,  # 10MB
        backupCount=10
    )

    # Use JSON formatter for audit logs
    formatter = JSONFormatter()
    handler.setFormatter(formatter)

    audit_logger.addHandler(handler)
    audit_logger.propagate = False  # Don't propagate to root logger

    return audit_logger


# Don't auto-initialize - let applications call setup_logging() explicitly
# This avoids circular import issues