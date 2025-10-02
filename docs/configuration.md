# Configuration Guide

This guide covers all configuration options for the RAG Production System.

## Configuration Methods

The system supports multiple configuration methods in order of precedence:

1. **Environment variables** (highest priority)
2. **`.env` file** (recommended for development)
3. **`config.yml` file** (structured configuration)
4. **Default values** (lowest priority)

## Environment Variables

### Core API Keys

```bash
# OpenAI API Key (Required)
OPENAI_API_KEY=your-openai-api-key-here

# Pinecone Configuration (v5.0+ - if using Pinecone)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_CLOUD=aws  # Options: aws, gcp, azure
PINECONE_REGION=us-east-1  # Your preferred region
PINECONE_INDEX_NAME=rag-documents

# Weaviate Configuration (if using Weaviate)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=optional-api-key

# ChromaDB Configuration (if using ChromaDB)
CHROMA_PERSIST_DIRECTORY=./data/chromadb
```

### Vector Store Configuration

```bash
# Backend Selection
VECTOR_STORE_BACKEND=pinecone  # Options: pinecone, weaviate, chroma
VECTOR_STORE_INDEX_NAME=rag-documents

# Vector Store Settings
VECTOR_DIMENSION=1536  # Must match embedding model
VECTOR_METRIC=cosine   # Options: cosine, euclidean, dotproduct
```

### Model Configuration

```bash
# Embedding Model
EMBEDDING_MODEL=text-embedding-3-small  # Options: text-embedding-3-small, text-embedding-3-large
EMBEDDING_BATCH_SIZE=100
EMBEDDING_DIMENSION=1536

# Language Model
LLM_MODEL=gpt-4.1  # Options: gpt-4.1, gpt-3.5-turbo
LLM_MAX_TOKENS=1000
LLM_TEMPERATURE=0.1
MAX_CONTEXT_LENGTH=8000
```

### Infrastructure Configuration

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0
REDIS_PASSWORD=optional-password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=300

# Security
API_KEYS=your-api-key-1,your-api-key-2,your-api-key-3
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
TRUSTED_HOSTS=localhost,127.0.0.1
```

### Cache Configuration

```bash
# Cache Settings
CACHE_ENABLED=true
CACHE_DEFAULT_TTL=3600  # seconds
CACHE_MAX_SIZE=10000    # number of items
CACHE_MEMORY_LIMIT=512  # MB

# Cache Strategy
CACHE_QUERY_EMBEDDINGS=true
CACHE_SEARCH_RESULTS=true
CACHE_LLM_RESPONSES=true
```

### Rate Limiting Configuration

```bash
# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=3600

# Per-endpoint limits
RATE_LIMIT_QUERY_ENDPOINT=50
RATE_LIMIT_UPLOAD_ENDPOINT=10
RATE_LIMIT_ADMIN_ENDPOINT=20
```

### Logging Configuration

```bash
# Logging
LOG_LEVEL=INFO          # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=structured   # Options: structured, simple
LOG_FILE=logs/rag-system.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5

# Specific loggers
LOG_LEVEL_UVICORN=WARNING
LOG_LEVEL_REDIS=INFO
LOG_LEVEL_OPENAI=WARNING
```

### Monitoring Configuration

```bash
# Prometheus Metrics
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
METRICS_INCLUDE_LABELS=true

# Health Checks
HEALTH_CHECK_INTERVAL=30    # seconds
HEALTH_CHECK_TIMEOUT=10     # seconds
HEALTH_CHECK_RETRIES=3

# Performance Monitoring
MONITOR_QUERY_TIMES=true
MONITOR_EMBEDDING_TIMES=true
MONITOR_RETRIEVAL_TIMES=true
```

### Development Configuration

```bash
# Environment
ENVIRONMENT=development  # Options: development, staging, production
DEBUG=true
RELOAD=true

# Development Tools
JUPYTER_ENABLE_LAB=yes
JUPYTER_PORT=8888
JUPYTER_TOKEN=your-secure-token
```

## .env File Configuration

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Example `.env` file:

```bash
# ===========================================
# Core Configuration
# ===========================================
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east1-gcp
VECTOR_STORE_BACKEND=pinecone

# ===========================================
# Infrastructure
# ===========================================
REDIS_URL=redis://localhost:6379
API_PORT=8000
LOG_LEVEL=INFO

# ===========================================
# Security
# ===========================================
API_KEYS=dev-key-1,dev-key-2
CORS_ORIGINS=http://localhost:3000

# ===========================================
# Performance
# ===========================================
EMBEDDING_BATCH_SIZE=50
CACHE_DEFAULT_TTL=1800
RATE_LIMIT_MAX_REQUESTS=200
```

## YAML Configuration

Create a `config.yml` file for structured configuration:

```yaml
# config.yml
embedding:
  model: "text-embedding-3-small"
  batch_size: 100
  dimension: 1536
  timeout: 30

vector_store:
  backend: "pinecone"  # pinecone, weaviate, chroma
  index_name: "rag-documents"
  metric: "cosine"
  replicas: 1

  # Backend-specific settings
  pinecone:
    environment: "us-east1-gcp"
    pod_type: "p1.x1"

  weaviate:
    url: "http://localhost:8080"
    scheme: "http"

  chroma:
    persist_directory: "./data/chromadb"
    collection_name: "rag_documents"

llm:
  model: "gpt-4.1"
  max_tokens: 1000
  temperature: 0.1
  max_context_length: 8000
  timeout: 60

retrieval:
  # Hybrid search weights
  vector_weight: 0.6
  keyword_weight: 0.4

  # Re-ranking
  rerank_enabled: true
  rerank_model: "BAAI/bge-reranker-base"
  rerank_top_k: 10

  # Search parameters
  top_k: 5
  similarity_threshold: 0.7

cache:
  enabled: true
  backend: "redis"  # redis, memory
  default_ttl: 3600
  max_size: 10000

  # Cache strategies
  query_embeddings: true
  search_results: true
  llm_responses: true

  # TTL overrides
  embedding_ttl: 86400  # 24 hours
  search_ttl: 1800      # 30 minutes
  llm_ttl: 3600         # 1 hour

rate_limit:
  enabled: true
  backend: "redis"

  # Global limits
  max_requests: 100
  window_seconds: 3600

  # Per-endpoint limits
  endpoints:
    "/query": 50
    "/upload": 10
    "/admin": 20

security:
  # API Keys
  api_keys:
    - "production-key-1"
    - "production-key-2"

  # CORS
  cors_origins:
    - "https://yourdomain.com"
    - "http://localhost:3000"

  # Trusted hosts
  trusted_hosts:
    - "localhost"
    - "127.0.0.1"
    - "yourdomain.com"

monitoring:
  # Prometheus
  prometheus:
    enabled: true
    port: 9090
    include_labels: true

  # Health checks
  health_check:
    interval: 30
    timeout: 10
    retries: 3

  # Performance monitoring
  performance:
    track_query_times: true
    track_embedding_times: true
    track_retrieval_times: true

    # Alerting thresholds
    slow_query_threshold: 5.0      # seconds
    high_error_rate_threshold: 0.1  # 10%

logging:
  level: "INFO"
  format: "structured"  # structured, simple

  # File logging
  file:
    enabled: true
    path: "logs/rag-system.log"
    max_size: "10MB"
    backup_count: 5
    rotation: "time"  # time, size

  # Specific loggers
  loggers:
    uvicorn: "WARNING"
    redis: "INFO"
    openai: "WARNING"
    rag_system: "INFO"

# Development settings
development:
  debug: true
  reload: true
  jupyter:
    enabled: true
    port: 8888
    lab: true

# Production settings
production:
  debug: false
  workers: 4

  # Security hardening
  security:
    hide_docs: true
    disable_admin: true
    require_https: true
```

## Configuration Loading

The system automatically loads configuration in this order:

```python
from rag_system.config import get_config

# Get configuration
config = get_config()

# Access configuration values
embedding_model = config.embedding.model
vector_backend = config.vector_store.backend
api_port = config.api.port
```

## Environment-Specific Configuration

### Development

```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
CACHE_DEFAULT_TTL=300
RATE_LIMIT_MAX_REQUESTS=1000
```

### Staging

```bash
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
CACHE_DEFAULT_TTL=1800
RATE_LIMIT_MAX_REQUESTS=500
```

### Production

```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
CACHE_DEFAULT_TTL=3600
RATE_LIMIT_MAX_REQUESTS=100
API_WORKERS=4
```

## Advanced Configuration

### Custom Embedding Models

```bash
# Use custom OpenAI model
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072

# Use Azure OpenAI
OPENAI_API_TYPE=azure
OPENAI_API_BASE=https://your-resource.openai.azure.com/
OPENAI_API_VERSION=2023-05-15
AZURE_DEPLOYMENT_NAME=your-embedding-deployment
```

### Custom Vector Stores

```bash
# Custom Pinecone settings
PINECONE_POD_TYPE=p1.x2
PINECONE_REPLICAS=2
PINECONE_SHARDS=1

# Custom Weaviate settings
WEAVIATE_GRPC_PORT=50051
WEAVIATE_SCHEME=https
WEAVIATE_ADDITIONAL_HEADERS='{"Authorization": "Bearer token"}'
```

### Advanced Caching

```bash
# Cache clustering
REDIS_CLUSTER_ENABLED=true
REDIS_CLUSTER_NODES=redis-1:6379,redis-2:6379,redis-3:6379

# Cache partitioning
CACHE_PARTITION_BY_USER=true
CACHE_PARTITION_BY_TENANT=true
```

### Performance Tuning

```bash
# Embedding optimization
EMBEDDING_BATCH_SIZE=200
EMBEDDING_PARALLEL_REQUESTS=4
EMBEDDING_RETRY_ATTEMPTS=3

# Vector search optimization
VECTOR_SEARCH_EF=200
VECTOR_SEARCH_NPROBE=10
VECTOR_INDEX_M=16

# API optimization
API_WORKERS=8
API_WORKER_CONNECTIONS=1000
API_KEEPALIVE=2
```

## Configuration Validation

The system validates configuration on startup:

```python
from rag_system.config import validate_config

# Validate current configuration
errors = validate_config()
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
```

Common validation errors:
- Missing required API keys
- Invalid model names
- Unreachable vector store URLs
- Invalid cache sizes
- Conflicting security settings

## Configuration Best Practices

1. **Use .env files for development**, environment variables for production
2. **Keep API keys secure** - never commit them to version control
3. **Set appropriate cache sizes** based on available memory
4. **Configure rate limits** based on your API quotas
5. **Use structured logging** in production
6. **Enable monitoring** for production deployments
7. **Validate configuration** before deployment

## Troubleshooting Configuration

### Common Issues

```bash
# Check current configuration
python -c "
from rag_system.config import get_config
config = get_config()
print(f'Vector backend: {config.vector_store.backend}')
print(f'API port: {config.api.port}')
"

# Test API key
python -c "
import os
if not os.getenv('OPENAI_API_KEY'):
    print('❌ OPENAI_API_KEY not set')
else:
    print('✅ OPENAI_API_KEY configured')
"

# Test Redis connection
python -c "
import redis
import os
r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
try:
    r.ping()
    print('✅ Redis connection successful')
except:
    print('❌ Redis connection failed')
"
```

### Configuration Debugging

Enable debug logging to see configuration loading:

```bash
export LOG_LEVEL=DEBUG
python -c "
from rag_system.config import get_config
config = get_config()
"
```

This will show:
- Which configuration files were loaded
- Which environment variables were used
- Any configuration validation warnings