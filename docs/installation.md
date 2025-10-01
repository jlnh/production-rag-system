# Installation Guide

This guide provides comprehensive installation instructions for the RAG Production System.

## Prerequisites

- **Python 3.9+** (Python 3.11 recommended)
- **Redis server** (for caching and rate limiting)
- **Vector database** (Pinecone, Weaviate, or ChromaDB)
- **OpenAI API key** (for embeddings and LLM)

## Quick Start with Docker

The fastest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd production-rag-system

# Copy environment file
cp .env.example .env

# Edit .env with your actual API keys
nano .env  # or use your preferred editor

# Start all services
docker-compose -f docker/docker-compose.yml up
```

This will start:
- RAG API server (port 8000)
- Redis cache (port 6379)
- Prometheus monitoring (port 9090)

## Python Environment Setup

### Option 1: Virtual Environment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd production-rag-system

# Create virtual environment
python -m venv rag-env

# Activate virtual environment
# On Windows:
rag-env\Scripts\activate
# On macOS/Linux:
source rag-env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# For development (includes testing, linting, docs)
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .
```

### Option 2: Conda Environment

```bash
# Clone the repository
git clone <repository-url>
cd production-rag-system

# Create conda environment
conda create -n rag-env python=3.11
conda activate rag-env

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# Install the package
pip install -e .
```

### Option 3: Poetry

```bash
# Clone the repository
git clone <repository-url>
cd production-rag-system

# Install poetry if not already installed
pip install poetry

# Install dependencies
poetry install

# Install development dependencies
poetry install --with dev

# Activate the environment
poetry shell
```

## Verification

### Test Basic Installation

```bash
# Test imports
python -c "from rag_system import DocumentProcessor; print('✅ Installation successful')"

# Run a simple example (dry run mode)
python examples/module1_basic_rag.py --dry-run
```

### Test with Real API Keys

```bash
# Set environment variables
export OPENAI_API_KEY="your-actual-api-key"

# Run basic RAG example
python examples/module1_basic_rag.py

# Run hybrid search example
python examples/module2_hybrid_search.py

# Test production API
python examples/module3_production_api.py
```

## Infrastructure Setup

### Redis Installation

#### Using Docker
```bash
docker run -d --name redis-rag -p 6379:6379 redis:alpine
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### macOS (Homebrew)
```bash
brew install redis
brew services start redis
```

#### Windows
Download and install from: https://redis.io/download

### Vector Database Setup

#### Pinecone
```bash
# Install Pinecone client
pip install pinecone-client

# Set environment variables
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_ENVIRONMENT="your-pinecone-environment"
```

#### Weaviate
```bash
# Start Weaviate with Docker
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest
```

#### ChromaDB (Local)
```bash
# ChromaDB is included in requirements.txt
# No additional setup needed for local usage
```

## Environment Configuration

### Create .env File

```bash
# Copy the example
cp .env.example .env
```

### Required Environment Variables

```bash
# Core API Keys (Required)
OPENAI_API_KEY=your-openai-api-key-here

# Vector Store Configuration (Choose one)
# For Pinecone:
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
VECTOR_STORE_BACKEND=pinecone

# For Weaviate:
WEAVIATE_URL=http://localhost:8080
VECTOR_STORE_BACKEND=weaviate

# For ChromaDB:
CHROMA_PERSIST_DIRECTORY=./data/chromadb
VECTOR_STORE_BACKEND=chroma

# Infrastructure
REDIS_URL=redis://localhost:6379
API_PORT=8000
LOG_LEVEL=INFO
```

### Optional Configuration

```bash
# Security
API_KEYS=your-api-key-1,your-api-key-2
CORS_ORIGINS=http://localhost:3000

# Performance
EMBEDDING_BATCH_SIZE=100
CACHE_DEFAULT_TTL=3600
RATE_LIMIT_MAX_REQUESTS=100

# Monitoring
PROMETHEUS_ENABLED=true
HEALTH_CHECK_INTERVAL=30
```

## Development Installation

### Additional Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

This includes:
- **pytest** - Testing framework
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking
- **jupyter** - Notebook support
- **sphinx** - Documentation generation

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Production Installation

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-dev build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# macOS
xcode-select --install
```

### Production Requirements

```bash
# Install with production dependencies only
pip install -r requirements.txt

# Set production environment
export ENVIRONMENT=production
export DEBUG=false
```

### Systemd Service (Linux)

Create `/etc/systemd/system/rag-system.service`:

```ini
[Unit]
Description=RAG Production System
After=network.target

[Service]
Type=simple
User=rag
WorkingDirectory=/opt/rag-system
Environment=PATH=/opt/rag-system/venv/bin
ExecStart=/opt/rag-system/venv/bin/uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-system
sudo systemctl start rag-system
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in editable mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Redis Connection Issues
```bash
# Test Redis connection
redis-cli ping

# Check Redis logs
redis-cli monitor
```

#### API Key Issues
```bash
# Verify environment variables
python -c "import os; print('OPENAI_API_KEY' in os.environ)"

# Test API key
python -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.Model.list())
"
```

#### Vector Store Connection
```bash
# Test Pinecone
python -c "
import pinecone
import os
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)
print(pinecone.list_indexes())
"
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
pip install psutil
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

#### Embedding Generation
```bash
# Use smaller batch sizes for limited memory
export EMBEDDING_BATCH_SIZE=10

# Use smaller embedding model
export EMBEDDING_MODEL=text-embedding-3-small
```

### Logs and Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check application logs
tail -f logs/rag-system.log

# Monitor API requests
curl -X GET http://localhost:8000/health -v
```

## Next Steps

After successful installation:

1. **[Configuration Guide](configuration.md)** - Detailed configuration options
2. **[Examples](examples.md)** - Usage examples and tutorials
3. **[Deployment Guide](deployment.md)** - Production deployment
4. **[Architecture Guide](architecture.md)** - System design overview

## Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting) above
2. Review the [examples](../examples/) directory
3. Open an issue on GitHub with:
   - Your environment details
   - Error messages
   - Steps to reproduce