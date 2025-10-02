# RAG Production System

A comprehensive, production-ready Retrieval-Augmented Generation (RAG) system built from the ground up. This repository contains all the code extracted and organized from the "RAG Development: From Prototype to Production" course.

## 🚀 Key Features

- **Multi-format document processing** (PDF, DOCX, TXT, Markdown)
- **Hybrid search** combining vector and keyword retrieval
- **Production FastAPI server** with monitoring and security
- **Redis-based rate limiting and caching**
- **Automated quality gates** for deployment validation
- **Docker containerization** and CI/CD pipelines

## ⚡ Quick Start

### Using Docker (Recommended)

```bash
git clone <repository-url>
cd production-rag-system
cp .env.example .env
# Edit .env with your API keys
docker-compose -f docker/docker-compose.yml up
```

### Python Installation

```bash
git clone <repository-url>
cd production-rag-system

# Create virtual environment
python -m venv rag-env
source rag-env/bin/activate  # Linux/Mac
# rag-env\Scripts\activate    # Windows

# Install the package in editable mode (includes all dependencies)
pip install -e .

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY and PINECONE_API_KEY are required)
```

### Basic Usage

```python
from rag_system.core import DocumentProcessor, EmbeddingGenerator, VectorStore, RAGQueryProcessor

# Initialize components
doc_processor = DocumentProcessor(chunk_size=500, overlap=50)
embedding_generator = EmbeddingGenerator(model="text-embedding-3-small")
vector_store = VectorStore(backend="pinecone", index_name="my-docs")
query_processor = RAGQueryProcessor(embedding_generator, vector_store)

# Process and query
chunks = doc_processor.process_file("document.pdf")
chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
vector_store.store_chunks(chunks_with_embeddings)

result = query_processor.query("What is machine learning?")
print(result['answer'])
```

## 📚 Examples

```bash
# Basic RAG implementation
python examples/module1_basic_rag.py

# Advanced hybrid search
python examples/module2_hybrid_search.py

# Production API server
python examples/module3_production_api.py
```

### Required Configuration

Before running the examples, ensure you have:

1. **API Keys** set in `.env`:
   - `OPENAI_API_KEY` - Required for embeddings and LLM
   - `PINECONE_API_KEY` - Required for vector storage (or use ChromaDB/Weaviate)

#### Getting Your Pinecone API Key

To get your `PINECONE_API_KEY`:

1. Go to [https://pinecone.io](https://pinecone.io) and log in (or sign up for a free account)
2. Select your project in the Pinecone console
3. Navigate to the **API Keys** tab in your project dashboard
4. Click **Create API Key**, give it a name, and set permissions (typically "All" for Starter plan)
5. Copy the generated API key immediately and save it securely (you won't be able to view it again)
6. Add it to your `.env` file: `PINECONE_API_KEY=your-pinecone-api-key-here`

2. **Optional Services** (for full production features):
   - **Redis** - For caching and rate limiting
     ```bash
     # Using Docker
     docker run -d -p 6379:6379 redis:latest
     ```

3. **Vector Store Setup**:
   - **Pinecone**: Create an index in your Pinecone dashboard
   - **ChromaDB**: No setup needed (local storage)
   - **Weaviate**: Run Weaviate instance locally or in cloud

## 🧪 Testing

```bash
pytest                    # All tests
pytest tests/unit/        # Unit tests only
pytest --cov=src         # With coverage
```

## 📊 API & Monitoring

Start the production server:
```bash
uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000
```

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## 📖 Documentation

### Quick Links
- **[Installation Guide](docs/installation.md)** - Complete installation with Docker, venv, conda, and Poetry
- **[Configuration](docs/configuration.md)** - All environment variables and settings explained
- **[Examples](docs/examples.md)** - Code examples and Jupyter notebooks
- **[Deployment](docs/deployment.md)** - Production deployment with Docker, Kubernetes, AWS
- **[Architecture](docs/architecture.md)** - System design, components, and data flow

### Key Documentation Topics

**Getting Started:**
- [Python Environment Setup](docs/installation.md#python-environment-setup)
- [Docker Quick Start](docs/installation.md#quick-start-with-docker)
- [API Keys & Configuration](docs/installation.md#environment-configuration)

**Vector Stores:**
- [Pinecone Setup](docs/installation.md#pinecone) (v5.0+ with updated API)
- [Weaviate Setup](docs/installation.md#weaviate)
- [ChromaDB Setup](docs/installation.md#chromadb-local)

**Production:**
- [Redis Configuration](docs/installation.md#redis-installation)
- [API Deployment](docs/deployment.md)
- [Monitoring & Metrics](docs/configuration.md#monitoring-configuration)

## 🛠️ Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run quality checks
black src/ tests/         # Format code
flake8 src/               # Lint code
mypy src/                 # Type checking
pytest                    # Run tests
```

## 🔒 Security & Production

- API key authentication
- Rate limiting with Redis
- Input validation and sanitization
- Security headers (CORS, CSP)
- Comprehensive audit logging
- Quality gates for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 🔧 Troubleshooting

### Common Issues

**Import Errors / Circular Imports**
```bash
# Reinstall in editable mode
pip install -e .
```

**Pinecone API Errors**
- Ensure you're using `pinecone>=5.0.0` (not the old `pinecone-client`)
- Update your `.env` with valid `PINECONE_API_KEY`
- Remove `PINECONE_ENVIRONMENT` from `.env` (no longer needed in v5+)

**Redis Connection Errors**
- Redis is optional for development
- Start Redis: `docker run -d -p 6379:6379 redis:latest`
- Or set `CACHE_ENABLED=false` and `RATE_LIMIT_ENABLED=false` in `.env`

**OpenAI Authentication Errors**
- Set valid `OPENAI_API_KEY` in `.env`
- Check your API key at https://platform.openai.com/api-keys

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for production RAG systems**

For detailed documentation, visit the [docs/](docs/) directory.