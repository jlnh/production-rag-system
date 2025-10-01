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

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Setup environment
cp .env.example .env
# Edit .env with your API keys
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

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Configuration](docs/configuration.md)** - Environment and config options
- **[Examples](docs/examples.md)** - Usage examples and tutorials
- **[Deployment](docs/deployment.md)** - Production deployment guide
- **[Architecture](docs/architecture.md)** - System design and components

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

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for production RAG systems**

For detailed documentation, visit the [docs/](docs/) directory.