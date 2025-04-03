# GraphRAG API

A FastAPI-powered REST API for Graph-based Retrieval Augmented Generation (GraphRAG).

## Features

- **Document Ingestion**: Upload files or provide URLs to extract knowledge graphs
- **Graph-Based RAG**: Utilize graph relationships for more contextual responses
- **Streaming Responses**: Support for streaming LLM responses with proper UTF-8 encoding
- **Detailed Debugging**: Access full retrieval results, graph context and entity relationships
- **Configuration API**: Tune system parameters via API
- **Containerized Deployment**: Easy deployment with Docker and Docker Compose
- **Multi-Model Support**: Works with OpenAI or Ollama models

## Containerized Deployment

This project is fully containerized for easy deployment and includes:

- GraphRAG API (Python 3.11, FastAPI)
- Neo4j (Graph Database)
- Qdrant (Vector Database)
- Ollama (Local LLM Provider)

### Prerequisites

- Docker and Docker Compose
- If using OpenAI: An OpenAI API key
- ~10GB disk space for Docker images and volumes

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd graphrag
   ```

2. Start all services:
   ```bash
   ./run-docker.sh up
   ```

3. Access the API:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474 (user: neo4j, pass: password)
   - Qdrant Dashboard: http://localhost:6333/dashboard

### Configuration Options

You can customize the deployment using environment variables:

```bash
# Use OpenAI instead of Ollama
DEFAULT_MODEL_PROVIDER=openai OPENAI_API_KEY=your-key ./run-docker.sh up

# Change the Ollama model
OLLAMA_INFERENCE_MODEL=llama3 ./run-docker.sh up
```

### Container Management

The included `run-docker.sh` script makes management easy:

```bash
./run-docker.sh status    # Check container status
./run-docker.sh logs      # View all logs
./run-docker.sh api-logs  # View only API logs
./run-docker.sh down      # Stop all containers
./run-docker.sh restart   # Restart all containers
```

### Persistent Storage

The Docker setup includes persistent volumes for:

- `neo4j_data`: Graph database data
- `qdrant_data`: Vector database data
- `ollama_data`: Downloaded LLM models
- Local directory mappings:
  - `./uploads`: Uploaded files
  - `./logs`: Application logs

### Container Details

| Container         | Image                | Ports        | Description           |
|-------------------|----------------------|--------------|------------------------|
| graph-rag-api     | Custom (Python 3.11) | 8000         | GraphRAG API           |
| graph-rag-neo4j   | neo4j:5.9.0          | 7474, 7687   | Graph Database         |
| graph-rag-qdrant  | qdrant/qdrant:v1.4.0 | 6333, 6334   | Vector Database        |
| graph-rag-ollama  | ollama/ollama:latest | 11434        | Local LLM Provider     |

## API Endpoints

### Data Ingestion

- `POST /ingest/file` - Ingest data from a file
- `POST /ingest/url` - Ingest data from a URL

### Querying

- `POST /query` - Ask a question and get a response
- `POST /query/stream` - Ask a question with streaming response
- `POST /query/detailed` - Ask a question and get detailed response with graph context and retrieval results

### Data Management

- `POST /clear` - Clear all data from the system

## Local Development

If you prefer to run the components locally instead of with Docker:

### Prerequisites

- Python 3.11+
- Neo4j Database
- Qdrant Vector Database
- Ollama or OpenAI API access

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your configuration:

```
# Database Settings
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=graphRAGstore

# Model Provider Settings
DEFAULT_MODEL_PROVIDER=ollama  # or 'openai'

# OpenAI Settings (if using OpenAI)
OPENAI_API_KEY=your_key_here
OPENAI_INFERENCE_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_VECTOR_DIMENSION=1536

# Ollama Settings (if using Ollama)
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_INFERENCE_MODEL=llama2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_VECTOR_DIMENSION=768

# Processing Settings
PARALLEL_PROCESSING=true
MAX_WORKERS=4
BATCH_SIZE=100
CHUNK_SIZE=5000
USE_STREAMING=true
REQUEST_TIMEOUT=60
```

### Running Locally

```bash
uvicorn api:app --reload
```

## International Language Support

The API fully supports non-ASCII characters and international languages. All text processing maintains UTF-8 encoding throughout the pipeline, ensuring compatibility with any language supported by the underlying LLM models.

## Error Handling

- Robust error handling for connection issues
- Proper timeout management
- Graceful degradation when services are unavailable
