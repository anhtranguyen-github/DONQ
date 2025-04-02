# GraphRAG API

A FastAPI-powered REST API for Graph-based Retrieval Augmented Generation (GraphRAG).

## Features

- **Document Ingestion**: Upload files or provide URLs to extract knowledge graphs
- **Graph-Based RAG**: Utilize graph relationships for more contextual responses
- **Streaming Responses**: Support for streaming LLM responses with proper UTF-8 encoding
- **Detailed Debugging**: Access full retrieval results, graph context and entity relationships
- **Configuration API**: Tune system parameters via API
- **Provider Support**: Works with OpenAI or Ollama models
- **Robust Error Handling**: Graceful handling of timeouts and connection issues
- **Containerized Deployment**: Easy deployment with Docker and Docker Compose

## Quick Start

### Prerequisites

- Python 3.8+ (for local installation)
- Neo4j database
- Qdrant vector database
- OpenAI API key or Ollama setup
- Docker and Docker Compose (for containerized deployment)

### Installation

#### Option 1: Local Installation

```bash
# Clone the repository
git clone [repository-url]
cd graphrag-api

# Install dependencies
pip install -e .
```

#### Option 2: Docker Deployment

```bash
# Clone the repository
git clone [repository-url]
cd graphrag-api

# Start all services with Docker Compose
docker-compose up -d

# Check the status of the services
docker-compose ps
```

The application will be available at http://localhost:8000

### Configuration

Create a `.env` file based on the provided `.env.example`:

```
# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Collection Name
COLLECTION_NAME=graphRAGstoreds

# Default Model Provider: "openai" or "ollama"
DEFAULT_MODEL_PROVIDER=openai

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key
OPENAI_INFERENCE_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_VECTOR_DIMENSION=1536

# Ollama Configuration (if using Ollama)
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_INFERENCE_MODEL=llama2:13b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_VECTOR_DIMENSION=768

# Processing Options
PARALLEL_PROCESSING=true
MAX_WORKERS=8
BATCH_SIZE=100
CHUNK_SIZE=5000
USE_STREAMING=true

# API Performance
REQUEST_TIMEOUT=60
```

## Docker Deployment

The project includes Docker support for easy deployment of the entire stack:

### Components

- **GraphRAG API**: The main application
- **Neo4j**: Graph database for entity relationships
- **Qdrant**: Vector database for semantic search
- **Ollama**: Local LLM provider for inference and embeddings

### Deployment Options

#### Full Stack Deployment

```bash
# Deploy all services
docker-compose up -d
```

#### Using External Services

If you already have Neo4j or Qdrant running:

1. Edit the `docker-compose.yml` file to remove the services you don't need
2. Update the environment variables for the API to point to your existing services

#### Environment Variables

For Docker deployment, you can override default environment variables:

```bash
# Use OpenAI instead of Ollama
DEFAULT_MODEL_PROVIDER=openai OPENAI_API_KEY=your_key docker-compose up -d
```

#### Volumes

The Docker Compose setup creates the following volumes:

- `neo4j_data`: Persistent Neo4j database
- `qdrant_data`: Persistent Qdrant vector database
- `ollama_data`: Ollama models
- Local directory mapping for uploads and logs

### Scaling

For production environments, consider:

- Increasing Neo4j memory settings in docker-compose.yml
- Using Docker Swarm or Kubernetes for orchestration

## International Language Support

The API fully supports non-ASCII characters and international languages in both input and output. All text processing is handled with proper UTF-8 encoding to ensure compatibility with languages like Chinese, Vietnamese, Japanese, and others.

### Running the API

```bash
# Start the API server
graphrag-api --host 0.0.0.0 --port 8000

# For development with auto-reload
graphrag-api --reload
```

### Using the Client

```bash
# Run the interactive client
graphrag-client

# Set custom API URL
export GRAPHRAG_API_URL=http://localhost:8000
graphrag-client
```

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

### Configuration

- `GET /config` - Get current configuration
- `PATCH /config` - Update configuration settings

### System

- `GET /status` - Get system status and version information

For detailed API documentation, visit `/docs` when the server is running.

## Architecture

This application combines:

1. **FastAPI**: For RESTful API endpoints and request validation
2. **Neo4j**: Graph database for storing extracted entities and relationships
3. **Qdrant**: Vector database for semantic search
4. **DocLing**: Document processor for handling various file formats
5. **LLM Integration**: OpenAI API or Ollama for graph extraction and RAG

## Error Handling

The application includes comprehensive error handling:

- Automatic retries for extraction failures
- Fallback processing for parallel extraction issues
- Detailed logging to `graphrag_api.log`
- Graceful handling of connection issues
- Proper UTF-8 encoding for international text

## Performance Optimization

- **Parallel Processing**: Extract graph components in parallel
- **Batch Operations**: Optimize database operations
- **Streaming Responses**: Deliver content without waiting for complete generation
- **Configurable Parameters**: Tune chunk size, batch size, and worker count
- **Request Timeouts**: Configurable timeout handling for all requests
- **Nginx Compatible**: Includes proper headers for Nginx streaming support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
