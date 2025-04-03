# GraphRAG API

A FastAPI-powered REST API for Graph-based Retrieval Augmented Generation (GraphRAG).

## Features

- **Document Ingestion**: Upload files or provide URLs to extract knowledge graphs
- **Graph-Based RAG**: Utilize graph relationships for more contextual responses
- **Streaming Responses**: Support for streaming LLM responses with proper UTF-8 encoding
- **Detailed Debugging**: Access full retrieval results, graph context and entity relationships
- **Configuration API**: Tune system parameters via API
- **Containerized Deployment**: Easy deployment with Docker and Docker Compose
- **Multi-Model Support**: Works with OpenAI, Anthropic, or Ollama models

## Containerized Deployment

This project is fully containerized for easy deployment and includes:

- GraphRAG API (Python, FastAPI)
- Neo4j (Graph Database)
- Qdrant (Vector Database)
- Ollama (Local LLM Provider)

### Prerequisites

- Docker and Docker Compose
- If using OpenAI or Anthropic: An API key
- ~10GB disk space for Docker images and volumes

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd graphrag
   ```

2. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

3. Start all services:
   ```bash
   docker compose up -d
   ```

4. Access the API:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474 (user: neo4j, pass: password)
   - Qdrant Dashboard: http://localhost:6333/dashboard

### Environment Configuration

The system uses environment variables for configuration:

1. Create a `.env` file based on the provided `.env.example`
2. Docker Compose will automatically load variables from this file
3. Container-specific variables (like host connections) are automatically overridden in docker-compose.yml

#### Configuration Options

Important configuration variables:

```
# Model Provider Options
DEFAULT_MODEL_PROVIDER=ollama  # options: 'openai', 'anthropic', 'ollama'

# OpenAI Configuration
OPENAI_API_KEY=your_key_here
OPENAI_INFERENCE_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Anthropic Configuration
ANTHROPIC_API_KEY=your_key_here

# Ollama Configuration
OLLAMA_INFERENCE_MODEL=qwen2.5:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Processing Settings
PARALLEL_PROCESSING=true
MAX_WORKERS=4
CHUNK_SIZE=5000
```

### Container Management

Basic Docker Compose commands:

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps

# Stop services
docker compose down

# Restart services
docker compose restart
```

### Persistent Storage

The Docker setup includes persistent volumes for:

- `neo4j_data`: Graph database data
- `neo4j_logs`: Neo4j logs
- `qdrant_data`: Vector database data
- `ollama_data`: Downloaded LLM models
- Local directory mappings:
  - `./uploads`: Uploaded files
  - `./logs`: Application logs

### Container Details

| Container        | Image                | Ports        | Description           |
|------------------|----------------------|--------------|------------------------|
| graphrag-api     | Custom (Dockerfile)  | 8000         | GraphRAG API           |
| graphrag-neo4j   | neo4j:5.9.0          | 7474, 7687   | Graph Database         |
| graphrag-qdrant  | qdrant/qdrant:v1.13.3| 6333, 6334   | Vector Database        |
| graphrag-ollama  | ollama/ollama:0.6.2  | 11434        | Local LLM Provider     |

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

- Python 3.10+
- Neo4j Database
- Qdrant Vector Database
- Ollama or OpenAI/Anthropic API access

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your configuration (use the provided `.env.example` as a template).

For local development without Docker, ensure the host settings point to your locally running services:

```
NEO4J_URI=bolt://localhost:7687
QDRANT_HOST=localhost
OLLAMA_HOST=localhost
```

### Running Locally

```bash
uvicorn app.main:app --reload
```

## International Language Support

The API fully supports non-ASCII characters and international languages. All text processing maintains UTF-8 encoding throughout the pipeline, ensuring compatibility with any language supported by the underlying LLM models.

## Error Handling

- Robust error handling for connection issues
- Proper timeout management
- Graceful degradation when services are unavailable
