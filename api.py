from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from pydantic import BaseModel, Field
import asyncio
import traceback
from dotenv import load_dotenv

# Version information
API_VERSION = "1.0.0"
API_TITLE = "GraphRAG API"
API_DESCRIPTION = """
API for ingesting data, running queries, and managing a graph-based RAG system.

This API allows you to:
- Upload and process documents (files or URLs)
- Clear stored data
- Ask questions using graph-based retrieval augmented generation
- Configure system parameters
"""

# Import existing functionality from graph_rag.py
from graph_rag import (
    initialize_clients,
    create_collection,
    extract_graph_components,
    extract_graph_components_parallel,
    ingest_to_neo4j,
    ingest_to_qdrant,
    retriever_search,
    fetch_related_graph,
    format_graph_context,
    graphRAG_run,
    clear_data,
    VECTOR_DIMENSION
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("graphrag_api.log")
    ]
)
logger = logging.getLogger("graphrag.api")

# Import the processor factory and reader
from processors.processor_factory import get_processor, reload_config
from reader.docling_reader import DoclingReader

# Load environment variables
load_dotenv()

# Initialize application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for requests and responses
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the GraphRAG system")
    use_streaming: bool = Field(True, description="Whether to stream the response")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The answer from the GraphRAG system")
    processing_time: float = Field(..., description="Total processing time in seconds")

class DetailedQueryResponse(BaseModel):
    answer: str = Field(..., description="The answer from the GraphRAG system")
    processing_time: float = Field(..., description="Total processing time in seconds")
    retrieval_items: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved items from vector store")
    graph_context: Optional[Dict[str, Any]] = Field(None, description="Graph context used for generation")
    entity_ids: Optional[List[str]] = Field(None, description="Entity IDs used in the query")
    
class ConfigRequest(BaseModel):
    parallel_processing: Optional[bool] = Field(None, description="Enable parallel processing")
    max_workers: Optional[int] = Field(None, description="Number of workers for parallel processing")
    batch_size: Optional[int] = Field(None, description="Batch size for database operations")
    chunk_size: Optional[int] = Field(None, description="Chunk size for text processing")
    use_streaming: Optional[bool] = Field(None, description="Enable response streaming")

class ConfigResponse(BaseModel):
    settings: Dict[str, Any] = Field(..., description="Current configuration settings")
    message: str = Field(..., description="Status message")

class StatusResponse(BaseModel):
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Status message")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds if applicable")
    info: Optional[Dict[str, Any]] = Field(None, description="Additional information about the system")

# Global configuration with default values
config = {
    "parallel_processing": os.getenv("PARALLEL_PROCESSING", "true").lower() == "true",
    "max_workers": int(os.getenv("MAX_WORKERS", "8")),
    "batch_size": int(os.getenv("BATCH_SIZE", "100")),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "5000")),
    "use_streaming": os.getenv("USE_STREAMING", "true").lower() == "true"
}

# Dependencies
async def get_db_clients():
    """Dependency to get database clients"""
    neo4j_driver, qdrant_client, collection_name = initialize_clients()
    try:
        yield (neo4j_driver, qdrant_client, collection_name)
    finally:
        neo4j_driver.close()

# Initialize clients at startup
@app.on_event("startup")
async def startup_event():
    """Initialize resources at startup"""
    logger.info("Starting GraphRAG API server")
    try:
        # Initialize clients
        neo4j_driver, qdrant_client, collection_name = initialize_clients()
        logger.info(f"Connected to Neo4j at {os.getenv('NEO4J_URI')}")
        logger.info(f"Connected to Qdrant at {os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT')}")
        
        # Ensure collection exists
        create_collection(qdrant_client, collection_name, VECTOR_DIMENSION)
        logger.info(f"Initialized Qdrant collection '{collection_name}' with vector dimension {VECTOR_DIMENSION}")
        
        # Close Neo4j driver
        neo4j_driver.close()
    except Exception as e:
        log_and_handle_error(e, "startup_event")
        logger.error("Failed to initialize resources at startup. API may not function correctly.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources at shutdown"""
    logger.info("Shutting down GraphRAG API server")

# API Routes
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to docs"""
    logger.info("Root endpoint called - redirecting to /docs")
    return RedirectResponse(url="/docs")

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def status():
    """Health check endpoint - return system status"""
    logger.info("Status endpoint called")
    
    # Get environment info
    model_provider, _ = reload_config()
    processor = get_processor()
    
    # Add additional information to the response
    info = {
        "api_version": API_VERSION,
        "model_provider": model_provider.upper(),
        "llm_model": processor["LLM_MODEL"],
        "embedding_model": processor["EMBEDDING_MODEL"],
        "vector_dimension": VECTOR_DIMENSION,
    }
    
    return {
        "success": True,
        "message": "GraphRAG API is running and healthy",
        "info": info
    }

@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """Get current configuration settings"""
    logger.info("Getting current configuration")
    try:
        model_provider, _ = reload_config()
        processor = get_processor()
        
        # Add model info to config
        config_with_models = {
            **config,
            "model_provider": model_provider.upper(),
            "llm_model": processor["LLM_MODEL"],
            "embedding_model": processor["EMBEDDING_MODEL"],
        }
        
        logger.info(f"Configuration retrieved with model provider: {model_provider.upper()}")
        return {
            "settings": config_with_models,
            "message": "Current configuration retrieved successfully"
        }
    except Exception as e:
        log_and_handle_error(e, "get_config")
        raise HTTPException(status_code=500, detail=f"Error retrieving configuration: {str(e)}")

@app.patch("/config", response_model=ConfigResponse, tags=["Configuration"])
async def update_config(config_request: ConfigRequest):
    """Update configuration settings"""
    logger.info(f"Updating configuration: {config_request.dict(exclude_unset=True)}")
    try:
        # Update config with provided values
        updated_settings = {}
        for key, value in config_request.dict(exclude_unset=True).items():
            if value is not None:
                config[key] = value
                updated_settings[key] = value
        
        logger.info(f"Updated settings: {updated_settings}")
        
        model_provider, _ = reload_config()
        processor = get_processor()
        
        # Add model info to response
        config_with_models = {
            **config,
            "model_provider": model_provider.upper(),
            "llm_model": processor["LLM_MODEL"],
            "embedding_model": processor["EMBEDDING_MODEL"],
        }
        
        return {
            "settings": config_with_models,
            "message": "Configuration updated successfully"
        }
    except Exception as e:
        log_and_handle_error(e, "update_config")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@app.post("/ingest/file", response_model=StatusResponse, tags=["Data Ingestion"])
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db_clients: tuple = Depends(get_db_clients)
):
    """Ingest data from an uploaded file"""
    logger.info(f"Received file upload request: {file.filename}")
    neo4j_driver, qdrant_client, collection_name = db_clients
    
    # Save the uploaded file
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved uploaded file to {file_path} ({len(contents)} bytes)")
    except Exception as e:
        log_and_handle_error(e, "save_uploaded_file")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Process the file in a background task
    logger.info(f"Starting background processing task for file: {file.filename}")
    background_tasks.add_task(
        process_file, 
        file_path, 
        neo4j_driver, 
        qdrant_client, 
        collection_name,
        config
    )
    
    return {
        "success": True,
        "message": f"File '{file.filename}' uploaded and processing started in the background"
    }

@app.post("/ingest/url", response_model=StatusResponse, tags=["Data Ingestion"])
async def ingest_url(
    background_tasks: BackgroundTasks,
    url: str = Body(..., embed=True),
    db_clients: tuple = Depends(get_db_clients)
):
    """Ingest data from a URL"""
    logger.info(f"Received URL ingestion request: {url}")
    neo4j_driver, qdrant_client, collection_name = db_clients
    
    # Process the URL in a background task
    logger.info(f"Starting background processing task for URL: {url}")
    background_tasks.add_task(
        process_url, 
        url, 
        neo4j_driver, 
        qdrant_client, 
        collection_name,
        config
    )
    
    return {
        "success": True,
        "message": f"URL ingestion started in the background"
    }

@app.post("/clear", response_model=StatusResponse, tags=["Data Management"])
async def clear_all_data(db_clients: tuple = Depends(get_db_clients)):
    """Clear all data from Neo4j and Qdrant"""
    logger.info("Received request to clear all data")
    neo4j_driver, qdrant_client, collection_name = db_clients
    
    try:
        logger.info("Clearing all data from Neo4j and Qdrant")
        clear_data(neo4j_driver, qdrant_client, collection_name)
        logger.info("All data cleared successfully")
        return {
            "success": True,
            "message": "All data cleared successfully"
        }
    except Exception as e:
        log_and_handle_error(e, "clear_data")
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["Querying"])
async def query(
    request: QueryRequest,
    db_clients: tuple = Depends(get_db_clients)
):
    """Ask a question to the GraphRAG system"""
    neo4j_driver, qdrant_client, collection_name = db_clients
    query = request.query
    use_streaming = request.use_streaming
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    logger.info(f"Processing query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    try:
        # Start retriever search
        retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
        
        if not hasattr(retriever_result, 'items') or not retriever_result.items:
            logger.warning("No relevant information found for the query")
            return {
                "answer": "No relevant information found. Try ingesting some data first.",
                "processing_time": time.time() - start_time
            }
        
        logger.info(f"Found {len(retriever_result.items)} relevant items")
        
        # Extract entity IDs
        try:
            entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
            logger.info(f"Extracted {len(entity_ids)} entity IDs")
        except Exception as parsing_error:
            return log_and_handle_error(
                parsing_error, 
                "entity_id_extraction",
                lambda: {
                    "answer": f"Error parsing search results: {str(parsing_error)}",
                    "processing_time": time.time() - start_time
                }
            )
        
        # Fetch related graph
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        logger.info(f"Fetched subgraph with {len(subgraph)} connections")
        
        # Format graph context
        graph_context = format_graph_context(subgraph)
        logger.info(f"Formatted graph context with {len(graph_context['nodes'])} nodes and {len(graph_context['edges'])} edges")
        
        # Run GraphRAG
        start_answer_time = time.time()
        logger.info(f"Running GraphRAG query with streaming={use_streaming}")
        
        try:
            if use_streaming:
                # Just get the full answer for now in the REST API
                # In a future version we could implement streaming responses
                model_provider, _ = reload_config()
                
                if model_provider == "openai":
                    stream_response = graphRAG_run(graph_context, query, stream=True)
                    full_answer = ""
                    for chunk in stream_response:
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            full_answer += chunk.choices[0].delta.content
                else:
                    stream_response = graphRAG_run(graph_context, query, stream=True)
                    full_answer = ""
                    for content in stream_response:
                        full_answer += content
            else:
                full_answer = graphRAG_run(graph_context, query, stream=False)
        except Exception as llm_error:
            return log_and_handle_error(
                llm_error, 
                "llm_query_execution",
                lambda: {
                    "answer": f"Error generating answer: {str(llm_error)}",
                    "processing_time": time.time() - start_time
                }
            )
        
        end_time = time.time()
        query_time = end_time - start_time
        answer_time = end_time - start_answer_time
        logger.info(f"Query completed in {query_time:.2f}s (answer generation: {answer_time:.2f}s)")
        
        return {
            "answer": full_answer,
            "processing_time": query_time
        }
    
    except Exception as e:
        log_and_handle_error(e, "query_endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/stream", tags=["Querying"])
async def stream_query(
    request: QueryRequest,
    db_clients: tuple = Depends(get_db_clients)
):
    """Ask a question to the GraphRAG system with streaming response"""
    neo4j_driver, qdrant_client, collection_name = db_clients
    query = request.query
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Processing streaming query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    try:
        # Start retriever search
        retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
        
        if not hasattr(retriever_result, 'items') or not retriever_result.items:
            logger.warning("No relevant information found for the streaming query")
            async def error_stream():
                yield "No relevant information found. Try ingesting some data first.".encode('utf-8')
            return StreamingResponse(error_stream(), media_type="text/plain; charset=utf-8")
        
        logger.info(f"Found {len(retriever_result.items)} relevant items for streaming query")
        
        # Extract entity IDs
        try:
            entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
            logger.info(f"Extracted {len(entity_ids)} entity IDs")
        except Exception as parsing_error:
            log_and_handle_error(parsing_error, "entity_id_extraction_streaming")
            async def parsing_error_stream():
                yield f"Error parsing search results: {str(parsing_error)}".encode('utf-8')
            return StreamingResponse(parsing_error_stream(), media_type="text/plain; charset=utf-8")
        
        # Fetch related graph
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        logger.info(f"Fetched subgraph with {len(subgraph)} connections")
        
        # Format graph context
        graph_context = format_graph_context(subgraph)
        logger.info(f"Formatted graph context with {len(graph_context['nodes'])} nodes and {len(graph_context['edges'])} edges")
        
        # Run GraphRAG with streaming
        async def response_stream():
            try:
                model_provider, _ = reload_config()
                logger.info(f"Starting streaming response with model provider: {model_provider}")
                stream_response = graphRAG_run(graph_context, query, stream=True)
                
                if model_provider == "openai":
                    for chunk in stream_response:
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            yield content.encode('utf-8')
                else:
                    for content in stream_response:
                        if content:
                            yield content.encode('utf-8')
                
                logger.info("Streaming response completed")
            except Exception as stream_error:
                error_msg = f"\nError occurred during streaming: {str(stream_error)}"
                log_and_handle_error(stream_error, "streaming_response")
                yield error_msg.encode('utf-8')
        
        return StreamingResponse(
            response_stream(),
            media_type="text/plain; charset=utf-8",
            headers={"X-Accel-Buffering": "no"}  # Disable buffering for Nginx
        )
    
    except Exception as e:
        log_and_handle_error(e, "stream_query_endpoint")
        async def error_stream():
            yield f"Error processing streaming query: {str(e)}".encode('utf-8')
        return StreamingResponse(
            error_stream(),
            media_type="text/plain; charset=utf-8",
            status_code=500
        )

@app.post("/query/detailed", response_model=DetailedQueryResponse, tags=["Querying"])
async def detailed_query(
    request: QueryRequest,
    db_clients: tuple = Depends(get_db_clients)
):
    """Ask a question and get detailed response including retrieval results and graph data"""
    neo4j_driver, qdrant_client, collection_name = db_clients
    query = request.query
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    logger.info(f"Processing detailed query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    try:
        # Start retriever search
        retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
        
        if not hasattr(retriever_result, 'items') or not retriever_result.items:
            logger.warning("No relevant information found for the detailed query")
            return {
                "answer": "No relevant information found. Try ingesting some data first.",
                "processing_time": time.time() - start_time,
                "retrieval_items": [],
                "graph_context": {"nodes": [], "edges": []},
                "entity_ids": []
            }
        
        logger.info(f"Found {len(retriever_result.items)} relevant items")
        
        # Extract entity IDs
        try:
            entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
            logger.info(f"Extracted {len(entity_ids)} entity IDs")
        except Exception as parsing_error:
            return log_and_handle_error(
                parsing_error, 
                "entity_id_extraction",
                lambda: {
                    "answer": f"Error parsing search results: {str(parsing_error)}",
                    "processing_time": time.time() - start_time,
                    "retrieval_items": [],
                    "graph_context": {"nodes": [], "edges": []},
                    "entity_ids": []
                }
            )
        
        # Convert retriever items to serializable format
        retrieval_items = []
        for item in retriever_result.items:
            retrieval_items.append({
                "content": item.content,
                "score": item.score if hasattr(item, "score") else None,
                "metadata": item.metadata if hasattr(item, "metadata") else {}
            })
        
        # Fetch related graph
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        logger.info(f"Fetched subgraph with {len(subgraph)} connections")
        
        # Format graph context
        graph_context = format_graph_context(subgraph)
        logger.info(f"Formatted graph context with {len(graph_context['nodes'])} nodes and {len(graph_context['edges'])} edges")
        
        # Run GraphRAG
        start_answer_time = time.time()
        logger.info(f"Running GraphRAG query")
        
        try:
            # Get the full answer (non-streaming for detailed response)
            full_answer = graphRAG_run(graph_context, query, stream=False)
        except Exception as llm_error:
            return log_and_handle_error(
                llm_error, 
                "llm_query_execution",
                lambda: {
                    "answer": f"Error generating answer: {str(llm_error)}",
                    "processing_time": time.time() - start_time,
                    "retrieval_items": retrieval_items,
                    "graph_context": graph_context,
                    "entity_ids": entity_ids
                }
            )
        
        end_time = time.time()
        query_time = end_time - start_time
        answer_time = end_time - start_answer_time
        logger.info(f"Detailed query completed in {query_time:.2f}s (answer generation: {answer_time:.2f}s)")
        
        return {
            "answer": full_answer,
            "processing_time": query_time,
            "retrieval_items": retrieval_items,
            "graph_context": graph_context,
            "entity_ids": entity_ids
        }
    
    except Exception as e:
        log_and_handle_error(e, "detailed_query_endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing detailed query: {str(e)}")

# Helper functions for background tasks
async def process_file(file_path, neo4j_driver, qdrant_client, collection_name, config):
    """Process a file in the background"""
    try:
        start_time = time.time()
        docling = DoclingReader()
        raw_data = docling.read_from_file_as_text(file_path)
        print(raw_data)
        
        if not raw_data.strip():
            logger.warning(f"No data found in file {file_path}")
            return
        
        try:
            # Extract graph components
            if config["parallel_processing"]:
                logger.info(f"Processing file with parallel extraction (max_workers={config['max_workers']}, chunk_size={config['chunk_size']})")
                nodes, relationships = extract_graph_components_parallel(
                    raw_data, 
                    chunk_size=config["chunk_size"], 
                    max_workers=config["max_workers"]
                )
            else:
                logger.info("Processing file with sequential extraction")
                nodes, relationships = extract_graph_components(raw_data)
            
            # Check if we have valid nodes and relationships
            if not nodes:
                return log_and_handle_error(
                    Exception("No valid nodes were extracted"), 
                    "extract_graph_components"
                )
            
            logger.info(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
            
            # Ingest to Neo4j
            logger.info(f"Ingesting to Neo4j (batch_size={config['batch_size']})")
            node_id_mapping = ingest_to_neo4j(
                neo4j_driver, 
                nodes, 
                relationships, 
                batch_size=config["batch_size"]
            )
            
            # Ingest to Qdrant
            logger.info(f"Ingesting to Qdrant collection '{collection_name}'")
            ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"File processing completed in {processing_time:.2f} seconds")
        
        except Exception as extraction_error:
            log_and_handle_error(extraction_error, "file_extraction_process")
            
            # Try with smaller chunks or without parallel processing if that was the issue
            if config["parallel_processing"]:
                logger.info("Retrying with single-threaded processing...")
                try:
                    nodes, relationships = extract_graph_components(raw_data)
                    
                    if nodes:
                        # Ingest to Neo4j
                        node_id_mapping = ingest_to_neo4j(
                            neo4j_driver, 
                            nodes, 
                            relationships, 
                            batch_size=config["batch_size"]
                        )
                        
                        # Ingest to Qdrant
                        ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        logger.info(f"File processing completed with fallback method in {processing_time:.2f} seconds")
                except Exception as fallback_error:
                    log_and_handle_error(fallback_error, "fallback_extraction_process")
    
    except Exception as e:
        log_and_handle_error(e, "process_file")

async def process_url(url, neo4j_driver, qdrant_client, collection_name, config):
    """Process a URL in the background"""
    try:
        start_time = time.time()
        docling = DoclingReader()
        raw_data = docling.read_from_url_as_text(url)
        
        if not raw_data.strip():
            logger.warning(f"No data found from URL {url}")
            return
        
        try:
            # Extract graph components
            if config["parallel_processing"]:
                logger.info(f"Processing URL with parallel extraction (max_workers={config['max_workers']}, chunk_size={config['chunk_size']})")
                nodes, relationships = extract_graph_components_parallel(
                    raw_data, 
                    chunk_size=config["chunk_size"], 
                    max_workers=config["max_workers"]
                )
            else:
                logger.info("Processing URL with sequential extraction")
                nodes, relationships = extract_graph_components(raw_data)
            
            # Check if we have valid nodes and relationships
            if not nodes:
                return log_and_handle_error(
                    Exception("No valid nodes were extracted"), 
                    "extract_graph_components"
                )
            
            logger.info(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
            
            # Ingest to Neo4j
            logger.info(f"Ingesting to Neo4j (batch_size={config['batch_size']})")
            node_id_mapping = ingest_to_neo4j(
                neo4j_driver, 
                nodes, 
                relationships, 
                batch_size=config["batch_size"]
            )
            
            # Ingest to Qdrant
            logger.info(f"Ingesting to Qdrant collection '{collection_name}'")
            ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"URL processing completed in {processing_time:.2f} seconds")
        
        except Exception as extraction_error:
            log_and_handle_error(extraction_error, "url_extraction_process")
            
            # Try with smaller chunks or without parallel processing if that was the issue
            if config["parallel_processing"]:
                logger.info("Retrying with single-threaded processing...")
                try:
                    nodes, relationships = extract_graph_components(raw_data)
                    
                    if nodes:
                        # Ingest to Neo4j
                        node_id_mapping = ingest_to_neo4j(
                            neo4j_driver, 
                            nodes, 
                            relationships, 
                            batch_size=config["batch_size"]
                        )
                        
                        # Ingest to Qdrant
                        ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        logger.info(f"URL processing completed with fallback method in {processing_time:.2f} seconds")
                except Exception as fallback_error:
                    log_and_handle_error(fallback_error, "fallback_extraction_process")
    
    except Exception as e:
        log_and_handle_error(e, "process_url")

# Utility function for error handling
def log_and_handle_error(error: Exception, context: str = "", return_func: Optional[Callable] = None):
    """
    Log error consistently and optionally return a value if return_func is provided.
    
    Args:
        error: The exception that was caught
        context: Context information about where the error occurred
        return_func: Optional function to call to generate a return value
    
    Returns:
        The result of return_func if provided, otherwise None
    """
    # Get the stack trace
    trace = traceback.format_exc()
    
    # Log the error with context and stack trace
    logger.error(f"Error in {context}: {str(error)}\nStack trace:\n{trace}")
    
    # Return a value if a return function is provided
    if return_func:
        return return_func()
    return None

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Middleware to handle request timeouts"""
    try:
        # Default timeout is 60 seconds
        timeout_seconds = int(os.getenv("REQUEST_TIMEOUT", "60"))
        
        # Exclude streaming endpoints from timeout
        if request.url.path in ["/query/stream"]:
            return await call_next(request)
            
        # Apply timeout to non-streaming requests
        return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"Request to {request.url.path} timed out after {timeout_seconds} seconds")
        return JSONResponse(
            status_code=504,
            content={"detail": f"Request timed out after {timeout_seconds} seconds"}
        )
    except Exception as e:
        logger.error(f"Unhandled error in middleware: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

def run_server():
    """Run the server - used as an entry point for the console script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the GraphRAG API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()
    
    logger.info(f"Starting GraphRAG API server on {args.host}:{args.port}")
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    run_server()
