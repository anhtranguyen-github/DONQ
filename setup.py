from setuptools import setup, find_packages

setup(
    name="graphrag-api",
    version="1.0.0",
    description="FastAPI application for Graph-based RAG system",
    author="GraphRAG Team",
    packages=find_packages(),
    install_requires=[
        "neo4j-graphrag[qdrant]",
        "python-dotenv",
        "pydantic",
        "openai",
        "requests",
        "typing-extensions",
        "spacy",
        "docling",
        "fastapi",
        "uvicorn",
        "python-multipart",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "graphrag-api=api:run_server",
            "graphrag-client=client:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 