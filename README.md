# RAG for edgenAI

This repository contains the implementation of the Retriever-And-Generator (RAG) system for the edgenAI project. The RAG system combines a powerful vector database with a sophisticated language model to provide accurate and contextually relevant responses to user queries.

## Prerequisites
To run this project, you will need:

- EdgenAI must be running on your machine.
- Docker installed on your machine.
- Python 3 installed on your machine.

## Getting Started

Follow these steps to get your RAG system up and running:

### 1. Start the Qdrant Database

Run the Qdrant vector search engine using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

This command pulls the Qdrant image from Docker Hub (if not already locally available), starts a container, and exposes the service on port 6333.

### 2. Start the Unstructured-API Server

Pull and run the unstructured-api server using Docker:

```bash
docker pull downloads.unstructured.io/unstructured-io/unstructured-api:latest
docker run -p 8000:8000 -d --rm --name unstructured-api downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0
```

### 3. Running Your RAG Implementation

After ensuring that both the Qdrant database and the unstructured-API server are running, you can start your RAG system. Ensure you have all necessary Python dependencies installed, and run your main script:

```bash
python3 main_script.py
```
