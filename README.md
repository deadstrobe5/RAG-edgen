# RAG for edgenAI

This repository contains the implementation of the Retriever-And-Generator (RAG) system for the edgenAI project. The RAG system combines a powerful vector database with a sophisticated language model to provide accurate and contextually relevant responses to user queries.

## Prerequisites
To run this project, you will need:

- EdgenAI must be running on your machine.
- Docker installed on your machine.
- Python 3 installed on your machine.

## Getting Started

Follow these steps to get your RAG system up and running:

### 1. Install Dependencies

Run this command to install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the Qdrant Database

Run the Qdrant vector search engine using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

This command pulls the Qdrant image from Docker Hub (if not already locally available), starts a container, and exposes the service on port 6333.

### 3. Start the Unstructured-API Server

#### Using Docker

To initiate the Unstructured-API server, utilize Docker by executing the following commands:
```bash
docker pull downloads.unstructured.io/unstructured-io/unstructured-api:latest
docker run -p 8000:8000 -d --rm --name unstructured-api downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0
```

#### Configuring Vector Sizes and Other Settings

For specialized configurations, such as adjusting vector sizes, it's necessary to run the Unstructured-API directly, as the Docker image does not support these modifications.

Execute these steps to clone the repository, install dependencies, and run the application:

```bash
git clone https://github.com/Unstructured-IO/unstructured-api
cd unstructured-api
make install
make run-web-app
```
Note: Direct execution allows for more granular control over the Unstructured-API's configurations and parameters, suitable for custom and advanced setups.

### 4. Running Your RAG Implementation

After ensuring that both the Qdrant database and the unstructured-API server are running, you can start your RAG system. Ensure you have all necessary Python dependencies installed, and run your main script:

```bash
python3 rag-server.py
```
