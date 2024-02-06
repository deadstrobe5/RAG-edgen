# RAG for edgenAI

This repository contains the implementation of the Retriever-And-Generator (RAG) system for the edgenAI project. The RAG system combines a powerful vector database with a sophisticated language model to provide accurate and contextually relevant responses to user queries.

## Prerequisites
To run this project, you will need:

- EdgenAI must be running on your machine.
- Docker installed on your machine.
- Python 3 installed on your machine.
- Netcat (nc) installed on your machine for service health checks.

## Getting Started

Follow these steps to get your RAG system up and running:

### 1. Install Dependencies

First, you need to install the necessary Python packages. Run this command to install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the Server

The project utilizes Qdrant for the vector database and Unstructured-API for document processing. A Makefile is provided to simplify the process of starting these services along with the RAG server.

Run the following command to start all services:

```bash
make
```
This command will:

- Check if Qdrant is running and start it if it isn't already running.
- Check if Unstructured-API is installed and running, and handle the installation or startup as needed.
- Wait until both Qdrant and Unstructured-API are up and running.
- Start the RAG server.

The Makefile handles the intricacies of setting up the services, including cloning repositories if necessary, building the Unstructured-API, and ensuring that all services are healthy before starting the RAG server.

### 3. Usage

After starting the services, the RAG system will be ready to receive and process queries. 
