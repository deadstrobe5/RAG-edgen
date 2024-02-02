import sys
import requests
import json
from edgen import Edgen
import os
import argparse

client = Edgen()
sys.path.insert(0, "canopy")
sys.path.insert(0, "canopy/src")

from canopy.knowledge_base.qdrant.qdrant_knowledge_base import QdrantKnowledgeBase
from canopy.models.data_models import Document, Query
from canopy.knowledge_base.qdrant.constants import COLLECTION_NAME_PREFIX
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_chunker import StubChunker

QDRANT_COLLECTION_NAME = "canopy--test_collection"
DEFAULT_FILE_PATH = "sibs.pdf"
DEFAULT_QUERY = "What is a cyber asset anyway?"
DEFAULT_TOP_K = 10

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Process input for RAG.")
parser.add_argument("--file_path", type=str, default=DEFAULT_FILE_PATH, help="The file path to process")
parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="The query to process")
parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="The number of top results to return (default: 10)")
args = parser.parse_args()

# Assigning variables from arguments or default values
FILE_PATH = args.file_path
QUERY = args.query
TOP_K = args.top_k


# Function to check if Qdrant server is running
def qdrant_server_running() -> bool:
    """Check if Qdrant server is running."""
    try:
        response = requests.get("http://localhost:6333", timeout=10.0)
        response_json = response.json()
        return response_json.get("title") == "qdrant - vector search engine"
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False
    

def load_documents_from_json(data):
    documents = []
    for item in data:
        # WAS HAVING PROBLEM WITH LINKS
        metadata = item.get("metadata", {})
        if 'links' in metadata:
            if isinstance(metadata['links'], list):
                # Join list into a string or perform other transformations as required
                metadata['links'] = ', '.join(map(str, metadata['links']))
            # Add more checks or transformations if needed
        
        document = Document(
            id=item.get("element_id", ""),
            text=item.get("text", ""),
            source=item.get("type", ""),
            metadata=metadata
        )
        documents.append(document)
    
    return documents

def document_with_text(text):
    return [
        Document(id=text, text=text),
    ]

def fetch_documents_from_unstructured(file_path):
    url = 'http://localhost:8000/general/v0/general'
    files = {'files': open(file_path, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return response.json()  # Parses the JSON response and returns it
    else:
        print(f"Error fetching documents: {response.status_code}")
        return None

def call_edgen(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True
            )

        content = ""
        for chunk in completion:
            content += chunk.choices[0].delta.content

        print(content)
        return content.strip()
    except Exception as e:  # Catch a general exception as the specific OpenAIError is not available
        print(f"Error calling EdgenAI API: {e}")
        return None

def initialize_qdrant():
    if not qdrant_server_running():
        print("Qdrant server is not running. Please start the server and try again.")
        return

    # Initialize stub components for the test
    chunker = StubChunker(num_chunks_per_doc=1)
    dense_encoder = StubDenseEncoder(dimension=768)  # Adjust if your encoder has a different dimension
    encoder = StubRecordEncoder(dense_encoder)

    # Initialize QdrantKnowledgeBase
    kb = QdrantKnowledgeBase(
        collection_name=QDRANT_COLLECTION_NAME,
        record_encoder=encoder,
        chunker=chunker,
        location="http://localhost:6333",  # Adjust if your Qdrant instance is hosted elsewhere
    )

    # Create collection if it doesn't exist
    try:
        kb._client.get_collection(kb.collection_name)
    except Exception as e:  # You might want to catch a more specific exception if possible
        print(f"Collection does not exist: {e}. Creating new collection.")
        kb.create_canopy_collection()

    return kb


# Main test function
def main():

    # Prepare prompt for LLM
    prompt = (
        "I need to generate a query for a vector database where each document is represented as a high-dimensional vector. "
        "The query should retrieve documents based on semantic similarity, focusing on the most relevant information related to the user's query. "
        "Please construct a precise and optimized query, using only necessary words that are highly relevant to the user's intent. "
        "Here are some examples of how a natural language query should be converted into a concise query for the vector database: "
        "\n\n"
        "- Natural Language Query: 'Find documents about renewable energy sources.'\n"
        "  Vector Database Query: 'renewable energy sources'\n"
        "- Natural Language Query: 'What are the economic impacts of climate change?'\n"
        "  Vector Database Query: 'economic impacts climate change'\n"
        "- Natural Language Query: 'Show studies on the effects of meditation on stress.'\n"
        "  Vector Database Query: 'meditation effects stress'\n"
        "\n"
        f"Now, convert the following user's natural language query into a concise query for the vector database:"
        f"\n\nUser's Natural Language Query: \"{QUERY}\""
        "\n\n"
        "Generated Query for Vector Database:"
    )   

    # Call OpenAI API
    vector_query = call_edgen(prompt)

    # Qdrant vector db
    kb = initialize_qdrant()
   
    # Create and insert documents
    #json = fetch_documents_from_unstructured(FILE_PATH)
    #docs = load_documents_from_json(json)
    #kb.upsert(docs)


    
    # Query the knowledge base
    query = Query(text=vector_query, top_k=TOP_K)
    query_results = kb.query([query])
    #print(query_results)

    # Prepare data for LLM
    context = ""
    for result in query_results:
        for doc in result.documents:
            # Assuming you have 'page_number', 'file_name' in your document's metadata
            page_info = f"Page number: {doc.metadata['page_number']}" if 'page_number' in doc.metadata else "Page number not available"
            file_info = f"File name: {doc.metadata['filename']}" if 'filename' in doc.metadata else "File name not available"
            context += f"{file_info}, {page_info}, Text: {doc.text}\n\n"

    #print(context)

    
    # Prepare prompt for LLM to generate the final response
    final_prompt = (
        "Based on the user's initial query, the following documents were retrieved:\n\n"
        f"{context}\n"
        "Each document snippet is accompanied by its file name and page number for reference. "
        "Using the information from these documents and considering the user's initial query: "
        f"'{QUERY}', generate a comprehensive response that accurately references the source documents:"
    )

    # Call OpenAI API for the final response
    final_answer = call_edgen(final_prompt)

    # Print LLM output
    print(f"LLM Response: {final_answer}")
    
    

# Run the main function
if __name__ == "__main__":
    main()
