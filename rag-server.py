import sys
sys.path.insert(0, "canopy")
sys.path.insert(0, "canopy/src")

import json
from edgen import Edgen
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger
from canopy.knowledge_base.qdrant.qdrant_knowledge_base import QdrantKnowledgeBase
from canopy.models.data_models import Document, Query
from canopy.knowledge_base.qdrant.constants import COLLECTION_NAME_PREFIX
from tests.unit.stubs.stub_record_encoder import StubRecordEncoder
from tests.unit.stubs.stub_dense_encoder import StubDenseEncoder
from tests.unit.stubs.stub_chunker import StubChunker
import asyncio
import http3
import aiofiles
from io import BytesIO
import httpx
from qdrant_client import QdrantClient



app = FastAPI()
client = Edgen()
httpclient = http3.AsyncClient()
qdrantclient = QdrantClient(url="http://localhost:6333")

QDRANT_COLLECTION_NAME = "canopy--sibs2"
FILE_PATH = "sibs2.pdf"
QUERY = "What is a cyber asset anyway?"
TOP_K = 10
kb = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global QUERY
    global kb
    data = await request.json()
    QUERY = data.get('messages', '')[-1]["content"]
    print(f"User input: {QUERY}")
    if kb == None:
        kb = await initialize_vectordb()
    return StreamingResponse(run_rag(QUERY, kb), media_type="text/event-stream")


async def initialize_vectordb():
    global kb
    # Qdrant vector db
    kb = await initialize_qdrant()
   
    # Create and insert documents
    #print("Loading Documents...")
    #json = await fetch_documents_from_unstructured(FILE_PATH)
    #docs = await load_documents_from_json(json)
    #kb.upsert(docs)

    return kb



async def run_rag(prompt, kb):
    # Prepare prompt for LLM
    prompt = (
        "Objective: Process a natural language query to identify and extract its core informational intent, "
        "then convert this essence into a clean, concise set of keywords suitable for a vector database query. "
        "Omit non-informative words (like expletives, fillers, etc.) and focus solely on meaningful, descriptive keywords. "
        "These keywords are used for retrieving documents based on semantic similarity. Provide only the essential keywords as output, "
        "ensuring they accurately represent the user's informational need. See examples for reference: "
        "\n\n"
        "- Natural: 'Find documents about renewable energy sources.'\n"
        "  Vector: 'renewable energy sources'\n"
        "- Natural: 'What are the economic impacts of climate change?'\n"
        "  Vector: 'economic impacts climate change'\n"
        "- Natural: 'Show studies on the effects of meditation on stress.'\n"
        "  Vector: 'meditation effects stress'\n"
        "\n"
        f"Input User's Natural Language Query: \"{QUERY}\"\n"
        "Expected Output (Clean Vector Database Query Keywords):"
    )

    # Call Edgen
    vector_query = await call_edgen(prompt)
    print(f"Vector query: {vector_query}")
    
    
    # Query the knowledge base
    query = Query(text=vector_query, top_k=TOP_K)
    query_results = kb.query([query])


    # Prepare data for LLM
    context = ""
    for result in query_results:
        for doc in result.documents:
            #print(doc)
            #if doc.source != "NarrativeText":
            #    continue
            # Assuming you have 'page_number', 'file_name' in your document's metadata
            page_info = f"Page number: {doc.metadata['page_number']}" if 'page_number' in doc.metadata else "Page number not available"
            file_info = f"File name: {doc.metadata['filename']}" if 'filename' in doc.metadata else "File name not available"
            context += f"{file_info}, {page_info}, Text: {doc.text}\n\n\n\n\n"

    
    # Prepare prompt for LLM to generate the final response
    final_prompt = (
        "Based on these contextual documents:\n\n"
        f"{context}\n"
        "Based on the user's initial query: "
        f"'{QUERY}', "
        "Synthesize the key information from the provided document snippets into a coherent response. "
    )


    print(final_prompt)
    
    """Call the Edgen API and stream the response."""
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ],
            stream=True
        )

        for chunk in completion:
            
            # Assuming chunk is an object with the structure you provided, construct the dictionary
            chunk_dict = {
                "choices": [
                    {
                        "delta": {
                            "content":choice.delta.content,
                        },
                        "finish_reason":choice.finish_reason,
                        "index":choice.index
                    } for choice in chunk.choices
                ],
                "created":chunk.created,
                "id":chunk.id,
                "model":chunk.model,
                "object":chunk.object
            }
            
            if chunk_dict is not None:
                chunk_json = json.dumps(chunk_dict)
                chunk_sse = f"data: {chunk_json}\n\n"
                yield chunk_sse.encode()
            else:
                break


    except Exception as e:
        print(f"Error calling EdgenAI API: {e}")
        error_message = json.dumps({
            "error": "Error generating response",
            "details": str(e)
        })
        yield error_message



async def qdrant_server_running() -> bool:
    """Check if Qdrant server is running."""
    try:
        response = await httpclient.get("http://localhost:6333", timeout=10.0)
        response_json = response.json()
        return response_json.get("title") == "qdrant - vector search engine"
    except:
        return False

  
async def load_documents_from_json(data):
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


import subprocess
import json
from shlex import quote

async def fetch_documents_from_unstructured(file_path):
    url = 'http://localhost:8000/general/v0/general'
    file_path_escaped = quote(file_path)

    curl_command = f"""
    curl -X 'POST' \
      '{url}' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'files=@{file_path_escaped}' \
      -F 'chunking_strategy=by_title' \
      -F 'combine_under_n_chars=1000' \
      -F 'new_after_n_chars=10000' \
      -F 'max_characters=10000'
    """

    process = subprocess.run(curl_command, shell=True, text=True, capture_output=True)

    if process.returncode == 0:
        try:
            return json.loads(process.stdout)
        except json.JSONDecodeError:
            print("Error decoding JSON from response")
            return None
    else:
        print(f"Error fetching documents: {process.stderr}")
        return None


async def initialize_qdrant():
    global kb
    if not await qdrant_server_running():
        print("Qdrant server is not running. Please start the server and try again.")
        return None


    chunker = StubChunker(num_chunks_per_doc=1)
    dense_encoder = StubDenseEncoder(dimension=1000)  # Adjust if your encoder has a different dimension
    encoder = StubRecordEncoder(dense_encoder)

    kb = QdrantKnowledgeBase(
        collection_name=QDRANT_COLLECTION_NAME,
        record_encoder=encoder,
        chunker=chunker,
        location="http://localhost:6333",  # Adjust if your Qdrant instance is hosted elsewhere
    )

    # Create collection if it doesn't exist
    try:
        kb._client.get_collection(QDRANT_COLLECTION_NAME)
    except Exception as e:  # You might want to catch a more specific exception if possible
        print(f"Collection does not exist: {e}. Creating new collection.")
        await kb.create_canopy_collection()

    return kb




async def call_edgen(prompt, stream=False):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=stream
            )
        
        
        content = completion.choices[0].message.content
        
        return content
    except Exception as e:  # Catch a general exception as the specific OpenAIError is not available
        print(f"Error calling EdgenAI API: {e}")
        return None



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=22333)
