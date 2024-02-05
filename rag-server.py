import sys
sys.path.insert(0, "canopy")
sys.path.insert(0, "canopy/src")

from threading import Thread
import asyncio
import json
import requests
from edgen import Edgen
import time
from fastapi import FastAPI, Request, BackgroundTasks
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


app = FastAPI()
client = Edgen()
messages = asyncio.Queue()


QDRANT_COLLECTION_NAME = "canopy--test_collection"
FILE_PATH = "sibs.pdf"
QUERY = "What is a cyber asset anyway?"
TOP_K = 20


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    QUERY = data.get('messages', '')[-1]["content"]
    print(f"User input: {QUERY}")

    return StreamingResponse(run_rag(QUERY), media_type="text/event-stream")


async def run_rag(prompt):
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


    # Qdrant vector db
    kb = initialize_qdrant()
   
    # Create and insert documents
    # json = fetch_documents_from_unstructured(FILE_PATH)
    # docs = load_documents_from_json(json)
    # kb.upsert(docs)

    
    
    # Query the knowledge base
    query = Query(text=vector_query, top_k=TOP_K)
    query_results = kb.query([query])

    # Prepare data for LLM
    context = ""
    for result in query_results:
        for doc in result.documents:
            # Assuming you have 'page_number', 'file_name' in your document's metadata
            page_info = f"Page number: {doc.metadata['page_number']}" if 'page_number' in doc.metadata else "Page number not available"
            file_info = f"File name: {doc.metadata['filename']}" if 'filename' in doc.metadata else "File name not available"
            context += f"{file_info}, {page_info}, Text: {doc.text}\n\n"

    
    # Prepare prompt for LLM to generate the final response
    final_prompt = (
        "Based on the user's initial query: "
        f"'{QUERY}', "
        "synthesize the key information from the provided document snippets into a coherent and direct response. "
        "Start by stating the document's name once if all snippets are from the same file, then list the relevant page numbers and their associated content. "
        "Ensure each point is supported by a direct reference to the document snippet (including page number), formatted neatly and concisely. The documents are as follows:\n\n"
        f"{context}\n"
        "Craft a response that integrates insights from the documents in a structured and informative manner."
    )
    print("final_prompt:", final_prompt, "\n")
    
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
                print(chunk_sse)
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




async def call_edgen_stream(prompt):
    global messages
    """Call the Edgen API and stream the response."""
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in completion:
            # Assuming chunk is an object with the structure you provided, construct the dictionary
            chunk_dict = {
                "id": chunk.id,
                "choices": [
                    {
                        "delta": {
                            "content": choice.delta.content,
                            "role": choice.delta.role
                        },
                        "finish_reason": choice.finish_reason,
                        "index": choice.index
                    } for choice in chunk.choices
                ],
                "created": chunk.created,
                "model": chunk.model,
                "system_fingerprint": chunk.system_fingerprint,
                "object": chunk.object
            }
            
            # Convert the dictionary to a JSON string and then to bytes
            chunk_json = json.dumps(chunk_dict)
            print(chunk_json)
            await messages.put(chunk_json)     

        # Signal the end of the stream
        await messages.put("!It is the end!")

    except Exception as e:
        print(f"Error calling EdgenAI API: {e}")
        error_message = json.dumps({
            "error": "Error generating response",
            "details": str(e)
        })
        await messages.put(error_message)

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

def fetch_documents_from_unstructured(file_path):
    url = 'http://localhost:8000/general/v0/general'
    files = {'files': open(file_path, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return response.json()  # Parses the JSON response and returns it
    else:
        print(f"Error fetching documents: {response.status_code}")
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

def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

def _corsify_actual_response(response, plain_text=False):
    if plain_text:
        # Create a plain text response
        resp = make_response(response)
        resp.mimetype = 'text/plain'
    else:
        # Create a JSON response
        resp = make_response(jsonify(response))
        resp.mimetype = 'application/json'
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp

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
    uvicorn.run(app, host="0.0.0.0", port=9003)
