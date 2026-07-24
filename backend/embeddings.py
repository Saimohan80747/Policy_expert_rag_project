import os
from pathlib import Path
from dotenv import load_dotenv
import uuid
import time

from pinecone import Pinecone, ServerlessSpec

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))



def embedding_chunks(json):
    result=[]
    
    if(json.get("status","")=="duplicate" or json.get("status","")=="failed" or len(json.get("chunks",[]))==0):
            return json

    start = time.time()
     
    json.setdefault("metadata", {})
    json["metadata"].setdefault("embedding_metadata", {})

    metadata = json["metadata"]["embedding_metadata"]
    metadata.setdefault("embedding_model", "")
    metadata.setdefault("embedding_model_type", "")
    metadata.setdefault("processing_time", "")
    metadata["embedding_model"] = "llama-text-embed-v2"
    metadata["embedding_model_type"] = "HuggingFaceEmbeddings"

    chunks = json.get("chunks", [])

    documents = []
    ids = []

    

    hybrid_index_name = "hybrid-index"

    if not pc.has_index(hybrid_index_name):
        pc.create_index(
            name=hybrid_index_name,
            vector_type="dense",
            dimension=1024,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
   
    index = pc.Index(hybrid_index_name)
    records = []
    # Convert the chunk_text into dense vectors
    dense_embeddings = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=chunks,
        parameters={"input_type": "passage", "truncate": "END"}
    )

    # Convert the chunk_text into sparse vectors
    sparse_embeddings = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=chunks,
        parameters={"input_type": "passage", "truncate": "END"}
    )

    records = []
    for chunk, de, se in zip(chunks, dense_embeddings, sparse_embeddings):
        
        doc_id = str(uuid.uuid4())
        records.append({
            "id": doc_id,
            "values": de['values'],
            "sparse_values": {'indices': se['sparse_indices'], 'values': se['sparse_values']},
            "metadata": {
                'text': chunk, 
                "file_url": json.get("metadata", {}).get("file_metadata", {}).get("url", ""),
                "file_name": json.get("metadata", {}).get("file_metadata", {}).get("file_name", ""),
            }
        })

    # Upsert the records into the index
    index.upsert(
        vectors=records,
        namespace="first-namespace"
    )

    
    metadata["processing_time"] = round(
        time.time() - start, 4)
    
    return json


