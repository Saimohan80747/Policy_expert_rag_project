
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))




def retrieve_documents(query: str):
    start = time.time()
    result={
        "query": query,
        "retrieved_documents": [],
    }
   
    index_name = "hybrid-index"
    index = pc.Index(index_name)
    
     # Convert the query into a dense vector
    dense_query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=query,
        parameters={"input_type": "query", "truncate": "END"}
    )

    # Convert the query into a sparse vector
    sparse_query_embedding = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=query,
        parameters={"input_type": "query", "truncate": "END"}
    )
    retrieved_documents = []
    
   

    for d, s in zip(dense_query_embedding, sparse_query_embedding):
        query_response = index.query(
            namespace="first-namespace",
            top_k=5,
            vector=d['values'],
            sparse_vector={'indices': s['sparse_indices'], 'values': s['sparse_values']},
            include_values=False,
            include_metadata=True
        )
    
    for match in query_response.matches:
        retrieved_documents.append({
            "score": match.score,
            "text": match.metadata["text"],
            "file_url": match.metadata.get("file_url", None),
            "file_name": match.metadata.get("file_name", None)
        })

    print(retrieved_documents)
    
    result["retrieved_documents"] = retrieved_documents
    return result