
import time
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_DENSE_MODEL,
    PINECONE_NAMESPACE,
    PINECONE_RETRIEVAL_INDEX,
    PINECONE_SPARSE_MODEL,
    RETRIEVAL_TOP_K,
)

pc = Pinecone(api_key=PINECONE_API_KEY)




def retrieve_documents(query: str):
    start = time.time()
  
   
    index_name = PINECONE_RETRIEVAL_INDEX
    index = pc.Index(index_name)
    
     # Convert the query into a dense vector
    dense_query_embedding = pc.inference.embed(
        model=PINECONE_DENSE_MODEL,
        inputs=query,
        parameters={"input_type": "query", "truncate": "END"}
    )

    # Convert the query into a sparse vector
    sparse_query_embedding = pc.inference.embed(
        model=PINECONE_SPARSE_MODEL,
        inputs=query,
        parameters={"input_type": "query", "truncate": "END"}
    )
    retrieved_documents = []
    
   

    for d, s in zip(dense_query_embedding, sparse_query_embedding):
        query_response = index.query(
            namespace=PINECONE_NAMESPACE,
            top_k=RETRIEVAL_TOP_K,
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
    
    return retrieved_documents
