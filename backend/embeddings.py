import uuid
import time
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_DIMENSION,
    PINECONE_DENSE_MODEL,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
    PINECONE_SPARSE_MODEL,
    PINECONE_UPLOAD_INDEX,
)

pc = Pinecone(api_key=PINECONE_API_KEY)



def embedding_chunks(json):
    hybrid_index_name = PINECONE_UPLOAD_INDEX
    
    result=[]

    start = time.time()
     
    json.setdefault("metadata", {})
    json["metadata"].setdefault("embedding_metadata", {})

    metadata = json["metadata"]["embedding_metadata"]
    metadata.setdefault("embedding_model", "")
    metadata.setdefault("embedding_model_type", "")
    metadata.setdefault("processing_time", "")
    metadata["embedding_model"] = PINECONE_DENSE_MODEL
    metadata["embedding_model_type"] = "HuggingFaceEmbeddings"
    
    chunks = json.get("chunks", [])

    documents = []
    ids = []

    if not pc.has_index(hybrid_index_name):
        pc.create_index(
            name=hybrid_index_name,
            vector_type="dense",
            dimension=PINECONE_DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
    index = pc.Index(
           host=pc.describe_index(hybrid_index_name).host
       )  

    records = []
    # Convert the chunk_text into dense vectors
    dense_embeddings = pc.inference.embed(
        model=PINECONE_DENSE_MODEL,
        inputs=chunks,
        parameters={"input_type": "passage", "truncate": "END"}
    )

    # Convert the chunk_text into sparse vectors
    sparse_embeddings = pc.inference.embed(
        model=PINECONE_SPARSE_MODEL,
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
        namespace=PINECONE_NAMESPACE
    )

    
    metadata["processing_time"] = round(
        time.time() - start, 4)
    
    return json
