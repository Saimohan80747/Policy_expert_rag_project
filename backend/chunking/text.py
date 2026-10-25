import re
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter,TokenTextSplitter

import time

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def semantic_chunks(data):
#     sentences = data.get("chunks", [])

#     if not sentences:
#         return []

#     embeddings = model.encode(sentences)

#     groups = [[0]]
#     threshold = 0.25

#     for i in range(1, len(sentences)):
#         centroid = np.mean(embeddings[groups[-1]], axis=0)

#         sim = cosine_similarity(
#             [embeddings[i]],
#             [centroid]
#         )[0][0]

#         if sim >= threshold:
#             groups[-1].append(i)
#         else:
#             groups.append([i])

#     return [
#         " ".join(sentences[idx] for idx in group)
#         for group in groups
#     ]


# def fixed_chunks(text, chunk_size):
#     chunks = []

#     for i in range(0, len(text), chunk_size):
#         chunks.append(text[i:i + chunk_size])

#     return chunks

def overlap_chunks(text, chunk_size, overlap_size):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap_size-1

    return chunks

def sentence_chunks(text, sentences_per_chunk=1):
    start_time = time.time()
    json1 = {}
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    json1["chunks"] = chunks
    json1["metadata"]["chunking_metadata"] = {
        "chunking_strategy": "sentence_based",
        "sentences_per_chunk": sentences_per_chunk,
        "total_chunks": len(chunks),
        "processing_time_seconds": round(time.time() - start_time, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_text_length": len(text),
    }

    return json1


def recursive_chunks(json, chunk_size=2000, chunk_overlap=300):
    chunks = []
    start_time = time.time()

    text = json.get("text", "")
    print(f"Original text length: {len(text)}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # fall back to sentences, words, then characters so no chunk exceeds chunk_size
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    

    chunks = splitter.split_text(text)

    json["chunks"] = chunks

    if "metadata" not in json:
        json["metadata"] = {}

    json["metadata"]["chunking_metadata"] = {
        "chunking_strategy": "recursive",
        "chunk_size": chunk_size,
        "total_chunks": len(chunks),
        "processing_time_seconds": round(time.time() - start_time, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_text_length": len(text),
    }

    return json

def token_chunks(text):

    splitter = TokenTextSplitter(
        chunk_size=15,
        chunk_overlap=3
    )

    chunks = splitter.split_text(text)

    for chunk in chunks:
        print(chunk)
    
    return chunks


