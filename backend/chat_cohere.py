import cohere
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
import pprint
co = cohere.ClientV2(
    api_key=os.getenv("COHERE_API_KEY")
)

def chat_with_cohere(json):
    retrieved_documents = json.get("documents", [])
    memory = json.get("memory", [])
    knowledge_context=json.get("knowledge_context", "")
    context=[]
    context.append({
        "data": knowledge_context
    })
    for doc in retrieved_documents:
        context.append({
            "data": {
                "text": doc.get("text", ""),
                "score": doc.get("score", 0),
                "memory": memory
            }
        })

    
    question = json.get("query", "")

    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": question}],
        documents=context,
    )
    answer = response.message.content[0].text
    # first_document = retrieved_documents[0] if retrieved_documents else {}
    return {
        "query": question,
        "answer": answer,
        # "file_url": first_document.get("file_url", ""),
        # "file_name": first_document.get("file_name", "")
    }
