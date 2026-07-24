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
    answer = ""
    

    retrieved_documents = json.get("retrieved_documents", [])
    context=[]
    for doc in retrieved_documents:
        context.append({
            "data": {
                "text": doc.get("text", ""),
                "score": doc.get("score", 0)
            }
        })
    question = json.get("query", "")

    response = co.chat(
        model="command-a-plus-05-2026",
        messages=[{"role": "user", "content": question}],
        documents=context,
    )
    answer += f"{response.message.content[1].text}\n\n"
    return {
        "query": question,
        "answer": answer,
        "file_url": json.get("retrieved_documents", [{}])[0].get("file_url", None),
        "file_name": json.get("retrieved_documents", [{}])[0].get("file_name", None)
    }
