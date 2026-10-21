import cohere
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


def _get_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing COHERE_API_KEY. Set it in backend_analyser/.env before generating chat titles."
        )
    return cohere.ClientV2(api_key=api_key)

def generate_title(question, answer):

    prompt = f"""
        Generate a concise title (maximum 5 words).

        Question:
        {question}

        Answer:
        {answer}

        Rules:
        - Maximum 5 words
        - Return only the title
        - No quotes
    """

    co = _get_client()

    response = co.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content[0].text.strip()