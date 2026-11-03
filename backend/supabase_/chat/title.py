import cohere
from config import COHERE_API_KEY, COHERE_MODEL


def _get_client():
    if not COHERE_API_KEY:
        raise RuntimeError(
            "Missing COHERE_API_KEY. Set it in backend_analyser/.env before generating chat titles."
        )
    return cohere.ClientV2(api_key=COHERE_API_KEY)

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
        model=COHERE_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content[0].text.strip()