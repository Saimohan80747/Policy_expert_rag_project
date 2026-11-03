import cohere
import json
from config import COHERE_API_KEY, COHERE_MODEL



def _get_client():
    if not COHERE_API_KEY:
        raise RuntimeError(
            "Missing COHERE_API_KEY. Set it in backend_analyser/.env before generating chat titles."
        )
    return cohere.ClientV2(api_key=COHERE_API_KEY)
def router_decision(question):
    prompt = """
    You are a query routing classifier for a company knowledge assistant.

    Classify the user's question into exactly one of these routes:

    1. knowledge_graph
    Use this route for questions about:
    - Employees, interns, departments, and projects
    - Reporting relationships, managers, and team members
    - Company policies and their owners or approvers
    - Connections and relationships between entities
    - Facts stored in the knowledge graph

    2. document_retrieval
    Use this route for questions about:
    - Information contained in uploaded documents
    - Detailed explanations of document content
    - Summaries of policies, reports, and other files
    - Information that requires searching document chunks

    3. both
    Use this route when the question requires BOTH:
    - Entity relationships or structured company information
    - Detailed information from documents

    Rules:
    - Select the most appropriate route based on the question.
    - Do not answer the question.
    - Return valid JSON only.
    - Do not include Markdown or explanations.

    User question:
    $question

    Output format:
    {
    "route": "document_retrieval" | "both" |"knowledge_graph"
    }
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
    return json.loads(response.message.content[0].text)


    # return response.message.content[0].text.strip()