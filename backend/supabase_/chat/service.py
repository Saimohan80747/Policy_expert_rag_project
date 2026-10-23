from .message import save_chat
from .session import create_session
from .title import generate_title


def process_chat(
    session_id: str | None,
    question: str,
    answer: str,
    file_name: str | None = None,
    file_url: str | None = None,
    sources: list | None = None,
):
    """Create a session if needed, save the Q&A, and return the session id."""
    if not session_id:  # first message of a new chat
        session = create_session(title=generate_title(question, answer))
        session_id = session["id"]

    save_chat(
        session_id=session_id,
        question=question,
        answer=answer,
        file_name=file_name,
        file_url=file_url,
        sources=sources,
    )

    return session_id
