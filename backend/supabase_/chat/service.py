
from .session import create_session
from .message import save_chat
from .title import generate_title


def process_chat(
    session_id: str | None,
    question: str,
    answer: str,
    file_name: str | None = None,
    file_url: str | None = None,
):
    """
    Creates a new session if needed and saves the chat.
    Returns the session_id.
    """

    # First message of a new chat
    if not session_id :
        title_ = generate_title(question, answer)
        session = create_session(title=title_)
        session_id = session["id"]
    
    # Save question & answer
    save_chat(
        session_id=session_id,
        question=question,
        answer=answer,
        file_name=file_name,
        file_url=file_url,
    )

    return session_id