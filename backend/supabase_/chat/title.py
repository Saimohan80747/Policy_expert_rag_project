import logging

import llm
from app_config import GEMINI_MODEL

logger = logging.getLogger(__name__)


def generate_title(question, answer):
    """Short chat title. Falls back to the first words of the question if the LLM is unavailable."""
    fallback = " ".join((question or "").split()[:6]) or "New Chat"
    prompt = (
        "Write a title of at most 5 words for this conversation. "
        "Return only the title, with no quotes.\n\n"
        f"Question: {question}\n\nAnswer: {(answer or '')[:500]}"
    )
    try:
        title = llm.generate_text(prompt, model=GEMINI_MODEL).splitlines()[0].strip(" \"'")
        return title[:60] or fallback
    except Exception:  # noqa: BLE001 - a missing title must never lose the chat
        logger.exception("Title generation failed; using the question")
        return fallback
