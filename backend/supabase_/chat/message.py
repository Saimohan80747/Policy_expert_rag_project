import logging

from ..supabase_db import supabase_var

logger = logging.getLogger(__name__)


def save_chat(
    session_id,
    question,
    answer,
    file_name=None,
    file_url=None,
    sources=None,
):
    row = {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "file_name": file_name,
        "file_url": file_url,
    }
    try:
        response = (
            supabase_var.table("chat_messages")
            .insert({**row, "sources": sources or []})
            .execute()
        )
    except Exception:  # noqa: BLE001 - `sources` column missing: run backend/supabase_/schema.sql
        logger.warning(
            "Saving sources failed (is the chat_messages.sources column created?); saving without them",
            exc_info=True,
        )
        response = supabase_var.table("chat_messages").insert(row).execute()

    return response.data[0]


def get_chat_history(session_id):

    response = (
        supabase_var
        .table("chat_messages")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )

    return response.data
