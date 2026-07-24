from ..supabase_db import supabase_var


def save_chat(
    session_id,
    question,
    answer,
    file_name=None,
    file_url=None
):

    response = (
        supabase_var
        .table("chat_messages")
        .insert({
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "file_name": file_name,
            "file_url": file_url
        })
        .execute()
    )

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