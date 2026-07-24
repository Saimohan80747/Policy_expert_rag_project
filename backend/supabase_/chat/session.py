import uuid
from ..supabase_db import supabase_var


def create_session(title="New Chat"):
    session_id = str(uuid.uuid4())

    response = (
        supabase_var
        .table("chat_sessions")
        .insert({
            "id": session_id,
            "title": title
        })
        .execute()
    )

    return response.data[0]

def get_session(session_id):
    
    response = (
        supabase_var
        .table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .execute()
    )

    if response.data:
        return response.data[0]

    return None
    
def get_all_sessions():

    response = (
        supabase_var
        .table("chat_sessions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def delete_session(session_id):

    response = (
        supabase_var
        .table("chat_sessions")
        .delete()
        .eq("id", session_id)
        .execute()
    )

    return response.data