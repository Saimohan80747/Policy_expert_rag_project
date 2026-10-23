import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

_client = None


def get_supabase():
    """Create the Supabase client on first use so the API can start without keys."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "Missing SUPABASE_URL or SUPABASE_KEY. Set them in backend/.env."
            )
        _client = create_client(url, key)
    return _client


class _LazySupabase:
    """Forwards attribute access to the real client, creating it on demand.

    Keeps `from .supabase_db import supabase_var` working everywhere.
    """

    def __getattr__(self, name):
        return getattr(get_supabase(), name)


supabase_var = _LazySupabase()
