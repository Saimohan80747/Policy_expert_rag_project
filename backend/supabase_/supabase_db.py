import logging
import os
import time
from pathlib import Path

import httpx
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


# Calls that actually hit the network. Everything else on the client just builds a request.
_NETWORK_CALLS = {"execute", "upload", "download", "remove", "list", "create_signed_url"}
_PLAIN = (str, bytes, bytearray, int, float, bool, list, dict, tuple, set, type(None))
_RETRIES = 3

log = logging.getLogger(__name__)


def _call_with_retry(fn, *args, **kwargs):
    """Retry on dropped connections.

    Supabase closes idle HTTP/2 connections; the next request that reuses one fails
    with "Server disconnected". httpx discards the dead connection, so a retry works.
    """
    for attempt in range(1, _RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except httpx.TransportError as exc:
            if attempt == _RETRIES:
                raise
            log.warning("Supabase connection error (%s); retry %d/%d", type(exc).__name__, attempt, _RETRIES - 1)
            time.sleep(0.3 * attempt)


class _Retrying:
    """Wraps the client (and the builders it returns) so network calls retry."""

    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return _wrap(attr)
        if name in _NETWORK_CALLS:
            return lambda *a, **k: _call_with_retry(attr, *a, **k)
        return lambda *a, **k: _wrap(attr(*a, **k))


def _wrap(value):
    return value if isinstance(value, _PLAIN) else _Retrying(value)


class _LazySupabase:
    """Forwards attribute access to the real client, creating it on demand.

    Keeps `from .supabase_db import supabase_var` working everywhere.
    """

    def __getattr__(self, name):
        return getattr(_Retrying(get_supabase()), name)


supabase_var = _LazySupabase()
