"""Optional Cohere reranking. Never raises: any problem means "keep the order you were given".

Cohere's trial key allows 10 rerank requests/minute and about 1,000 API calls/month, so calls are
rate-limited in-process and counted in a small JSON file; past either limit the caller falls back
to its own ordering.
"""
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path

from app_config import COHERE_RERANK_MODEL, RERANK_MONTHLY_LIMIT, RERANK_PER_MINUTE

logger = logging.getLogger(__name__)
USAGE_FILE = Path(__file__).resolve().parent / ".rerank_usage.json"
_lock = threading.Lock()
_recent = deque()


def rerank_enabled() -> bool:
    return bool(os.getenv("COHERE_API_KEY")) and os.getenv("RERANK", "on").lower() not in ("off", "0", "false")


def _client():
    import chat_cohere
    return chat_cohere._client()


def _allow_call() -> bool:
    """Per-minute limiter plus a monthly counter. Returns False when either budget is spent."""
    now = time.time()
    with _lock:
        while _recent and now - _recent[0] > 60:
            _recent.popleft()
        if len(_recent) >= RERANK_PER_MINUTE:
            return False
        month = date.today().strftime("%Y-%m")
        try:
            usage = json.loads(USAGE_FILE.read_text())
        except (OSError, ValueError):
            usage = {}
        if usage.get("month") != month:
            usage = {"month": month, "calls": 0}
        if usage["calls"] >= RERANK_MONTHLY_LIMIT:
            return False
        usage["calls"] += 1
        try:
            USAGE_FILE.write_text(json.dumps(usage))
        except OSError:
            logger.warning("Could not write %s; monthly rerank count is not persisted", USAGE_FILE)
        _recent.append(now)
        return True


def rerank(query: str, chunks: list[dict], top_n: int) -> tuple[list[dict], bool]:
    """Return (chunks, reranked). On any failure returns the first top_n chunks unchanged."""
    fallback = chunks[:top_n]
    if len(chunks) <= 1 or not rerank_enabled():
        return fallback, False
    if not _allow_call():
        logger.info("Rerank budget spent (per-minute or monthly); keeping fused order")
        return fallback, False
    try:
        response = _client().rerank(
            model=COHERE_RERANK_MODEL, query=query, documents=[c["text"] for c in chunks], top_n=top_n,
        )
        ordered = []
        for hit in response.results:
            chunk = dict(chunks[hit.index])
            chunk["score"] = float(hit.relevance_score)
            ordered.append(chunk)
        return (ordered or fallback), bool(ordered)
    except Exception as exc:  # noqa: BLE001 - 429, bad key, network: the fused order is still good
        logger.warning("Rerank failed (%s: %s); keeping fused order", type(exc).__name__, str(exc)[:200])
        return fallback, False
