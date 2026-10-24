import logging
import os
import time
from datetime import datetime, timezone

from pinecone import Pinecone, ServerlessSpec

from app_config import (
    DELETE_BATCH,
    DENSE_MODEL,
    EMBED_BATCH,
    EMBED_DIM,
    PINECONE_INDEX,
    PINECONE_NAMESPACE,
    SPARSE_MODEL,
    UPSERT_BATCH,
)

logger = logging.getLogger(__name__)

_pc = None
_index = None


def get_pc():
    global _pc
    if _pc is None:
        key = os.getenv("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("PINECONE_API_KEY is not set (backend/.env).")
        _pc = Pinecone(api_key=key)
    return _pc


def get_index():
    """Return the index handle, creating the serverless index on first use."""
    global _index
    if _index is None:
        pc = get_pc()
        if not pc.has_index(PINECONE_INDEX):
            pc.create_index(
                name=PINECONE_INDEX,
                vector_type="dense",
                dimension=EMBED_DIM,
                metric="dotproduct",  # required for dense+sparse hybrid
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(PINECONE_INDEX).status["ready"]:
                time.sleep(1)
        _check_index_type(pc.describe_index(PINECONE_INDEX))
        _index = pc.Index(PINECONE_INDEX)
    return _index


def _check_index_type(description):
    """This app needs a classic dense vector index (Vectors API, dotproduct)."""
    dimension = getattr(description, "dimension", None)
    metric = getattr(description, "metric", None)
    if dimension != EMBED_DIM or metric != "dotproduct":
        raise RuntimeError(
            f"Pinecone index '{PINECONE_INDEX}' is not a {EMBED_DIM}-dimension dotproduct "
            f"vector index (found dimension={dimension}, metric={metric}). It was probably "
            "created in the Pinecone console with a document schema, which this app does not "
            "support. Delete that index in the console (or set PINECONE_INDEX to a new name "
            "in backend/.env); the app then creates the right kind itself."
        )


def _field(obj, key):
    try:
        return obj[key]
    except (TypeError, KeyError):
        return getattr(obj, key)


def _is_rate_limit(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "ratelimit" in text or "resource_exhausted" in text or "rate limit" in text


def _with_retry(fn, attempts=4, rate_limit_attempts=6):
    """Retry transient errors quickly; wait out rate limits (they reset every minute)."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - rate limits / transient errors
            limited = _is_rate_limit(exc)
            attempt += 1
            if attempt >= (rate_limit_attempts if limited else attempts):
                raise
            time.sleep(20 * attempt if limited else 2 ** (attempt - 1))


class _TokenBudget:
    """Keeps passage embedding under the provider's tokens-per-minute cap (free tier: 250K)."""

    def __init__(self, limit=200_000, window=60.0):
        self.limit, self.window, self.events = limit, window, []  # events: (time, tokens)

    def wait(self, tokens, now=time.monotonic, sleep=time.sleep):
        tokens = min(tokens, self.limit)
        while True:
            t = now()
            self.events = [(at, n) for at, n in self.events if t - at < self.window]
            if sum(n for _, n in self.events) + tokens <= self.limit:
                self.events.append((t, tokens))
                return
            sleep(max(self.events[0][0] + self.window - t, 0.1))


_budgets = {}
MAX_REQUEST_TOKENS = 60_000  # estimated tokens per embed call; well under the 250K/min cap


def estimate_tokens(text):
    """Pessimistic: tables full of names, ids and dates cost ~1.4 characters per token."""
    return int(len(text) / 1.25) + 1


def _batches(texts):
    """Slices of `texts` limited by count (EMBED_BATCH) and by estimated tokens per request."""
    batch, used = [], 0
    for text in texts:
        cost = estimate_tokens(text)
        if batch and (len(batch) >= EMBED_BATCH or used + cost > MAX_REQUEST_TOKENS):
            yield batch
            batch, used = [], 0
        batch.append(text)
        used += cost
    if batch:
        yield batch


def _throttle(model, batch, input_type):
    if input_type != "passage":
        return
    _budgets.setdefault(model, _TokenBudget()).wait(sum(estimate_tokens(t) for t in batch))


def _embed(model, texts, input_type):
    results = []
    for batch in _batches(texts):
        _throttle(model, batch, input_type)
        res = _with_retry(
            lambda b=batch: get_pc().inference.embed(
                model=model,
                inputs=b,
                parameters={"input_type": input_type, "truncate": "END"},
            )
        )
        results.extend(res)
    return results


def embed_dense(texts, input_type="passage"):
    return [_field(r, "values") for r in _embed(DENSE_MODEL, texts, input_type)]


def embed_sparse(texts, input_type="passage"):
    return [
        {"indices": _field(r, "sparse_indices"), "values": _field(r, "sparse_values")}
        for r in _embed(SPARSE_MODEL, texts, input_type)
    ]


def chunk_id(doc_id, version, index):
    """Deterministic chunk id: re-ingesting the same version overwrites, never duplicates."""
    return f"{doc_id}:v{version}:{index}"


def list_ids(doc_id, version=None):
    prefix = f"{doc_id}:v{version}:" if version is not None else f"{doc_id}:"
    ids = []
    for page in get_index().list(prefix=prefix, namespace=PINECONE_NAMESPACE):
        # SDK >= 8 yields ListResponse pages (.vectors of items with .id);
        # older SDKs yield plain lists of id strings. Handle both.
        for item in getattr(page, "vectors", page):
            ids.append(item if isinstance(item, str) else _field(item, "id"))
    return ids


def delete_doc_vectors(doc_id, version=None):
    """Delete all chunk vectors of a document (or one version). Returns the count."""
    ids = list_ids(doc_id, version)
    index = get_index()
    for i in range(0, len(ids), DELETE_BATCH):
        index.delete(ids=ids[i : i + DELETE_BATCH], namespace=PINECONE_NAMESPACE)
    return len(ids)


def retire_older_versions(doc_id, keep_version):
    """Mark chunks of earlier versions as not current (kept for history, hidden from search)."""
    index = get_index()
    try:  # one server-side update; the write is applied asynchronously by Pinecone
        index.update(
            filter={"$and": [{"doc_id": {"$eq": doc_id}}, {"version": {"$lt": keep_version}}]},
            set_metadata={"is_current": False},
            namespace=PINECONE_NAMESPACE,
        )
        return
    except Exception:  # noqa: BLE001 - older SDK/index: fall back to per-vector updates
        logger.info("Filter update unavailable; retiring versions one vector at a time")
    for vec_id in list_ids(doc_id):
        if int(vec_id.split(":")[1].lstrip("v")) < keep_version:
            index.update(
                id=vec_id,
                set_metadata={"is_current": False},
                namespace=PINECONE_NAMESPACE,
            )


def set_superseded(doc_id, flag=True):
    """Flag every chunk of a document as replaced by a newer, differently named document."""
    index = get_index()
    meta = {"superseded": bool(flag)}
    try:
        index.update(filter={"doc_id": {"$eq": doc_id}}, set_metadata=meta, namespace=PINECONE_NAMESPACE)
        return
    except Exception:  # noqa: BLE001 - older SDK/index: fall back to per-vector updates
        logger.info("Filter update unavailable; flagging %s one vector at a time", doc_id)
    for vec_id in list_ids(doc_id):
        index.update(id=vec_id, set_metadata=meta, namespace=PINECONE_NAMESPACE)


def cleanup_version(doc_id, version):
    """Best-effort removal of one version's vectors; failures are logged, not raised."""
    try:
        delete_doc_vectors(doc_id, version)
    except Exception:  # noqa: BLE001
        logger.exception("Could not clean up vectors for %s v%s", doc_id, version)


def _cleanup(state):
    cleanup_version(state["doc_id"], state["version"])


def embedding_chunks(state):
    """Step 5: embed chunks (dense + sparse) and upsert them with deterministic ids."""
    start = time.time()
    chunks = state["chunks"]
    meta = state["metadata"]["file_metadata"]

    dense = embed_dense(chunks)
    sparse = embed_sparse(chunks)

    ingested_at = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "id": chunk_id(state["doc_id"], state["version"], i),
            "values": d,
            "sparse_values": s,
            "metadata": {
                "text": chunk,
                "file_name": state["file_name"],
                "file_url": meta.get("url", ""),
                "doc_id": state["doc_id"],
                "version": state["version"],
                "is_current": True,
                "chunk_index": i,
                "doc_type": state["extension"].lstrip("."),
                "ingested_at": ingested_at,
            },
        }
        for i, (chunk, d, s) in enumerate(zip(chunks, dense, sparse))
    ]

    index = get_index()
    try:
        for i in range(0, len(records), UPSERT_BATCH):
            index.upsert(
                vectors=records[i : i + UPSERT_BATCH], namespace=PINECONE_NAMESPACE
            )
    except Exception:
        _cleanup(state)  # no partial documents; never hides the original error
        raise

    state["embedded_chunks"] = len(records)
    state["metadata"]["embedding_metadata"] = {
        "dense_model": DENSE_MODEL,
        "sparse_model": SPARSE_MODEL,
        "index": PINECONE_INDEX,
        "namespace": PINECONE_NAMESPACE,
        "processing_time": round(time.time() - start, 4),
    }
    return state
