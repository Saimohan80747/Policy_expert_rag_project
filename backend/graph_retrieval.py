"""Graph side of retrieval: link entities, ask for a read-only Cypher query, run it, return facts."""
import logging
import time

from app_config import GRAPH_MAX_ATTEMPTS, GRAPH_MAX_ROWS, GRAPH_QUERY_TIMEOUT
from knowledge_graph import db
from knowledge_graph.linker import link_entities
from knowledge_graph.text2cypher import RestrictedQuery, generate_cypher, get_schema, neighbourhood, validate_cypher

logger = logging.getLogger(__name__)
MAX_FACT_CHARS = 300
_ready = {"at": 0.0, "value": False, "as_of": None}


def _check_ready(ttl=60):
    if time.time() - _ready["at"] > ttl:
        try:
            rows = db.run("MATCH (m:GraphMeta {name: 'domain'}) RETURN m.as_of AS as_of LIMIT 1")
            _ready.update(value=bool(rows), as_of=rows[0]["as_of"] if rows else None)
        except db.GraphNotConfigured:
            _ready.update(value=False, as_of=None)
        except Exception as exc:  # noqa: BLE001 - unreachable or failing: no graph route, but say why
            logger.warning("Graph unavailable, using document search only: %s: %s", type(exc).__name__, exc)
            _ready.update(value=False, as_of=None)
        else:
            if not _ready["value"]:
                logger.warning("Neo4j is reachable but the company graph is not loaded (run the loader).")
        _ready["at"] = time.time()
    return _ready


def graph_available() -> bool:
    return _check_ready()["value"]


def reset_ready_cache():
    _ready.update(at=0.0, value=False, as_of=None)


def format_facts(rows) -> list[str]:
    facts = []
    for row in rows[:GRAPH_MAX_ROWS]:
        parts = [f"{k}={str(v)[:MAX_FACT_CHARS]}" for k, v in row.items() if v not in (None, [], "")]
        facts.append("; ".join(parts) or "(empty row)")
    return facts


def only_zeros(rows) -> bool:
    """True when every value is 0, empty or null: a count of nothing usually means a wrong path, not a real zero."""
    values = [v for row in rows for v in row.values()]
    return bool(rows) and all(v in (None, 0, [], "", 0.0) for v in values)


def retrieve_graph_context(question: str, entities: list[str] | None = None) -> dict:
    """Never raises. Returns {linked, cypher, rows, facts, as_of, restricted, error}."""
    ctx = {"linked": [], "cypher": None, "rows": [], "facts": [], "as_of": None, "restricted": False, "error": None,
           "llm_calls": 0}
    try:
        as_of = _check_ready()["as_of"] or "today"
        ctx["as_of"] = as_of
        ctx["linked"] = link_entities(question, entities)
        schema = get_schema()
        nearby = neighbourhood(ctx["linked"])
        feedback, fallback_rows, fallback_cypher = None, [], None
        for attempt in range(GRAPH_MAX_ATTEMPTS):
            ctx["llm_calls"] += 1
            try:
                cypher = validate_cypher(generate_cypher(question, ctx["linked"], as_of, schema, feedback, nearby=nearby))
            except RestrictedQuery:
                ctx["restricted"] = True
                return ctx
            except ValueError as exc:
                feedback = str(exc)
                continue
            ctx["cypher"] = cypher
            try:
                rows = db.run_read(cypher, limit=GRAPH_MAX_ROWS, timeout=GRAPH_QUERY_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - syntax errors etc. go back to the model once
                feedback = f"{type(exc).__name__}: {str(exc)[:300]}"
                ctx["rows"] = []
                continue
            ctx["rows"], feedback = rows, None
            if rows and not only_zeros(rows):
                break
            if rows:  # all zeros: keep it as the answer if no attempt does better
                fallback_rows, fallback_cypher = rows, cypher
                feedback = ("It returned only zeros or empty values. Check the relationship names and directions against "
                            "what the entities are connected to, and remove any filter the question did not ask for.")
            else:
                feedback = "It returned no rows. Re-check entity uids, relationship direction, labels and date filters."
        if not ctx["rows"] and fallback_rows:
            ctx["rows"], ctx["cypher"] = fallback_rows, fallback_cypher
        elif ctx["rows"] and only_zeros(ctx["rows"]) and fallback_rows:
            ctx["rows"], ctx["cypher"] = fallback_rows, fallback_cypher
        ctx["facts"] = format_facts(ctx["rows"])
        if not ctx["rows"]:
            ctx["error"] = feedback
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph retrieval failed")
        ctx["error"] = f"{type(exc).__name__}: {exc}"
    return ctx
