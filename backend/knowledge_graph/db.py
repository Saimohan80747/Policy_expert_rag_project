"""Neo4j connection (lazy, so the API starts without graph credentials)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

_driver = None


class GraphNotConfigured(RuntimeError):
    """Neo4j credentials are missing; callers that treat the graph as optional skip quietly."""


def get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase

        uri, user, password = (os.getenv(k) for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"))
        if not all([uri, user, password]):
            raise GraphNotConfigured("Missing NEO4J_URI, NEO4J_USER or NEO4J_PASSWORD. Set them in backend/.env.")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            driver.verify_connectivity()
        except Exception:
            driver.close()
            raise
        _driver = driver
    return _driver


def run(query: str, **params) -> list[dict]:
    """Run one Cypher statement in its own auto-commit transaction and return the rows."""
    with get_driver().session() as session:
        return session.run(query, params).data()


def run_read(query: str, params: dict | None = None, limit: int = 50, timeout: float = 15) -> list[dict]:
    """Run a query in a READ transaction (the server rejects writes) with a time and row limit."""
    import itertools

    from neo4j import unit_of_work

    @unit_of_work(timeout=timeout)  # transaction timeout; Query(timeout=...) only works with session.run
    def work(tx):
        result = tx.run(query, params or {})
        return [record.data() for record in itertools.islice(result, limit)]

    with get_driver().session() as session:
        return session.execute_read(work)


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
