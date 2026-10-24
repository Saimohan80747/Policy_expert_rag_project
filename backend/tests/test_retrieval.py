"""Entity-anchored chunks, RRF fusion and optional Cohere rerank."""
import types

import pytest

import rag
import rerank
import retrieval

LINKED = [{"uid": "Person:LS00433", "name": "Harini Subramanyam", "label": "Person", "how": "alias", "surface": "Harini"}]


def chunk(doc, idx, text="t", **extra):
    return {"id": f"{doc}:{idx}", "score": 1.0, "text": text, "file_name": f"{doc}.txt", "file_url": "u",
            "doc_id": doc, "version": 1, "chunk_index": idx, **extra}


def test_fuse_boosts_chunks_found_by_both_lists_and_dedupes():
    vector = [chunk("a", 0), chunk("b", 0), chunk("c", 0)]
    graph = [chunk("c", 0, via_graph=True), chunk("d", 0, via_graph=True)]
    fused = retrieval.fuse(vector, graph)
    keys = [(c["doc_id"], c["chunk_index"]) for c in fused]
    assert keys[0] == ("c", 0) or keys[0] == ("a", 0)
    assert len(keys) == len(set(keys)) == 4
    assert fused[[k[0] for k in keys].index("c")]["via_graph"] is True
    assert keys.index(("c", 0)) < keys.index(("b", 0))     # boosted above its plain vector rank


def test_chunk_key_ignores_float_indexes_from_pinecone():
    assert retrieval.chunk_key(chunk("a", 3.0)) == retrieval.chunk_key(chunk("a", 3))


def test_entity_chunks_only_uses_confident_links(fake_graph):
    fake_graph.chunk_rows = [{"cid": "a:v1:0", "text": "x", "idx": 0, "doc_id": "a", "version": 1,
                              "file_name": "a.txt", "file_url": "u"}]
    assert retrieval.entity_chunks([{**LINKED[0], "how": "fuzzy"}]) == []
    out = retrieval.entity_chunks(LINKED)
    assert out[0]["via_graph"] and out[0]["file_name"] == "a.txt"
    _, params = fake_graph.queries("RETURN c.cid AS cid")[0]
    assert params["uids"] == ["Person:LS00433"] and params["min_hits"] == 1


def test_entity_chunks_require_all_entities_when_several_are_linked(fake_graph):
    two = LINKED + [{"uid": "Client:CL-001", "name": "DFL", "label": "Client", "how": "exact", "surface": "DFL"}]
    retrieval.entity_chunks(two)
    assert fake_graph.queries("RETURN c.cid AS cid")[0][1]["min_hits"] == 2


def test_entity_chunk_failure_is_swallowed(fake_graph, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("neo4j down")
    monkeypatch.setattr(retrieval.db, "run", boom)
    assert retrieval.entity_chunks(LINKED) == []


# ---------------------------------- rerank ----------------------------------
@pytest.fixture
def cohere(monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "test")
    monkeypatch.setattr(rerank, "USAGE_FILE", tmp_path / "usage.json")
    rerank._recent.clear()
    calls = []

    class Client:
        def rerank(self, **kw):
            calls.append(kw)
            order = list(reversed(range(len(kw["documents"]))))[: kw["top_n"]]
            return types.SimpleNamespace(results=[types.SimpleNamespace(index=i, relevance_score=0.9 - i / 100)
                                                  for i in order])
    monkeypatch.setattr(rerank, "_client", lambda: Client())
    return calls


def test_rerank_reorders_and_sets_scores(cohere):
    chunks = [chunk("a", 0, "A"), chunk("b", 0, "B"), chunk("c", 0, "C")]
    out, used = rerank.rerank("q", chunks, 2)
    assert used and [c["doc_id"] for c in out] == ["c", "b"]
    assert cohere[0]["documents"] == ["A", "B", "C"] and cohere[0]["top_n"] == 2


def test_rerank_without_key_keeps_order(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    out, used = rerank.rerank("q", [chunk("a", 0), chunk("b", 0)], 1)
    assert not used and out[0]["doc_id"] == "a"


def test_rerank_falls_back_on_errors(cohere, monkeypatch):
    class Bad:
        def rerank(self, **kw):
            raise RuntimeError("429 too many requests")
    monkeypatch.setattr(rerank, "_client", lambda: Bad())
    out, used = rerank.rerank("q", [chunk("a", 0), chunk("b", 0)], 1)
    assert not used and out[0]["doc_id"] == "a"


def test_rerank_respects_per_minute_and_monthly_budgets(cohere, monkeypatch):
    monkeypatch.setattr(rerank, "RERANK_PER_MINUTE", 2)
    pair = [chunk("a", 0), chunk("b", 0)]
    assert [rerank.rerank("q", pair, 2)[1] for _ in range(3)] == [True, True, False]
    rerank._recent.clear()
    monkeypatch.setattr(rerank, "RERANK_MONTHLY_LIMIT", 2)
    assert rerank.rerank("q", pair, 2)[1] is False


# ----------------------------- wired into answers -----------------------------
def test_answer_uses_graph_anchored_chunks_for_vanilla_route(env, fake_graph, fake_llm, monkeypatch):
    import io
    from fastapi import UploadFile
    from ingestion import ingest_file
    ingest_file(UploadFile(filename="leave.txt", file=io.BytesIO(b"Casual leave is 12 days per year.")))
    fake_graph.chunk_rows = [{"cid": "z:v1:0", "text": "Harini approves leave requests.", "idx": 0, "doc_id": "z",
                              "version": 1, "file_name": "approvals.txt", "file_url": "u"}]
    monkeypatch.setattr(rag, "link_entities", lambda q, e: LINKED)
    fake_llm.analysis = {"standalone_question": "Who approves leave?", "route": "vanilla", "entities": ["Harini"]}
    result = rag.answer_question("Who approves leave?", [{"question": "q", "answer": "a"}])
    names = [s["file_name"] for s in result["sources"]]
    assert "approvals.txt" in names and "leave.txt" in names
