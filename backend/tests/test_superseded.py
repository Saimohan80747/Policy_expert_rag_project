import json
import sys
from pathlib import Path

import rag
import retrieval
from retrieve_docs import retrieve_documents

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mark_superseded  # noqa: E402


def upload(name, text):
    import io
    from fastapi import UploadFile
    from ingestion import ingest_file
    return ingest_file(UploadFile(filename=name, file=io.BytesIO(text.encode())))


def write_meta(tmp_path):
    meta = tmp_path / "_meta"
    meta.mkdir()
    (meta / "LS-POL-2018-0007.json").write_text(json.dumps({"doc_id": "LS-POL-2018-0007", "supersedes": "LS-POL-2015-0004"}))
    (meta / "LS-POL-2025-0005.json").write_text(json.dumps({"doc_id": "LS-POL-2025-0005", "supersedes": "LS-POL-2018-0007"}))
    (meta / "list.json").write_text("[]")
    (meta / "LS-X.json").write_text(json.dumps({"doc_id": "LS-X", "supersedes": None}))
    return meta


def test_superseded_ids_follow_the_chain_and_skip_odd_files(tmp_path):
    assert mark_superseded.superseded_ids(write_meta(tmp_path)) == {"LS-POL-2015-0004", "LS-POL-2018-0007"}


def test_corpus_id_comes_from_the_file_name_prefix():
    assert mark_superseded.corpus_id("LS-POL-2018-0007__pol-hr-003-leave-policy-v3.pdf") == "LS-POL-2018-0007"
    assert mark_superseded.corpus_id("Leave_Policy.txt") is None


def test_flagged_documents_are_hidden_unless_history_is_wanted(env, fake_graph, tmp_path):
    upload("LS-POL-2018-0007__leave-v3.txt", "Casual leave is 7 days per year.")
    upload("LS-POL-2025-0005__leave-v5.txt", "Casual leave is 8 days per year.")
    rows = mark_superseded.current_documents(env.supabase)
    docs = mark_superseded.select_documents(rows, mark_superseded.superseded_ids(write_meta(tmp_path)))
    assert [d["file_name"] for d in docs] == ["LS-POL-2018-0007__leave-v3.txt"]

    from embeddings import set_superseded
    assert mark_superseded.apply_flag(docs, True, set_superseded, mark_superseded._set_graph) == 1

    now = {r["file_name"] for r in retrieve_documents("casual leave", top_k=5)}
    assert now == {"LS-POL-2025-0005__leave-v5.txt"}
    past = {r["file_name"] for r in retrieve_documents("casual leave", top_k=5, include_superseded=True)}
    assert len(past) == 2
    assert fake_graph.queries("SET d.superseded")[0][1]["flag"] is True

    mark_superseded.apply_flag(docs, False, set_superseded, mark_superseded._set_graph)
    assert len(retrieve_documents("casual leave", top_k=5)) == 2


def test_retrieval_falls_back_when_only_superseded_documents_match(env, fake_graph):
    from embeddings import set_superseded
    upload("old.txt", "Casual leave is 7 days per year.")
    doc_id = env.supabase.tables["documents"][0]["doc_id"]
    set_superseded(doc_id, True)
    assert retrieve_documents("casual leave", top_k=3) == []
    assert [c["file_name"] for c in retrieval.retrieve_context("casual leave", None)] == ["old.txt"]


def test_history_flag_reaches_retrieval(env, fake_llm, fake_graph, monkeypatch):
    seen = {}
    monkeypatch.setattr(rag, "retrieve_context", lambda q, linked, top_k, include_history: seen.update(h=include_history) or [])
    fake_llm.analysis = {"standalone_question": "What was the leave policy in 2018?", "route": "vanilla",
                         "entities": [], "include_history": True}
    rag.answer_question("What was the leave policy in 2018?", [{"question": "q", "answer": "a"}])
    assert seen["h"] is True
