"""Lexical layer: gazetteer matching, graph sync during ingestion, delete."""
import io

from fastapi import UploadFile

from ingestion import ingest_file
from knowledge_graph import lexical
from knowledge_graph.gazetteer import Gazetteer


def upload(name, text):
    return UploadFile(filename=name, file=io.BytesIO(text.encode()))


def gaz(fake_graph):
    return Gazetteer(fake_graph.alias_rows)


# ---------------------------------- gazetteer ----------------------------------
def test_finds_names_ids_and_codenames_and_counts(fake_graph):
    mentions, _ = gaz(fake_graph).find(
        "Harini Subramanyam (LS00433) led Project Beacon. Later Beacon shipped.")
    by_uid = {m["uid"]: m for m in mentions}
    assert by_uid["Person:LS00433"]["count"] == 2            # name + employee id
    assert by_uid["Project:PRJ-2012-01"]["count"] == 2       # 'Project Beacon' (longest) + 'Beacon'


def test_longest_match_does_not_double_count(fake_graph):
    mentions, _ = gaz(fake_graph).find("Project Beacon")
    assert mentions == [{"uid": "Project:PRJ-2012-01", "count": 1, "via": "codename"}]


def test_ambiguous_generic_and_role_aliases_are_not_linked(fake_graph):
    mentions, unresolved = gaz(fake_graph).find("HS met Priya Sharma about the Platform and the CTO.")
    assert mentions == []
    assert "priya sharma" in unresolved                       # reported for the LLM tier


def test_three_letter_acronym_needs_exact_case(fake_graph):
    g = gaz(fake_graph)
    assert [m["uid"] for m in g.find("DFL renewed the contract.")[0]] == ["Client:CL-001"]
    assert g.find("the dfl word in lowercase")[0] == []


def test_matching_is_accent_and_punctuation_insensitive(fake_graph):
    mentions, _ = gaz(fake_graph).find("harini-subramanyam, approved.")
    assert mentions[0]["uid"] == "Person:LS00433"


# ---------------------------------- graph sync ----------------------------------
STATE = {"doc_id": "abc123", "version": 1, "file_name": "LS-ADR-2023-0001__adr-0040-use-redis.txt",
         "extension": ".txt", "url": "https://x/f", "chunks": ["Harini Subramanyam chose Redis.", "DFL agreed."]}


def test_sync_writes_doc_chunks_next_mentions_and_record_link(fake_graph):
    result = lexical.sync_document(dict(STATE))
    assert result == {"chunks": 2, "mentions": 2, "entities": 2, "records": ["ADR-0040"]}
    doc = fake_graph.queries("MERGE (d:SourceDoc")[0][1]
    assert doc["sid"] == "abc123:v1" and doc["props"]["ref"] == "LS-ADR-2023-0001"
    chunks = fake_graph.queries("MERGE (c:Chunk")[0][1]["rows"]
    assert [c["cid"] for c in chunks] == ["abc123:v1:0", "abc123:v1:1"]
    assert fake_graph.queries("(a)-[:NEXT]->(b)")[0][1]["pairs"] == [{"a": "abc123:v1:0", "b": "abc123:v1:1"}]
    mentions = fake_graph.queries("MERGE (c)-[m:MENTIONS]->(e)")[0][1]["rows"]
    assert {(m["cid"], m["uid"]) for m in mentions} == {("abc123:v1:0", "Person:LS00433"), ("abc123:v1:1", "Client:CL-001")}


def test_sync_retires_older_versions_when_superseding(fake_graph):
    lexical.sync_document({**STATE, "version": 2, "supersedes": "abc123:v1"})
    retire = fake_graph.queries("SET d.is_current = false")
    assert retire and retire[0][1] == {"doc_id": "abc123", "v": 2}


def test_graph_not_configured_raises_before_writing(env):
    import pytest
    from knowledge_graph.db import GraphNotConfigured

    with pytest.raises(GraphNotConfigured):
        lexical.sync_document(dict(STATE))


# ------------------------------ ingestion integration ------------------------------
def test_ingest_reports_graph_summary(env, fake_graph):
    result = ingest_file(upload("Notes.txt", "Harini Subramanyam owns Project Beacon."))
    assert result["status"] == "ingested"
    assert result["graph"]["chunks"] == 1 and result["graph"]["mentions"] == 2


def test_ingest_without_graph_still_succeeds(env):
    result = ingest_file(upload("Notes.txt", "Harini Subramanyam owns Project Beacon."))
    assert result["status"] == "ingested" and result["graph"] == "skipped"


def test_graph_failure_is_a_warning_not_an_ingest_failure(env, fake_graph):
    fake_graph.fail = True
    result = ingest_file(upload("Notes.txt", "Some text about nothing in particular."))
    assert result["status"] == "ingested" and result["graph"] == "failed"
    assert any("Graph sync failed" in w for w in result["warnings"])
    assert env.index.all_ids()                                # vectors are still stored


def test_delete_document_also_removes_graph_nodes(env, fake_graph):
    from supabase_.delete_documents import delete_document

    ingest_file(upload("Notes.txt", "Harini Subramanyam owns Project Beacon."))
    result = delete_document("Notes.txt")
    assert result["graph"] == "deleted"
    assert fake_graph.queries("DETACH DELETE c, d")


def test_delete_without_graph_reports_skipped(env):
    from supabase_.delete_documents import delete_document

    ingest_file(upload("Notes.txt", "hello world text"))
    assert delete_document("Notes.txt")["graph"] == "skipped"
