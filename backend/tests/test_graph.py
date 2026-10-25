"""Graph loader tests: pure preparation plus the Cypher it would send (no Neo4j needed)."""
import csv
import json

import pytest

from knowledge_graph import loader
from knowledge_graph.schema import STATEMENTS, ensure_schema


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def corpus(tmp_path):
    write_csv(tmp_path / "kg_node.csv", ["label", "id", "name", "props"], [
        ["Person", "LS1", "Asha Rao", json.dumps({"status": "active", "exit_date": None, "office": "HYD"})],
        ["Person", "LS2", "Bala Iyer", json.dumps({"status": "former"})],
        ["Project", "PRJ-1", "Beacon", json.dumps({"codename": "Beacon", "rag": None})],
        ["PolicyVersion", "POL-1-v1", "POL-1 v1", json.dumps({"key_values": {"values_count": 6}, "id": "evil"})],
    ])
    write_csv(tmp_path / "kg_edge.csv",
              ["src_label", "src_id", "rel", "dst_label", "dst_id", "valid_from", "valid_to", "props"], [
        ["Person", "LS2", "REPORTS_TO", "Person", "LS1", "2020-01-01", "2022-01-01", ""],
        ["Person", "LS2", "REPORTS_TO", "Person", "LS1", "2022-02-01", "", ""],
        ["Person", "LS2", "REPORTS_TO", "Person", "LS1", "2022-02-01", "", ""],   # exact duplicate
        ["Person", "LS1", "WORKS_ON", "Project", "PRJ-1", "2021-01-01", "", json.dumps({"allocation_pct": 30})],
        ["Person", "LS1", "WORKS_ON", "Project", "PRJ-404", "2021-01-01", "", ""],   # dangling
    ])
    write_csv(tmp_path / "alias.csv",
              ["entity_label", "entity_id", "alias", "alias_type", "valid_from", "valid_to", "ambiguous_with"], [
        ["Person", "LS1", "Asha Rao", "canonical", "", "", ""],
        ["Person", "LS1", "AR", "initials", "", "", "Person:LS2"],
        ["Person", "LS9", "Ghost", "canonical", "", "", ""],                       # unknown entity
    ])
    write_csv(tmp_path / "meta.csv", ["key", "value"], [["as_of", "2026-09-30"], ["company", "Lumenstack"]])
    return str(tmp_path)


def test_normalize():
    assert loader.normalize("Dr. S. Pothineni") == "dr s pothineni"
    assert loader.normalize("José  Núñez") == "jose nunez"


def test_nodes_get_uid_flat_props_and_cannot_override_identity(corpus):
    data = loader.load_corpus(corpus)
    people = {n["id"]: n for n in data["nodes"]["Person"]}
    assert people["LS1"]["uid"] == "Person:LS1" and people["LS1"]["status"] == "active"
    assert "exit_date" not in people["LS1"]                       # nulls are dropped
    pv = data["nodes"]["PolicyVersion"][0]
    assert pv["id"] == "POL-1-v1"                                 # props cannot override id
    assert json.loads(pv["key_values"]) == {"values_count": 6}   # nested dict kept as JSON text


def test_edges_dedupe_skip_dangling_and_keep_validity(corpus):
    data = loader.load_corpus(corpus)
    assert data["dangling_edges"] == 1 and data["duplicate_edges"] == 1
    reports = data["edges"]["REPORTS_TO"]
    assert len(reports) == 2
    old, new = sorted(reports, key=lambda e: e["props"]["valid_from"])
    assert old["props"]["valid_to"] == "2022-01-01" and "valid_to" not in new["props"]
    assert old["props"]["eid"] != new["props"]["eid"]
    assert data["edges"]["WORKS_ON"][0]["props"]["allocation_pct"] == 30


def test_aliases_normalised_and_unknown_entities_skipped(corpus):
    data = loader.load_corpus(corpus)
    assert data["alias_skipped"] == 1
    by_alias = {a["alias"]: a for a in data["aliases"]}
    assert by_alias["AR"]["norm"] == "ar" and by_alias["AR"]["ambiguous"] is True
    assert by_alias["AR"]["ambiguous_with"] == ["Person:LS2"]
    assert by_alias["Asha Rao"]["ambiguous"] is False


def test_unsafe_labels_and_relationship_types_are_rejected(corpus, tmp_path):
    with pytest.raises(ValueError):
        loader.prepare_nodes([{"label": "Person) DETACH DELETE n //", "id": "1", "name": "x", "props": ""}])
    with pytest.raises(ValueError):
        loader.prepare_edges([{"src_label": "A", "src_id": "1", "rel": "X]-() DELETE", "dst_label": "A",
                               "dst_id": "1", "valid_from": "", "valid_to": "", "props": ""}], {"A:1"})


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, query, **params):
        self.calls.append((query, params))
        return [{"c": 0}]


def test_write_graph_sends_batched_merge_statements(corpus):
    data = loader.load_corpus(corpus)
    rec = Recorder()
    loader.write_graph(data, rec)
    queries = [q for q, _ in rec.calls]
    assert any("MERGE (n:Entity:Person {uid: row.uid})" in q for q in queries)
    assert any("MERGE (a)-[r:REPORTS_TO {eid: row.props.eid}]->(b)" in q for q in queries)
    assert any("MERGE (a:Alias {aid: row.aid})" in q for q in queries)
    assert any("GraphMeta" in q for q in queries)
    assert all("DELETE" not in q for q in queries)               # writing never deletes


def test_large_inputs_are_batched(monkeypatch, corpus):
    monkeypatch.setattr(loader, "BATCH", 1)
    data = loader.load_corpus(corpus)
    rec = Recorder()
    loader.write_graph(data, rec)
    person_calls = [p for q, p in rec.calls if "Entity:Person" in q]
    assert len(person_calls) == 2 and all(len(p["rows"]) == 1 for p in person_calls)


def test_reset_only_touches_domain_layer():
    rec = Recorder()
    loader.reset_graph(rec)
    assert "n:Entity OR n:Alias" in rec.calls[0][0]


def test_schema_statements_are_idempotent_and_all_run():
    rec = Recorder()
    ensure_schema(rec)
    assert len(rec.calls) == len(STATEMENTS)
    assert all("IF NOT EXISTS" in q for q, _ in rec.calls)


def test_verify_reports_mismatch(corpus):
    data = loader.load_corpus(corpus)
    s = loader.summary(data)

    def run(query, **_):
        if "Entity) RETURN" in query:
            return [{"c": s["nodes"]}]
        if "WHERE r.eid" in query:
            return [{"c": s["edges"] - 1}]
        return [{"c": s["aliases"]}]

    result = loader.verify(data, run)
    assert result["ok"] is False and result["edges"] == (s["edges"] - 1, s["edges"])


def test_dry_run_cli_needs_no_neo4j(corpus, capsys):
    assert loader.main(["--csv-dir", corpus, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"nodes": 4' in out and "Dry run" in out


def test_graph_stats_endpoint_reports_errors_as_json(monkeypatch):
    from fastapi.testclient import TestClient

    import api
    from knowledge_graph import stats

    monkeypatch.setattr(stats, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no neo4j")))
    r = TestClient(api.app, raise_server_exceptions=False).get("/graph/stats")
    assert r.status_code == 500 and "no neo4j" in r.json()["detail"]


def test_aliases_are_derived_for_types_the_alias_table_omits(tmp_path):
    write_csv(tmp_path / "kg_node.csv", ["label", "id", "name", "props"], [
        ["Technology", "TECH-048", "Elasticsearch", ""],
        ["Technology", "TECH-049", "Kafka", ""],
        ["Person", "LS1", "Asha Rao", ""],
        ["Document", "ADR-0040", "Use Redis Streams", ""],
    ])
    write_csv(tmp_path / "kg_edge.csv", ["src_label", "src_id", "rel", "dst_label", "dst_id", "valid_from", "valid_to", "props"], [])
    write_csv(tmp_path / "alias.csv", ["entity_label", "entity_id", "alias", "alias_type", "valid_from", "valid_to", "ambiguous_with"],
              [["Person", "LS1", "Kafka", "nickname", "", "", ""]])      # existing alias keeps its surface
    data = loader.load_corpus(str(tmp_path))
    derived = {a["alias"]: a for a in data["aliases"] if a["alias_type"].startswith("derived")}
    assert derived["Elasticsearch"]["uid"] == "Technology:TECH-048"
    assert derived["ADR-0040"]["alias_type"] == "derived_id" and "Use Redis Streams" not in derived
    assert "Kafka" not in derived                                         # claimed by the alias table


def test_graph_stats_includes_the_document_layer(monkeypatch):
    from knowledge_graph import stats

    def run(query, **_):
        if "UNWIND labels" in query:
            return [{"label": "Person", "count": 2}]
        if "type(r) AS rel" in query:
            return [{"rel": "REPORTS_TO", "count": 1}]
        if "GraphMeta" in query:
            return []
        if "Alias" in query:
            return [{"c": 5}]
        return [{"c": 7}]

    monkeypatch.setattr(stats, "run", run)
    out = stats.graph_stats()
    assert out["nodes"] == 2 and out["aliases"] == 5
    assert out["documents"]["chunks"] == 7 and set(out["documents"]) >= {"source_docs", "chunk_mentions", "adr_rfc_links"}
