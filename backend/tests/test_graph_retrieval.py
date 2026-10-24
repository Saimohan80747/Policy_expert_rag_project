"""Graph retrieval: query safety, entity linking, query generation with retries, and the answer flow."""
import pytest

import graph_retrieval
import rag
from knowledge_graph import text2cypher
from knowledge_graph.linker import fuzzy_query, link_entities
from knowledge_graph.text2cypher import RestrictedQuery, validate_cypher

GOOD = "MATCH (p:Entity {uid: 'Person:LS00433'})-[r:REPORTS_TO]->(m:Person) WHERE r.valid_to IS NULL RETURN m.name AS manager"


# ------------------------------- validation -------------------------------
def test_valid_query_gets_a_row_limit_and_fences_are_stripped():
    out = validate_cypher("```cypher\n" + GOOD + ";\n```")
    assert out.startswith("MATCH") and out.endswith("LIMIT 50")
    assert validate_cypher(GOOD + " LIMIT 5").count("LIMIT") == 1


@pytest.mark.parametrize("bad", [
    "MATCH (n) DETACH DELETE n",
    "MATCH (n) SET n.name = 'x' RETURN n",
    "CREATE (n:Person) RETURN n",
    "MATCH (n) RETURN n; MATCH (m) RETURN m",
    "CALL db.labels() YIELD label RETURN label",
    "MATCH (n) RETURN n UNION MATCH (m) RETURN m",
    "MATCH (a:Alias) RETURN a.alias",
    "MATCH (m:GraphMeta) RETURN m",
    "LOAD CSV FROM 'http://x' AS row RETURN row",
    "MATCH (n) WHERE n.name = 'x'",
    "",
])
def test_unsafe_or_malformed_queries_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_cypher(bad)


@pytest.mark.parametrize("q", [
    "MATCH (p:Person)-[e:EARNS]->(l:Level) RETURN e.base_amount",
    "MATCH (p:Person) RETURN p.salary",
    "MATCH (p)-[e]->(l) RETURN e.variable_pct",
])
def test_compensation_queries_are_restricted(q):
    with pytest.raises(RestrictedQuery):
        validate_cypher(q)


def test_keywords_inside_string_values_do_not_trigger_the_filter():
    assert "Set Theory" in validate_cypher("MATCH (s:Skill {name: 'Set Theory'}) RETURN s.name")


def test_schema_text_hides_compensation_and_internal_properties(fake_graph):
    text = text2cypher.get_schema()
    assert "ACCOUNT_MANAGER_OF" in text and "(Person)->(Client)" in text
    assert "EARNS" not in text and "base_amount" not in text and "eid" not in text


# --------------------------------- linking ---------------------------------
def test_links_entities_found_in_the_question_text(fake_graph):
    linked = link_entities("Who reports to Harini Subramanyam?")
    assert [(e["uid"], e["how"]) for e in linked] == [("Person:LS00433", "alias")]


def test_router_surfaces_resolve_exactly_ambiguously_or_fuzzily(fake_graph):
    fake_graph.entities["Person:LS00900"] = ("Priya Sharma", "Person")
    fake_graph.exact["priya sharma"] = [
        {"uid": "Person:LS00900", "name": "Priya Sharma", "label": "Person", "ambiguous": True},
        {"uid": "Person:LS00901", "name": "Priya Sharma", "label": "Person", "ambiguous": True}]
    fake_graph.fuzzy = [{"uid": "Project:PRJ-1", "name": "Rudra", "label": "Project", "score": 2.0}]
    linked = link_entities("what about it", ["Priya Sharma", "Rudhra"])
    assert [(e["uid"], e["how"]) for e in linked] == [
        ("Person:LS00900", "ambiguous"), ("Person:LS00901", "ambiguous"), ("Project:PRJ-1", "fuzzy")]


def test_fuzzy_query_escapes_lucene_syntax_and_drops_short_tokens():
    assert fuzzy_query('Acme (Pvt) "Ltd" a') == "Acme~ Pvt~ Ltd~"


# ----------------------------- query generation -----------------------------
def test_retrieval_runs_the_generated_query_and_formats_facts(fake_graph, fake_llm):
    fake_llm.cypher = [GOOD]
    fake_graph.read_results = [[{"manager": "Madhav Bandi", "since": None}]]
    ctx = graph_retrieval.retrieve_graph_context("Who does Harini Subramanyam report to?", [])
    assert ctx["facts"] == ["manager=Madhav Bandi"] and ctx["cypher"].endswith("LIMIT 50")
    assert ctx["as_of"] == "2026-09-30" and ctx["linked"][0]["uid"] == "Person:LS00433"
    assert ctx["error"] is None and len(fake_graph.read_calls) == 1


def test_a_failing_query_is_fixed_by_a_second_attempt(fake_graph, fake_llm):
    fake_llm.cypher = ["MATCH (p:Person) RETURN p.nme AS x", GOOD]
    fake_graph.read_results = [RuntimeError("Unknown property"), [{"manager": "Madhav Bandi"}]]
    ctx = graph_retrieval.retrieve_graph_context("who is the manager", [])
    assert ctx["facts"] == ["manager=Madhav Bandi"]
    assert "Unknown property" in fake_llm.calls[-1][2]       # the error was shown to the model


def test_an_unsafe_generated_query_is_never_executed(fake_graph, fake_llm):
    fake_llm.cypher = ["MATCH (n) DETACH DELETE n", GOOD]
    fake_graph.read_results = [[{"manager": "Madhav Bandi"}]]
    ctx = graph_retrieval.retrieve_graph_context("who is the manager", [])
    assert fake_graph.read_calls == [ctx["cypher"]] and "DELETE" not in ctx["cypher"]


def test_empty_results_after_retry_report_no_facts(fake_graph, fake_llm):
    fake_llm.cypher = [GOOD]
    fake_graph.read_results = [[]]
    ctx = graph_retrieval.retrieve_graph_context("who is the manager", [])
    assert ctx["facts"] == [] and "no rows" in ctx["error"] and len(fake_graph.read_calls) == 3


def test_compensation_questions_are_flagged_restricted(fake_graph, fake_llm):
    fake_llm.cypher = ["MATCH (n) RETURN n.salary"]
    ctx = graph_retrieval.retrieve_graph_context("what does Harini earn?", [])
    assert ctx["restricted"] is True and fake_graph.read_calls == []


def test_retrieval_never_raises_when_the_llm_is_down(fake_graph, fake_llm):
    fake_llm.fail_json = True
    ctx = graph_retrieval.retrieve_graph_context("anything", [])
    assert ctx["facts"] == [] and "llm down" in ctx["error"]


# ------------------------------- answer flow -------------------------------
def seed_doc(env, text="Casual leave is 12 days per year."):
    import io

    from fastapi import UploadFile

    from ingestion import ingest_file

    ingest_file(UploadFile(filename="Leave Policy.txt", file=io.BytesIO(text.encode())))


def test_graph_route_answers_from_facts_without_searching_documents(env, fake_graph, fake_llm):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "Who does Harini report to?", "route": "graph", "entities": ["Harini"]}
    fake_llm.cypher, fake_graph.read_results = [GOOD], [[{"manager": "Madhav Bandi"}]]
    fake_llm.text = "Harini reports to Madhav Bandi [G1]."
    result = rag.answer_question("Who does Harini report to?")
    assert result["route_used"] == "graph" and result["answer"].startswith("Harini reports")
    graph_src = [s for s in result["sources"] if s.get("type") == "graph"]
    assert len(graph_src) == 1 and graph_src[0]["facts"] == ["manager=Madhav Bandi"]
    assert graph_src[0]["cypher"] and result["file_name"] == ""
    answer_prompt = [c for c in fake_llm.calls if c[0] == "text" and c[3] == rag.ANSWER_SYSTEM][0][2]
    assert "[G1] manager=Madhav Bandi" in answer_prompt and "Casual leave" not in answer_prompt


def test_hybrid_route_combines_graph_facts_and_document_passages(env, fake_graph, fake_llm):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "leave policy owner", "route": "hybrid", "entities": []}
    fake_llm.cypher, fake_graph.read_results = [GOOD], [[{"manager": "Madhav Bandi"}]]
    result = rag.answer_question("leave policy owner")
    assert result["route_used"] == "hybrid"
    kinds = [s.get("type", "document") for s in result["sources"]]
    assert kinds.count("document") >= 1 and kinds.count("graph") == 1 and result["file_name"] == "Leave_Policy.txt"


def test_graph_route_with_no_facts_falls_back_to_document_search(env, fake_graph, fake_llm):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "casual leave days", "route": "graph", "entities": []}
    fake_llm.cypher, fake_graph.read_results = [GOOD], [[]]
    result = rag.answer_question("casual leave days")
    assert result["route_used"] == "vanilla" and result["route"] == "graph"
    assert all(s.get("type") != "graph" for s in result["sources"]) and result["file_name"] == "Leave_Policy.txt"


def test_restricted_question_gets_a_fixed_answer_without_calling_the_answer_model(env, fake_graph, fake_llm):
    fake_llm.analysis = {"standalone_question": "what does Harini earn", "route": "graph", "entities": []}
    fake_llm.cypher = ["MATCH (n) RETURN n.salary"]
    result = rag.answer_question("what does Harini earn")
    assert result["answer"] == rag.RESTRICTED_ANSWER and result["route_used"] == "restricted"
    assert not [c for c in fake_llm.calls if c[3] == rag.ANSWER_SYSTEM]


def test_without_a_loaded_graph_everything_stays_vanilla(env, fake_graph, fake_llm):
    seed_doc(env)
    fake_graph.loaded = False
    fake_llm.analysis = {"standalone_question": "Who does Harini report to?", "route": "graph", "entities": []}
    result = rag.answer_question("Who does Harini report to?")
    assert result["route_used"] == "vanilla" and not fake_graph.read_calls


def test_empty_graph_attempt_is_explained_in_the_result(env, fake_graph, fake_llm):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "casual leave days", "route": "graph", "entities": []}
    fake_llm.cypher, fake_graph.read_results = [GOOD], [[]]
    debug = rag.answer_question("casual leave days")["graph_debug"]
    assert debug["cypher"] and "no rows" in debug["error"]


def test_unavailable_graph_is_logged_when_the_router_wanted_it(env, fake_llm, caplog):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "who reports to X", "route": "graph", "entities": []}
    history = [{"question": "hi", "answer": "hello"}]  # a history makes the router run
    rag.answer_question("who reports to X", history)    # no graph configured in this test
    assert "graph is not available" in caplog.text


def test_a_broken_connection_is_logged_not_silent(monkeypatch, caplog):
    from knowledge_graph import db

    monkeypatch.setattr(db, "run", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("ssl: bad certificate")))
    assert graph_retrieval.graph_available() is False
    assert "ssl: bad certificate" in caplog.text


def test_run_read_uses_a_read_transaction_with_a_transaction_timeout(monkeypatch):
    """Guards the real driver contract: managed transactions take a unit_of_work timeout, not Query objects."""
    from knowledge_graph import db

    seen = {}

    class Record:
        def __init__(self, d):
            self.d = d

        def data(self):
            return self.d

    class Tx:
        def run(self, query, params):
            seen["query"], seen["params"] = query, params
            assert isinstance(query, str)                 # a Query object raises TypeError in the real driver
            return iter(Record({"n": i}) for i in range(100))

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute_read(self, work):
            seen["timeout"] = getattr(work, "timeout", None)
            return work(Tx())

    class Driver:
        def session(self):
            return Session()

    monkeypatch.setattr(db, "get_driver", lambda: Driver())
    rows = db.run_read("MATCH (n) RETURN n", {"a": 1}, limit=3, timeout=9)
    assert rows == [{"n": 0}, {"n": 1}, {"n": 2}]
    assert seen["timeout"] == 9 and seen["params"] == {"a": 1}


def test_individual_pay_questions_are_refused_before_any_search(env, fake_graph, fake_llm):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "Harini's salary", "route": "vanilla", "entities": [],
                         "asks_individual_pay": True}
    result = rag.answer_question("What is Harini Subramanyam's salary?")
    assert result["answer"] == rag.RESTRICTED_ANSWER and result["sources"] == []
    assert not [c for c in fake_llm.calls if c[3] == rag.ANSWER_SYSTEM] and not fake_graph.read_calls


def test_pay_policy_questions_are_still_answered(env, fake_graph, fake_llm):
    seed_doc(env, "Salary is paid on the last working day of each month.")
    fake_llm.analysis = {"standalone_question": "When is salary paid?", "route": "vanilla", "entities": [],
                         "asks_individual_pay": False}
    assert rag.answer_question("When is salary paid?")["route_used"] == "vanilla"


def test_answer_prompt_includes_the_query_that_explains_the_rows(env, fake_graph, fake_llm):
    fake_llm.analysis = {"standalone_question": "Who reported to X in 2018?", "route": "graph", "entities": []}
    fake_llm.cypher, fake_graph.read_results = [GOOD], [[{"employee": "Sravani Pothineni"}]]
    rag.answer_question("Who reported to X in 2018?")
    prompt = [c for c in fake_llm.calls if c[3] == rag.ANSWER_SYSTEM][0][2]
    assert "REPORTS_TO" in prompt and "[G1] employee=Sravani Pothineni" in prompt


def test_neighbourhood_is_shown_to_the_model(fake_graph, fake_llm):
    from knowledge_graph import text2cypher

    text2cypher.reset_neighbourhood_cache()
    fake_graph.neighbour_rows = [{"rel": "ACCOUNT_MANAGER_OF", "outgoing": False, "label": "Person", "n": 3, "current": 1}]
    fake_llm.cypher = [GOOD]
    fake_graph.read_results = [[{"manager": "A"}]]
    graph_retrieval.retrieve_graph_context("who manages Deccan Freight Logistics", ["Deccan Freight Logistics"])
    prompt = next(c[2] for c in fake_llm.calls if "Linked entities" in c[2])
    assert "<-[:ACCOUNT_MANAGER_OF]-(Person) x3 (1 current)" in prompt
    text2cypher.reset_neighbourhood_cache()


def test_a_count_of_zero_is_retried_once_and_kept_if_it_stays_zero(fake_graph, fake_llm):
    fake_llm.cypher = [GOOD]
    fake_graph.read_results = [[{"n": 0}]]
    ctx = graph_retrieval.retrieve_graph_context("how many records describe it", [])
    assert len(fake_graph.read_calls) == 3
    assert ctx["rows"] == [{"n": 0}] and ctx["error"] is None


def test_a_zero_count_is_replaced_by_a_better_retry(fake_graph, fake_llm):
    fake_llm.cypher = [GOOD]
    fake_graph.read_results = [[{"n": 0}], [{"n": 2}]]
    ctx = graph_retrieval.retrieve_graph_context("how many records describe it", [])
    assert ctx["rows"] == [{"n": 2}] and len(fake_graph.read_calls) == 2
    assert "only zeros" in next(c[2] for c in fake_llm.calls[1:] if "did not work" in c[2])
