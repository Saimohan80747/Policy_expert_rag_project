"""Evaluation harness: sampling, the LLM cache, running configurations and reporting."""
import json
from types import SimpleNamespace

import pytest

import rag
from evaluation import llm_cache, report, run_eval, sample
from evaluation.llm_cache import CachedLLM, QuotaReached
from evaluation.modes import options_for

GOOD = ("MATCH (p:Entity {uid: 'Person:LS00433'})-[r:REPORTS_TO]->(m:Person) "
        "WHERE r.valid_to IS NULL RETURN m.name AS manager")


def question(qid, category="relation_lookup", hops=1, behavior="answer", answer_type="entity", **extra):
    base = {"qid": qid, "category": category, "hops": hops, "expected_behavior": behavior, "question": f"Question {qid}?",
            "answer": {"type": answer_type, "value": "Madhav Bandi", "aliases": []}, "evidence": [{"doc_id": "LS-DOC-2026-0001"}],
            "language": "en", "access_role": "all", "gold_path": [], "requires_waves": [], "requires_formats": [],
            "corpus_state": "wave_0", "valid_through_wave": 6}
    return {**base, **extra}


# --------------------------------- sampling ---------------------------------
def test_sample_is_stratified_reproducible_and_skips_unscorable_questions():
    qs = [question(f"Q{i:03d}") for i in range(30)]
    qs += [question(f"M{i}", "multi_hop", hops=4) for i in range(10)]
    qs += [question("R1", behavior="refuse", answer_type="refuse"), question("C1", behavior="clarify", answer_type="clarify"),
           question("J1", answer_type="rubric"), question("L1", language="de"), question("A1", access_role="CHRO"),
           question("F1", gold_path=[{"s": "LS00433"}]), question("D1", evidence=[{"doc_id": "LS-HRX-2026-0018"}])]
    quotas = {("relation_lookup", None): 12, ("multi_hop", 4): 5}
    first = sample.select(qs, lambda *a: True, quotas=quotas, exclude_docs=["LS-HRX-2026-0018"])
    again = sample.select(qs, lambda *a: True, quotas=quotas, exclude_docs=["LS-HRX-2026-0018"])
    assert [q["qid"] for q in first] == [q["qid"] for q in again]
    assert sum(q["category"] == "relation_lookup" for q in first) == 12
    assert sum(q["category"] == "multi_hop" for q in first) == 5
    assert not {"R1", "C1", "J1", "L1", "A1", "F1", "D1"} & {q["qid"] for q in first}
    assert {q["split"] for q in first} <= {"dev", "test"} and any(q["split"] == "dev" for q in first)


def test_split_is_stable_for_a_question():
    assert sample.split_of("Q0001") == sample.split_of("Q0001")


# -------------------------------- the LLM cache --------------------------------
def fake_module(counter):
    def text(prompt, *, system=None, model=None, temperature=0.2):
        counter.append(("text", prompt))
        return f"answer to {prompt}"

    def js(prompt, schema, *, system=None, model=None):
        counter.append(("json", prompt))
        return schema(standalone_question=prompt, route="vanilla", entities=[], asks_individual_pay=False,
                      include_history=False)
    return SimpleNamespace(generate_text=text, generate_json=js)


def test_cache_answers_repeats_from_disk_and_survives_restarts(tmp_path):
    calls = []
    module = fake_module(calls)
    cache = CachedLLM(tmp_path / "c.jsonl", rpm=600, sleep=lambda s: None)
    cache.install(module)
    assert module.generate_text("a", model="m") == "answer to a"
    assert module.generate_text("a", model="m") == "answer to a"
    assert module.generate_json("q", rag.Analysis, model="m").standalone_question == "q"
    assert len(calls) == 2 and cache.new_calls == 2 and cache.logical == 3

    module2 = fake_module(calls)                      # a new process: cache loaded from the file
    again = CachedLLM(tmp_path / "c.jsonl", sleep=lambda s: None)
    again.install(module2)
    assert module2.generate_text("a", model="m") == "answer to a" and len(calls) == 2 and again.new_calls == 0
    assert module2.generate_text("a", model="other") == "answer to a" and len(calls) == 3   # model is part of the key


def test_cache_stops_at_the_call_budget_and_on_daily_quota(tmp_path):
    calls = []
    module = fake_module(calls)
    cache = CachedLLM(tmp_path / "c.jsonl", max_new_calls=1, sleep=lambda s: None)
    cache.install(module)
    module.generate_text("a")
    with pytest.raises(QuotaReached):
        module.generate_text("b")

    def daily(*a, **k):
        raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")
    daily_module = SimpleNamespace(generate_text=daily, generate_json=daily)
    CachedLLM(tmp_path / "d.jsonl", sleep=lambda s: None).install(daily_module)
    with pytest.raises(QuotaReached):
        daily_module.generate_text("x")


def test_cache_throttles_uncached_calls(tmp_path):
    waits, now = [], [0.0]
    module = fake_module([])
    cache = CachedLLM(tmp_path / "c.jsonl", rpm=30, sleep=waits.append, clock=lambda: now[0])
    cache.install(module)
    module.generate_text("a")
    module.generate_text("b")
    assert waits == [pytest.approx(2.0)]
    module.generate_text("a")                          # cached: no wait
    assert len(waits) == 1


# ------------------------------ prediction mapping ------------------------------
def test_prediction_maps_sources_abstention_and_refusal():
    result = {"answer": "I couldn't find anything relevant in the uploaded documents. Try...", "route_used": "vanilla",
              "route": "vanilla", "sources": [{"file_name": "LS-POL-2025-0005__leave-v5.pdf"}, {"file_name": "plain.txt"},
                                              {"file_name": "LS-POL-2025-0005__leave-v5.pdf"}, {"type": "graph", "rows": 3}]}
    p = run_eval.to_prediction("Q1", dict(result, graph_llm_calls=2), 1.234)
    assert p["abstained"] and p["sources"] == ["LS-POL-2025-0005"] and p["llm_calls"] == 3 and p["graph_rows"] == 3   # analysis + 2 Cypher; no answer call
    refused = run_eval.to_prediction("Q2", {"answer": rag.RESTRICTED_ANSWER, "route_used": "restricted", "sources": []}, 0)
    assert refused["refused"] and not refused["abstained"]
    ok = run_eval.to_prediction("Q3", {"answer": "Madhav Bandi [G1]", "route_used": "graph", "sources": []}, 0)
    assert not ok["abstained"] and not ok["refused"]
    failed = run_eval.to_prediction("Q4", None, 0, error="RuntimeError: boom")
    assert failed["abstained"] and failed["error"] and failed["llm_calls"] == 1


# ------------------------------ configurations ------------------------------
def seed_doc(env):
    import io
    from fastapi import UploadFile
    from ingestion import ingest_file
    ingest_file(UploadFile(filename="LS-POL-2025-0005__leave.txt", file=io.BytesIO(b"Casual leave is 12 days per year.")))


def test_graph_only_mode_does_not_fall_back_to_documents(env, fake_graph, fake_llm):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "Who does Harini report to?", "route": "vanilla", "entities": ["Harini"]}
    fake_llm.cypher, fake_graph.read_results = [GOOD], [[]]
    analysis = rag.analyze_question("Who does Harini report to?", None)
    out = rag.answer_question("Who does Harini report to?", None, options_for("graph", analysis=analysis))
    assert out["answer"] == rag.NO_RESULTS_ANSWER and out["route_used"] == "graph" and out["sources"] == []
    normal = rag.answer_question("Who does Harini report to?", None, options_for("router", analysis=analysis.model_copy(update={"route": "graph"})))
    assert normal["route_used"] == "vanilla" and normal["sources"]       # the shipped fallback is unchanged


def test_vanilla_mode_never_touches_the_graph_and_hybrid_uses_both(env, fake_graph, fake_llm):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "Who does Harini report to?", "route": "graph", "entities": ["Harini"]}
    fake_llm.cypher, fake_graph.read_results = [GOOD], [[{"manager": "Madhav Bandi"}]]
    analysis = rag.analyze_question("Who does Harini report to?", None)
    cache = {}
    vanilla = rag.answer_question("q", None, options_for("vanilla", analysis=analysis, graph_cache=cache))
    assert not [s for s in vanilla["sources"] if s.get("type") == "graph"] and cache == {} and not fake_graph.read_calls
    hybrid = rag.answer_question("q", None, options_for("hybrid", analysis=analysis, graph_cache=cache))
    assert [s for s in hybrid["sources"] if s.get("type") == "graph"] and [s for s in hybrid["sources"] if "file_name" in s]
    reads = len(fake_graph.read_calls)
    rag.answer_question("q", None, options_for("graph", analysis=analysis, graph_cache=cache))
    assert len(fake_graph.read_calls) == reads                          # graph retrieval shared between modes


def test_evaluate_runs_modes_resumes_and_reports(env, fake_graph, fake_llm, tmp_path):
    seed_doc(env)
    fake_llm.analysis = {"standalone_question": "Who does Harini report to?", "route": "graph", "entities": ["Harini"]}
    fake_llm.cypher, fake_graph.read_results = [GOOD], [[{"manager": "Madhav Bandi"}]]
    fake_llm.text = "Harini reports to Madhav Bandi [G1]."
    import llm
    cached = CachedLLM(tmp_path / "cache.jsonl", rpm=6000, sleep=lambda s: None)
    cached.install(llm)
    qs = [question("Q1", evidence=[{"doc_id": "LS-POL-2025-0005"}])]
    modes = ["vanilla", "graph", "hybrid", "router"]
    run_eval.evaluate(qs, modes, tmp_path, cached, rag, log=lambda m: None)
    preds = {m: run_eval.read_jsonl(tmp_path / f"{m}.jsonl") for m in modes}
    assert all(len(v) == 1 for v in preds.values())
    assert preds["graph"][0]["route_used"] == "graph" and preds["router"][0]["graph_rows"] == 1
    assert preds["vanilla"][0]["graph_rows"] == 0 and preds["vanilla"][0]["sources"] == ["LS-POL-2025-0005"]
    # one analysis + one Cypher + three distinct answers (vanilla, graph, hybrid; router reuses graph's)
    assert cached.new_calls <= 6
    assert preds["graph"][0]["llm_calls"] == 1 + 1 + 1 and preds["vanilla"][0]["llm_calls"] == 2
    assert preds["hybrid"][0]["llm_calls"] == 3                         # charged for the Cypher it reused
    before = cached.new_calls
    run_eval.evaluate(qs, modes, tmp_path, cached, rag, log=lambda m: None)
    assert cached.new_calls == before and all(len(run_eval.read_jsonl(tmp_path / f"{m}.jsonl")) == 1 for m in modes)

    scorer = SimpleNamespace(score_one=lambda q, p: {"qid": q["qid"], "category": q["category"], "behavior": q["expected_behavior"],
                                                     "correct": "Madhav Bandi" in p["answer"], "evidence_hit": True, "evidence_recall": 1.0})
    results = report.write_report(tmp_path, scorer, qs, modes, {0})
    assert results["graph"]["overall"]["accuracy"] == 1.0 and set(results) == set(modes)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| graph |" in text and "relation_lookup" in text and "Router decisions" in text


def test_cached_calls_are_priced_by_their_recorded_duration(tmp_path):
    now = [0.0]
    module = SimpleNamespace(generate_text=lambda p, **k: now.__setitem__(0, now[0] + 4.0) or "x", generate_json=None)
    first = CachedLLM(tmp_path / "c.jsonl", rpm=6000, sleep=lambda s: None, clock=lambda: now[0])
    first.install(module)
    module.generate_text("a")
    assert first.real_seconds == pytest.approx(4.0) and first.charged == pytest.approx(4.0)
    module.generate_text("a")                                            # cached: free in real time, charged 4s
    assert first.real_seconds == pytest.approx(4.0) and first.charged == pytest.approx(8.0)
    second = CachedLLM(tmp_path / "c.jsonl", sleep=lambda s: None)
    second.install(module)
    module.generate_text("a")
    assert second.charged == pytest.approx(4.0) and second.new_calls == 0


def test_failures_lists_wrong_answers_with_context(tmp_path, monkeypatch, capsys):
    from evaluation import failures

    run_dir = tmp_path / "evaluation" / "runs" / "t"
    run_dir.mkdir(parents=True)
    q = question("Q1", split="dev")
    (run_dir / "sample.json").write_text(json.dumps([q]))
    (run_dir / "router.jsonl").write_text(json.dumps({"qid": "Q1", "answer": "Wrong person", "abstained": False, "sources": [],
                                                      "route": "graph", "route_used": "vanilla"}) + "\n")
    scorer_dir = tmp_path / "corpus_eval"
    scorer_dir.mkdir()
    (scorer_dir / "score.py").write_text("def score_one(q, p):\n    return {'correct': 'Madhav' in p['answer'], 'reason': 'no match'}\n")
    monkeypatch.setattr(failures, "__file__", str(tmp_path / "evaluation" / "failures.py"))
    failures.main(["--name", "t", "--corpus-eval", str(scorer_dir)])
    out = capsys.readouterr().out
    assert "1 wrong answers" in out and "Gold: Madhav Bandi" in out and "Wrong person" in out and "chose graph, used vanilla" in out
