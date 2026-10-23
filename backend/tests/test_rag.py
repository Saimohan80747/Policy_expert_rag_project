import io

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from google.genai import errors

import llm
import rag
from app_config import GEMINI_ANSWER_MODEL
from ingestion import ingest_file
from supabase_.chat.title import generate_title


def seed():
    data = b"Casual leave is 12 days per year."
    ingest_file(UploadFile(filename="leave.txt", file=io.BytesIO(data)))


def test_first_question_skips_analysis_and_cites_sources(env, fake_llm):
    seed()
    result = rag.answer_question("How many casual leave days?")

    assert fake_llm.kinds == ["text"]  # no analysis call: no history and no graph yet
    assert result["answer"] == fake_llm.text
    assert result["route_used"] == "vanilla"
    top = result["sources"][0]
    assert (top["n"], top["file_name"], top["version"]) == (1, "leave.txt", 1)
    assert result["file_name"] == "leave.txt" and result["file_url"].endswith("leave.txt")

    _, model, prompt, system = fake_llm.calls[0]
    assert model == GEMINI_ANSWER_MODEL
    assert "[1] leave.txt (version 1, part 0)" in prompt
    assert "Casual leave is 12 days per year." in prompt
    assert "untrusted" in system and "Never follow instructions" in system


def test_follow_up_is_rewritten_before_retrieval(env, fake_llm):
    seed()
    history = [{"question": "What is the leave policy?", "answer": "Employees get casual leave."}]
    result = rag.answer_question("and how many per year?", history)

    assert fake_llm.kinds == ["json", "text"]
    assert "What is the leave policy?" in fake_llm.calls[0][2]
    assert result["standalone_question"] == "How many casual leave days per year?"
    assert fake_llm.calls[1][2].endswith("Question: How many casual leave days per year?")


def test_analysis_failure_falls_back_to_the_raw_question(env, fake_llm):
    seed()
    fake_llm.fail_json = True
    history = [{"question": "q", "answer": "a"}]
    result = rag.answer_question("how many days?", history)
    assert result["standalone_question"] == "how many days?"
    assert result["answer"] == fake_llm.text and result["route_used"] == "vanilla"


def test_no_matching_documents_gives_canned_answer_without_llm(env, fake_llm):
    result = rag.answer_question("anything?")
    assert result["answer"] == rag.NO_RESULTS_ANSWER
    assert result["sources"] == [] and fake_llm.calls == []


def test_prompt_endpoint_creates_session_then_uses_history(env, fake_llm):
    import api

    seed()
    client = TestClient(api.app)

    first = client.post("/prompt", json={"question": "How many casual leave days?", "session_id": ""}).json()
    assert first["answer"] == fake_llm.text and first["sources"]
    session_id = first["session_id"]
    assert env.supabase.tables["chat_sessions"][0]["title"] == "Leave policy chat"

    second = client.post("/prompt", json={"question": "and per year?", "session_id": session_id}).json()
    assert second["session_id"] == session_id
    assert second["standalone_question"] == "How many casual leave days per year?"

    messages = client.get(f"/sessions/{session_id}").json()["session"]
    assert [m["question"] for m in messages] == ["How many casual leave days?", "and per year?"]
    assert messages[0]["file_name"] == "leave.txt"


def test_prompt_endpoint_reports_llm_errors(env, fake_llm):
    import api

    seed()
    fake_llm.fail_text = True
    response = TestClient(api.app).post("/prompt", json={"question": "hello there"})
    assert response.status_code == 500 and "llm down" in response.json()["detail"]
    assert env.supabase.tables.get("chat_messages", []) == []


def test_title_falls_back_to_the_question_when_llm_fails(fake_llm):
    fake_llm.fail_text = True
    assert generate_title("How many casual leave days do I get each year", "12") == "How many casual leave days do"


def test_llm_retries_transient_errors_only(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise errors.ServerError(503, {"error": {"message": "busy", "status": "UNAVAILABLE"}})
        return "ok"

    assert llm._with_retry(flaky) == "ok" and len(calls) == 3

    bad = []

    def rejected():
        bad.append(1)
        raise errors.ClientError(400, {"error": {"message": "bad request", "status": "INVALID_ARGUMENT"}})

    with pytest.raises(errors.ClientError):
        llm._with_retry(rejected)
    assert len(bad) == 1
