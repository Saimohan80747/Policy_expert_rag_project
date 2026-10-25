import pytest
import io
import zipfile

from fastapi.testclient import TestClient


def client():
    import api  # must import without any API keys configured

    return TestClient(api.app)


def test_pipeline_search_and_zip_endpoints(env):
    c = client()
    r = c.post("/pipeline", files={"file": ("notes.txt", b"Remote work needs manager approval.")})
    assert r.status_code == 200 and r.json()["status"] == "ingested"

    r = c.post("/pipeline", files={"file": ("notes.txt", b"Remote work needs manager approval.")})
    assert r.json()["status"] == "duplicate"

    hits = c.get("/search", params={"q": "remote work"}).json()["results"]
    assert hits and hits[0]["file_name"] == "notes.txt"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.txt", "first")
        z.writestr("y.bin", "nope")
    r = c.post("/pipeline", files={"file": ("pack.zip", buf.getvalue())}).json()
    assert len(r["ingested_files"]) == 1 and len(r["failed_files"]) == 1, r


# ---------------------- dropped connections / error visibility ----------------------
def test_supabase_retries_dropped_connection(supabase):
    import httpx

    from supabase_.supabase_db import supabase_var

    supabase.tables["chat_sessions"] = [{"id": "s1", "title": "t"}]
    real = supabase_var.table("chat_sessions")
    calls = {"n": 0}
    inner = real._target
    original = inner.execute

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.RemoteProtocolError("Server disconnected")
        return original()

    inner.execute = flaky
    assert real.select("*").execute().data[0]["id"] == "s1"
    assert calls["n"] == 3


def test_supabase_gives_up_after_retries(supabase):
    import httpx

    from supabase_.supabase_db import supabase_var

    q = supabase_var.table("chat_sessions")

    def always():
        raise httpx.RemoteProtocolError("Server disconnected")

    q._target.execute = always
    with pytest.raises(httpx.RemoteProtocolError):
        q.select("*").execute()


def test_unhandled_error_still_carries_cors_header(monkeypatch):
    from fastapi.testclient import TestClient

    import api

    monkeypatch.setattr(api, "get_all_sessions", lambda: (_ for _ in ()).throw(RuntimeError("db exploded")))
    client = TestClient(api.app, raise_server_exceptions=False)
    r = client.get("/history", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 500
    assert "db exploded" in r.json()["detail"]
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
