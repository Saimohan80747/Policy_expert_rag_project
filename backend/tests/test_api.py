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
