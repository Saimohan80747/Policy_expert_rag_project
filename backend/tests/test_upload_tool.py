import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import upload_folder  # noqa: E402


def make_tree(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "_meta").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("Casual leave is 12 days per year.")
    (tmp_path / "docs" / "b.txt").write_text("Sick leave is 8 days per year.")
    (tmp_path / "docs" / "ignore.json").write_text("{}")
    (tmp_path / "_meta" / "a.txt").write_text("sidecar, must be skipped")
    return tmp_path


def test_find_files_skips_meta_and_unsupported(tmp_path):
    names = [p.name for p in upload_folder.find_files(make_tree(tmp_path))]
    assert names == ["a.txt", "b.txt"]


def test_sample_is_deterministic_and_limit_applies(tmp_path):
    root = make_tree(tmp_path)
    assert upload_folder.find_files(root, sample=1) == upload_folder.find_files(root, sample=1)
    assert len(upload_folder.find_files(root, limit=1)) == 1


def test_upload_is_resumable_and_reports_failures(env, tmp_path):
    import api

    client = TestClient(api.app)
    files = upload_folder.find_files(make_tree(tmp_path))

    def post(path):
        return client.post("/pipeline", files={"file": (path.name, path.read_bytes())}).json()

    lines = []
    first = upload_folder.upload_all(files, post, out=lines.append)
    again = upload_folder.upload_all(files, post, out=lines.append)
    assert first["ingested"] == 2 and again["duplicate"] == 2

    def broken(path):
        raise ConnectionError("API down")

    assert upload_folder.upload_all(files, broken, out=lines.append)["failed"] == 2


def test_cleanup_removes_bare_named_zip_members_but_not_the_new_ones(env, tmp_path):
    import io
    import zipfile

    import api
    import cleanup_zip_members as cz
    from ingestion import ingest_file as ingest

    def make_zip(path, members):
        with zipfile.ZipFile(path, "w") as z:
            for name, text in members.items():
                z.writestr(name, text)

    make_zip(tmp_path / "LS-ZIP-1__pack.zip", {"Lessons_learned.txt": "one", "dir/Notes.txt": "two"})
    assert cz.old_style_names(tmp_path) == ["Lessons_learned.txt", "Notes.txt"]

    client = TestClient(api.app)
    for name, text in (("Lessons_learned.txt", "old style"), ("LS-ZIP-1__Lessons_learned.txt", "new style")):
        client.post("/pipeline", files={"file": (name, io.BytesIO(text.encode()))})
    out = []
    cz.delete_all(cz.old_style_names(tmp_path), lambda n: client.delete(f"/delete-document/{n}").json(), out.append)
    remaining = [r["file_name"] for r in env.supabase.tables["documents"]]
    assert remaining == ["LS-ZIP-1__Lessons_learned.txt"]
