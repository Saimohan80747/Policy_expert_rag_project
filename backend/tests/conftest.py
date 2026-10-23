"""Fakes for Supabase and Pinecone so ingestion can be tested offline."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ----------------------------- fake Supabase -----------------------------
class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    reject_sources_column = False

    def __init__(self, rows):
        self.rows, self.filters, self.mode = rows, [], "select"
        self.payload, self._order, self._limit = None, None, None

    def select(self, *_):
        self.mode = "select"
        return self

    def insert(self, row):
        self.mode, self.payload = "insert", row
        return self

    def update(self, values):
        self.mode, self.payload = "update", values
        return self

    def delete(self):
        self.mode = "delete"
        return self

    def eq(self, col, val):
        self.filters.append(lambda r: r.get(col) == val)
        return self

    def in_(self, col, values):
        self.filters.append(lambda r: r.get(col) in values)
        return self

    def lt(self, col, val):
        self.filters.append(lambda r: r.get(col) is not None and r[col] < val)
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        match = [r for r in self.rows if all(f(r) for f in self.filters)]
        if self.mode == "insert":
            if self.reject_sources_column and "sources" in self.payload:
                raise RuntimeError("PGRST204: Could not find the 'sources' column")
            row = dict(self.payload)
            row.setdefault("id", len(self.rows) + 1)
            row.setdefault("created_at", len(self.rows) + 1)
            self.rows.append(row)
            return _Result([row])
        if self.mode == "update":
            for r in match:
                r.update(self.payload)
            return _Result(match)
        if self.mode == "delete":
            for r in match:
                self.rows.remove(r)
            return _Result(match)
        if self._order:
            col, desc = self._order
            match.sort(key=lambda r: r.get(col, 0), reverse=desc)
        return _Result(match[: self._limit] if self._limit else match)


class _Bucket:
    def __init__(self, files, fail_upload):
        self.files, self.fail_upload = files, fail_upload

    def upload(self, path, file, file_options=None):
        if self.fail_upload:
            raise RuntimeError("storage down")
        self.files[path] = file

    def get_public_url(self, path):
        return f"https://fake.storage/documents/{path}"

    def download(self, path):
        if path not in self.files:
            raise RuntimeError("Object not found")
        return self.files[path]

    def remove(self, paths):
        for p in paths:
            self.files.pop(p, None)
        return [{"name": p} for p in paths]


class FakeSupabase:
    def __init__(self):
        self.tables = {}
        self.files = {}
        self.fail_upload = False
        self.reject_sources_column = False
        self.storage = SimpleNamespace(from_=lambda _bucket: _Bucket(self.files, self.fail_upload))

    def table(self, name):
        query = _Query(self.tables.setdefault(name, []))
        query.reject_sources_column = self.reject_sources_column
        return query


# ----------------------------- fake Pinecone -----------------------------
def _matches(meta, flt):
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(_matches(meta, c) for c in cond):
                return False
        elif "$eq" in cond and meta.get(key) != cond["$eq"]:
            return False
        elif "$lt" in cond and not (meta.get(key) is not None and meta[key] < cond["$lt"]):
            return False
    return True


class FakeIndex:
    def __init__(self):
        self.ns = {}
        self.upsert_calls = 0
        self.fail_on_upsert_call = None
        self.fail_list = False
        self.legacy_list = False
        self.filter_update_supported = True

    def upsert(self, vectors, namespace):
        self.upsert_calls += 1
        if self.fail_on_upsert_call and self.upsert_calls >= self.fail_on_upsert_call:
            raise RuntimeError("pinecone down")
        store = self.ns.setdefault(namespace, {})
        for v in vectors:
            store[v["id"]] = {
                "values": v["values"],
                "sparse": v["sparse_values"],
                "metadata": dict(v["metadata"]),
            }

    def list(self, prefix, namespace):
        if self.fail_list:
            raise RuntimeError("reads not supported")
        ids = sorted(i for i in self.ns.get(namespace, {}) if i.startswith(prefix))
        for i in range(0, len(ids), 2):  # small pages exercise pagination
            page = ids[i : i + 2]
            if self.legacy_list:  # old SDKs: plain lists of id strings
                yield page
            else:  # current SDK: ListResponse pages with .vectors items
                yield SimpleNamespace(vectors=[SimpleNamespace(id=v) for v in page])

    def update(self, namespace, set_metadata, id=None, filter=None):
        if filter is not None:
            if not self.filter_update_supported:
                raise TypeError("filter updates not supported")
            for rec in self.ns.get(namespace, {}).values():
                if _matches(rec["metadata"], filter):
                    rec["metadata"].update(set_metadata)
        else:
            self.ns[namespace][id]["metadata"].update(set_metadata)

    def delete(self, ids, namespace):
        for i in ids:
            self.ns.get(namespace, {}).pop(i, None)

    def query(self, namespace, top_k, vector, sparse_vector, include_metadata, filter=None):
        def ok(meta):
            return _matches(meta, filter)

        scored = []
        for vid, rec in self.ns.get(namespace, {}).items():
            if not ok(rec["metadata"]):
                continue
            score = sum(a * b for a, b in zip(vector, rec["values"]))
            scored.append(SimpleNamespace(id=vid, score=score, metadata=rec["metadata"]))
        scored.sort(key=lambda m: m.score, reverse=True)
        return SimpleNamespace(matches=scored[:top_k])

    def all_ids(self):
        return sorted(i for store in self.ns.values() for i in store)


def _fake_dense(texts, input_type="passage"):
    return [[float(len(t) % 5), float(sum(map(ord, t)) % 7), 1.0] for t in texts]


def _fake_sparse(texts, input_type="passage"):
    return [{"indices": [0, 1], "values": [1.0, 2.0]} for _ in texts]


@pytest.fixture
def supabase(monkeypatch):
    from supabase_ import supabase_db

    fake = FakeSupabase()
    monkeypatch.setattr(supabase_db, "_client", fake)
    return fake


@pytest.fixture
def index(monkeypatch):
    import embeddings
    import retrieve_docs

    fake = FakeIndex()
    monkeypatch.setattr(embeddings, "get_index", lambda: fake)
    monkeypatch.setattr(embeddings, "embed_dense", _fake_dense)
    monkeypatch.setattr(embeddings, "embed_sparse", _fake_sparse)
    monkeypatch.setattr(retrieve_docs, "get_index", lambda: fake)
    monkeypatch.setattr(retrieve_docs, "embed_dense", _fake_dense)
    monkeypatch.setattr(retrieve_docs, "embed_sparse", _fake_sparse)
    return fake


@pytest.fixture
def env(supabase, index):
    return SimpleNamespace(supabase=supabase, index=index)


# ------------------------------- fake LLM --------------------------------
class FakeLLM:
    def __init__(self):
        self.calls = []  # (kind, model, prompt, system)
        self.text = "Casual leave is 12 days per year [1]."
        self.analysis = {"standalone_question": "How many casual leave days per year?", "route": "vanilla", "entities": []}
        self.fail_json = False
        self.fail_text = False

    @property
    def kinds(self):
        return [c[0] for c in self.calls]

    def generate_text(self, prompt, *, system=None, model=None, temperature=0.2):
        self.calls.append(("text", model, prompt, system))
        if self.fail_text:
            raise RuntimeError("llm down")
        return "Leave policy chat" if system is None else self.text

    def generate_json(self, prompt, schema, *, system=None, model=None):
        self.calls.append(("json", model, prompt, system))
        if self.fail_json:
            raise RuntimeError("llm down")
        return schema(**self.analysis)


@pytest.fixture
def fake_llm(monkeypatch):
    import llm

    fake = FakeLLM()
    monkeypatch.setattr(llm, "generate_text", fake.generate_text)
    monkeypatch.setattr(llm, "generate_json", fake.generate_json)
    return fake
