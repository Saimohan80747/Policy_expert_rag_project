# Company Docs RAG (Lumenstack)

Question answering over company documents. **Vanilla RAG** answers single-fact and policy questions from document chunks; **graph RAG** answers multi-hop questions (who reports to whom, which clients a person manages, headcounts) from a Neo4j knowledge graph via read-only Text2Cypher. A router picks `vanilla`, `graph` or `hybrid` per question, and answers cite documents `[1]` and graph facts `[G1]`.

The test corpus is a synthetic software company, generated separately (`C:\company_corpus`, not part of this repo).

## Architecture

```
question ─► analyse (Gemini flash-lite): standalone rewrite · route · entities · asks_individual_pay
              │
              ├─ restricted (individual pay) ─► fixed refusal
              ├─ graph / hybrid ─► link entities ─► Text2Cypher (validated, read-only) ─► Neo4j facts
              └─ vanilla / hybrid ─► Pinecone hybrid search (dense + sparse, current versions only)
                                     + chunks mentioning the linked entities (graph)
                                     ─► RRF fusion ─► optional Cohere rerank ─► top 8
              ─► Gemini answer with citations
```

| Part | Tech |
| --- | --- |
| API / UI | FastAPI, React + Vite |
| Vectors | Pinecone: `llama-text-embed-v2` (dense, 1024-d) + `pinecone-sparse-english-v0` |
| Files, chats | Supabase (storage bucket `documents`, tables `documents`, `chat_sessions`, `chat_messages`) |
| Graph | Neo4j Aura Free |
| LLM | Gemini (`google-genai`) |
| Rerank (optional) | Cohere trial |

### Graph layout

- **Domain:** `(:Entity:<Label> {uid, id, name, ...})` with temporal relationships `[:REL {valid_from, valid_to}]` (current = `valid_to IS NULL`). Loaded from `world/csv` by a deterministic loader, no LLM.
- **Aliases:** `(:Alias)-[:ALIAS_OF]->(:Entity)` drives entity linking and the gazetteer. Ambiguous aliases are never auto-matched.
- **Documents:** `(:SourceDoc)-[:HAS_CHUNK]->(:Chunk)-[:NEXT]->(:Chunk)`, `(:Chunk)-[:MENTIONS]->(:Entity)`, `(:SourceDoc)-[:RECORDS]->(:Document)` for ADR/RFC.
- **Versioning:** the doc id is a hash of the file name; same bytes = duplicate, new bytes under the same name = next version. Older versions are retired (`is_current=false`) in Pinecone and Neo4j.

### Safety

- Text2Cypher accepts one read-only statement; writes, `CALL`, `UNION`, the `Alias`/`GraphMeta` labels and anything touching compensation are rejected.
- Questions about an individual's pay get a fixed refusal, from documents and from the graph.
- Document text is treated as untrusted in the answer prompt.

## Setup

Requires Python 3.10+, Node 18+, and accounts for Pinecone, Supabase, Neo4j Aura and a Gemini API key (Cohere is optional).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
Copy-Item backend\.env.example backend\.env      # then fill in the keys
cd frontend; npm install
```

`backend\.env`:

```dotenv
PINECONE_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=
GEMINI_API_KEY=
NEO4J_URI=neo4j+s://<id>.databases.neo4j.io     # use neo4j+ssc:// behind an SSL-intercepting network
NEO4J_USER=neo4j
NEO4J_PASSWORD=
COHERE_API_KEY=                                  # optional: enables reranking
```

Optional overrides: `GEMINI_MODEL`, `GEMINI_ANSWER_MODEL`, `GEMINI_CYPHER_MODEL`, `COHERE_RERANK_MODEL` (default `rerank-v3.5`), `RERANK=off`, `PINECONE_INDEX`, `PINECONE_NAMESPACE`. Never commit `.env`.

## Run

```powershell
# terminal 1
.\.venv\Scripts\Activate.ps1
cd backend
python -m uvicorn api:app --reload

# terminal 2
cd frontend
npm run dev
```

API docs: `http://127.0.0.1:8000/docs`. UI: the URL Vite prints (usually `http://localhost:5173`).

### Load the graph (once, and after the corpus's CSVs change)

```powershell
cd backend
python -m knowledge_graph.loader --csv-dir C:\company_corpus\world\csv --reset --dry-run   # preview
python -m knowledge_graph.loader --csv-dir C:\company_corpus\world\csv --reset
```

`--reset` clears the domain graph and the lexical layer, so re-upload documents afterwards.

### Upload documents in bulk

With the API running:

```powershell
cd backend
python tools/upload_folder.py C:\company_corpus\corpus\wave_0_initial --sample 40   # try a sample
python tools/upload_folder.py C:\company_corpus\corpus\wave_0_initial               # everything
```

Safe to re-run: duplicates are skipped and changed files become new versions. Results are logged to `upload_log.jsonl`. `python tools/cleanup_zip_members.py` previews old-style zip member names; add `--yes` to delete them.

Free-tier notes: Pinecone Starter accepts 250K embedding tokens/minute, so uploads are batched by estimated tokens and wait out 429s. A very large spreadsheet can still be too big to embed; leave it out.

### Tests

```powershell
cd backend
python -m pytest -q
```

Tests use fakes for Supabase, Pinecone, Gemini and Neo4j; no keys needed.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/pipeline` | Upload and ingest a file (or zip) |
| `POST` | `/prompt` | Ask in a chat session; returns answer, `route_used`, `sources` (including the graph item with its Cypher) |
| `GET` | `/search` | Raw hybrid document search |
| `GET` | `/graph/stats` | Graph counts, labels, relationship types, document layer |
| `GET` / `DELETE` | `/history`, `/sessions/{id}` | List, read, delete chats |
| `GET` | `/download/{file}`, `/preview/{file}` | Source files |
| `DELETE` | `/delete-document/{file}` | Remove a document from Supabase, Pinecone and Neo4j |

## Status

Built: ingestion with versioning, hybrid vector search, graph loader, alias gazetteer, lexical layer, router, Text2Cypher, entity-anchored retrieval with RRF fusion and optional rerank, UI citations.
Not yet: ingestion waves 1-6 (html/markdown/pptx/eml/vtt/json/yaml readers, access control by confidentiality), LLM relation extraction for prose-only relationships, and marking superseded policy versions from the corpus sidecar metadata.

## Troubleshooting

- **Browser shows CORS errors:** usually a backend 500; read the uvicorn log.
- **`CERTIFICATE_VERIFY_FAILED` to Neo4j:** switch the URI scheme to `neo4j+ssc://`.
- **Graph route falls back to documents:** check the log line starting `Graph route found nothing`; `GET /graph/stats` shows whether the graph is loaded.
- **Pinecone 429 during upload:** wait; the client retries. Re-running the upload skips what's done.
