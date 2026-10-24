"""Flag documents that a newer document replaces, so search prefers the current one.

The corpus's sidecar files (corpus/_meta/<id>.json) say which document supersedes which, e.g. the
Leave Policy v5 file supersedes v4. Those are different files, so the upload's own versioning does not
know they are related. This marks the replaced documents `superseded=true` in Pinecone and Neo4j;
questions about the past ("in 2018", "what changed") still see them.

    python tools/mark_superseded.py C:\\company_corpus\\corpus              # preview
    python tools/mark_superseded.py C:\\company_corpus\\corpus --yes        # apply
    python tools/mark_superseded.py C:\\company_corpus\\corpus --yes --clear  # undo

Safe to re-run, and re-run it after uploading new waves.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORPUS_ID = re.compile(r"^(LS-[A-Z]+-\d{4}-\d{4})__")
PAGE = 1000


def find_meta_dir(folder) -> Path:
    folder = Path(folder)
    for candidate in (folder / "_meta", folder.parent / "_meta"):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"No _meta folder in {folder} or its parent")


def superseded_map(meta_dir) -> dict[str, set[str]]:
    """{old corpus id: ids of the documents that supersede it}, from the `supersedes` fields."""
    out = {}
    for path in Path(meta_dir).glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        value = data.get("supersedes")
        for old in ([value] if isinstance(value, str) else value or []):
            if old:
                out.setdefault(old, set()).add(data.get("doc_id") or path.stem)
    return out


def superseded_ids(meta_dir) -> set[str]:
    """Corpus ids that another document names in its `supersedes` field."""
    return set(superseded_map(meta_dir))


def corpus_id(file_name):
    match = CORPUS_ID.match(file_name or "")
    return match.group(1) if match else None


def current_documents(supabase) -> list[dict]:
    rows, start = [], 0
    while True:
        page = (supabase.table("documents").select("doc_id,file_name,version").eq("is_current", True)
                .range(start, start + PAGE - 1).execute().data or [])
        rows += page
        if len(page) < PAGE:
            return rows
        start += PAGE


def select_documents(rows, ids) -> list[dict]:
    return [r for r in rows if corpus_id(r["file_name"]) in ids]


def replaceable_ids(rows, mapping) -> set[str]:
    """Old ids whose replacement is actually uploaded.

    A document replaced by one we do not have (a later wave not ingested yet) must stay visible, or
    nothing current would be left to answer from.
    """
    uploaded = {corpus_id(r["file_name"]) for r in rows}
    return {old for old, news in mapping.items() if news & uploaded}


def apply_flag(docs, flag, set_vectors, set_graph, log=print):
    for doc in docs:
        set_vectors(doc["doc_id"], flag)
    doc_ids = [d["doc_id"] for d in docs]
    try:
        set_graph(doc_ids, flag)
    except Exception as exc:  # noqa: BLE001 - Pinecone is what search uses; the graph is a bonus
        log(f"Graph not updated ({type(exc).__name__}: {exc})")
    return len(docs)


def _set_graph(doc_ids, flag):
    from knowledge_graph import db
    for i in range(0, len(doc_ids), 200):
        db.run(
            "MATCH (d:SourceDoc) WHERE d.doc_id IN $ids SET d.superseded = $flag "
            "WITH d OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c) SET c.superseded = $flag",
            ids=doc_ids[i:i + 200], flag=flag,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="the corpus folder (or a wave folder) next to _meta")
    parser.add_argument("--yes", action="store_true", help="apply (default is a preview)")
    parser.add_argument("--clear", action="store_true", help="remove the flag instead of setting it")
    args = parser.parse_args(argv)

    from embeddings import set_superseded
    from supabase_.supabase_db import supabase_var

    mapping = superseded_map(find_meta_dir(args.folder))
    rows = current_documents(supabase_var)
    ids = set(mapping) if args.clear else replaceable_ids(rows, mapping)   # --clear undoes any earlier flagging
    docs = select_documents(rows, ids)
    print(f"{len(mapping)} documents are superseded in the corpus metadata; {len(docs)} of them are uploaded "
          f"and have their replacement uploaded too." if not args.clear else f"Clearing {len(docs)} uploaded documents.")
    for doc in docs[:15]:
        print("  ", doc["file_name"])
    if len(docs) > 15:
        print(f"   ... and {len(docs) - 15} more")
    if not args.yes:
        print("Preview only. Add --yes to apply.")
        return 0
    n = apply_flag(docs, not args.clear, set_superseded, _set_graph)
    print(f"{'Cleared' if args.clear else 'Flagged'} {n} documents. Pinecone applies metadata updates asynchronously; "
          "give it a minute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
