"""Upload a folder of documents through the running API (POST /pipeline).

Safe to re-run: files already ingested come back as 'duplicate' and are skipped, a changed file with
the same name becomes the next version. Waves of new files can be dropped into the same folder.

    python tools/upload_folder.py C:\\company_corpus\\corpus\\wave_0_initial --sample 40
    python tools/upload_folder.py C:\\company_corpus\\corpus\\wave_0_initial            # everything
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `supabase_` imports work from tools/

SUPPORTED = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".zip"}


def find_files(folder, exts=None, sample=None, seed=42, limit=None):
    files = sorted(p for p in Path(folder).rglob("*")
                   if p.is_file() and p.suffix.lower() in (exts or SUPPORTED) and "_meta" not in p.parts)
    if sample:
        random.Random(seed).shuffle(files)
        files = sorted(files[:sample])
    return files[:limit] if limit else files


def upload_all(files, post, log=None, pause=0.0, out=print):
    """post(path) -> result dict. Returns a Counter of statuses."""
    totals, started = Counter(), time.time()
    for i, path in enumerate(files, 1):
        try:
            result = post(path)
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        status = result.get("status") or ("zip" if "ingested_files" in result else "unknown")
        totals[status] += 1
        for key in ("ingested_files", "duplicate_files", "failed_files"):   # ZIP summaries
            for item in result.get(key, []) or []:
                totals[f"zip:{key.split('_')[0]}"] += 1
                if key == "failed_files":
                    out(f"      zip member failed: {item}")
        detail = result.get("error") or (f"{result.get('chunks', '')} chunks" if status == "ingested" else "")
        out(f"[{i}/{len(files)}] {status:9} {path.name[:70]} {detail}")
        if log:
            log.write(json.dumps({"file": str(path), **{k: v for k, v in result.items() if k != "url"}}) + "\n")
            log.flush()
        if pause:
            time.sleep(pause)
    out(f"Done in {time.time() - started:.0f}s: " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    return totals


def delete_existing(client, api, name):
    """Remove a document (Supabase, Pinecone, graph) so the same file can be ingested again."""
    from urllib.parse import quote

    from supabase_.file_upload import safe_filename

    r = client.delete(f"{api}/delete-document/{quote(safe_filename(name), safe='')}")
    if r.status_code not in (200, 404):
        r.raise_for_status()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--sample", type=int, help="upload a random (seeded) sample of N files")
    ap.add_argument("--limit", type=int, help="stop after N files")
    ap.add_argument("--ext", nargs="*", help="only these extensions, e.g. .txt .docx")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between files")
    ap.add_argument("--replace", action="store_true",
                    help="delete each document first so it is re-ingested even if its bytes are unchanged")
    ap.add_argument("--log", default="upload_log.jsonl")
    args = ap.parse_args(argv)

    import httpx

    exts = {e if e.startswith(".") else f".{e}" for e in args.ext} if args.ext else None
    files = find_files(args.folder, exts, args.sample, limit=args.limit)
    if not files:
        print("No supported files found.")
        return 1
    print(f"Uploading {len(files)} files to {args.api} ...")

    with httpx.Client(timeout=600) as client, open(args.log, "a", encoding="utf-8") as log:
        def post(path):
            if args.replace:
                delete_existing(client, args.api, path.name)
            r = client.post(f"{args.api}/pipeline", files={"file": (path.name, path.read_bytes())})
            r.raise_for_status()
            return r.json()

        totals = upload_all(files, post, log, args.pause)
    return 1 if totals.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
