"""One-off cleanup for ZIP members that were stored under their bare file names.

Before the fix, 'Lessons_learned.txt' from every close-out pack shared one name, so each upload
replaced the previous one as a "new version". This deletes those bare-named documents (all
versions, from Pinecone, Supabase storage and the graph) so the ZIPs can be uploaded again with
names qualified by their archive.

    python tools/cleanup_zip_members.py C:\\company_corpus\\corpus\\wave_0_initial            # preview
    python tools/cleanup_zip_members.py C:\\company_corpus\\corpus\\wave_0_initial --yes      # delete
    python tools/upload_folder.py       C:\\company_corpus\\corpus\\wave_0_initial --ext .zip
"""
import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supabase_.file_upload import safe_filename  # noqa: E402
from zip_file_breaker import _is_junk  # noqa: E402


def _members(data: bytes, depth=0):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if _is_junk(info.filename):
                continue
            base = os.path.basename(info.filename)
            if base.lower().endswith(".zip"):
                if depth < 3:
                    yield from _members(z.read(info), depth + 1)
            else:
                yield base


def old_style_names(folder) -> list[str]:
    names = set()
    for z in sorted(Path(folder).rglob("*.zip")):
        if "_meta" in z.parts:
            continue
        names.update(safe_filename(m) for m in _members(z.read_bytes()))
    return sorted(names)


def delete_all(names, delete, out=print):
    results = {}
    for name in names:
        results[name] = delete(name)
        out(f"{results[name].get('status', '?'):10} {name}  versions={results[name].get('versions_deleted', 0)}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--yes", action="store_true", help="actually delete (default is a preview)")
    args = ap.parse_args(argv)

    names = old_style_names(args.folder)
    print(f"{len(names)} bare member names found in the ZIPs:")
    for n in names:
        print("  ", n)
    if not args.yes:
        print("Preview only. Add --yes to delete them.")
        return 0

    import httpx

    with httpx.Client(timeout=300) as client:
        def delete(name):
            r = client.delete(f"{args.api}/delete-document/{quote(name)}")
            r.raise_for_status()
            return r.json()

        delete_all(names, delete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
