"""Run RAG configurations over a sample of the corpus's question set and score them.

    cd backend
    python -m evaluation.run_eval --split dev                 # about 60 questions, all four modes
    python -m evaluation.run_eval --split test --modes router
    python -m evaluation.run_eval --split dev --modes no_neighbours no_anchored --name ablations

Everything runs on Gemini Flash Lite (free tier: 500 requests/day) and is resumable: LLM calls are
cached in <out>/llm_cache.jsonl, so when the daily limit is reached, run the same command again
tomorrow. The backend server does not need to be running; the real Pinecone, Neo4j and Supabase are used.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_CORPUS_EVAL = r"C:\company_corpus\eval"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
CORPUS_ID = re.compile(r"^(LS-[A-Z]+-\d{4}-\d{4})__")
ABSTAIN = re.compile(
    r"(could(?: not|n't) find|can(?:not|'t) find|unable to find|no (?:relevant )?(?:information|record|data)|"
    r"not (?:found|available|mentioned|stated|specified|recorded)|does not (?:contain|mention|specify|exist))", re.I)


def load_scorer(path):
    spec = importlib.util.spec_from_file_location("corpus_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_ids(sources):
    seen, out = set(), []
    for s in sources:
        match = CORPUS_ID.match(s.get("file_name") or "")
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            out.append(match.group(1))
    return out


def to_prediction(qid, result, elapsed, error=None):
    """llm_calls = analysis (shared by every mode, so charged to each) + Cypher generations + the answer call."""
    answer = result.get("answer", "") if result else ""
    refused = bool(result) and result.get("route_used") == "restricted"
    no_results = bool(result) and answer.startswith("I couldn't find anything relevant")
    abstained = bool(error) or no_results or bool(ABSTAIN.search(answer[:300])) and not refused
    answered = bool(result) and not refused and not no_results
    llm_calls = 1 + (result or {}).get("graph_llm_calls", 0) + (1 if answered else 0)
    graph = next((s for s in (result or {}).get("sources", []) if s.get("type") == "graph"), None)
    debug = (result or {}).get("graph_debug")
    return {
        "qid": qid, "answer": answer, "abstained": abstained, "refused": refused, "clarified": False,
        "sources": corpus_ids([s for s in (result or {}).get("sources", []) if s.get("type") != "graph"]),
        "route": (result or {}).get("route"), "route_used": (result or {}).get("route_used"),
        "graph_rows": graph["rows"] if graph else 0,
        "graph_attempted": bool(graph or debug),
        "graph_error": (debug or {}).get("error"),
        "graph_cypher": (graph or {}).get("cypher") or (debug or {}).get("cypher"),
        "graph_linked": [e["name"] for e in (graph or {}).get("entities", [])] or (debug or {}).get("linked"),
        "graph_facts": (graph or {}).get("facts", [])[:3],
        "seconds": round(elapsed, 2), "llm_calls": llm_calls, "error": error,
    }


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(questions, modes, out_dir, cached_llm, rag, log=print):
    """Run each mode over each question; skips (qid, mode) pairs already in <out>/<mode>.jsonl."""
    from evaluation.llm_cache import QuotaReached
    from evaluation.modes import options_for

    out_dir = Path(out_dir)
    done = {m: {p["qid"] for p in read_jsonl(out_dir / f"{m}.jsonl")} for m in modes}
    for n, q in enumerate(questions, start=1):
        todo = [m for m in modes if q["qid"] not in done[m]]
        if not todo:
            continue
        graph_cache = {}
        analysis = None
        analysis_seconds = 0.0
        for mode in todo:
            started = time.time()
            before = (cached_llm.sleep_total, cached_llm.real_seconds, cached_llm.charged)
            error, result = None, None
            try:
                if analysis is None:
                    analysis = rag.analyze_question(q["question"], None)
                    analysis_seconds = cached_llm.charged - before[2]
                result = rag.answer_question(q["question"], None, options_for(mode, analysis=analysis, graph_cache=graph_cache))
            except QuotaReached:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
            # Latency as if every call had run live: wall time without rate-limit waits and without the
            # real call durations, plus the recorded duration of every call used (cached or not). The shared
            # analysis is charged to every mode.
            wall = time.time() - started
            estimate = (wall - (cached_llm.sleep_total - before[0]) - (cached_llm.real_seconds - before[1])
                        + (cached_llm.charged - before[2]) + (analysis_seconds if mode != todo[0] else 0.0))
            pred = to_prediction(q["qid"], result, max(estimate, 0.0), error)
            with (out_dir / f"{mode}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
            done[mode].add(q["qid"])
        log(f"[{n}/{len(questions)}] {q['qid']} {q['category']} done ({cached_llm.new_calls} new LLM calls so far)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-eval", default=DEFAULT_CORPUS_EVAL, help="folder with questions.jsonl and score.py")
    ap.add_argument("--waves", default="0")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--modes", nargs="*", default=["vanilla", "graph", "hybrid", "router"])
    ap.add_argument("--name", default="main", help="run folder name under evaluation/runs/")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rpm", type=int, default=10, help="uncached LLM requests per minute")
    ap.add_argument("--max-new-calls", type=int, default=450, help="stop after this many uncached LLM calls")
    ap.add_argument("--exclude-docs", nargs="*", default=[], help="corpus doc ids that were not ingested")
    ap.add_argument("--limit", type=int, help="only the first N questions of the split (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    # The app reads its model names from the environment when it is imported, so set them first.
    for var in ("GEMINI_MODEL", "GEMINI_ANSWER_MODEL", "GEMINI_CYPHER_MODEL"):
        os.environ[var] = args.model
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import llm
    import rag
    from evaluation.llm_cache import CachedLLM, QuotaReached
    from evaluation.report import write_report
    from evaluation.sample import select

    corpus_eval = Path(args.corpus_eval)
    scorer = load_scorer(corpus_eval / "score.py")
    questions = [json.loads(l) for l in (corpus_eval / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    waves = {int(w) for w in args.waves.split(",")}

    out_dir = Path(__file__).resolve().parent / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_file = out_dir / "sample.json"
    if sample_file.exists():
        sample = json.loads(sample_file.read_text(encoding="utf-8"))
    else:
        sample = select(questions, scorer.applies, waves, scorer.BASE_FORMATS, seed=args.seed, exclude_docs=args.exclude_docs)
        sample_file.write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
    chosen = [q for q in sample if args.split == "all" or q["split"] == args.split]
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"{len(chosen)} questions ({args.split}) from a sample of {len(sample)}; modes: {', '.join(args.modes)}; model {args.model}")

    cached = CachedLLM(out_dir / "llm_cache.jsonl", rpm=args.rpm, max_new_calls=args.max_new_calls)
    restore = cached.install(llm)
    try:
        evaluate(chosen, args.modes, out_dir, cached, rag)
    except QuotaReached as exc:
        print(f"\nStopped: {exc}\nRun the same command again later; finished questions and LLM calls are cached.")
    finally:
        restore()
    write_report(out_dir, scorer, chosen, args.modes, waves)
    print(f"Report: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
