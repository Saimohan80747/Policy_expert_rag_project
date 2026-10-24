"""List the questions a configuration got wrong, with the gold answer, what it said and why it may have failed.

    python -m evaluation.failures --mode router --split dev
    python -m evaluation.failures --mode graph --category multi_hop --limit 10
Writes the same text to evaluation/runs/<name>/failures_<mode>.md so it can be pasted or attached.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def gold_text(answer):
    value = answer.get("value")
    if isinstance(value, list):
        value = ", ".join(map(str, value))
    return f"{value}" + (f" (also accepted: {', '.join(map(str, answer['aliases'][:3]))})" if answer.get("aliases") else "")


def describe(q, pred, row):
    sources = pred.get("sources", [])
    evidence = [e["doc_id"] for e in q["evidence"]]
    found = [d for d in evidence if d in sources]
    lines = [
        f"### {q['qid']} · {q['category']}/{q.get('subtype', '')} · {q['hops']} hop(s) · split {q.get('split', '?')}",
        f"Q: {q['question']}",
        f"Gold: {gold_text(q['answer'])}",
        f"Said: {(pred.get('answer') or '').strip()[:300]!r}",
        f"Route: chose {pred.get('route')}, used {pred.get('route_used')}; abstained={pred.get('abstained')}",
        f"Evidence docs retrieved: {len(found)}/{len(evidence)}" + (f" (missing: {', '.join(d for d in evidence if d not in sources)[:160]})" if len(found) < len(evidence) else ""),
    ]
    if pred.get("graph_attempted"):
        lines.append(f"Graph: rows={pred.get('graph_rows')} error={pred.get('graph_error')} linked={pred.get('graph_linked')}")
        if pred.get("graph_cypher"):
            lines.append("Cypher: " + " ".join(str(pred["graph_cypher"]).split())[:600])
        if pred.get("graph_facts"):
            lines.append(f"Facts: {pred['graph_facts']}")
    if pred.get("error"):
        lines.append(f"Error: {pred['error']}")
    if row.get("reason"):
        lines.append(f"Scorer: {row['reason']}")
    return "\n".join(lines)


def main(argv=None):
    from evaluation.run_eval import DEFAULT_CORPUS_EVAL, load_scorer, read_jsonl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="main")
    ap.add_argument("--mode", default="router")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--category")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--corpus-eval", default=DEFAULT_CORPUS_EVAL)
    args = ap.parse_args(argv)

    run_dir = Path(__file__).resolve().parent / "runs" / args.name
    scorer = load_scorer(Path(args.corpus_eval) / "score.py")
    sample = {q["qid"]: q for q in json.loads((run_dir / "sample.json").read_text(encoding="utf-8"))}
    preds = {p["qid"]: p for p in read_jsonl(run_dir / f"{args.mode}.jsonl")}
    out = []
    for qid, pred in preds.items():
        q = sample.get(qid)
        if not q or (args.split != "all" and q["split"] != args.split) or (args.category and q["category"] != args.category):
            continue
        row = scorer.score_one(q, pred)
        if row.get("correct") is False:
            out.append(describe(q, pred, row))
    header = f"{len(out)} wrong answers for mode '{args.mode}' ({args.split}{', ' + args.category if args.category else ''}); showing {min(len(out), args.limit)}\n"
    text = header + "\n\n".join(out[: args.limit])
    (run_dir / f"failures_{args.mode}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
