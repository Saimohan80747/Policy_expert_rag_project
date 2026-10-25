"""Score predictions with the corpus's score_one() and write tables (summary.md / summary.json)."""
import json
from collections import defaultdict
from pathlib import Path


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def score_mode(scorer, questions, preds):
    by_qid = {p["qid"]: p for p in preds}
    rows = []
    for q in questions:
        p = by_qid.get(q["qid"])
        if p is None:
            continue
        r = scorer.score_one(q, p)
        r.update(hops=q["hops"], seconds=p.get("seconds"), llm_calls=p.get("llm_calls"), route_used=p.get("route_used"),
                 abstained=bool(p.get("abstained")), graph_attempted=p.get("graph_attempted"), graph_rows=p.get("graph_rows", 0),
                 graph_error=p.get("graph_error"), error=p.get("error"))
        rows.append(r)
    return rows


def _acc(rows):
    graded = [r for r in rows if r.get("correct") is not None]
    return (sum(1 for r in graded if r["correct"]) / len(graded)) if graded else None


def summarize(rows):
    graded = [r for r in rows if r.get("correct") is not None]
    answerable = [r for r in graded if r["behavior"] == "answer"]
    abstain_q = [r for r in graded if r["behavior"] == "abstain"]
    wrong = [r for r in answerable if not r["correct"] and not r["abstained"]]
    ev = [r for r in rows if "evidence_hit" in r]
    graph = [r for r in rows if r.get("graph_attempted")]
    return {
        "n": len(rows), "graded": len(graded), "accuracy": _acc(rows),
        "answerable_accuracy": _acc(answerable),
        "wrong_but_confident": len(wrong) / len(answerable) if answerable else None,
        "abstain_correct": _acc(abstain_q),
        "evidence_hit_rate": sum(r["evidence_hit"] for r in ev) / len(ev) if ev else None,
        "evidence_recall": sum(r["evidence_recall"] for r in ev) / len(ev) if ev else None,
        "graph_attempted": len(graph),
        "graph_non_empty": sum(1 for r in graph if r["graph_rows"]),
        "seconds_p50": percentile([r["seconds"] for r in rows if r.get("seconds") is not None], 50),
        "seconds_p95": percentile([r["seconds"] for r in rows if r.get("seconds") is not None], 95),
        "llm_calls_mean": (sum(r["llm_calls"] or 0 for r in rows) / len(rows)) if rows else None,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def breakdown(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return {k: {"n": len(v), "accuracy": _acc(v)} for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def num(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def render(results, order):
    modes = [m for m in order if m in results]
    lines = ["# Evaluation summary", ""]
    lines += ["## Overall", "", "| Mode | n | Accuracy | Answerable acc. | Wrong but confident | Abstain correct | Evidence hit | "
              "Graph non-empty | p50 s | p95 s | LLM calls/q | Errors |", "|" + " --- |" * 12]
    for m in modes:
        s = results[m]["overall"]
        gn = f"{s['graph_non_empty']}/{s['graph_attempted']}" if s["graph_attempted"] else "-"
        lines.append(f"| {m} | {s['n']} | {pct(s['accuracy'])} | {pct(s['answerable_accuracy'])} | {pct(s['wrong_but_confident'])} | "
                     f"{pct(s['abstain_correct'])} | {pct(s['evidence_hit_rate'])} | {gn} | {num(s['seconds_p50'])} | "
                     f"{num(s['seconds_p95'])} | {num(s['llm_calls_mean'])} | {s['errors']} |")
    cats = sorted({c for m in modes for c in results[m]["by_category"]})
    lines += ["", "## Accuracy by category", "", "| Category | n | " + " | ".join(modes) + " |", "|" + " --- |" * (len(modes) + 2)]
    for c in cats:
        n = max(results[m]["by_category"].get(c, {}).get("n", 0) for m in modes)
        lines.append(f"| {c} | {n} | " + " | ".join(pct(results[m]["by_category"].get(c, {}).get("accuracy")) for m in modes) + " |")
    hops = sorted({h for m in modes for h in results[m]["by_hops"]})
    lines += ["", "## Accuracy by reasoning depth (hops)", "", "| Hops | n | " + " | ".join(modes) + " |", "|" + " --- |" * (len(modes) + 2)]
    for h in hops:
        n = max(results[m]["by_hops"].get(h, {}).get("n", 0) for m in modes)
        lines.append(f"| {h} | {n} | " + " | ".join(pct(results[m]["by_hops"].get(h, {}).get("accuracy")) for m in modes) + " |")
    if "router" in results:
        used = defaultdict(int)
        for r in results["router"]["rows"]:
            used[r.get("route_used")] += 1
        lines += ["", "## Router decisions", "", ", ".join(f"{k}: {v}" for k, v in sorted(used.items(), key=lambda kv: str(kv[0])))]
    return "\n".join(lines) + "\n"


def write_report(out_dir, scorer, questions, modes, waves):
    from evaluation.modes import ABLATIONS, MODES
    out_dir = Path(out_dir)
    results = {}
    for m in modes:
        path = out_dir / f"{m}.jsonl"
        if not path.exists():
            continue
        preds = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = score_mode(scorer, questions, preds)
        if rows:
            results[m] = {"overall": summarize(rows), "by_category": breakdown(rows, lambda r: r["category"]),
                          "by_hops": breakdown(rows, lambda r: r["hops"]), "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "summary.md").write_text(render(results, list(MODES) + list(ABLATIONS)), encoding="utf-8")
    return results
