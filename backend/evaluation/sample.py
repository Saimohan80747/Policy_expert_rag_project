"""Pick a stratified, reproducible sample of the corpus's questions that this system can be scored on."""
import hashlib
import re

# Question types a string match can score. Rubric/clarify/refuse answers need a judge or features we scoped out.
SCORABLE_ANSWERS = {"entity", "number", "date", "text", "list", "ordered_list", "range", "abstain"}
SCORABLE_BEHAVIORS = {"answer", "abstain"}

# (category, hops or None) -> how many questions to draw. About 200 in total.
DEFAULT_QUOTAS = {
    ("relation_lookup", None): 36, ("single_fact", None): 30, ("list", None): 12, ("aggregation", None): 12,
    ("multi_hop", 3): 15, ("multi_hop", 4): 25, ("multi_hop", 5): 10,
    ("temporal", None): 30, ("disambiguation", None): 14, ("unanswerable", None): 16, ("needle", None): 6,
}
# Uids used as examples in the Text2Cypher prompt: keep questions about them out of the evaluation.
FEW_SHOT_IDS = ("LS00121", "LS00433", "ENG-PLAT")


def _rank(seed, qid):
    return hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()


def split_of(qid, seed=42, dev_share=0.3):
    return "dev" if int(_rank(seed + 1, qid)[:6], 16) / 0xFFFFFF < dev_share else "test"


def eligible(q, applies, waves, formats, exclude_docs=()):
    if not applies(q, waves, formats):
        return False
    if q["expected_behavior"] not in SCORABLE_BEHAVIORS or q["answer"]["type"] not in SCORABLE_ANSWERS:
        return False
    if q.get("language", "en") != "en" or q.get("access_role", "all") != "all":
        return False
    if any(e["doc_id"] in set(exclude_docs) for e in q["evidence"]):
        return False
    text = repr(q.get("gold_path", ""))
    return not any(i in text for i in FEW_SHOT_IDS)


def select(questions, applies, waves=frozenset({0}), formats=frozenset({"pdf", "docx", "txt", "csv", "xlsx", "zip"}),
           quotas=None, seed=42, exclude_docs=()):
    quotas = quotas or DEFAULT_QUOTAS
    pools = {}
    for q in questions:
        if not eligible(q, applies, waves, formats, exclude_docs):
            continue
        for (cat, hops), _ in quotas.items():
            if q["category"] == cat and (hops is None or q["hops"] == hops):
                pools.setdefault((cat, hops), []).append(q)
    chosen = []
    for key, n in quotas.items():
        pool = sorted(pools.get(key, []), key=lambda q: _rank(seed, q["qid"]))
        chosen += pool[:n]
    return [dict(q, split=split_of(q["qid"], seed)) for q in sorted(chosen, key=lambda q: q["qid"])]
