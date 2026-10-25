"""Tier-2 entity spotting: match chunk text against the alias table. No LLM.

Only unambiguous aliases become MENTIONS. Ambiguous surfaces (shared initials, 'the CTO', a bare
first name) are returned as `unresolved` so the later LLM step can decide them.
"""
import re
import time

from .loader import normalize

ID_LIKE = {"emp_id", "id", "code", "legacy_id", "email", "slack_channel"}
SKIP_TYPES = {"initials", "nickname", "role", "first_name_only"}
GENERIC_WORDS = {
    "platform", "backend", "web", "mobile", "finance", "legal", "sales", "security", "delivery", "support",
    "product", "people", "design", "data", "cloud", "core", "operations", "engineering", "marketing",
    "research", "quality", "growth", "board", "audit", "review", "program", "project",
}
MAX_NGRAM = 8
MAX_UNRESOLVED = 20

ALIAS_QUERY = (
    "MATCH (a:Alias)-[:ALIAS_OF]->(e:Entity) "
    "RETURN a.norm AS norm, a.alias AS alias, a.alias_type AS alias_type, a.ambiguous AS ambiguous, "
    "e.uid AS uid, [l IN labels(e) WHERE l <> 'Entity'][0] AS label"
)


class Gazetteer:
    def __init__(self, rows):
        self.entries = {}       # norm -> {uid, alias, alias_type, case_sensitive}
        self.ambiguous = set()  # norms we saw but cannot resolve
        self.max_n = 1
        for r in rows:
            self._add(r)

    def _add(self, r):
        norm, alias, kind = r["norm"], r["alias"] or "", r["alias_type"]
        if not norm or kind in SKIP_TYPES:
            return
        if r["label"] == "Department" and kind in ("short", "abbrev"):
            return
        tokens = norm.split()
        if len(tokens) == 1 and norm in GENERIC_WORDS and kind not in ID_LIKE:
            return
        short_acronym = len(norm) == 3 and alias.isupper()
        if len(norm) < 4 and not short_acronym:
            return
        if r["ambiguous"] or (norm in self.entries and self.entries[norm]["uid"] != r["uid"]):
            self.ambiguous.add(norm)
            self.entries.pop(norm, None)
            return
        if norm in self.ambiguous:
            return
        self.entries[norm] = {"uid": r["uid"], "alias": alias, "alias_type": kind, "case_sensitive": short_acronym}
        self.max_n = max(self.max_n, min(len(tokens), MAX_NGRAM))

    def find(self, text: str):
        """Return (mentions, unresolved). Longest match wins; matches never overlap."""
        tokens = normalize(text).split()
        mentions, unresolved, i = {}, [], 0
        while i < len(tokens):
            hit = None
            for n in range(min(self.max_n, len(tokens) - i), 0, -1):
                key = " ".join(tokens[i : i + n])
                entry = self.entries.get(key)
                if entry and (not entry["case_sensitive"] or re.search(rf"\b{re.escape(entry['alias'])}\b", text)):
                    hit = (n, entry)
                    break
                if key in self.ambiguous and (len(unresolved) < MAX_UNRESOLVED) and key not in unresolved:
                    unresolved.append(key)
            if hit:
                n, entry = hit
                m = mentions.setdefault(entry["uid"], {"uid": entry["uid"], "count": 0, "via": entry["alias_type"]})
                m["count"] += 1
                i += n
            else:
                i += 1
        return list(mentions.values()), unresolved


_cache = {"at": 0.0, "gaz": None}


def get_gazetteer(run, ttl=600):
    """Build from the Alias nodes in Neo4j, cached for `ttl` seconds."""
    now = time.time()
    if _cache["gaz"] is None or now - _cache["at"] > ttl:
        _cache["gaz"], _cache["at"] = Gazetteer(run(ALIAS_QUERY)), now
    return _cache["gaz"]


def reset_cache():
    _cache["gaz"], _cache["at"] = None, 0.0
