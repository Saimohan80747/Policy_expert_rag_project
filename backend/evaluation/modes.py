"""The configurations compared in the evaluation, as rag.Options.

Rerank is off in the main comparison (the Cohere trial allows about 1,000 calls a month) and only
turned on in the 'rerank' ablation.
"""
from rag import Options


def _options(base, extra):
    return Options(**{"rerank": False, **base, **extra})


MODES = {
    # Documents only: hybrid vector search + neighbours. No graph facts, no entity-anchored chunks.
    "vanilla": {"force_route": "vanilla", "anchored": False},
    # Graph facts only; when the graph has nothing, the answer is "not found" (no fallback to documents).
    "graph": {"force_route": "graph", "graph_fallback": False},
    # Always both: graph facts plus document chunks.
    "hybrid": {"force_route": "hybrid", "graph_fallback": False},
    # What the app does: the router picks, with its normal fallbacks.
    "router": {},
}

# Ablations of the document side, run on top of the router configuration.
ABLATIONS = {
    "no_neighbours": {"neighbours": False},
    "no_expand": {"expand": False},
    "no_anchored": {"anchored": False},
    "no_superseded_filter": {"superseded_filter": False},
    "rerank": {"rerank": True},
}


def options_for(name, **extra):
    base = MODES.get(name)
    if base is None:
        base = ABLATIONS[name]
    return _options(base, extra)
