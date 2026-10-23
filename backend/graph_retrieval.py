"""Graph side of retrieval.

Placeholder until the graph is built from documents (steps 3-6). The router
already produces graph/hybrid routes; until this module reports the graph as
ready, those routes fall back to vanilla retrieval.
"""


def graph_available() -> bool:
    return False  # switched on in step 6, once graph retrieval exists


def retrieve_graph_context(question: str, entities: list[str]):
    raise NotImplementedError("Graph retrieval is not built yet.")
