import embeddings


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_budget_lets_small_batches_through_without_waiting():
    clock, b = Clock(), embeddings._TokenBudget(limit=1000, window=60)
    for _ in range(4):
        b.wait(200, clock.now, clock.sleep)
    assert clock.slept == []


def test_budget_waits_for_the_window_to_clear_when_over_the_limit():
    clock, b = Clock(), embeddings._TokenBudget(limit=1000, window=60)
    b.wait(800, clock.now, clock.sleep)
    clock.t = 10
    b.wait(800, clock.now, clock.sleep)          # 800 + 800 > 1000: must wait for the first to expire
    assert clock.slept and abs(sum(clock.slept) - 50) < 1


def test_one_oversized_request_is_still_allowed():
    clock, b = Clock(), embeddings._TokenBudget(limit=1000, window=60)
    b.wait(50_000, clock.now, clock.sleep)       # capped to the limit instead of waiting forever
    assert clock.slept == []


def test_rate_limit_errors_are_waited_out_longer_than_other_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr(embeddings.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def limited():
        calls["n"] += 1
        if calls["n"] < 5:
            raise RuntimeError("RateLimitError: [429 RESOURCE_EXHAUSTED] max tokens per minute")
        return "ok"

    assert embeddings._with_retry(limited) == "ok"
    assert sleeps == [20, 40, 60, 80]


def test_other_errors_still_fail_fast(monkeypatch):
    import pytest

    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        embeddings._with_retry(broken)
    assert calls["n"] == 4


def test_query_embeddings_are_not_throttled(monkeypatch):
    seen = []
    monkeypatch.setattr(embeddings._TokenBudget, "wait", lambda self, *a, **k: seen.append(1))
    embeddings._throttle("m", ["q"], "query")
    embeddings._throttle("m", ["p"], "passage")
    assert seen == [1]


def test_batches_are_capped_by_estimated_tokens_not_just_count():
    chunks = ["x" * 1990 for _ in range(106)]                # the headcount spreadsheet's shape
    batches = list(embeddings._batches(chunks))
    assert [c for b in batches for c in b] == chunks          # nothing lost or reordered
    assert all(sum(embeddings.estimate_tokens(c) for c in b) <= embeddings.MAX_REQUEST_TOKENS for b in batches)
    assert len(batches) > 2                                  # was 2 requests of up to 96 chunks


def test_short_texts_still_batch_up_to_the_count_limit():
    batches = list(embeddings._batches(["short"] * 200))
    assert [len(b) for b in batches] == [embeddings.EMBED_BATCH, embeddings.EMBED_BATCH, 200 - 2 * embeddings.EMBED_BATCH]


def test_embed_sends_each_request_under_the_token_cap(monkeypatch):
    sizes = []

    class Inference:
        def embed(self, model, inputs, parameters):
            sizes.append(sum(embeddings.estimate_tokens(t) for t in inputs))
            return [{"values": [0.0]} for _ in inputs]

    monkeypatch.setattr(embeddings, "get_pc", lambda: type("P", (), {"inference": Inference()})())
    monkeypatch.setattr(embeddings, "_budgets", {})
    monkeypatch.setattr(embeddings._TokenBudget, "wait", lambda self, *a, **k: None)
    out = embeddings.embed_dense(["y" * 1990] * 106)
    assert len(out) == 106 and max(sizes) <= embeddings.MAX_REQUEST_TOKENS
