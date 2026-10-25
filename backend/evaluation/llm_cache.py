"""Disk cache + throttle around llm.generate_text / generate_json.

Identical calls (same model, system prompt and prompt) are answered from the cache, so
 - a rerun after a crash or a daily-quota stop continues where it left off;
 - the router configuration reuses the answer of whichever mode it picks, at no cost.
Only uncached calls count against the free-tier limits.
"""
import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class QuotaReached(RuntimeError):
    """The run's call budget or the provider's daily quota is spent. Rerun later; the cache keeps the progress."""


class CachedLLM:
    def __init__(self, path, rpm=10, max_new_calls=450, sleep=time.sleep, clock=time.monotonic):
        self.path, self.interval = Path(path), 60.0 / max(rpm, 1)
        self.max_new_calls, self.sleep, self.clock = max_new_calls, sleep, clock
        self.new_calls, self._last = 0, None
        self.logical = 0                 # every call, cached or not
        self.sleep_total = 0.0           # seconds spent waiting for the rate limit
        self.real_seconds = 0.0          # seconds spent inside real LLM calls
        self.charged = 0.0               # seconds every call (cached or not) would take: used for latency estimates
        self._known = []                 # durations seen so far, to price cached calls from older runs
        self.store = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.store[row["key"]] = (row["value"], row.get("t"))
                    if row.get("t") is not None:
                        self._known.append(row["t"])

    @staticmethod
    def key(kind, model, system, prompt, schema=None):
        raw = json.dumps([kind, model, system, prompt, getattr(schema, "__name__", None)], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _fetch(self, key, call):
        self.logical += 1
        if key in self.store:
            value, seconds = self.store[key]
            self.charged += seconds if seconds is not None else (sum(self._known) / len(self._known) if self._known else 3.0)
            return value
        if self.new_calls >= self.max_new_calls:
            raise QuotaReached(f"Reached this run's limit of {self.max_new_calls} new LLM calls.")
        if self._last is not None:
            wait = self.interval - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
                self.sleep_total += wait
        started = self.clock()
        try:
            value = call()
        except Exception as exc:  # noqa: BLE001
            if "PerDay" in str(exc) or "per day" in str(exc).lower():
                raise QuotaReached("The provider's daily quota is spent.") from exc
            raise
        finally:
            self._last, self.new_calls = self.clock(), self.new_calls + 1
        seconds = max(self.clock() - started, 0.0)
        self.real_seconds += seconds
        self.charged += seconds
        self._known.append(seconds)
        self.store[key] = (value, seconds)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "value": value, "t": round(seconds, 3)}, ensure_ascii=False) + "\n")
        return value

    def install(self, llm_module):
        """Patch the module in place; returns a function that restores the originals."""
        real_text, real_json = llm_module.generate_text, llm_module.generate_json

        def generate_text(prompt, *, system=None, model=None, temperature=0.2):
            key = self.key("text", model, system, prompt)
            return self._fetch(key, lambda: real_text(prompt, system=system, model=model, temperature=temperature))

        def generate_json(prompt, schema, *, system=None, model=None):
            key = self.key("json", model, system, prompt, schema)
            raw = self._fetch(key, lambda: real_json(prompt, schema, system=system, model=model).model_dump_json())
            return schema.model_validate_json(raw)

        llm_module.generate_text, llm_module.generate_json = generate_text, generate_json

        def restore():
            llm_module.generate_text, llm_module.generate_json = real_text, real_json
        return restore
