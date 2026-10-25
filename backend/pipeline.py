TERMINAL_STATUSES = {"duplicate", "failed"}

_HEAVY_KEYS = ("file_bytes", "text", "chunks", "uploaded_file")


def fail(state, error, step=""):
    """Turn a pipeline state into a lightweight 'failed' result."""
    out = {k: v for k, v in (state or {}).items() if k not in _HEAVY_KEYS}
    out["status"] = "failed"
    out["error"] = error
    if step:
        out["step"] = step
    return out


class Pipeline:
    """Runs steps in order. Each step takes and returns a state dict.

    The run stops early when a step marks the state 'duplicate' or 'failed',
    and any exception is converted into a 'failed' result instead of
    propagating.
    """

    def __init__(self, steps=None):
        self.steps = list(steps or [])

    def run(self, data):
        file_name = getattr(data, "filename", None)
        if file_name is None and isinstance(data, dict):
            file_name = data.get("file_name", "")

        for step in self.steps:
            step_name = getattr(step, "__name__", str(step))
            try:
                data = step(data)
            except Exception as exc:  # noqa: BLE001 - report every failure uniformly
                return {
                    "status": "failed",
                    "file_name": file_name or "",
                    "step": step_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if isinstance(data, dict) and data.get("status") in TERMINAL_STATUSES:
                break
        return data
