"""Thin wrapper around the Gemini API (google-genai)."""
import logging
import os
import time

from google import genai
from google.genai import errors, types

from app_config import GEMINI_MODEL

logger = logging.getLogger(__name__)

_client = None
RETRYABLE = {429, 500, 502, 503, 504}


def get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (backend/.env).")
        _client = genai.Client(api_key=key)
    return _client


def _with_retry(call, attempts=3):
    """Retry rate-limit and transient server errors with a short backoff."""
    for attempt in range(attempts):
        try:
            return call()
        except errors.APIError as exc:
            if exc.code in RETRYABLE and attempt < attempts - 1:
                delay = 2 * 2**attempt
                logger.warning("Gemini error %s; retrying in %ss", exc.code, delay)
                time.sleep(delay)
                continue
            raise


_cooldown = {}   # model -> time until which it is skipped after its quota ran out
COOLDOWN_SECONDS = 300


def _call(prompt, config, model):
    return _with_retry(
        lambda: get_client().models.generate_content(model=model, contents=prompt, config=config)
    )


def _generate(prompt, config, model):
    """Use the requested model; if its quota is exhausted, fall back to the default (cheaper) model.

    Free-tier quotas differ a lot by model (the answer model may allow only a handful of requests a
    day), so a spent quota should degrade the answer quality, not break the chat.
    """
    model = model or GEMINI_MODEL
    if model != GEMINI_MODEL and _cooldown.get(model, 0) > time.time():
        model_to_use = GEMINI_MODEL
    else:
        model_to_use = model
    try:
        response = _call(prompt, config, model_to_use)
    except errors.APIError as exc:
        if exc.code != 429 or model_to_use == GEMINI_MODEL:
            raise
        _cooldown[model_to_use] = time.time() + COOLDOWN_SECONDS
        logger.warning("%s quota exhausted; falling back to %s for %ss", model_to_use, GEMINI_MODEL, COOLDOWN_SECONDS)
        response = _call(prompt, config, GEMINI_MODEL)
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response (it may have been blocked).")
    return text


def generate_text(prompt, *, system=None, model=None, temperature=0.2):
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    return _generate(prompt, config, model).strip()


def generate_json(prompt, schema, *, system=None, model=None):
    """Return an instance of the pydantic `schema`, parsed from the model's JSON output."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
    )
    return schema.model_validate_json(_generate(prompt, config, model))
