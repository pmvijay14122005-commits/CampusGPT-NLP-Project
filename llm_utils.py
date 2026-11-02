"""
llm_utils.py
Small shared helpers for talking to the Groq LLM:
- get_llm(): create a ChatGroq client (reads GROQ_API_KEY from the environment)
- safe_invoke(): call the LLM and retry when the free tier rate-limits us
- extract_json(): pull a JSON object/list out of an LLM reply

Models can be changed without editing code by setting environment variables
in your .env file:
    CAMPUSGPT_MODEL=openai/gpt-oss-120b     (answers, verification, quizzes)
    JUDGE_MODEL=openai/gpt-oss-20b          (independent judge used only by evaluate.py)
"""

import json
import os
import re
import time

from langchain_groq import ChatGroq

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"


def main_model_name():
    return os.getenv("CAMPUSGPT_MODEL", DEFAULT_MODEL)


def judge_model_name():
    return os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def get_llm(model=None, temperature=0.0):
    """Create the LLM client lazily, so .env is already loaded when this runs."""
    return ChatGroq(model=model or main_model_name(), temperature=temperature)


_RETRY_WORDS = ("429", "rate", "overloaded", "timeout", "timed out", "503", "502", "connection")


def safe_invoke(llm, messages, retries=6, base_wait=5):
    """Invoke the LLM and return its text. Waits and retries on rate-limit style errors."""
    last_error = None
    for attempt in range(retries):
        try:
            return llm.invoke(messages).content or ""
        except Exception as e:  # noqa: BLE001 - we re-raise anything that is not retryable
            last_error = e
            if any(w in str(e).lower() for w in _RETRY_WORDS):
                time.sleep(base_wait * (attempt + 1))
                continue
            raise
    raise last_error


def extract_json(text):
    """Return the parsed JSON found in an LLM reply, or None if there is none."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # fall back: take the span from the first [ or { to the last ] or }
    starts = [i for i in (cleaned.find("["), cleaned.find("{")) if i != -1]
    ends = [i for i in (cleaned.rfind("]"), cleaned.rfind("}")) if i != -1]
    if not starts or not ends:
        return None
    start, end = min(starts), max(ends)
    if end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
