"""Retrieval-Augmented Generation with a pluggable LLM backend.

RAG = retrieve relevant text first, then ask an LLM to answer *using only that
text*. This grounds the answer in real Wikipedia content and lets us show
citations, instead of relying on whatever the model happened to memorize.

The retrieval half is done by search.py (BM25 + ranking); this module handles
the "augmented generation" half and supports two interchangeable backends:

    * "ollama" — a model running locally on your machine (best for development)
    * "groq"   — hosted open models via Groq's free API (best for deployment)

Both speak the same internal interface, selected by config.active_provider(), so
the rest of the app never needs to know which one is in use.
"""
from typing import List, Dict, Optional

import httpx

from . import config


SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY "
    "the numbered Wikipedia sources provided. Cite the sources you use inline like "
    "[1] or [2]. If the sources do not contain the answer, say so plainly instead "
    "of guessing. Write a clear, well-structured answer in 2-5 sentences."
)


def _build_user_prompt(question: str, passages: List[Dict]) -> str:
    """Assemble the question + numbered sources into one grounded prompt."""
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(f"[{i}] {p['title']}\n{p['text']}")
    sources = "\n\n".join(blocks)
    return (
        f"Sources:\n{sources}\n\n"
        f"Question: {question}\n\n"
        f"Answer (cite sources inline with [n]):"
    )


# --- Ollama backend --------------------------------------------------------

def _ollama_available() -> bool:
    try:
        resp = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return False
    base = config.OLLAMA_MODEL.split(":")[0]
    return any(n == config.OLLAMA_MODEL or n.split(":")[0] == base for n in names)


def _ollama_generate(question: str, passages: List[Dict]) -> Dict[str, Optional[str]]:
    payload = {
        "model": config.OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": _build_user_prompt(question, passages),
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        resp = httpx.post(
            f"{config.OLLAMA_URL}/api/generate",
            json=payload,
            timeout=config.LLM_TIMEOUT,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        return {"answer": answer or None, "error": None}
    except httpx.ConnectError:
        return {
            "answer": None,
            "error": (
                "Could not reach Ollama. Start it with 'ollama serve' and make "
                f"sure the model '{config.OLLAMA_MODEL}' is pulled."
            ),
        }
    except httpx.HTTPStatusError as exc:
        return {"answer": None, "error": f"Ollama returned an error: {exc}"}
    except httpx.TimeoutException:
        return {
            "answer": None,
            "error": "The model took too long to respond. Try a simpler query.",
        }


# --- Groq backend (OpenAI-compatible chat completions) ---------------------

def _groq_available() -> bool:
    return bool(config.GROQ_API_KEY)


def _groq_generate(question: str, passages: List[Dict]) -> Dict[str, Optional[str]]:
    if not config.GROQ_API_KEY:
        return {
            "answer": None,
            "error": "GROQ_API_KEY is not set. Get a free key at https://console.groq.com",
        }
    payload = {
        "model": config.GROQ_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, passages)},
        ],
    }
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    try:
        resp = httpx.post(
            config.GROQ_URL, json=payload, headers=headers, timeout=config.LLM_TIMEOUT
        )
        resp.raise_for_status()
        choices = resp.json().get("choices", [])
        answer = choices[0]["message"]["content"].strip() if choices else ""
        return {"answer": answer or None, "error": None}
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:200] if exc.response is not None else str(exc)
        return {"answer": None, "error": f"Groq API error ({exc.response.status_code}): {detail}"}
    except httpx.TimeoutException:
        return {"answer": None, "error": "Groq took too long to respond. Try again."}
    except Exception as exc:  # network, JSON shape, etc.
        return {"answer": None, "error": f"Groq request failed: {exc}"}


# --- Public interface ------------------------------------------------------

def model_available() -> bool:
    """Is the currently selected backend ready to generate answers?"""
    return _groq_available() if config.active_provider() == "groq" else _ollama_available()


def active_model_name() -> str:
    """Human-readable name of the active model, for the UI/health endpoint."""
    if config.active_provider() == "groq":
        return f"groq:{config.GROQ_MODEL}"
    return f"ollama:{config.OLLAMA_MODEL}"


def generate_answer(question: str, passages: List[Dict]) -> Dict[str, Optional[str]]:
    """Generate a grounded answer from retrieved passages using the active backend.

    Returns {"answer": str|None, "error": str|None}. Failures are returned as a
    helpful error string rather than raised, so the API/UI degrade gracefully —
    the ranked passages are still useful on their own.
    """
    if not passages:
        return {
            "answer": None,
            "error": "No relevant Wikipedia passages were found for this query.",
        }
    if config.active_provider() == "groq":
        return _groq_generate(question, passages)
    return _ollama_generate(question, passages)
