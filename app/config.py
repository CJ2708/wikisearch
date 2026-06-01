"""Central configuration for the Wikipedia search engine.

Everything tunable lives here so you can experiment without hunting through code.
Override any value with an environment variable of the same name.
"""
import os

try:
    # Load variables from a local .env file if present (handy for a Groq key in
    # development). In production the host injects env vars directly, so this is
    # a no-op. Safe to skip if python-dotenv isn't installed.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Wikipedia source ---
WIKI_LANG = _env("WIKI_LANG", "en")
WIKI_API_URL = f"https://{WIKI_LANG}.wikipedia.org/w/api.php"
WIKI_REST_URL = f"https://{WIKI_LANG}.wikipedia.org/api/rest_v1"
# How many candidate articles to pull from Wikipedia's own search for each query.
CANDIDATE_ARTICLES = int(_env("CANDIDATE_ARTICLES", "8"))
# Be a good API citizen: Wikipedia asks for a descriptive User-Agent.
USER_AGENT = _env(
    "USER_AGENT",
    "WikiSearchEngine/1.0 (personal learning project; contact: you@example.com)",
)

# --- BM25 retrieval ---
# Articles are chopped into passages of roughly this many words before ranking.
PASSAGE_WORDS = int(_env("PASSAGE_WORDS", "120"))
# How many top passages to keep as context for the answer.
TOP_PASSAGES = int(_env("TOP_PASSAGES", "5"))

# --- RAG: which LLM backend generates the answer ---
# "ollama" -> local model on your machine (great for development)
# "groq"   -> hosted open models via Groq's free API (great for live deployment)
# "auto"   -> use Groq if GROQ_API_KEY is set, otherwise fall back to Ollama.
LLM_PROVIDER = _env("LLM_PROVIDER", "auto").lower()

# Upper bound on generation time so the UI never hangs forever.
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "120"))

# Local Ollama backend
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2")

# Groq cloud backend (OpenAI-compatible API). Get a free key at https://console.groq.com
GROQ_API_KEY = _env("GROQ_API_KEY", "")
GROQ_URL = _env("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
# 70B model on Groq's free tier — far stronger answers than a local 3B model.
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")


def active_provider() -> str:
    """Resolve which backend to actually use for this process."""
    if LLM_PROVIDER == "auto":
        return "groq" if GROQ_API_KEY else "ollama"
    return LLM_PROVIDER
