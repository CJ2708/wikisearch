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

# --- Retrieval / ranking ---
# Articles are chopped into passages of roughly this many words before ranking.
PASSAGE_WORDS = int(_env("PASSAGE_WORDS", "120"))
# How many top passages to keep as context for the answer.
TOP_PASSAGES = int(_env("TOP_PASSAGES", "6"))

# Cap passages per article so huge pages (e.g. "World War II") don't generate
# hundreds of chunks. Wikipedia front-loads the key facts, so the lead plus the
# first several sections are what matter — and this keeps embedding fast.
MAX_PASSAGES_PER_ARTICLE = int(_env("MAX_PASSAGES_PER_ARTICLE", "10"))

# Semantic reranking: combine BM25 (lexical) with embedding similarity (meaning).
# Set USE_EMBEDDINGS=false on very memory-constrained hosts to fall back to
# lexical-only ranking (the model needs a few hundred MB of RAM).
USE_EMBEDDINGS = _env("USE_EMBEDDINGS", "true").lower() in ("1", "true", "yes")
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Small extra weight for an article's lead/summary passage, where Wikipedia puts
# the most direct answer to factual questions.
LEAD_BOOST = float(_env("LEAD_BOOST", "0.015"))

# --- RAG: which LLM backend generates the answer ---
# "ollama" -> local model on your machine (great for development)
# "groq"   -> hosted open models via Groq's free API (great for live deployment)
# "auto"   -> use Groq if GROQ_API_KEY is set, otherwise fall back to Ollama.
LLM_PROVIDER = _env("LLM_PROVIDER", "auto").lower()

# Upper bound on generation time so the UI never hangs forever.
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "120"))

# Local Ollama backend
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
# A 7B model gives far better grounded answers than a 3B one; override via env
# if you need something lighter (e.g. OLLAMA_MODEL=llama3.2).
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5:7b")

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
