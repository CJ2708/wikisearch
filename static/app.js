// Front-end controller: talks to the FastAPI backend and renders results.

const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const submitBtn = document.getElementById("submit-btn");
const ragToggle = document.getElementById("rag-toggle");
const statusPill = document.getElementById("status-pill");
const sourceToggle = document.getElementById("source-toggle");

// Currently selected search source ("web" or "wikipedia").
let currentSource = "web";

sourceToggle.addEventListener("click", (e) => {
  const btn = e.target.closest(".seg");
  if (!btn) return;
  currentSource = btn.dataset.source;
  sourceToggle.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b === btn));
});

const loading = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");
const answerSection = document.getElementById("answer-section");
const answerText = document.getElementById("answer-text");
const answerMeta = document.getElementById("answer-meta");
const resultsSection = document.getElementById("results-section");
const resultsList = document.getElementById("results-list");
const errorBox = document.getElementById("error-box");

// --- helpers ---------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Turn inline [1], [2] citations into styled superscripts.
function renderCitations(text) {
  return escapeHtml(text).replace(/\[(\d+)\]/g, '<sup class="cite">[$1]</sup>');
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function resetView() {
  hide(answerSection);
  hide(resultsSection);
  hide(errorBox);
  resultsList.innerHTML = "";
}

function showError(message) {
  errorBox.textContent = message;
  show(errorBox);
}

// --- health check ----------------------------------------------------------

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();

    // Reflect the real default source; if web search has no key, steer to Wikipedia.
    currentSource = data.default_source || "web";
    const webBtn = sourceToggle.querySelector('[data-source="web"]');
    const wikiBtn = sourceToggle.querySelector('[data-source="wikipedia"]');
    if (!data.web_search) {
      webBtn.title = "Set a TAVILY_API_KEY to enable web search";
      webBtn.style.opacity = "0.5";
    }
    sourceToggle.querySelectorAll(".seg").forEach((b) =>
      b.classList.toggle("active", b.dataset.source === currentSource)
    );

    if (data.llm_ready) {
      statusPill.textContent = `${data.model} ✓`;
      statusPill.className = "pill ready";
    } else {
      statusPill.textContent = `${data.model} not ready`;
      statusPill.className = "pill down";
      ragToggle.checked = false;
    }
  } catch {
    statusPill.textContent = "backend offline";
    statusPill.className = "pill down";
  }
}

// --- rendering -------------------------------------------------------------

function renderResults(results) {
  resultsList.innerHTML = "";
  results.forEach((r) => {
    const li = document.createElement("li");
    li.className = "result";
    // Show semantic relevance as a percentage when available, else the fused score.
    const badge =
      r.relevance != null
        ? `${Math.round(r.relevance * 100)}% match`
        : `score ${r.score}`;
    li.innerHTML = `
      <div class="result-head">
        <div class="result-title">
          <span class="result-rank">#${r.rank}</span>
          <a href="${escapeHtml(r.url)}" target="_blank" rel="noopener">${escapeHtml(r.title)}</a>
        </div>
        <span class="score">${badge}</span>
      </div>
      <p class="snippet">${escapeHtml(r.text)}</p>
    `;
    resultsList.appendChild(li);
  });
  show(resultsSection);
}

// --- main search flow ------------------------------------------------------

async function runSearch(query) {
  resetView();
  submitBtn.disabled = true;
  const useRag = ragToggle.checked;
  loadingText.textContent = useRag
    ? "Searching Wikipedia and generating an answer…"
    : "Searching Wikipedia…";
  show(loading);

  const endpoint = useRag ? "/api/ask" : "/api/search";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, source: currentSource }),
    });

    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Request failed (${res.status}): ${detail}`);
    }

    const data = await res.json();
    hide(loading);

    if (useRag) {
      if (data.answer) {
        answerText.innerHTML = renderCitations(data.answer);
        answerMeta.textContent =
          `Model: ${data.model} · ${data.articles_considered} articles considered · ${data.took_ms} ms`;
        show(answerSection);
      } else if (data.error) {
        showError(data.error);
      }
      renderResults(data.sources || []);
    } else {
      renderResults(data.results || []);
      if (!data.results || data.results.length === 0) {
        showError("No matching passages found. Try rephrasing your query.");
      }
    }
  } catch (err) {
    hide(loading);
    showError(err.message || "Something went wrong.");
  } finally {
    submitBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (query) runSearch(query);
});

checkHealth();
