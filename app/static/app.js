const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const form = $("#ask-form");
const input = $("#question");
const sendBtn = $("#send");

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try { const body = await res.json(); detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

// ---------------------------------------------------------------- bootstrap
async function loadStatus() {
  try {
    const h = await api("/health");
    const badge = $("#mode-badge");
    badge.textContent = h.mode === "mock" ? "MOCK MODE" : "LIVE";
    badge.className = `badge ${h.mode}`;
    badge.title = h.mode === "mock" ? "No API key: offline embeddings and extractive answers" : `LLM: ${h.llm}`;
    $("#stack-info").textContent = `${h.llm} · ${h.reranker} · ${h.chunks} sections`;
  } catch {
    $("#mode-badge").textContent = "offline";
  }
}

async function loadDocuments() {
  try {
    const docs = await api("/api/documents");
    $("#doc-list").innerHTML = docs
      .map((d) => `<li><span class="code">${escapeHtml(d.doc_id)}</span>${escapeHtml(d.title)}
        <div class="muted small">v${escapeHtml(d.version ?? "-")} · ${d.sections} sections</div></li>`)
      .join("");
  } catch {
    $("#doc-list").innerHTML = '<li class="muted">Unavailable</li>';
  }
}

// ---------------------------------------------------------------- rendering
function addMessage(role, html) {
  messagesEl.querySelector(".empty-state")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = html;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function renderAnswer(text) {
  return escapeHtml(text).replace(
    /\[([A-Z]{2,5}-\d+(?:\.\d+)*(?:\.p\d+)?)\]/g,
    (_, id) => `<button type="button" class="cite" data-cite="${id}" title="Show source ${id}">${id}</button>`
  );
}

function sourceCard(c, { related = false } = {}) {
  const score = c.score != null ? `relevance ${c.score.toFixed(2)}` : "";
  return `<article class="source-card" id="src-${c.id}">
    <header><span class="title">${escapeHtml(c.doc_title)}</span><span class="score">${score}</span></header>
    <div><span class="cite" data-cite="${c.id}">${c.id}</span> ${escapeHtml(c.heading)}</div>
    <p>${escapeHtml(c.snippet)}</p>
    ${related ? "" : `<button class="expand" data-full="${c.id}">Show full section</button>`}
  </article>`;
}

function renderSources(data) {
  const box = $("#sources");
  box.classList.remove("muted");
  let html = "";
  if (data.used_fallback) html += `<div class="notice">Fallback retrieval used: the question was rewritten into policy vocabulary and searched again.</div>`;
  if (data.citations.length) {
    html += data.citations.map((c) => sourceCard(c)).join("");
  } else if (data.related.length) {
    html += `<div class="notice">No policy answered this directly. Closest sections:</div>`;
    html += data.related.map((c) => sourceCard(c, { related: true })).join("");
  } else {
    html += '<p class="muted">No sources.</p>';
  }
  box.innerHTML = html;

  const trace = $("#trace");
  trace.innerHTML = data.trace
    .map((t) => `<li><code>${escapeHtml(t.node)}</code> <span class="muted">${t.ms} ms</span><br>${escapeHtml(t.detail)}</li>`)
    .join("");
  $("#trace-box").hidden = data.trace.length === 0;
}

function focusSource(id) {
  const card = document.getElementById(`src-${id}`);
  if (!card) return;
  document.querySelectorAll(".source-card.active").forEach((el) => el.classList.remove("active"));
  card.classList.add("active");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function feedbackHtml(queryId) {
  if (!queryId) return "";
  return `<span class="feedback" data-qid="${queryId}">
    <button type="button" data-rating="1" aria-label="Helpful">👍</button>
    <button type="button" data-rating="-1" aria-label="Not helpful">👎</button></span>`;
}

// ---------------------------------------------------------------- actions
async function ask(question) {
  addMessage("user", escapeHtml(question));
  const pending = addMessage("bot", '<span class="typing"><span></span><span></span><span></span></span>');
  sendBtn.disabled = true;
  try {
    const data = await api("/api/ask", { method: "POST", body: JSON.stringify({ question }) });
    const conf = `<span class="badge ${data.confidence}">${data.confidence} confidence</span>`;
    const fb = data.used_fallback ? '<span class="badge medium">fallback</span>' : "";
    pending.innerHTML = `${renderAnswer(data.answer)}
      <div class="meta">${conf}${fb}<span>${Math.round(data.latency_ms)} ms</span>${feedbackHtml(data.query_id)}</div>`;
    renderSources(data);
  } catch (err) {
    pending.classList.add("error");
    pending.textContent = `Something went wrong: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q.length < 3 || sendBtn.disabled) return;
  input.value = "";
  input.style.height = "auto";
  ask(q);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${input.scrollHeight}px`; });

document.querySelectorAll("#examples li").forEach((li) => {
  li.tabIndex = 0;
  const go = () => { if (!sendBtn.disabled) ask(li.textContent.trim()); };
  li.addEventListener("click", go);
  li.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
});

document.addEventListener("click", async (e) => {
  const cite = e.target.closest("[data-cite]");
  if (cite) { focusSource(cite.dataset.cite); return; }

  const expand = e.target.closest("[data-full]");
  if (expand) {
    try {
      const chunk = await api(`/api/chunks/${encodeURIComponent(expand.dataset.full)}`);
      expand.previousElementSibling.textContent = chunk.text;
      expand.remove();
    } catch { expand.textContent = "Could not load section"; }
    return;
  }

  const rate = e.target.closest("[data-rating]");
  if (rate) {
    const wrap = rate.parentElement;
    try {
      await api("/api/feedback", { method: "POST", body: JSON.stringify({ query_id: wrap.dataset.qid, rating: Number(rate.dataset.rating) }) });
      wrap.querySelectorAll("button").forEach((b) => { b.disabled = true; b.classList.toggle("chosen", b === rate); });
    } catch { rate.title = "Feedback failed"; }
  }
});

loadStatus();
loadDocuments();
