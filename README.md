# Enterprise RAG Knowledge Assistant

**Answers employee questions about company policy, with a cited source for every claim.**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent%20workflow-1C3C3C)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-36%20passing-2ea44f)

Employees lose time searching policy PDFs, and HR, Finance and IT answer the same questions again and again. This project is a retrieval-augmented generation (RAG) assistant that answers those questions from the policy documents themselves.

- **Hybrid retrieval.** BM25 finds exact terms such as "USD 250" or "FIDO2". FAISS vector search finds paraphrases. The two ranked lists are merged with Reciprocal Rank Fusion.
- **Cross-encoder reranking.** `BAAI/bge-reranker-base` rescores the top candidates, and its score decides what happens next.
- **LangGraph agent with a fallback path.** If retrieval is weak, the graph rewrites the question into policy vocabulary ("PTO" → "annual leave entitlement"), searches wider, and reranks again. If the evidence is still weak, it says it doesn't know instead of guessing.
- **Validated citations.** Each answer cites section IDs such as `[TRV-5.1]`. Any ID that wasn't in the retrieved context is removed before the answer is returned.
- **Mock mode.** With no API key the whole system still runs offline: local embeddings, a lexical reranker, and extractive answers. Anyone can clone it and try it in about a minute.

> All documents describe **Northwind Dynamics**, a fictional company. No real company data is used.

---

## Contents
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Design decisions & limitations](#design-decisions--limitations)
- [Tech stack](#tech-stack)

---

## Screenshots

| Cited answer | Fallback path + pipeline trace |
|---|---|
| ![Cited answer](docs/screenshots/02-cited-answer.png) | ![Fallback and trace](docs/screenshots/03-fallback-trace.png) |
| **Out-of-scope question: the assistant abstains** | **Dark mode** |
| ![Abstain](docs/screenshots/04-abstain.png) | ![Dark mode](docs/screenshots/05-dark-mode.png) |

These screenshots were captured in **mock mode**, so the answers are extractive: sentences quoted from the policies. In live mode GPT-4o writes a concise answer from the same cited sections.

---

## Architecture

### System overview

```mermaid
flowchart LR
    subgraph Client
        UI[Web UI<br/>HTML/JS]
        CLI[CLI / curl]
    end

    subgraph API[FastAPI service]
        EP["/api/ask · /api/feedback<br/>/api/documents · /health"]
        G[[LangGraph workflow]]
    end

    subgraph Retrieval
        BM25[(BM25<br/>rank_bm25)]
        FAISS[(FAISS<br/>IndexFlatIP)]
        RRF{{Reciprocal Rank<br/>Fusion}}
        CE[BGE cross-encoder<br/>reranker]
    end

    subgraph Models
        LLM[OpenAI GPT-4o<br/>or Mock LLM]
        EMB[OpenAI embeddings<br/>or hashing embedder]
    end

    DOCS[/data/policies/*.md/] -->|section-aware chunking| BM25
    DOCS -->|embed + cache| EMB --> FAISS
    UI --> EP
    CLI --> EP
    EP --> G
    G --> BM25 & FAISS --> RRF --> CE --> G
    G --> LLM
    EP --> DB[(SQLite / PostgreSQL<br/>query + feedback log)]
```

### Agent workflow (LangGraph)

This is the compiled graph as reported by LangGraph (`graph.get_graph().draw_mermaid()`):

```mermaid
flowchart TD
    START([start]) --> A[analyze_query]
    A --> R[hybrid_retrieve<br/>BM25 + FAISS → RRF]
    R --> RR[rerank<br/>BGE cross-encoder]
    RR -->|top score ≥ threshold| GEN[generate<br/>GPT-4o, cite section IDs]
    RR -.->|weak & not yet retried| FB[fallback_retrieve<br/>query rewrite · 2× depth · BM25-weighted]
    FB --> RR
    RR -.->|still weak after retry| AB[abstain<br/>'not in the policies']
    GEN --> V[validate_citations<br/>drop invented IDs · set confidence]
    V --> END([end])
    AB --> END
```

---

## How it works

| Stage | What happens | Why |
|---|---|---|
| **Chunking** | Markdown policies are split on their numbered headings. Each chunk is one section with a stable ID (`EXP-2.3`). Long sections are split on sentence boundaries (`EXP-2.3.p2`). | Citations point to a real, human-readable section rather than an arbitrary 512-token window. |
| **Hybrid retrieval** | BM25 and FAISS each return their top 50 results. RRF (`k=60`) merges the two rankings into the top 20. | BM25 catches exact figures and jargon, and vectors catch paraphrase. RRF needs no score calibration between the two. |
| **Reranking** | A cross-encoder reads the question and each passage *together* and outputs a relevance probability. The top 5 are kept. | Much more precise than bi-encoder similarity, and the probability works as a confidence signal. |
| **Routing** | If the top score is at or above the threshold, the graph generates an answer. Otherwise it takes the fallback path once, and abstains if that also comes back weak. | Vocabulary gaps get a second chance, and genuinely unanswerable questions don't produce hallucinated answers. |
| **Fallback retrieval** | The LLM rewrites the question into up to 3 queries in policy vocabulary. Every query is searched, BM25 is weighted 1.5×, and the candidate pool is doubled. Reranking takes each passage's best score across all the queries. | Recovers questions like "How much PTO can I roll over?" when the policies say "carry over unused annual leave". |
| **Generation** | GPT-4o at temperature 0, with a strict system prompt: cite every claim, only cite supplied IDs, and give an exact "not found" reply when the answer isn't there. | Grounded, auditable answers. |
| **Citation validation** | Citation IDs are parsed from the answer. Any ID not in the retrieved context is removed and logged as a warning. Confidence is set to high, medium or low. | A guard that doesn't rely on trusting the model. |
| **Resilience** | If the LLM call fails, the node falls back to an extractive answer over the same sources and returns a warning. If the database is down, answers are still served without logging. | Degrades gracefully instead of returning a 500. |

### Mock mode vs live mode

Mock mode is enabled automatically when `OPENAI_API_KEY` is empty, or when `MOCK_MODE=true`.

| Component | Live | Mock (offline) |
|---|---|---|
| Embeddings | `text-embedding-3-small` | Deterministic feature hashing (word, bigram and char-trigram features) |
| Reranker | `BAAI/bge-reranker-base` | BGE if already cached locally, otherwise an IDF-weighted lexical scorer |
| Query rewrite | GPT-4o (JSON mode) | Synonym table (`PTO → annual leave`, `WFH → remote work`, …) |
| Answer | GPT-4o, written answer with citations | Extractive: best-matching sentences plus their section IDs |

Mock mode runs the **same graph**: the same routing, fallback, citation validation, API and UI. Only the model-backed components are swapped out.

---

## Quickstart

### Option A: local (Python 3.11+)

```bash
git clone <your-repo-url> enterprise-rag-assistant
cd enterprise-rag-assistant

python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env            # optional: add OPENAI_API_KEY for live mode
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**. The API docs are at **http://localhost:8000/docs**.

Without a key the badge in the header shows **MOCK MODE**. Add `OPENAI_API_KEY` to `.env` and restart to switch to **LIVE**.

Optionally, cache the BGE reranker (about 1.1 GB) so it is also used in mock mode and on the first live request:

```bash
python scripts/download_models.py
```

### Option B: Docker

```bash
cp .env.example .env            # optional: leave the key empty for mock mode
docker compose up --build
```

To use **PostgreSQL** for the query and feedback logs instead of SQLite, set `DATABASE_URL=postgresql+psycopg://rag:rag@postgres:5432/rag` in `.env` and run:

```bash
docker compose --profile postgres up --build
```

To bake the reranker weights into the image, run `PRELOAD_RERANKER=true docker compose build`.

---

## Usage

### Web UI
- Ask a question or click one of the examples.
- Citation chips such as `TRV-5.1` highlight the matching source card. **Show full section** loads the complete policy text.
- Badges show the confidence level and whether the fallback path ran. **Pipeline trace** lists every graph node with its latency.
- Use 👍 / 👎 to record feedback in the database.

### API

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I fly business class to Singapore?"}'
```

```jsonc
{
  "answer": "Economy class is the standard for all flights. Business class is permitted only for flights with a scheduled duration over 10 hours and requires department head approval. [TRV-3.1]",
  "citations": [{ "id": "TRV-3.1", "doc_title": "Business Travel Policy", "heading": "Class of Travel", "score": 0.98, "...": "..." }],
  "confidence": "medium",
  "used_fallback": true,
  "queries": ["Can I fly business class to Singapore?", "... air travel class of travel"],
  "mode": "mock",
  "latency_ms": 16.2,
  "trace": [{ "node": "analyze_query", "ms": 0.0, "detail": "7 words" }, "..."],
  "query_id": "8e0d928e-..."
}
```

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/ask` | Ask a question and get the answer, citations, confidence and trace |
| `POST` | `/api/feedback` | `{query_id, rating: 1 \| -1, comment?}` |
| `GET` | `/api/documents` | List the indexed policies with version and section count |
| `GET` | `/api/chunks/{id}` | Full text of one section, e.g. `TRV-5.1` |
| `GET` | `/health` | Mode, models, reranker, threshold and corpus size |

### CLI

```bash
python scripts/ask.py --trace "How much PTO can I roll over?"
python scripts/build_index.py      # pre-build / refresh the embedding cache
```

---

## Configuration

All settings are read from environment variables or `.env` (see [.env.example](.env.example)).

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | Empty → mock mode |
| `MOCK_MODE` | `false` | Force mock mode even when a key is set |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | Model for answers and query rewrites |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (live mode) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Any Hugging Face cross-encoder, e.g. `BAAI/bge-reranker-v2-m3` |
| `USE_CROSS_ENCODER` | `true` | `false` → lexical reranker (no torch model) |
| `RELEVANCE_THRESHOLD` | per reranker | Routing threshold (BGE 0.20, lexical 0.35) |
| `TOP_K_RETRIEVE` / `TOP_K_RERANK` | `20` / `5` | Candidates after fusion / sections sent to the LLM |
| `DATABASE_URL` | `sqlite:///storage/rag.db` | SQLAlchemy URL; PostgreSQL supported via `psycopg` |

To add or update documents, drop Markdown files with a `doc_id`/`title` front-matter block into `data/policies/`. The embedding cache is fingerprinted against the corpus and rebuilds automatically.

---

## Evaluation

```bash
python -m eval.run_ragas               # offline metrics; + RAGAS when OPENAI_API_KEY is set
python -m eval.run_ragas --no-ragas    # offline only
python -m eval.run_ragas --judge-model gpt-4o
```

The golden set ([eval/golden_qa.jsonl](eval/golden_qa.jsonl)) has **22 questions**: 20 answerable, each with a reference answer and the expected section IDs, and 2 out-of-scope questions. The script writes `eval/results/report.md` and `results.json`.

- **Offline metrics** (always computed): hit rate@k, MRR, citation hit rate, grounded-answer rate, false-abstention rate, abstention accuracy, fallback rate and latency.
- **RAGAS metrics** (live mode, with an LLM judge): faithfulness, answer relevancy, context precision and context recall.

**Mock-mode baseline** (offline embeddings and lexical reranker, 2026-09-30):

| Metric | Value |
|---|---|
| Hit rate@5 | 1.00 |
| MRR | 0.867 |
| Citation hit rate | 1.00 |
| Grounded-answer rate | 1.00 |
| Abstention accuracy (out-of-scope) | 1.00 |
| False-abstention rate | 0.00 |
| Fallback rate | 0.18 |
| Median latency | ~3 ms |

> **Read these numbers with care.** The golden set is small and was written by the same author who tuned the mock components, so it is a regression check, not a benchmark. Run the live-mode RAGAS evaluation with your own key to get LLM-judged scores. The RAGAS path hasn't been run as part of this repository's build, because no API key was available.

---

## Testing

```bash
pytest          # 36 tests, ~2 s, fully offline (mock mode, temp index + DB)
ruff check .
```

| File | Coverage |
|---|---|
| `test_ingestion.py` | Front-matter parsing, section IDs, long-section splitting, validation errors |
| `test_retrieval.py` | Tokenizer, hashing embedder, RRF math, BM25, FAISS, hybrid recall@5 (parametrized), reranker behaviour |
| `test_graph.py` | Direct answer path, fallback path, abstention, trace, hallucinated-citation stripping, LLM outage degradation |
| `test_api.py` | Health, UI, `/api/ask`, input validation, feedback round-trip, documents and chunks |

---

## Project structure

```
enterprise-rag-assistant/
├── app/
│   ├── main.py               # FastAPI app: API + static UI
│   ├── config.py             # pydantic-settings (.env)
│   ├── pipeline.py           # wires everything; RAGPipeline.ask()
│   ├── schemas.py            # request/response models
│   ├── ingestion/            # loader (front matter) + section-aware chunker
│   ├── retrieval/            # bm25, embeddings, vector_store (FAISS), hybrid (RRF), reranker
│   ├── llm/                  # prompts, OpenAI client, mock LLM
│   ├── graph/                # LangGraph state, nodes, workflow
│   ├── db/                   # SQLAlchemy query + feedback log
│   └── static/               # web UI (HTML/CSS/JS, no build step)
├── data/policies/            # 8 synthetic policy documents
├── eval/                     # golden_qa.jsonl + run_ragas.py
├── tests/                    # pytest suite (offline)
├── scripts/                  # ask.py, build_index.py, download_models.py
├── docs/screenshots/
├── Dockerfile · docker-compose.yml · Makefile
└── requirements.txt · requirements-dev.txt · pyproject.toml · .env.example
```

---

## Design decisions & limitations

- **RRF instead of a weighted score sum.** BM25 scores and cosine similarities are on different scales. RRF uses only ranks, so it is robust without per-corpus tuning.
- **Using the reranker score as the router.** Cross-encoder probabilities are much better calibrated than fused retrieval scores, so they make a good "do we actually know this?" signal.
- **One retry, then abstain.** Looping more costs latency for diminishing returns. For a policy assistant, a clear "not covered, contact People Ops" is better than a confident wrong answer.
- **`IndexFlatIP`.** Exact search is instant at this corpus size (145 sections). For millions of chunks, switch to HNSW or IVF. The `VectorStore` interface stays the same.
- **Mock mode is a lexical approximation.** Without the cross-encoder, questions that share no words with the policy ("How many days of annual leave do I get?" vs. "*entitled* to 25 days") can rank a neighbouring section first. Live mode (BGE + GPT-4o) handles this; mock mode exists so the system can be tried for free.
- **Not included (next steps).** Authentication and per-department access control on documents, streaming responses, conversation memory for follow-up questions, PDF/DOCX ingestion, and an admin view of the feedback log.

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic v2, pydantic-settings |
| Orchestration | LangGraph (`StateGraph` with conditional routing and a retry loop) |
| LLM | OpenAI GPT-4o (answers, query rewriting) |
| Embeddings | OpenAI `text-embedding-3-small` · offline hashing embedder |
| Retrieval | FAISS (`faiss-cpu`), BM25 (`rank_bm25`), Reciprocal Rank Fusion |
| Reranking | `BAAI/bge-reranker-base` via Hugging Face Transformers + PyTorch (CPU) |
| Persistence | SQLAlchemy 2 · SQLite (default) · PostgreSQL (optional) |
| Evaluation | RAGAS · custom retrieval/grounding metrics |
| Front end | Vanilla HTML/CSS/JS served by FastAPI (light/dark) |
| Quality | pytest, Ruff |
| Deployment | Docker (non-root, health check, CPU torch), Docker Compose (optional Postgres profile) |

---

<p align="center">
  Built by <strong>Magnum Minds AI</strong>
</p>
