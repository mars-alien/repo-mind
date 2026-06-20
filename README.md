# repo-mind

> Ask anything about any GitHub repository. Get precise, cited answers backed by the actual source code.

repo-mind is a Chrome extension that indexes GitHub repositories using a production-grade RAG pipeline and lets you chat with the codebase through a side panel — with answers that cite exact file paths and line numbers.

---

## What it does

Navigate to any GitHub repository, click **Index this repo**, and start asking questions:

- *"How does authentication work?"*
- *"Where is the database connection initialized?"*
- *"What does the routing service do?"*

The extension retrieves the most relevant code chunks, reranks them, and streams a grounded answer — citing `filepath:L{start}-L{end}` for every claim.

---

## Architecture

```
Chrome Extension (React 18 + Vite + CRXJS)
        │
        │  REST / SSE
        ▼
FastAPI Backend
   ├── GitHub Fetcher      → downloads repo files via GitHub API (up to 200 files)
   ├── AST Chunker         → splits Python at function/class boundaries; JS/TS via regex
   ├── BGE Embedder        → bge-small-en-v1.5  (384-dim, CPU-friendly, ~33 MB)
   ├── Weaviate            → BM25 + HNSW hybrid index  (Docker)
   ├── Hybrid Retriever    → BM25 (k1=1.5, b=0.4) + dense fused via RRF (k=60)
   ├── Cross-Encoder       → ms-marco-MiniLM-L-6-v2  reranks top-20 → top-8
   ├── Query Processor     → intent detection + HyDE for "explain" queries
   └── Groq LLM            → llama-3.3-70b-versatile / deepseek-r1 (design intent)
```

### Query pipeline (per request)

```
User question
   │
   ├─ 1. Intent detection    explain / find / design / general
   ├─ 2. HyDE expansion      hypothetical code snippet  (explain intent only)
   ├─ 3. Embed               384-dim BGE dense vector
   ├─ 4. Hybrid search       BM25 + HNSW → top-20 candidates via RRF fusion
   ├─ 5. Cross-encoder       rerank top-20 → top-8
   ├─ 6. Build context       heading-prefixed code snippets with file/line metadata
   └─ 7. Stream              Groq SSE response with per-chunk citations
```

**Query latency:** under 2 s end-to-end — retrieval <500 ms, Groq first token <800 ms

---

## Tech stack

| Layer | Technology |
|---|---|
| Extension | React 18, Tailwind CSS 3, Vite + CRXJS, Chrome MV3 Side Panel API |
| Backend | FastAPI 0.115, Python 3.11+, Uvicorn |
| Vector DB | Weaviate 1.28 (Docker) — HNSW cosine + BM25 |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API — `llama-3.3-70b-versatile`, `deepseek-r1-distill-llama-70b` |
| Auth | JWT + bcrypt, SQLite |
| Code parsing | Python `ast` module for exact function/class line boundaries |

---

## Project structure

```
repo-mind/
├── backend/
│   ├── main.py              # FastAPI app — all endpoints + query pipeline
│   ├── chunker.py           # AST-aware chunker (Python, JS/TS, sliding-window fallback)
│   ├── embedder.py          # BGE embedding singleton
│   ├── retriever.py         # Hybrid search + cross-encoder reranking
│   ├── query_processor.py   # Intent detection + HyDE generation
│   ├── llm.py               # Groq streaming, model routing by intent
│   ├── fetcher_github.py    # GitHub API fetcher + smart file filter
│   ├── fetcher.py           # Web page content extractor
│   ├── auth.py              # JWT + bcrypt helpers
│   ├── database.py          # SQLite — users + document metadata
│   ├── metrics.py           # Per-request latency tracking
│   ├── evaluate_rag.py      # RAGAS-style evaluation script
│   ├── questions.txt        # Evaluation questions (one per line)
│   └── requirements.txt
├── src/
│   ├── sidepanel/
│   │   ├── App.jsx          # Main React UI — chat, library, GitHub detection
│   │   ├── index.css        # Tailwind base + custom animations
│   │   └── index.html
│   ├── background/
│   │   └── service-worker.js
│   ├── content/
│   │   └── launcher.js      # Floating launcher button (Shadow DOM)
│   └── lib/
│       ├── api.js           # Backend API client (REST + SSE)
│       └── embedder.js
├── manifest.json            # Chrome MV3 manifest
├── docker-compose.yml       # Weaviate container
├── vite.config.js
└── tailwind.config.js
```

---

## Getting started

### Prerequisites

- Node.js 18+
- Python 3.11+
- Docker Desktop
- [Groq API key](https://console.groq.com) — free tier, ~250 tok/s

### 1. Start Weaviate

```bash
docker compose up -d
```

Weaviate runs on `localhost:8080` (HTTP) and `localhost:50051` (gRPC). Data persists in a named Docker volume across restarts.

### 2. Backend setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create `backend/.env`:

```env
GROQ_API_KEY=gsk_your_key_here
SECRET_KEY=any_random_32_char_string
```

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The Weaviate schema is created automatically on first startup. The BGE embedding model (~33 MB) downloads on the first indexing request.

### 3. Build the extension

```bash
# from project root
npm install
npm run build
```

### 4. Load in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `dist/` folder

---

## Usage

### Index a repository

1. Open the extension side panel (click the extension icon or use the floating launcher)
2. Register / log in
3. Navigate to any GitHub repo — `https://github.com/owner/repo`
4. Click **Index this repo** and wait for completion

### Ask questions

Type any question about the codebase. Answers include clickable citations:

```
src/auth/middleware.py:L42-L67
internal/routes/routes.go:L15-L38
```

### Manage sources

Use the **Library** tab to view all indexed repos, see chunk counts, and delete sources.

### GitHub token (optional but recommended)

Without a token: 55-file limit, 60 req/h GitHub API rate limit.  
With a PAT: 200-file limit, 5,000 req/h.

Paste your token in the token field in the index dialog.

---

## API reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | — | Create account |
| `POST` | `/auth/login` | — | Get JWT token |
| `POST` | `/ingest` | JWT | Index a web page |
| `POST` | `/ingest/github` | JWT | Index a GitHub repo |
| `POST` | `/query` | JWT | Ask a question (SSE stream) |
| `GET` | `/sources` | JWT | List indexed sources |
| `DELETE` | `/sources/{id}` | JWT | Remove a source |
| `GET` | `/metrics` | — | Last 20 request latencies |
| `GET` | `/health` | — | Health + vector count |

---

## RAG evaluation

Measure retrieval quality against your indexed repos:

```bash
cd backend
python evaluate_rag.py --username YOUR_USERNAME
```

Three RAGAS-style metrics (LLM-as-judge via Groq):

| Metric | What it measures |
|---|---|
| **Faithfulness** | Every claim in the answer is supported by the retrieved context |
| **Answer relevancy** | The answer directly addresses the question asked |
| **Context precision** | The retrieved chunks are relevant to the question |

Customize `questions.txt` with questions specific to your indexed repos. Results saved to `eval_results.json`.

Options:

```bash
python evaluate_rag.py --username alice --questions questions.txt --delay 10
```

`--delay` (default 8 s) adds a buffer between questions to stay under Groq free tier rate limits.

---

## Indexing latency (bge-small-en-v1.5, CPU)

Indexing is a **one-time cost per repo**. Query latency after indexing is <2 s.

| Repo | Files | Chunks | GitHub fetch | Embedding | Total |
|---|---|---|---|---|---|
| eco-route | 49 | 151 | 50.8 s | 12.0 s | **72.8 s** |
| page-mind | 26 | 214 | 24.2 s | 35.7 s | **61.1 s** |
| next-js-portfolio | 38 | 106 | 34.0 s | 19.5 s | **54.6 s** |
| user-api-go | 11 | 36 | 10.8 s | 7.3 s | **19.1 s** |
| **Average** | **31** | **127** | **30.0 s** | **18.6 s** | **51.9 s** |

The embedding model loads once per server session (~8 s on first request, zero cost after).  
GitHub network fetch is the dominant cost — use a PAT to unlock higher file limits.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key — get one free at console.groq.com |
| `SECRET_KEY` | Yes | JWT signing secret (any random string, min 32 chars) |
| `HF_TOKEN` | No | Hugging Face token — removes the unauthenticated warning on model download |

---

## License

MIT
