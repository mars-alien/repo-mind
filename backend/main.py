"""
PageMind API v8.0 — Production GitHub RAG
──────────────────────────────────────────
Stack:
  • FastAPI                – HTTP + SSE streaming
  • Weaviate (Docker)      – HNSW dense + BM25 sparse + hybrid search
  • Groq (llama-3.3-70b)  – LLM generation  (free tier, ~250 tok/s)
  • sentence-transformers  – BAAI/bge-m3 dense embeddings (1024-dim)
  • cross-encoder          – ms-marco-MiniLM-L-6-v2 reranking
  • SQLite                 – user accounts + document metadata
  • JWT / bcrypt           – authentication

/query pipeline (code-optimized):
  1. process_query()         → intent detection + HyDE snippet
  2. embed_query()           → 1024-dim bge-m3 dense vector
                               (HyDE vector used for "explain" intent)
  3. hybrid_retrieve()       → BM25 + HNSW fused (top 20 candidates)
  4. rerank_hits()           → cross-encoder reranks → top 8
  5. Build RAG context       → heading-prefixed code snippets
  6. stream_with_context()   → Groq streams answer (model per intent)
  7. SSE sources event       → per-chunk citations (filepath:L{start}-L{end})

Endpoints:
  POST /auth/register
  POST /auth/login
  POST /ingest          (JWT required)
  POST /kb/add          (JWT required)
  POST /ingest/github   (JWT required)
  POST /query           (JWT required) → SSE stream
  GET  /sources         (JWT required)
  DELETE /sources/{id}  (JWT required)
  GET  /metrics
  GET  /health
"""

import json
import os
import time
import uuid
import warnings
from contextlib import asynccontextmanager

# Suppress noisy third-party deprecation/upgrade warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="weaviate")
warnings.filterwarnings("ignore", category=UserWarning,        module="weaviate")

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5 as wv_uuid

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from auth import create_token, get_current_user, hash_password, verify_password
from chunker import chunk_by_type, chunk_code
from fetcher_github import fetch_repo
from database import (
    create_user, delete_document, get_user_by_username,
    get_user_documents, init_db, upsert_document,
)
from embedder import embed_query, embed_texts, EMBED_DIM
from fetcher import fetch_and_extract
from retriever import hybrid_retrieve, rerank_hits, CANDIDATE_K, RERANK_TOP_N
from llm import (
    build_generation_kwargs,
    stream_with_context,
    get_model_for_intent,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GEN_FREQUENCY_PENALTY,
    GEN_MAX_OUTPUT_TOKENS,
    GROQ_MODEL,
    SYSTEM_PROMPT,
)
from query_processor import process_query
import metrics as _metrics

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION       = "KnowledgeChunk"
MAX_HISTORY_TURNS = 12   # messages (= 6 full exchanges)

# ── Weaviate client ───────────────────────────────────────────────────────────

wv: weaviate.WeaviateClient | None = None

# ── Lazy Groq client for HyDE ─────────────────────────────────────────────────

_groq_client = None

def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global wv
    init_db()

    weaviate_url = os.environ.get("WEAVIATE_URL", "").strip()
    weaviate_key = os.environ.get("WEAVIATE_API_KEY", "").strip()

    if weaviate_url:
        wv = weaviate.connect_to_weaviate_cloud(
            cluster_url      = weaviate_url,
            auth_credentials = weaviate.auth.AuthApiKey(weaviate_key),
        )
        print(f"[startup] Weaviate Cloud connected: {weaviate_url}")
    else:
        wv = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
        print("[startup] Weaviate local (Docker) connected.")

    _ensure_collection()
    print("[startup] DB ready.")
    yield

    if wv is not None:
        wv.close()
        wv = None


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="PageMind API", version="8.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ── Schema ────────────────────────────────────────────────────────────────────

def _ensure_collection() -> None:
    """
    Schema v5-code:
      HNSW  efConstruction=256, maxConnections=16, ef=128
      BM25  k1=1.5, b=0.4  (code-optimized: lower b reduces length bias)
      New props: start_line (INT), end_line (INT), chunk_type (TEXT)
      GitHub props: filepath, language, repo

    Migration strategy:
      - Missing BM25 → drop + recreate (unavoidable — index config is immutable)
      - BM25 present  → add missing props non-destructively via add_property()
    """
    _SCHEMA_VERSION = "v6-small"   # bge-small-en-v1.5 (384-dim)

    if wv.collections.exists(COLLECTION):
        col = wv.collections.get(COLLECTION)
        try:
            cfg        = col.config.get()
            prop_names = {p.name for p in cfg.properties}
            text_prop  = next((p for p in cfg.properties if p.name == "text"), None)
            bm25_ok    = (
                text_prop is not None
                and getattr(text_prop, "index_searchable", False) is True
            )
            # Check vector dimension by sampling an existing object's vector
            dim_ok = True
            try:
                sample = col.query.fetch_objects(limit=1, include_vector=True)
                if sample.objects:
                    vec = sample.objects[0].vector
                    if isinstance(vec, dict):
                        vec = next(iter(vec.values()), None)
                    if vec is not None and len(vec) != EMBED_DIM:
                        dim_ok = False
            except Exception:
                pass
        except Exception:
            bm25_ok    = False
            dim_ok     = False
            prop_names = set()

        if not bm25_ok or not dim_ok:
            reason = "BM25 not enabled" if not bm25_ok else f"vector dim mismatch (need {EMBED_DIM})"
            print(
                f"\n[migration] Upgrading '{COLLECTION}' to {_SCHEMA_VERSION}.\n"
                f"            Reason: {reason} — dropping and recreating.\n"
                f"            All chunks cleared — please re-index your sources.\n"
            )
            wv.collections.delete(COLLECTION)
        else:
            # Non-destructive: add missing properties
            _new_text_props = {
                "filepath":   (DataType.TEXT, True,  False),
                "language":   (DataType.TEXT, True,  False),
                "repo":       (DataType.TEXT, True,  False),
                "chunk_type": (DataType.TEXT, False, False),
            }
            _new_int_props = {
                "start_line": DataType.INT,
                "end_line":   DataType.INT,
            }
            added = []
            for pname, (dtype, filterable, searchable) in _new_text_props.items():
                if pname not in prop_names:
                    try:
                        col.config.add_property(Property(
                            name=pname, data_type=dtype,
                            index_filterable=filterable,
                            index_searchable=searchable,
                        ))
                        added.append(pname)
                    except Exception:
                        pass
            for pname, dtype in _new_int_props.items():
                if pname not in prop_names:
                    try:
                        col.config.add_property(Property(
                            name=pname, data_type=dtype,
                        ))
                        added.append(pname)
                    except Exception:
                        pass
            if added:
                print(f"[migration] Added properties to existing collection: {added}")
            return

    # ── Create fresh collection ───────────────────────────────────────────────
    wv.collections.create(
        name=COLLECTION,

        vectorizer_config=Configure.Vectorizer.none(),

        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric  = VectorDistances.COSINE,
            ef_construction  = 256,   # higher build quality for code retrieval
            max_connections  = 16,    # M — lower = less memory, still high recall
            ef               = 128,   # query beam width (raise at > 100K vecs)
        ),

        # BM25 params: k1=1.5 (moderate TF saturation), b=0.4 (weak length norm)
        # Code files vary wildly in length; low b prevents long files dominating.
        inverted_index_config=Configure.inverted_index(
            bm25_k1=1.5,
            bm25_b=0.4,
        ),

        properties=[
            # ── Metadata-filter-only ──────────────────────────────────────────
            Property(name="user_id",      data_type=DataType.TEXT,
                     index_filterable=True,  index_searchable=False),
            Property(name="doc_id",       data_type=DataType.TEXT,
                     index_filterable=True,  index_searchable=False),
            Property(name="source_type",  data_type=DataType.TEXT,
                     index_filterable=True,  index_searchable=False),
            Property(name="content_type", data_type=DataType.TEXT,
                     index_filterable=True,  index_searchable=False),
            Property(name="chunk_index",  data_type=DataType.INT),
            Property(name="url",          data_type=DataType.TEXT,
                     index_filterable=False, index_searchable=False),

            # ── BM25 sparse index ─────────────────────────────────────────────
            Property(name="title",        data_type=DataType.TEXT,
                     index_filterable=False, index_searchable=True),
            Property(name="heading_path", data_type=DataType.TEXT,
                     index_filterable=False, index_searchable=True),
            Property(name="text",         data_type=DataType.TEXT,
                     index_filterable=False, index_searchable=True),

            # ── GitHub repo metadata ──────────────────────────────────────────
            Property(name="filepath",   data_type=DataType.TEXT,
                     index_filterable=True,  index_searchable=False),
            Property(name="language",   data_type=DataType.TEXT,
                     index_filterable=True,  index_searchable=False),
            Property(name="repo",       data_type=DataType.TEXT,
                     index_filterable=True,  index_searchable=False),
            Property(name="chunk_type", data_type=DataType.TEXT,
                     index_filterable=False, index_searchable=False),

            # ── Code line citations ───────────────────────────────────────────
            Property(name="start_line", data_type=DataType.INT),
            Property(name="end_line",   data_type=DataType.INT),
        ],
    )
    print(
        f"[startup] '{COLLECTION}' ({_SCHEMA_VERSION}) created.\n"
        f"          Embed : bge-small-en-v1.5  dim={EMBED_DIM}\n"
        f"          Dense : HNSW cosine  efConstruction=256  M=16  ef=128\n"
        f"          Sparse: BM25 k1=1.5 b=0.4  on text, title, heading_path\n"
        f"          Search: query.hybrid(alpha=0.7, fusion=RANKED, candidates={CANDIDATE_K})\n"
        f"          Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2  top={RERANK_TOP_N}"
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str

class IngestRequest(BaseModel):
    text:         str
    title:        str
    url:          str
    source_type:  str = "webpage"
    content_type: str = "general"
    sections:     list[dict] | None = None

class IngestResponse(BaseModel):
    doc_id:        str
    chunks_stored: int
    status:        str
    content_type:  str

class KBAddRequest(BaseModel):
    url: str

class KBAddResponse(BaseModel):
    doc_id:        str
    chunks_stored: int
    status:        str
    content_type:  str
    title:         str

class GitHubIngestRequest(BaseModel):
    repo_url:     str
    github_token: str | None = None

class GitHubIngestResponse(BaseModel):
    doc_id:        str
    chunks_stored: int
    files_indexed: int
    status:        str
    repo:          str

class QueryRequest(BaseModel):
    question:   str
    source_ids: list[str] | None = None
    top_k:      int = RERANK_TOP_N
    history:    list[dict] | None = None

class SourceItem(BaseModel):
    doc_id:       str
    title:        str
    url:          str
    chunk_count:  int
    created_at:   str
    source_type:  str
    content_type: str = "general"


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse)
def register(body: RegisterRequest):
    body.username = body.username.strip().lower()
    if len(body.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if get_user_by_username(body.username):
        raise HTTPException(400, "Username already taken.")
    user_id = create_user(body.username, hash_password(body.password))
    return TokenResponse(access_token=create_token(user_id), username=body.username)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    body.username = body.username.strip().lower()
    user = get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid username or password.")
    return TokenResponse(access_token=create_token(user["id"]), username=user["username"])


# ── Shared batch-insert helper ────────────────────────────────────────────────

def _batch_insert(
    collection,
    chunks:       list[dict],
    embeddings:   list[list[float]],
    user_id:      str,
    doc_id:       str,
    title:        str,
    url:          str,
    source_type:  str,
    content_type: str,
) -> int:
    try:
        collection.data.delete_many(
            where=(
                Filter.by_property("user_id").equal(user_id) &
                Filter.by_property("doc_id").equal(doc_id)
            )
        )
    except Exception:
        pass

    inserted = 0
    with collection.batch.fixed_size(batch_size=200) as batch:
        for chunk, embedding in zip(chunks, embeddings):
            batch.add_object(
                properties={
                    "user_id":      user_id,
                    "doc_id":       doc_id,
                    "title":        title,
                    "url":          url,
                    "source_type":  source_type,
                    "content_type": chunk.get("content_type", content_type),
                    "chunk_index":  chunk["index"],
                    "heading_path": chunk.get("heading_path", ""),
                    "text":         chunk["text"],
                },
                vector=embedding,
                uuid=wv_uuid(f"{doc_id}:{chunk['index']}"),
            )
            inserted += 1
            if getattr(batch, "number_errors", 0) > 10:
                print(f"[batch] Aborting after 10 errors on doc {doc_id}")
                break

    return inserted


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest, user_id: str = Depends(get_current_user)):
    t0 = time.time()
    print(f"\n[ingest] START  user={user_id}  title={body.title[:60]}")

    chunks = chunk_by_type(
        text=body.text,
        source=body.url,
        content_type=body.content_type,
        title=body.title,
        sections=body.sections,
    )
    if not chunks:
        raise HTTPException(400, "Could not extract any content from the provided text.")

    doc_id     = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{body.url}"))
    collection = wv.collections.get(COLLECTION)
    embeddings = embed_texts([c["text"] for c in chunks])
    inserted   = _batch_insert(
        collection, chunks, embeddings,
        user_id, doc_id, body.title, body.url,
        body.source_type, body.content_type,
    )
    upsert_document(
        doc_id, user_id, body.title, body.url,
        body.source_type, inserted, body.content_type,
    )
    print(f"[ingest] DONE — {inserted} chunks | {time.time()-t0:.2f}s")
    return IngestResponse(doc_id=doc_id, chunks_stored=inserted,
                          status="success", content_type=body.content_type)


# ── KB Add (server-side URL fetch) ───────────────────────────────────────────

@app.post("/kb/add", response_model=KBAddResponse)
def kb_add(body: KBAddRequest, user_id: str = Depends(get_current_user)):
    import requests as _req
    t0 = time.time()
    print(f"\n[kb/add] START  user={user_id}  url={body.url[:60]}")

    try:
        page = fetch_and_extract(body.url)
    except _req.exceptions.HTTPError as exc:
        raise HTTPException(400, f"HTTP error fetching URL: {exc}")
    except _req.exceptions.ConnectionError:
        raise HTTPException(400, "Cannot connect to the URL.")
    except _req.exceptions.Timeout:
        raise HTTPException(400, "URL timed out (> 15 s).")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Fetch error: {exc}")

    if not page["text"] or len(page["text"].strip()) < 100:
        raise HTTPException(400, "Not enough readable content at this URL.")

    chunks = chunk_by_type(
        text=page["text"], source=body.url,
        content_type=page["content_type"], title=page["title"],
        sections=page["sections"],
    )
    if not chunks:
        raise HTTPException(400, "Could not extract usable content from this URL.")

    doc_id     = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{body.url}"))
    collection = wv.collections.get(COLLECTION)
    embeddings = embed_texts([c["text"] for c in chunks])
    inserted   = _batch_insert(
        collection, chunks, embeddings,
        user_id, doc_id, page["title"], body.url,
        "url", page["content_type"],
    )
    upsert_document(doc_id, user_id, page["title"], body.url, "url", inserted, page["content_type"])
    print(f"[kb/add] DONE — {inserted} chunks | {time.time()-t0:.2f}s")
    return KBAddResponse(doc_id=doc_id, chunks_stored=inserted,
                         status="success", content_type=page["content_type"],
                         title=page["title"])


# ── GitHub repo ingestion ─────────────────────────────────────────────────────

def _batch_insert_github(
    collection,
    chunks:     list[dict],
    embeddings: list[list[float]],
    user_id:    str,
    doc_id:     str,
    title:      str,
    repo_url:   str,
    repo_slug:  str,
) -> int:
    """Insert GitHub code chunks, preserving filepath/start_line/end_line/chunk_type."""
    try:
        collection.data.delete_many(
            where=(
                Filter.by_property("user_id").equal(user_id) &
                Filter.by_property("doc_id").equal(doc_id)
            )
        )
    except Exception:
        pass

    inserted = 0
    with collection.batch.fixed_size(batch_size=200) as batch:
        for chunk, embedding in zip(chunks, embeddings):
            batch.add_object(
                properties={
                    "user_id":      user_id,
                    "doc_id":       doc_id,
                    "title":        title,
                    "url":          chunk.get("source", repo_url),
                    "source_type":  "github",
                    "content_type": chunk.get("content_type", "code"),
                    "chunk_index":  chunk["index"],
                    "heading_path": chunk.get("heading_path", ""),
                    "text":         chunk["text"],
                    "filepath":     chunk.get("filepath", ""),
                    "language":     chunk.get("language", ""),
                    "repo":         repo_slug,
                    "chunk_type":   chunk.get("chunk_type", "block"),
                    "start_line":   chunk.get("start_line", 0),
                    "end_line":     chunk.get("end_line", 0),
                },
                vector=embedding,
                uuid=wv_uuid(f"{doc_id}:{chunk['index']}"),
            )
            inserted += 1
            if getattr(batch, "number_errors", 0) > 10:
                print(f"[batch] Aborting after 10 errors on GitHub doc {doc_id}")
                break

    return inserted


@app.post("/ingest/github", response_model=GitHubIngestResponse)
def ingest_github(body: GitHubIngestRequest, user_id: str = Depends(get_current_user)):
    t0 = time.time()
    print(f"\n[github] START  user={user_id}  repo={body.repo_url}")

    m = _metrics.RequestMetrics()

    # Step 1 — Fetch repo
    t_fetch = time.time()
    try:
        files = fetch_repo(body.repo_url, body.github_token)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"GitHub fetch failed: {exc}")

    m.github_fetch_ms = (time.time() - t_fetch) * 1000

    if not files:
        raise HTTPException(400, "No indexable files found in this repository.")

    repo_slug = files[0]["repo"]
    print(f"  → {len(files)} files fetched in {m.github_fetch_ms:.0f}ms")

    # Step 2 — Chunk
    all_chunks: list[dict] = []
    for f in files:
        chunks = chunk_code(
            content=f["content"], filepath=f["filepath"],
            language=f["language"], repo=repo_slug,
        )
        all_chunks.extend(chunks)
    print(f"  → {len(all_chunks)} chunks from {len(files)} files")

    if not all_chunks:
        raise HTTPException(400, "No content could be extracted from this repository.")

    doc_id     = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:github:{repo_slug}"))
    collection = wv.collections.get(COLLECTION)
    repo_title = f"GitHub: {repo_slug}"
    repo_url   = f"https://github.com/{repo_slug}"

    # Step 3 — Embed
    t_emb = time.time()
    embeddings = embed_texts([c["text"] for c in all_chunks])
    m.embed_ms = (time.time() - t_emb) * 1000
    print(f"  → embedded in {m.embed_ms:.0f}ms")

    # Step 4 — Insert
    inserted = _batch_insert_github(
        collection, all_chunks, embeddings,
        user_id, doc_id, repo_title, repo_url, repo_slug,
    )
    upsert_document(doc_id, user_id, repo_title, repo_url, "github", inserted, "code")

    m.total_ms = (time.time() - t0) * 1000
    print(f"[github] DONE — {len(files)} files | {inserted} chunks | {m.total_ms:.0f}ms")

    return GitHubIngestResponse(
        doc_id=doc_id, chunks_stored=inserted,
        files_indexed=len(files), status="success", repo=repo_slug,
    )


# ── Query (SSE streaming) ─────────────────────────────────────────────────────

@app.post("/query")
def query(body: QueryRequest, user_id: str = Depends(get_current_user)):
    """
    Code-optimized hybrid RAG pipeline:

    1. Intent detection  → "explain" | "design" | "find" | "general"
    2. HyDE              → hypothetical code snippet for "explain" intent
    3. embed_query()     → 1024-dim bge-m3  (HyDE text if use_hyde)
    4. hybrid_retrieve() → BM25 + HNSW, top CANDIDATE_K=20 candidates
    5. rerank_hits()     → cross-encoder reranks → top RERANK_TOP_N=8
    6. Build context     → "[filepath | chunk_type fn]\\n{code}" per hit
    7. stream_with_context() → Groq (model per intent), streams tokens
    8. SSE sources       → per-chunk citations with filepath + line numbers
    """
    t0  = time.time()
    m   = _metrics.RequestMetrics(question=body.question[:120])

    # ── Step 1: Intent + HyDE ─────────────────────────────────────────────────
    t_hyde = time.time()
    qproc  = process_query(body.question, _get_groq())
    intent    = qproc["intent"]
    use_hyde  = qproc["use_hyde"]
    hyde_text = qproc["hyde_text"]
    m.intent    = intent
    m.hyde_ms   = (time.time() - t_hyde) * 1000 if use_hyde else 0.0
    _ms = m.hyde_ms
    print(f"\n[query] q={body.question[:80]!r}")
    print(f"  1. intent/HyDE   {_ms:7.0f} ms  intent={intent}  use_hyde={use_hyde}")

    # ── Step 2: Embed (use HyDE text for dense arm if applicable) ─────────────
    t_emb = time.time()
    try:
        embed_input = hyde_text if use_hyde else body.question
        q_vector    = embed_query(embed_input)
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {e}")
    m.embed_ms = (time.time() - t_emb) * 1000
    print(f"  2. embed (BGE-M3) {m.embed_ms:7.0f} ms")

    # ── Step 3: Hybrid retrieval (BM25 always uses original question) ─────────
    t_ann = time.time()
    try:
        collection = wv.collections.get(COLLECTION)
        hits = hybrid_retrieve(
            collection   = collection,
            query_text   = body.question,   # BM25 sparse arm — original question
            query_vector = q_vector,        # HNSW dense arm  — HyDE or question
            user_id      = user_id,
            source_ids   = body.source_ids or None,
            top_k        = CANDIDATE_K,
        )
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {e}")
    m.ann_ms         = (time.time() - t_ann) * 1000
    m.chunks_retrieved = len(hits)
    print(f"  3. hybrid search  {m.ann_ms:7.0f} ms  hits={len(hits)}")

    # ── Empty knowledge base ──────────────────────────────────────────────────
    if not hits:
        def _empty():
            msg = (
                "My knowledge base is empty. "
                "Navigate to a GitHub repository and click **Index this repo**, "
                "then ask me anything about its code."
            )
            yield f"data: {json.dumps({'text': msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # ── Step 4: Cross-encoder reranking ──────────────────────────────────────
    t_rerank = time.time()
    hits = rerank_hits(hits, body.question, top_n=RERANK_TOP_N)
    m.rerank_ms      = (time.time() - t_rerank) * 1000
    m.chunks_reranked = len(hits)
    print(f"  4. rerank (CE)    {m.rerank_ms:7.0f} ms  kept={len(hits)}")

    # ── Step 5: Build RAG context + deduplicated chunk-level citations ─────────
    context_parts: list[str] = []
    citations:     list[dict] = []

    for hit in hits:
        label  = hit.get("heading_path") or hit.get("title") or hit.get("url") or "Source"
        context_parts.append(f"[{label}]\n{hit['text']}")

        filepath   = hit.get("filepath", "")
        start_line = hit.get("start_line") or 0
        end_line   = hit.get("end_line")   or 0
        repo       = hit.get("repo", "")

        # Build clickable GitHub blob URL with line anchor
        if filepath and start_line:
            blob_url = (
                f"https://github.com/{repo}/blob/HEAD/{filepath}"
                f"#L{start_line}"
                if repo else hit.get("url", "")
            )
        else:
            blob_url = hit.get("url", "")

        citations.append({
            "doc_id":       hit.get("doc_id", ""),
            "title":        hit.get("title", ""),
            "url":          blob_url,
            "content_type": hit.get("content_type", "general"),
            "score":        hit.get("_rerank_score", hit.get("_hybrid_score", 0.0)),
            "filepath":     filepath,
            "start_line":   start_line if start_line else None,
            "end_line":     end_line   if end_line   else None,
            "chunk_type":   hit.get("chunk_type", ""),
            "repo":         repo,
        })

    rag_context = "\n\n---\n\n".join(context_parts)

    # ── Step 6: Build conversation context ────────────────────────────────────
    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Repository code context:\n\n{rag_context}"
    )
    conversation: list[dict] = [{"role": "system", "content": system_content}]

    if body.history:
        valid_turns = [
            t for t in body.history
            if isinstance(t, dict)
            and t.get("role") in ("user", "assistant")
            and isinstance(t.get("content"), str)
            and t["content"].strip()
        ]
        for turn in valid_turns[-MAX_HISTORY_TURNS:]:
            conversation.append({"role": turn["role"], "content": turn["content"]})

    # ── Step 7: Stream with intent-routed model ───────────────────────────────
    model      = get_model_for_intent(intent)
    m.model_used = model
    gen_kwargs = build_generation_kwargs(
        temperature    = GEN_TEMPERATURE,
        top_p          = GEN_TOP_P,
        max_new_tokens = GEN_MAX_OUTPUT_TOKENS,
    )

    def stream_answer():
        print(f"  5. LLM stream     model={model.split('-')[0]}…")
        t_llm = time.time()
        try:
            for text_chunk in stream_with_context(
                prompt  = body.question,
                context = conversation,
                role    = "user",
                model   = model,
                **gen_kwargs,
            ):
                yield f"data: {json.dumps({'text': text_chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        m.llm_ms   = (time.time() - t_llm) * 1000
        m.total_ms = (time.time() - t0) * 1000
        _metrics.record(m)

        # ── Latency summary ───────────────────────────────────────────────────
        pre_llm = m.hyde_ms + m.embed_ms + m.ann_ms + m.rerank_ms
        print(f"  5. LLM stream     {m.llm_ms:7.0f} ms")
        print(f"  {'─'*40}")
        print(f"  {'pre-LLM total':<18} {pre_llm:7.0f} ms  "
              f"(HyDE:{m.hyde_ms:.0f} + embed:{m.embed_ms:.0f} "
              f"+ search:{m.ann_ms:.0f} + rerank:{m.rerank_ms:.0f})")
        print(f"  {'TOTAL (wall)':<18} {m.total_ms:7.0f} ms\n")

        yield f"data: {json.dumps({'sources': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_answer(), media_type="text/event-stream")


# ── Source management ─────────────────────────────────────────────────────────

@app.get("/sources")
def list_sources(user_id: str = Depends(get_current_user)):
    docs = get_user_documents(user_id)
    return {
        "sources": [
            SourceItem(
                doc_id=d["doc_id"],
                title=d["title"] or "Untitled",
                url=d["url"] or "",
                chunk_count=d["chunk_count"],
                created_at=d["created_at"],
                source_type=d["source_type"],
                content_type=d.get("content_type", "general"),
            )
            for d in docs
        ]
    }


@app.delete("/sources/{doc_id}")
def delete_source(doc_id: str, user_id: str = Depends(get_current_user)):
    try:
        collection = wv.collections.get(COLLECTION)
        collection.data.delete_many(
            where=(
                Filter.by_property("user_id").equal(user_id) &
                Filter.by_property("doc_id").equal(doc_id)
            )
        )
    except Exception:
        pass

    if not delete_document(doc_id, user_id):
        raise HTTPException(404, "Source not found.")

    return {"deleted": True, "doc_id": doc_id}


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/metrics")
def get_metrics():
    """Return aggregated latency stats + recent request records."""
    return {
        "summary": _metrics.summary(),
        "recent":  _metrics.get_recent(20),
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        collection = wv.collections.get(COLLECTION)
        agg        = collection.aggregate.over_all(total_count=True)
        vec_count  = agg.total_count or 0
    except Exception:
        vec_count  = 0

    return {
        "status":   "ok",
        "pipeline": {
            "step1_db":       "Weaviate  (Docker or Cloud)",
            "step2_load":     "batch.fixed_size(200)",
            "step3_sparse":   "BM25 k1=1.5 b=0.4  (text, title, heading_path)",
            "step4_dense":    f"HNSW cosine  bge-m3 {EMBED_DIM}-dim  efC=256 M=16 ef=128",
            "step5_hybrid":   f"query.hybrid alpha=0.7 fusion=RANKED candidates={CANDIDATE_K}",
            "step6_rerank":   f"cross-encoder/ms-marco-MiniLM-L-6-v2  top={RERANK_TOP_N}",
        },
        "llm": {
            "provider":            "Groq",
            "model_general":       GROQ_MODEL,
            "model_reasoning":     "deepseek-r1-distill-llama-70b",
            "temperature":         GEN_TEMPERATURE,
            "top_p":               GEN_TOP_P,
            "frequency_penalty":   GEN_FREQUENCY_PENALTY,
            "max_tokens":          GEN_MAX_OUTPUT_TOKENS,
            "max_history_turns":   MAX_HISTORY_TURNS,
        },
        "collection": COLLECTION,
        "vectors":    vec_count,
        "metrics":    _metrics.summary(),
    }
