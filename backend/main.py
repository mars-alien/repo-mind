"""
PageMind API v7.0 — Groq LLM + Context-Aware Conversation
───────────────────────────────────────────────────────────
Stack:
  • FastAPI                – HTTP + SSE streaming
  • Weaviate (Docker)      – HNSW dense + BM25 sparse + hybrid search
  • Groq (llama-3.3-70b)  – LLM generation  (free tier, ~250 tok/s)
  • sentence-transformers  – BAAI/bge-m3 dense embeddings (1024-dim)
  • SQLite                 – user accounts + document metadata
  • JWT / bcrypt           – authentication
  • requests + lxml/BS4    – server-side URL fetching (internet KB)

Weaviate 5-step pipeline per query:
  1. DB Setup        Docker container on :8080 / :50051  (persistent volume)
  2. Load Documents  batch.fixed_size(200)  — ingest & /kb/add routes
  3. Sparse Vectors  BM25 inverted index  (index_searchable=True)
  4. Dense Vectors   bge-m3 1024-dim HNSW  (bring-your-own vector)
  5. Search          query.hybrid(alpha=0.7)  — single fused call

LLM Conversation pipeline (/query):
  1. embed_query()           → 1024-dim bge-m3 dense vector
  2. hybrid_retrieve()       → BM25 + HNSW fused by Weaviate (15 hits)
  3. Build RAG context       → heading-prefixed text snippets
  4. Build conversation      → system(RAG) + prior history + current question
  5. build_generation_kwargs → flexible parameter dict
  6. stream_with_context()   → Groq streams answer, context updated in-place
  7. SSE sources event       → deduplicated doc citations

Endpoints:
  POST /auth/register
  POST /auth/login
  POST /ingest          (JWT required) — index current browser tab text
  POST /kb/add          (JWT required) — fetch & index any URL server-side
  POST /query           (JWT required) → SSE stream  (hybrid retrieval)
  GET  /sources         (JWT required)
  DELETE /sources/{id}  (JWT required)
  GET  /health
"""

import json
import os
import uuid
from contextlib import asynccontextmanager

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
from chunker import chunk_by_type
from database import (
    create_user, delete_document, get_user_by_username,
    get_user_documents, init_db, upsert_document,
)
from embedder import embed_query, embed_texts, EMBED_DIM
from fetcher import fetch_and_extract
from retriever import hybrid_retrieve, TOP_K as RETRIEVER_TOP_K
from llm import (
    build_generation_kwargs,
    call_llm_with_context,
    stream_with_context,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GEN_FREQUENCY_PENALTY,
    GEN_MAX_OUTPUT_TOKENS,
    GROQ_MODEL,
)

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION = "KnowledgeChunk"

# Maximum prior conversation turns to inject into the LLM context window.
# Each exchange = 2 messages (user + assistant). 6 exchanges = 12 messages.
# Keeps the context focused without overflowing the 128K token window.
MAX_HISTORY_TURNS = 12   # messages (= 6 full exchanges)

SYSTEM_PROMPT = (
    "You are a precise research assistant called PageMind.\n"
    "Answer ONLY using the provided knowledge base context snippets.\n"
    "If the context does not contain enough information, say so clearly.\n"
    "Be concise. When citing facts, reference the source title if available.\n"
    "Do not hallucinate or add information not present in the context.\n"
    "You may use prior conversation turns to understand follow-up questions."
)

# ── Weaviate client (connected in lifespan, NOT at module level) ──────────────

wv: weaviate.WeaviateClient | None = None


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup → yield → shutdown. Modern FastAPI pattern."""
    global wv
    # ── Startup ───────────────────────────────────────────────────────────────
    init_db()

    weaviate_url = os.environ.get("WEAVIATE_URL", "").strip()
    weaviate_key = os.environ.get("WEAVIATE_API_KEY", "").strip()

    if weaviate_url:
        # ── Cloud / remote Weaviate (Render, Railway, etc.) ───────────────────
        wv = weaviate.connect_to_weaviate_cloud(
            cluster_url      = weaviate_url,
            auth_credentials = weaviate.auth.AuthApiKey(weaviate_key),
        )
        print(f"[startup] Weaviate Cloud connected: {weaviate_url}")
    else:
        # ── Local Docker Weaviate (development) ───────────────────────────────
        wv = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051)
        print("[startup] Weaviate local (Docker) connected.")

    _ensure_collection()
    print("[startup] DB ready.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    if wv is not None:
        wv.close()
        wv = None


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="PageMind API", version="7.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_collection() -> None:
    """
    Weaviate DB Setup — Steps 1-4 of the hybrid-search pipeline.

    Step 1  DB Setup
            Weaviate runs in Docker (docker-compose.yml at project root).
            Connect via weaviate.connect_to_local() in the lifespan handler.
            This function handles schema creation / migration.

    Step 2  Load Documents
            batch.fixed_size(batch_size=200) used in /ingest and /kb/add.
            Weaviate batches HNSW insertions and BM25 index updates together.

    Step 3  Sparse Vector (BM25)
            Properties with index_searchable=True get an inverted index
            automatically. Weaviate tokenises, stems and stores term
            frequencies as a sparse BM25 representation at insert time.
            Fields:  text  (weight 1.0)
                     title (weight 2.0 at query time via title^2)
                     heading_path (weight 1.5 at query time)

    Step 4  Dense Vector (HNSW / ANN)
            vectorizer_config = none()  →  bring-your-own vector.
            We compute 1024-dim bge-m3 embeddings externally and pass
            vector=embedding to batch.add_object().
            Weaviate builds the HNSW graph with:
              ef_construction = 128   (build quality)
              max_connections = 32    (M parameter — graph degree)
              ef              = 64    (query-time beam width)
            These give recall ~0.98 on collections up to ~500 K vectors.

    Step 5  Search (query.hybrid) is in retriever.py.

    Schema migration
    ────────────────
    If the collection exists without BM25 indexes (v1 schema), it is
    dropped and recreated here.  A console warning instructs the user
    to re-index their content.
    """
    _SCHEMA_VERSION = "v3-hnsw-bm25"

    if wv.collections.exists(COLLECTION):
        # ── Detect v1 schema (BM25 not enabled on text property) ────────────
        try:
            cfg       = wv.collections.get(COLLECTION).config.get()
            text_prop = next((p for p in cfg.properties if p.name == "text"), None)
            bm25_ok   = (
                text_prop is not None and
                getattr(text_prop, "index_searchable", False) is True
            )
        except Exception:
            bm25_ok = False   # unreadable config → recreate safely

        if bm25_ok:
            return   # schema is current, nothing to do

        print(
            f"\n[migration] Upgrading '{COLLECTION}' to {_SCHEMA_VERSION}.\n"
            f"            Adds HNSW explicit params + BM25 inverted index.\n"
            f"            All chunks cleared — please re-index your sources.\n"
        )
        wv.collections.delete(COLLECTION)

    # ── Step 1 + 3 + 4: Create collection ────────────────────────────────────
    wv.collections.create(
        name=COLLECTION,

        # Step 4a — no built-in vectorizer; we supply bge-m3 vectors ourselves
        vectorizer_config=Configure.Vectorizer.none(),

        # Step 4b — HNSW index for Approximate Nearest Neighbour (ANN) search
        #           ef_construction: controls build-time recall quality
        #           max_connections: node degree M in the HNSW graph
        #           ef:              query-time beam width (recall vs latency)
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric  = VectorDistances.COSINE,
            ef_construction  = 128,   # higher = better index, slower build
            max_connections  = 32,    # M = 32, memory/recall balance
            ef               = 64,    # query recall; raise to 128 for > 100K vecs
        ),

        properties=[
            # ── Metadata-filter-only (no text search needed) ──────────────────
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

            # ── Step 3: BM25 sparse index (index_searchable = True) ───────────
            # Weaviate tokenises these fields and stores term frequencies.
            # query.hybrid() uses these for the keyword arm of hybrid search.
            Property(name="title",        data_type=DataType.TEXT,
                     index_filterable=False, index_searchable=True),
            Property(name="heading_path", data_type=DataType.TEXT,
                     index_filterable=False, index_searchable=True),
            Property(name="text",         data_type=DataType.TEXT,
                     index_filterable=False, index_searchable=True),
        ],
    )
    print(
        f"[startup] '{COLLECTION}' ({_SCHEMA_VERSION}) created.\n"
        f"          Dense : HNSW cosine  ef_construction=128  M=32  ef=64\n"
        f"          Sparse: BM25 on text, title, heading_path\n"
        f"          Search: query.hybrid(alpha=0.7, fusion=RANKED, limit=15)"
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
    token_type: str = "bearer"
    username: str

class IngestRequest(BaseModel):
    text:         str
    title:        str
    url:          str
    source_type:  str = "webpage"
    content_type: str = "general"           # "docs" | "wiki" | "general"
    sections:     list[dict] | None = None  # [{heading, path, text}] from browser

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

class QueryRequest(BaseModel):
    question:   str
    source_ids: list[str] | None = None
    top_k:      int = 6
    # ── Conversation history ──────────────────────────────────────────────────
    # Prior turns sent from the frontend in OpenAI chat format.
    # Each element: {"role": "user"|"assistant", "content": "..."}
    # Injected into the LLM context so follow-up questions work correctly.
    # None / [] → stateless single-turn mode (backward-compatible default).
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
    """
    Step 2 — Load Documents (Weaviate hybrid pipeline).

    Inserts chunks into Weaviate using fixed_size batching (batch_size=200).
    Each object carries:
      • properties dict  → stored for retrieval context
      • vector           → 1024-dim bge-m3 dense embedding  (Step 4 / HNSW)
      • uuid             → deterministic  doc_id:chunk_index

    BM25 sparse vectors (Step 3) are built automatically by Weaviate for
    all properties marked  index_searchable=True  (text, title, heading_path).

    Returns the number of successfully inserted objects.
    """
    # ── Delete stale chunks first (idempotent re-index) ──────────────────────
    try:
        collection.data.delete_many(
            where=(
                Filter.by_property("user_id").equal(user_id) &
                Filter.by_property("doc_id").equal(doc_id)
            )
        )
    except Exception:
        pass   # empty collection on first run is fine

    inserted = 0
    errors   = 0

    # ── fixed_size(200): consistent batch size, easy error tracking ──────────
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
            # Abort if too many batch errors (Weaviate schema / network issue)
            if getattr(batch, "number_errors", 0) > 10:
                print(f"[batch] Aborting after 10 errors on doc {doc_id}")
                errors = batch.number_errors
                break

    if errors:
        print(f"[batch] {errors} failed objects for doc {doc_id}")

    return inserted


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest, user_id: str = Depends(get_current_user)):
    """
    Browser tab ingestion pipeline:
      1. chunk_by_type()    → docs/wiki/general chunking strategy
      2. embed_texts()      → 1024-dim bge-m3 dense vectors  (Step 4)
      3. _batch_insert()    → fixed_size(200) batch into Weaviate
                              BM25 sparse vectors built automatically  (Step 3)
      4. upsert_document()  → SQLite metadata record
    """
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

    inserted = _batch_insert(
        collection, chunks, embeddings,
        user_id, doc_id, body.title, body.url,
        body.source_type, body.content_type,
    )

    upsert_document(
        doc_id, user_id, body.title, body.url,
        body.source_type, inserted, body.content_type,
    )

    return IngestResponse(
        doc_id=doc_id,
        chunks_stored=inserted,
        status="success",
        content_type=body.content_type,
    )


# ── KB Add (server-side URL fetch) ───────────────────────────────────────────

@app.post("/kb/add", response_model=KBAddResponse)
def kb_add(body: KBAddRequest, user_id: str = Depends(get_current_user)):
    """
    Internet knowledge-base ingestion.

    The backend fetches the URL directly (no browser needed), extracts
    structured content, detects content type, chunks and embeds it, then
    stores vectors in Weaviate and metadata in SQLite.

    Idempotent: re-adding the same URL replaces the old chunks.
    """
    import requests as _req

    # ── Fetch & extract ──
    try:
        page = fetch_and_extract(body.url)
    except _req.exceptions.HTTPError as exc:
        raise HTTPException(400, f"HTTP error fetching URL: {exc}")
    except _req.exceptions.ConnectionError:
        raise HTTPException(400, "Cannot connect to the URL. Check if it is publicly accessible.")
    except _req.exceptions.Timeout:
        raise HTTPException(400, "The URL took too long to respond (> 15 s). Try again later.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Fetch error: {exc}")

    if not page["text"] or len(page["text"].strip()) < 100:
        raise HTTPException(400, "Not enough readable content at this URL.")

    # ── Chunk ──
    chunks = chunk_by_type(
        text=page["text"],
        source=body.url,
        content_type=page["content_type"],
        title=page["title"],
        sections=page["sections"],
    )
    if not chunks:
        raise HTTPException(400, "Could not extract any usable content from this URL.")

    doc_id     = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{body.url}"))
    collection = wv.collections.get(COLLECTION)

    embeddings = embed_texts([c["text"] for c in chunks])

    inserted = _batch_insert(
        collection, chunks, embeddings,
        user_id, doc_id, page["title"], body.url,
        "url", page["content_type"],
    )

    upsert_document(
        doc_id, user_id, page["title"], body.url,
        "url", inserted, page["content_type"],
    )

    print(
        f"[kb/add] {page['content_type'].upper()} | {inserted} chunks | "
        f"{body.url[:80]}"
    )

    return KBAddResponse(
        doc_id=doc_id,
        chunks_stored=inserted,
        status="success",
        content_type=page["content_type"],
        title=page["title"],
    )


# ── Query (SSE streaming) ─────────────────────────────────────────────────────

@app.post("/query")
def query(body: QueryRequest, user_id: str = Depends(get_current_user)):
    """
    Hybrid retrieval + context-aware LLM generation via SSE.

    Full pipeline
    ─────────────
    Step 1  embed_query()
            Encode the question into a 1024-dim bge-m3 dense vector.

    Step 2  hybrid_retrieve()
            collection.query.hybrid(
              query=text,         ← BM25  sparse arm  (30 %)
              vector=dense_vec,   ← HNSW  dense  arm  (70 %)
              alpha=0.7,
              fusion=RANKED,      ← RRF-style rank fusion
              filters=user_filter ← isolate this user's data
              limit=15)           ← top-15 after fusion

    Step 3  Build RAG context string
            Format each hit as "[Title]\n{text}" and join with "---".

    Step 4  Build conversation context list
            [
              {"role": "system",    "content": SYSTEM_PROMPT + RAG context},
              # ← inject prior turns (capped at MAX_HISTORY_TURNS)
              {"role": "user",      "content": "prev question"},
              {"role": "assistant", "content": "prev answer"},
              # ← stream_with_context appends the new question here
            ]

    Step 5  build_generation_kwargs()
            Returns flexible parameter dict for the LLM call.

    Step 6  stream_with_context()
            Appends new user question to context, streams Groq tokens via
            SSE, then appends the completed response to context.

    Step 7  SSE sources event
            Deduplicated doc citations sent as a final SSE event.

    Conversation history
    ────────────────────
    The frontend sends prior turns in body.history as
    [{"role": "user"|"assistant", "content": "..."}].
    These are injected between the system message and the new question,
    enabling follow-up questions like "tell me more" or "what about X?"
    to work correctly without the LLM losing track of prior context.
    """
    # ── Step 1: Embed query ───────────────────────────────────────────────────
    try:
        q_vector = embed_query(body.question)
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {e}")

    # ── Step 2: Hybrid retrieval (BM25 + HNSW + metadata filter) ─────────────
    try:
        collection = wv.collections.get(COLLECTION)
        hits = hybrid_retrieve(
            collection   = collection,
            query_text   = body.question,
            query_vector = q_vector,
            user_id      = user_id,
            source_ids   = body.source_ids or None,
            top_k        = RETRIEVER_TOP_K,   # 15 — precision/recall balance
        )
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {e}")

    # ── Empty knowledge base ──────────────────────────────────────────────────
    if not hits:
        def _empty():
            msg = (
                "My knowledge base is empty. "
                "Add URLs in the **Library** tab, or index the current page, "
                "then ask me anything about that content."
            )
            yield f"data: {json.dumps({'text': msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # ── Step 3: Build RAG context string + deduplicate sources ────────────────
    context_parts: list[str] = []
    seen_sources:  dict[str, dict] = {}

    for hit in hits:
        label = hit.get("title") or hit.get("url") or "Source"
        # Include heading path for extra precision when available
        prefix = f"[{label}]"
        if hit.get("heading_path"):
            prefix += f" [{hit['heading_path']}]"
        context_parts.append(f"{prefix}\n{hit['text']}")

        doc_id = hit["doc_id"]
        if doc_id not in seen_sources:
            seen_sources[doc_id] = {
                "doc_id":       doc_id,
                "title":        hit.get("title", ""),
                "url":          hit.get("url", ""),
                "content_type": hit.get("content_type", "general"),
                "score":        hit["_hybrid_score"],
            }

    rag_context = "\n\n---\n\n".join(context_parts)
    sources     = list(seen_sources.values())

    # ── Step 4: Build conversation context list ───────────────────────────────
    #
    # Structure:
    #   [0] system   — SYSTEM_PROMPT + freshly retrieved RAG context
    #   [1..N] prior turns (user / assistant) — capped at MAX_HISTORY_TURNS
    #   → stream_with_context() appends the new user question at [N+1]
    #
    # Why re-retrieve RAG context every turn?
    #   Each question may focus on different parts of the KB.
    #   Fresh retrieval ensures the most relevant chunks are always in context.
    #   Prior turns capture conversational follow-ups ("tell me more about…").

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Use the following knowledge base context to answer the question:\n\n"
        f"{rag_context}"
    )
    conversation: list[dict] = [{"role": "system", "content": system_content}]

    # Inject prior conversation turns (validate role + content)
    if body.history:
        valid_turns = [
            t for t in body.history
            if isinstance(t, dict)
            and t.get("role") in ("user", "assistant")
            and isinstance(t.get("content"), str)
            and t["content"].strip()
        ]
        # Cap at MAX_HISTORY_TURNS to avoid blowing the context window
        for turn in valid_turns[-MAX_HISTORY_TURNS:]:
            conversation.append({"role": turn["role"], "content": turn["content"]})

    # ── Step 5: Build generation kwargs ──────────────────────────────────────
    #
    # build_generation_kwargs() returns a flexible dict so parameters can be
    # changed in one place and automatically propagate to the LLM call below.
    # This decouples "what parameters to use" from "how to call the model."
    gen_kwargs = build_generation_kwargs(
        prompt         = body.question,   # informational only, not in dict
        temperature    = GEN_TEMPERATURE,
        top_p          = GEN_TOP_P,
        max_new_tokens = GEN_MAX_OUTPUT_TOKENS,
    )

    # ── Step 6 + 7: Stream answer + emit sources ──────────────────────────────
    #
    # stream_with_context():
    #   1. Appends {"role": "user", "content": question} to conversation
    #   2. Calls Groq with the full context list (RAG + history + question)
    #   3. Streams each token chunk back as it arrives
    #   4. On completion, appends {"role": "assistant", "content": full_reply}
    #      so conversation retains the full exchange for inspection/logging.
    def stream_answer():
        try:
            for text_chunk in stream_with_context(
                prompt   = body.question,
                context  = conversation,
                role     = "user",
                **gen_kwargs,
            ):
                yield f"data: {json.dumps({'text': text_chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        # Always send sources + DONE, even after an error
        yield f"data: {json.dumps({'sources': sources})}\n\n"
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
    # Remove all Weaviate vectors for this document
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


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    from retriever import ALPHA, TOP_K as RET_TOP_K
    try:
        collection = wv.collections.get(COLLECTION)
        agg        = collection.aggregate.over_all(total_count=True)
        vec_count  = agg.total_count or 0
    except Exception:
        vec_count  = 0
    return {
        "status":   "ok",
        "pipeline": {
            "step1_db":     "Weaviate embedded  (./weaviate_db)",
            "step2_load":   "batch.fixed_size(200)",
            "step3_sparse": "BM25 inverted index  (text, title, heading_path)",
            "step4_dense":  f"HNSW cosine  bge-m3 {EMBED_DIM}-dim  "
                            f"ef_construction=128  M=32  ef=64",
            "step5_search": f"query.hybrid  alpha={ALPHA}  "
                            f"fusion=RANKED  limit={RET_TOP_K}",
        },
        "llm": {
            "provider":           "Groq",
            "model":              GROQ_MODEL,
            "temperature":        GEN_TEMPERATURE,
            "top_p":              GEN_TOP_P,
            "frequency_penalty":  GEN_FREQUENCY_PENALTY,
            "max_tokens":         GEN_MAX_OUTPUT_TOKENS,
            "max_history_turns":  MAX_HISTORY_TURNS,
        },
        "collection": COLLECTION,
        "vectors":    vec_count,
    }
