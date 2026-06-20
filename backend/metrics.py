"""
metrics.py — Per-request latency tracking for PageMind RAG
═══════════════════════════════════════════════════════════
Stores the last 100 request records in a thread-safe in-memory deque.
Each record is a RequestMetrics dataclass populated incrementally by
main.py as each pipeline stage completes.

Stages tracked (all in milliseconds):
  github_fetch_ms  — time to download all repo files via GitHub API
  embed_ms         — BGE-M3 embedding time (dense vectors)
  hyde_ms          — HyDE snippet generation via Groq (0 if skipped)
  bm25_ms          — BM25 sparse retrieval stage
  ann_ms           — HNSW dense / hybrid ANN search
  rerank_ms        — cross-encoder reranking stage
  llm_ms           — Groq streaming response (first-token to [DONE])
  total_ms         — wall-clock from request received to response sent
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, asdict
from threading  import Lock

_MAXLEN = 100
_lock   = Lock()
_store: deque["RequestMetrics"] = deque(maxlen=_MAXLEN)


@dataclass
class RequestMetrics:
    question:        str   = ""
    intent:          str   = ""
    github_fetch_ms: float = 0.0
    embed_ms:        float = 0.0
    hyde_ms:         float = 0.0
    bm25_ms:         float = 0.0
    ann_ms:          float = 0.0
    rerank_ms:       float = 0.0
    llm_ms:          float = 0.0
    total_ms:        float = 0.0
    chunks_retrieved: int  = 0
    chunks_reranked:  int  = 0
    model_used:      str   = ""
    ts:              float = field(default_factory=time.time)


def record(m: RequestMetrics) -> None:
    """Append a completed RequestMetrics to the store."""
    with _lock:
        _store.append(m)


def get_recent(n: int = 20) -> list[dict]:
    """Return up to n most recent records as dicts (newest first)."""
    with _lock:
        items = list(_store)
    return [asdict(m) for m in reversed(items[-n:])]


def summary() -> dict:
    """Aggregate stats over all stored records."""
    with _lock:
        items = list(_store)

    if not items:
        return {"count": 0}

    def avg(key):
        vals = [getattr(m, key) for m in items if getattr(m, key) > 0]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    return {
        "count":               len(items),
        "avg_total_ms":        avg("total_ms"),
        "avg_embed_ms":        avg("embed_ms"),
        "avg_hyde_ms":         avg("hyde_ms"),
        "avg_ann_ms":          avg("ann_ms"),
        "avg_rerank_ms":       avg("rerank_ms"),
        "avg_llm_ms":          avg("llm_ms"),
        "avg_chunks_retrieved": round(
            sum(m.chunks_retrieved for m in items) / len(items), 1
        ),
    }
