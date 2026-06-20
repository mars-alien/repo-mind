"""
PageMind Hybrid Retriever  — Weaviate Native Hybrid Search + Cross-Encoder Reranking
─────────────────────────────────────────────────────────────────────────────────────
Pipeline
  1. Hybrid search  — BM25 (30 %) + HNSW (70 %) via Weaviate query.hybrid()
                      Retrieve CANDIDATE_K = 20 candidates
  2. Reranking      — cross-encoder/ms-marco-MiniLM-L-6-v2
                      Score every (question, chunk) pair; keep RERANK_TOP_N = 8

Hybrid Formula
  Weaviate HybridFusion.RANKED implements Reciprocal Rank Fusion (RRF):

    score(d) = alpha     × rank_score_dense(d)     ← HNSW / ANN
             + (1−alpha) × rank_score_sparse(d)    ← BM25

  RRF smoothing constant k=60 (Weaviate default for RANKED fusion).

BM25 field boosts
  title^2         →  exact title match is a strong relevance signal
  heading_path^1.5→  function/class breadcrumb path
  text            →  bulk content at base weight 1.0

Cross-encoder reranking
  Loads once at first rerank call (lazy singleton).
  Scores ALL CANDIDATE_K hits; returns sorted top RERANK_TOP_N with
  the raw cross-encoder score stored in _rerank_score.
"""

from __future__ import annotations

from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

# ── Configuration ─────────────────────────────────────────────────────────────

ALPHA         = 0.7   # dense vector weight  (1−ALPHA = 0.3 = BM25 weight)
CANDIDATE_K   = 20    # pre-reranking candidates retrieved from Weaviate
RERANK_TOP_N  = 8     # final results after cross-encoder reranking
TOP_K         = RERANK_TOP_N   # exposed alias for callers that import TOP_K

_BM25_PROPS = ["text", "title^2", "heading_path^1.5"]

# ── Lazy cross-encoder singleton ──────────────────────────────────────────────

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_retrieve(
    collection,
    query_text:   str,
    query_vector: list[float],
    user_id:      str,
    source_ids:   list[str] | None = None,
    top_k:        int               = CANDIDATE_K,
) -> list[dict]:
    """
    Hybrid search: BM25 (sparse) + HNSW (dense) fused by Weaviate RRF.

    Parameters
    ----------
    collection    : live Weaviate collection object
    query_text    : raw question string  → BM25 sparse search
    query_vector  : 1024-dim bge-m3 embedding → HNSW dense search
                    Pass the HyDE vector here when intent == "explain".
    user_id       : applied as pre-filter to isolate this user's chunks
    source_ids    : optional doc_id list for single-source chat mode
    top_k         : candidates to retrieve before reranking (default CANDIDATE_K)

    Returns
    -------
    list[dict] sorted by hybrid score (descending), NOT yet reranked.
    Each dict = all stored Weaviate properties + _hybrid_score (float).
    """
    f = Filter.by_property("user_id").equal(user_id)
    if source_ids:
        f = f & Filter.by_property("doc_id").contains_any(source_ids)

    result = collection.query.hybrid(
        query            = query_text,
        vector           = query_vector,
        alpha            = ALPHA,
        fusion_type      = HybridFusion.RANKED,
        query_properties = _BM25_PROPS,
        filters          = f,
        limit            = top_k,
        return_metadata  = MetadataQuery(score=True),
    )

    return [
        {
            **hit.properties,
            "_hybrid_score": round(hit.metadata.score or 0.0, 6),
        }
        for hit in result.objects
    ]


def rerank_hits(hits: list[dict], question: str, top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Cross-encoder reranking: score every (question, chunk_text) pair
    and return the top_n hits sorted by cross-encoder score descending.

    The cross-encoder model is loaded lazily on first call.
    Falls back to the original hybrid-score ordering on any error.

    Parameters
    ----------
    hits      : candidates from hybrid_retrieve()
    question  : original user question (not the HyDE snippet)
    top_n     : number of hits to return after reranking

    Returns
    -------
    list[dict]  length ≤ top_n, each hit has _rerank_score added.
    """
    if not hits:
        return hits

    try:
        ce = _get_cross_encoder()
        pairs  = [(question, h["text"]) for h in hits]
        scores = ce.predict(pairs)
        for hit, score in zip(hits, scores):
            hit["_rerank_score"] = float(score)
        ranked = sorted(hits, key=lambda h: h["_rerank_score"], reverse=True)
        return ranked[:top_n]
    except Exception as exc:
        print(f"[rerank] cross-encoder failed ({exc}), using hybrid order")
        for hit in hits:
            hit["_rerank_score"] = hit["_hybrid_score"]
        return hits[:top_n]
