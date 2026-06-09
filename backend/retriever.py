"""
PageMind Hybrid Retriever  — Weaviate Native Hybrid Search
───────────────────────────────────────────────────────────
Uses Weaviate's built-in  collection.query.hybrid()  which handles
all 5 vector-DB operations in a SINGLE round-trip:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Step 1 – DB Setup          EmbeddedOptions, persistent store   │
  │  Step 2 – Load Document     batch.fixed_size() in ingest routes │
  │  Step 3 – Sparse vector     BM25 inverted index (index_searchable│
  │                             = True) built automatically at ingest│
  │  Step 4 – Dense vector      bge-m3 1024-dim embeddings, stored   │
  │                             as HNSW vectors at ingest            │
  │  Step 5 – Run search        query.hybrid() — single call fuses  │
  │                             BM25 + HNSW via HybridFusion.RANKED  │
  └─────────────────────────────────────────────────────────────────┘

Hybrid Formula
──────────────
Weaviate's HybridFusion.RANKED implements Reciprocal Rank Fusion (RRF)
weighted by  alpha:

    score(d) = alpha     × rank_score_dense(d)     ← HNSW / ANN
             + (1−alpha) × rank_score_sparse(d)    ← BM25

  alpha = 0.7  →  70 % dense (semantic)  /  30 % BM25 (keyword)

  RRF smoothing constant k=60 (Weaviate default for RANKED fusion).
  Documents absent from one list score 0 from that retriever.

Configuration
─────────────
  ALPHA    = 0.7   →  semantic 70 %  /  keyword 30 %
  TOP_K    = 15    →  precision/recall balance (enough context, no noise)

BM25 property boosts  (keyword relevance prioritization)
───────────────────────────────────────────────────────
  title^2         →  exact title match is a strong relevance signal
  heading_path^1.5→  breadcrumb path is precise and section-specific
  text            →  bulk document content at base weight 1.0

Metadata filter  (applied BEFORE both retrievers)
─────────────────────────────────────────────────
  user_id  = current JWT user    (always)
  doc_id   in [source_ids]       (optional — single-source chat mode)
"""

from __future__ import annotations

from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

# ── Configuration ─────────────────────────────────────────────────────────────

ALPHA  = 0.7   # dense vector weight (1−ALPHA = 0.3 = BM25 weight)
TOP_K  = 15    # final results after hybrid fusion

# BM25 field boosts — higher = that field's keyword matches score more
_BM25_PROPS = ["text", "title^2", "heading_path^1.5"]


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    collection,
    query_text:   str,
    query_vector: list[float],
    user_id:      str,
    source_ids:   list[str] | None = None,
    top_k:        int               = TOP_K,
) -> list[dict]:
    """
    Single-call hybrid search: BM25 (sparse) + HNSW (dense) fused by Weaviate.

    Parameters
    ----------
    collection    : live Weaviate collection object
    query_text    : raw question string  → BM25 sparse search
    query_vector  : 1024-dim bge-m3 embedding → HNSW dense search
    user_id       : applied as pre-filter to isolate this user's chunks
    source_ids    : optional doc_id list for single-source chat mode
    top_k         : how many fused results to return (default = TOP_K = 15)

    Returns
    -------
    list[dict]  sorted by hybrid score (descending).
    Each dict = all stored Weaviate properties  +  _hybrid_score (float).
    """
    # ── Metadata filter (pre-applied to BOTH sparse and dense retrievers) ─────
    f = Filter.by_property("user_id").equal(user_id)
    if source_ids:
        f = f & Filter.by_property("doc_id").contains_any(source_ids)

    # ── Single hybrid call — Weaviate internally runs: ────────────────────────
    #    1. BM25  on  text / title^2 / heading_path^1.5  (sparse)
    #    2. near_vector with query_vector                 (dense / ANN)
    #    3. HybridFusion.RANKED (RRF-style) with alpha weighting
    result = collection.query.hybrid(
        query            = query_text,        # BM25 input
        vector           = query_vector,      # bring-your-own dense vector
        alpha            = ALPHA,             # 0.7 = 70% dense
        fusion_type      = HybridFusion.RANKED,   # RRF-style rank fusion
        query_properties = _BM25_PROPS,       # BM25 fields + boosts
        filters          = f,                 # metadata pre-filter
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
