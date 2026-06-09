"""
Local embeddings using sentence-transformers + BAAI/bge-m3.

Why sentence-transformers?
  - Loads bge-m3 as a standard dense encoder returning one 1024-dim vector
    per text — exactly what our Weaviate collection expects.

Model: BAAI/bge-m3
  - Embedding dimension : 1024
  - Model size          : ~570 MB (downloaded once to ~/.cache/huggingface)
  - Languages           : 100+ languages
  - Quality             : #1 open-source on MTEB multilingual benchmark
  - normalize_embeddings: True  → cosine similarity == dot product

First run downloads the model automatically — subsequent runs use the HF cache.
"""

from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM   = 1024   # must match Weaviate collection vector size

# Singleton — loaded once per process, reused for all requests
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading {EMBED_MODEL} (~570 MB on first run)…")
        _model = SentenceTransformer(EMBED_MODEL)
        print(f"[embedder] Model ready. Dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of documents.
    Returns a list of 1024-dimensional float vectors (L2-normalised).
    """
    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).tolist()


def embed_query(text: str) -> list[float]:
    """
    Embed a single search query.
    bge-m3 uses the same encoder for queries and documents (symmetric).
    """
    return _get_model().encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()
