"""
Local embeddings using sentence-transformers + BAAI/bge-small-en-v1.5.

Model: BAAI/bge-small-en-v1.5
  - Embedding dimension : 384
  - Model size          : ~33 MB (downloaded once to ~/.cache/huggingface)
  - normalize_embeddings: True  → cosine similarity == dot product

First run downloads the model automatically — subsequent runs use the HF cache.
"""

import os
import warnings

# Suppress HF warnings on Windows (no symlinks, no token)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning,        module="huggingface_hub")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning,      module="sentence_transformers")

from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM   = 384    # must match Weaviate collection vector size

# Singleton — loaded once per process, reused for all requests
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading {EMBED_MODEL} (~33 MB on first run)…")
        _model = SentenceTransformer(EMBED_MODEL)
        print(f"[embedder] Model ready. Dimension: {_model.get_embedding_dimension()}")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of documents.
    Returns a list of 1024-dimensional float vectors (L2-normalised).
    """
    print(f"[embedder] Encoding {len(texts)} chunks…")
    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
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
