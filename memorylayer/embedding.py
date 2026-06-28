"""
MemoryLayer — Embedding Utilities

Lazy-loads SentenceTransformer 'all-MiniLM-L6-v2' (384-dim, ~23MB, CPU-fast).
Falls back gracefully — returns None — when sentence-transformers is absent.
"""

from typing import Optional

_embed_model = None
EMBED_AVAILABLE: Optional[bool] = None


def get_embedder():
    """Lazy-load the embedding model. Safe to call repeatedly."""
    global _embed_model, EMBED_AVAILABLE
    if EMBED_AVAILABLE is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        EMBED_AVAILABLE = True
    except Exception:
        _embed_model = None
        EMBED_AVAILABLE = False
    return _embed_model


def embed(text: str) -> Optional[list]:
    """
    Return a 384-dim embedding for `text`, or None if the model is unavailable.
    Result is a plain Python list so it can be stored in metadata dicts.
    """
    model = get_embedder()
    if model is None:
        return None
    return model.encode(text, convert_to_numpy=True).tolist()
