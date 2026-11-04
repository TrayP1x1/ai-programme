"""Sentence embeddings for dense retrieval (Month 3 onwards).

Uses the small pretrained model all-MiniLM-L6-v2 (Apache-2.0), which runs fine on a
laptop CPU. Embeddings are cached to disk, because encoding a corpus is the slow part.
With COURSE_FAKE_DATA=1 a cheap hashing embedding is used instead (offline tests).
"""
import hashlib
from pathlib import Path

import numpy as np

from course.data import DATA_DIR, fake_mode

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def encode(texts, cache_name=None, batch_size=64):
    """Return an (N, d) float32 array of L2-normalised embeddings for ``texts``."""
    cache = DATA_DIR / "embeddings" / f"{cache_name}.npy" if cache_name else None
    if cache is not None and cache.exists() and not fake_mode():
        emb = np.load(cache)
        if len(emb) == len(texts):
            return emb
    if fake_mode():
        emb = _hash_embed(texts)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, device="cpu")
        emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                           normalize_embeddings=True).astype(np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)
    return emb


def _hash_embed(texts, dim=256):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)
