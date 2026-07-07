"""Embeddings: turn concept names and predicates into meaning-vectors.

Uses sentence-transformers (all-MiniLM-L6-v2, 384 dims). Each distinct text is
embedded once and cached — in memory for the session, and on disk (.npz) so
restarting the app doesn't recompute anything. All vectors are L2-normalised
on creation, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_FILE = Path("embeddings_cache.npz")


class EmbeddingStore:
    def __init__(self, cache_file: Path | None = CACHE_FILE) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._cache_file = Path(cache_file) if cache_file else None
        self._model = None  # loaded lazily; costs a few seconds + download on first run
        self._model_lock = threading.Lock()
        if self._cache_file and self._cache_file.exists():
            with np.load(self._cache_file) as data:
                self._cache = {key: data[key] for key in data.files}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Vectors for `texts` (rows in the same order). Cached texts are free."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self._get_model().encode(
                missing, normalize_embeddings=True, show_progress_bar=False
            )
            for text, vec in zip(missing, vectors):
                self._cache[text] = np.asarray(vec, dtype=np.float32)
            self._persist()
        return np.stack([self._cache[t] for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (vectors are already normalised)."""
        return float(np.dot(a, b))

    def _get_model(self):
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                try:
                    # Prefer the on-disk copy: no network traffic once the
                    # model has been downloaded (everything-local rule).
                    self._model = SentenceTransformer(
                        MODEL_NAME, local_files_only=True
                    )
                except Exception:
                    self._model = SentenceTransformer(MODEL_NAME)
            return self._model

    def _persist(self) -> None:
        if self._cache_file is None:
            return
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._cache_file, **self._cache)
