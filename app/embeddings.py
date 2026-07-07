"""Embeddings: turn concept names and predicates into meaning-vectors.

Uses sentence-transformers (Qwen3-Embedding-0.6B, 1024 dims). Each distinct
text is embedded once and cached — in memory for the session, and on disk
(.npz) so restarting the app doesn't recompute anything. All vectors are
L2-normalised on creation, so cosine similarity is a plain dot product.

The on-disk cache records which model produced it. Vectors from one model are
meaningless when compared against another's, so if MODEL_NAME changes the old
cache is discarded and everything is recomputed (instant at this graph size).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CACHE_FILE = Path("embeddings_cache.npz")


class EmbeddingStore:
    def __init__(self, cache_file: Path | None = CACHE_FILE) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._cache_file = Path(cache_file) if cache_file else None
        self._model = None  # loaded lazily; costs a few seconds + download on first run
        self._model_lock = threading.Lock()
        if self._cache_file and self._cache_file.exists():
            self._cache = self._load_cache(self._cache_file)

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
        if self._cache:
            keys = np.array(list(self._cache), dtype=str)
            vectors = np.stack(list(self._cache.values()))
        else:
            keys = np.array([], dtype=str)
            vectors = np.empty((0, 0), dtype=np.float32)
        # Store the producing model alongside the vectors so a later run with a
        # different MODEL_NAME can tell the cache is stale rather than mixing
        # incompatible vector spaces. Texts are kept as a separate keys array
        # (not archive names) so any concept wording is safe, including one that
        # happens to collide with a metadata field name.
        np.savez(
            self._cache_file,
            model=np.array(MODEL_NAME),
            keys=keys,
            vectors=vectors,
        )

    @staticmethod
    def _load_cache(cache_file: Path) -> dict[str, np.ndarray]:
        """Load a cache, but only if it was produced by the current model.
        A missing/foreign/legacy-format cache yields an empty dict, so its
        vectors get recomputed under MODEL_NAME instead of being trusted."""
        with np.load(cache_file, allow_pickle=False) as data:
            if "model" not in data.files or str(data["model"]) != MODEL_NAME:
                return {}
            keys = data["keys"]
            vectors = data["vectors"]
        return {
            str(text): np.asarray(vec, dtype=np.float32)
            for text, vec in zip(keys, vectors)
        }
