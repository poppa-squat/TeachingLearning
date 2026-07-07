"""Meaning-based 3D layout: UMAP squashes each concept's 384-number embedding
down to an (x, y, z) position, keeping similar concepts near each other.

These positions are derived, never saved — they are recomputed whenever the
graph changes. Manual positions live on the nodes themselves (graph.py).

UMAP needs a handful of points to work with; for very small graphs we fall
back to PCA (3 or 4 concepts) or fixed positions (fewer).
"""

from __future__ import annotations

import numpy as np

from app.embeddings import EmbeddingStore

SPREAD = 120.0  # target half-width of the layout in scene units
_MIN_POINTS_FOR_UMAP = 5


def meaning_positions(
    names: list[str],
    store: EmbeddingStore,
    texts: list[str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """3D positions for every concept, derived from embedding similarity.
    `texts`, when given, is what actually gets embedded for each name (e.g.
    name enriched with the user's definition); positions stay keyed by name."""
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: (0.0, 0.0, 0.0)}

    vectors = store.embed(texts if texts is not None else names)
    if len(names) < _MIN_POINTS_FOR_UMAP:
        coords = _pca_3d(vectors)
    else:
        import umap  # imported lazily: numba compilation makes this slow

        reducer = umap.UMAP(
            n_components=3,
            metric="cosine",
            n_neighbors=min(15, len(names) - 1),
            min_dist=0.3,
            random_state=42,  # same graph -> same layout, so the map feels stable
        )
        coords = reducer.fit_transform(vectors)

    coords = _rescale(np.asarray(coords, dtype=np.float64))
    return {
        name: (float(x), float(y), float(z))
        for name, (x, y, z) in zip(names, coords)
    }


def _pca_3d(vectors: np.ndarray) -> np.ndarray:
    """Project onto the 3 directions of greatest variation."""
    centred = vectors - vectors.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ components[:3].T
    if coords.shape[1] < 3:  # fewer points than dimensions requested
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    return coords


def _rescale(coords: np.ndarray) -> np.ndarray:
    """Centre on the origin and scale to a comfortable size for the 3D view."""
    coords = coords - coords.mean(axis=0)
    extent = np.abs(coords).max()
    if extent > 0:
        coords = coords * (SPREAD / extent)
    return coords
