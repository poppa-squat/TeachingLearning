"""Suggest missing edges: which unconnected concept pairs probably relate?

Two signals, both plain math (no generative model involved):
- Meaning: cosine similarity of the two concepts' embeddings.
- Structure: do they share neighbours? Adamic-Adar weighting (shared
  neighbours count for more when they have few other connections).

The graph is small, so we score every unconnected pair directly.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
from pydantic import BaseModel

from app.embeddings import EmbeddingStore
from app.graph import KnowledgeGraph

MEANING_WEIGHT = 0.6
STRUCTURE_WEIGHT = 0.4


class Suggestion(BaseModel):
    a: str
    b: str
    score: float      # combined, 0..1
    meaning: float    # cosine similarity rescaled to 0..1
    structure: float  # Adamic-Adar, rescaled 0..1 relative to the best pair


def suggest_edges(
    kg: KnowledgeGraph,
    store: EmbeddingStore,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[Suggestion]:
    """Rank unconnected pairs by how likely they are to relate."""
    names = kg.node_names()
    pairs = [
        (a, b) for a, b in combinations(names, 2) if not kg.has_connection(a, b)
    ]
    if not pairs:
        return []

    # Embed each concept as name + user definition (when given), so the
    # meaning signal knows what the user means by the name.
    vectors = dict(zip(names, store.embed([kg.node_text(n) for n in names])))
    undirected = kg.undirected_view()
    adamic_adar = {
        (u, v): score
        for u, v, score in nx.adamic_adar_index(undirected, pairs)
    }
    max_aa = max(adamic_adar.values(), default=0.0)

    suggestions = []
    for a, b in pairs:
        meaning = (EmbeddingStore.similarity(vectors[a], vectors[b]) + 1) / 2
        structure = adamic_adar[(a, b)] / max_aa if max_aa > 0 else 0.0
        score = MEANING_WEIGHT * meaning + STRUCTURE_WEIGHT * structure
        if score >= min_score:
            suggestions.append(
                Suggestion(
                    a=a, b=b, score=score, meaning=meaning, structure=structure
                )
            )
    suggestions.sort(key=lambda s: s.score, reverse=True)
    return suggestions[:top_k]
