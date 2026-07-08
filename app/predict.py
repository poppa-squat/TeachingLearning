"""Suggest missing edges: which unconnected concept pairs probably relate?
And missing memberships: which concepts probably belong inside a container?

Two signals each, both plain math (no generative model involved):
- Meaning: cosine similarity of embeddings (for membership, against the
  centroid of the container's current members).
- Structure: shared neighbours — Adamic-Adar for edges; for membership, the
  fraction of a node's neighbours already inside the container.

The graph is small, so we score every candidate pair directly.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
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


class MemberSuggestion(BaseModel):
    node: str
    container: str
    score: float      # combined, 0..1
    meaning: float    # cosine vs the container's member centroid, rescaled 0..1
    structure: float  # fraction of the node's neighbours already members


def suggest_members(
    kg: KnowledgeGraph,
    store: EmbeddingStore,
    top_k: int = 10,
    containers: list[str] | None = None,
) -> list[MemberSuggestion]:
    """Rank (node, container) pairs by how likely the node belongs inside.

    Math only — membership has no wording, so there is nothing for the
    generative model to do. By default candidates are the existing containers;
    pass `containers` to ask about specific (possibly still-empty) ones.
    """
    if containers is None:
        containers = kg.containers()
    if not containers:
        return []
    names = kg.node_names()
    vectors = dict(zip(names, store.embed([kg.node_text(n) for n in names])))

    suggestions = []
    for container in containers:
        members = set(kg.members(container))
        if members:
            centroid = np.mean([vectors[m] for m in sorted(members)], axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:  # keep the dot product a true cosine
                centroid = centroid / norm
        else:
            centroid = vectors[container]
        for node in names:
            # Skip pairs membership already links, in either direction and
            # transitively, so accepting a suggestion can never make a cycle.
            if node == container or kg.related_by_membership(node, container):
                continue
            meaning = (EmbeddingStore.similarity(vectors[node], centroid) + 1) / 2
            neighbours = kg.neighbors(node)
            structure = (
                len(neighbours & members) / len(neighbours) if neighbours else 0.0
            )
            score = MEANING_WEIGHT * meaning + STRUCTURE_WEIGHT * structure
            suggestions.append(
                MemberSuggestion(
                    node=node, container=container, score=score,
                    meaning=meaning, structure=structure,
                )
            )
    suggestions.sort(key=lambda s: s.score, reverse=True)
    return suggestions[:top_k]
