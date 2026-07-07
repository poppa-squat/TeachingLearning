"""Reasoning over relationships: paths, agreement between paths, linchpins.

Explaining a connection between two far-apart concepts:
1. Find every (simple) path between them, following asymmetric edges only in
   their stored direction and symmetric edges either way.
2. Turn each path into a readable chain and embed it.
3. Group paths whose embeddings are close — each group is one "facet" of the
   connection. One group means the paths agree; several mean the connection
   has genuinely distinct aspects.

The generative model may later verbalise these results; the grouping decision
itself is made here, by geometry alone.

Linchpins: relationships that many chains of reasoning pass through, found via
edge betweenness centrality (the fraction of shortest paths using each edge).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from pydantic import BaseModel

from app.embeddings import EmbeddingStore
from app.graph import Edge, KnowledgeGraph

MAX_PATH_EDGES = 5
AGREE_THRESHOLD = 0.6  # cosine similarity above which two paths count as agreeing


class PathStep(BaseModel):
    a: str          # concept we stepped from
    b: str          # concept we stepped to
    predicate: str
    directed: bool  # True if the relationship is asymmetric
    forward: bool = True  # False when we walked an asymmetric edge against
    #                       its arrow (the relationship truly reads b -> a)


class PathAnalysis(BaseModel):
    start: str
    end: str
    paths: list[list[PathStep]]
    groups: list[list[int]]  # indices into `paths`, one list per facet
    agree: bool              # True when every path lands in one group
    used_reverse: bool = False  # True when no arrow-respecting path existed
    #                             and steps against the arrows were allowed


def find_paths(
    kg: KnowledgeGraph,
    start: str,
    end: str,
    max_edges: int = MAX_PATH_EDGES,
    allow_reverse: bool = False,
) -> list[list[PathStep]]:
    """All simple paths from start to end. By default asymmetric edges are
    followed only in their stored direction; with allow_reverse=True they may
    also be walked backwards (each such step is marked forward=False)."""
    t = kg.traversal_view()
    if start not in t or end not in t:
        raise KeyError(f"Unknown concept: {start if start not in t else end!r}")
    if allow_reverse:
        for u, v, k, data in list(t.edges(keys=True, data=True)):
            if data["directed"] and not t.has_edge(v, u, k):
                t.add_edge(v, u, key=k, directed=True, reversed=True)
    paths = []
    for edge_path in nx.all_simple_edge_paths(t, start, end, cutoff=max_edges):
        steps = [
            PathStep(
                a=u,
                b=v,
                predicate=k,
                directed=t[u][v][k]["directed"],
                forward=not t[u][v][k].get("reversed", False),
            )
            for u, v, k in edge_path
        ]
        paths.append(steps)
    return paths


def path_text(path: list[PathStep]) -> str:
    """A path as one readable sentence-chain, e.g.
    'derivative -> (is the slope of) -> tangent line'."""
    if not path:
        return ""
    parts = [path[0].a]
    for step in path:
        if not step.directed:
            arrow_in, arrow_out = "<->", "<->"
        elif step.forward:
            arrow_in, arrow_out = "->", "->"
        else:  # walked against the arrow: the relationship reads b -> a
            arrow_in, arrow_out = "<-", "<-"
        parts.append(f" {arrow_in} ({step.predicate}) {arrow_out} {step.b}")
    return "".join(parts)


def analyse_connection(
    kg: KnowledgeGraph,
    store: EmbeddingStore,
    start: str,
    end: str,
    max_edges: int = MAX_PATH_EDGES,
    agree_threshold: float = AGREE_THRESHOLD,
) -> PathAnalysis:
    """Find all paths between two concepts and group them into facets.

    Arrow-respecting paths are preferred; when none exist, paths that walk
    against arrows are used instead (marked on the result)."""
    paths = find_paths(kg, start, end, max_edges)
    used_reverse = False
    if not paths:
        paths = find_paths(kg, start, end, max_edges, allow_reverse=True)
        used_reverse = bool(paths)
    if not paths:
        return PathAnalysis(start=start, end=end, paths=[], groups=[], agree=True)

    vectors = store.embed([path_text(p) for p in paths])
    groups = _group_by_similarity(vectors, agree_threshold)
    return PathAnalysis(
        start=start,
        end=end,
        paths=paths,
        groups=groups,
        agree=len(groups) == 1,
        used_reverse=used_reverse,
    )


def linchpins(kg: KnowledgeGraph, top_k: int = 5) -> list[tuple[Edge, float]]:
    """Relationships whose removal would break the most chains of reasoning,
    scored by edge betweenness centrality (0..1)."""
    t = kg.traversal_view()
    if t.number_of_edges() == 0:
        return []
    # Centrality is computed on the collapsed simple digraph; each stored
    # relationship then takes the score of the direction(s) it can be walked.
    collapsed = nx.DiGraph(t)
    centrality = nx.edge_betweenness_centrality(collapsed)
    scored = []
    for edge in kg.edges():
        score = centrality.get((edge.source, edge.target), 0.0)
        if not edge.directed:
            score = max(score, centrality.get((edge.target, edge.source), 0.0))
        scored.append((edge, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def _group_by_similarity(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Union-find grouping: two items share a group if their cosine similarity
    meets the threshold (directly or through a chain of similar items)."""
    n = len(vectors)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sims = vectors @ vectors.T
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                parent[root(j)] = root(i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)
