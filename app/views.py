"""Focus view: what the page shows when zoomed into one or more containers.

Zooming is a *view* of the flat graph, not a data-structure change. Python
computes everything the page needs — the member nodes with their Venn
signature, the ghost nodes at the boundary, the edges between rendered nodes,
and the legend groups — and the JS side only draws it.
"""

from __future__ import annotations

from collections import Counter

from app.graph import KnowledgeGraph


def focus_view(kg: KnowledgeGraph, focused: list[str]) -> dict:
    """Everything needed to render a zoom into `focused` (1+ container names).

    - Members: the direct members of the focused containers. Each carries its
      `signature` — the subset of `focused` it belongs to, so with two
      containers the three Venn groups fall out (unique to A, unique to B, in
      both). The focused containers themselves are not rendered; the
      breadcrumb names them.
    - Ghosts: outside nodes with at least one edge to a member, flagged
      `ghost` and rendered dimmed, so a container never feels like a silo.
    - Edges: flat-graph edges whose endpoints are both rendered and at least
      one of them is a member (member–member and member–ghost).
    - Groups: the non-empty signature groups with counts, for the legend.
    """
    ordered: list[str] = []
    for f in focused:
        if f not in ordered:
            ordered.append(f)
    focused = ordered
    if not focused:
        raise ValueError("Focus on at least one concept")
    for f in focused:
        if not kg.has_node(f):
            raise KeyError(f"Unknown concept: {f!r}")

    in_focus: dict[str, list[str]] = {}  # member -> focused containers it's in
    for f in focused:
        for m in kg.members(f):
            if m not in focused:
                in_focus.setdefault(m, []).append(f)
    signatures = {m: sorted(fs, key=focused.index) for m, fs in in_focus.items()}
    members = set(signatures)

    edges = []
    ghosts: set[str] = set()
    for e in kg.edges():
        ends = {e.source, e.target}
        inside = ends & members
        if not inside or ends & set(focused):
            continue  # no member endpoint, or touches a focused container
        ghosts |= ends - members  # the other endpoint (if any) is a ghost
        edges.append(e.model_dump())

    containers = set(kg.containers())
    descriptions = {n.name: n.description for n in kg.nodes()}

    def node_dict(name: str, ghost: bool) -> dict:
        return {
            "name": name,
            "description": descriptions[name],
            "ghost": ghost,
            "signature": signatures.get(name, []),
            "has_members": name in containers,  # zoomable one level deeper
        }

    group_counts = Counter(tuple(sig) for sig in signatures.values())
    groups = [
        {"signature": list(sig), "count": count}
        for sig, count in sorted(
            group_counts.items(),
            key=lambda item: (len(item[0]), [focused.index(f) for f in item[0]]),
        )
    ]

    return {
        "focused": focused,
        "nodes": [node_dict(m, ghost=False) for m in sorted(members)]
        + [node_dict(g, ghost=True) for g in sorted(ghosts)],
        "edges": edges,
        "groups": groups,
    }
