"""In-memory knowledge graph.

Nodes are concepts ("eigenvalues"); edges are relationships whose predicate is
the user's own free-text wording. A `directed` flag marks whether the
relationship is asymmetric (True: source -> target order matters) or symmetric
(False: readable either way).

Backed by a NetworkX MultiDiGraph so two concepts can be linked by several
distinct relationships at once.
"""

from __future__ import annotations

import networkx as nx
from pydantic import BaseModel


class Node(BaseModel):
    name: str
    description: str = ""  # the user's own definition of the concept; optional
    position: tuple[float, float, float] | None = None
    # Saved manual (x, y, z) for manual layout mode; None if never placed by
    # hand. Meaning-based positions are recomputed by UMAP, never stored.


class Edge(BaseModel):
    source: str
    target: str
    predicate: str  # free text, the user's own wording — NOT a category
    directed: bool  # True = asymmetric (one-way); False = symmetric


class KnowledgeGraph:
    def __init__(self) -> None:
        # Edge key is the predicate, so the same wording between the same pair
        # is stored once, while different wordings coexist.
        self._g = nx.MultiDiGraph()

    # -- nodes ---------------------------------------------------------------

    def add_node(self, name: str, description: str = "") -> bool:
        """Add a concept. Returns False if it already existed."""
        name = name.strip()
        if not name:
            raise ValueError("Concept name cannot be empty")
        if name in self._g:
            return False
        self._g.add_node(name, description=description.strip(), position=None)
        return True

    def remove_node(self, name: str) -> None:
        self._require(name)
        self._g.remove_node(name)

    def has_node(self, name: str) -> bool:
        return name in self._g

    def set_position(self, name: str, position: tuple[float, float, float] | None) -> None:
        self._require(name)
        self._g.nodes[name]["position"] = tuple(position) if position else None

    def set_description(self, name: str, description: str) -> None:
        self._require(name)
        self._g.nodes[name]["description"] = description.strip()

    def nodes(self) -> list[Node]:
        return [
            Node(
                name=n,
                description=data.get("description", ""),
                position=data.get("position"),
            )
            for n, data in self._g.nodes(data=True)
        ]

    def node_names(self) -> list[str]:
        return list(self._g.nodes)

    def node_text(self, name: str) -> str:
        """The text that stands for this concept in embedding space: the name,
        enriched with the user's definition when one exists."""
        self._require(name)
        description = self._g.nodes[name].get("description", "")
        return f"{name}: {description}" if description else name

    # -- edges ---------------------------------------------------------------

    def add_edge(self, edge: Edge) -> bool:
        """Add a relationship. Returns False if the same wording already links
        the pair (for symmetric edges, in either orientation)."""
        self._require(edge.source)
        self._require(edge.target)
        if edge.source == edge.target:
            raise ValueError("A concept cannot relate to itself")
        if self._find(edge.source, edge.target, edge.predicate) is not None:
            return False
        self._g.add_edge(
            edge.source, edge.target, key=edge.predicate, directed=edge.directed
        )
        return True

    def remove_edge(self, source: str, target: str, predicate: str) -> None:
        found = self._find(source, target, predicate)
        if found is None:
            raise KeyError(f"No edge {source!r} -[{predicate!r}]-> {target!r}")
        u, v = found
        self._g.remove_edge(u, v, key=predicate)

    def edges(self) -> list[Edge]:
        return [
            Edge(source=u, target=v, predicate=k, directed=data["directed"])
            for u, v, k, data in self._g.edges(keys=True, data=True)
        ]

    def edges_between(self, a: str, b: str) -> list[Edge]:
        """All relationships linking a and b, whichever way they were stored."""
        return [
            e
            for e in self.edges()
            if {e.source, e.target} == {a, b}
        ]

    def has_connection(self, a: str, b: str) -> bool:
        return self._g.has_edge(a, b) or self._g.has_edge(b, a)

    def neighbors(self, name: str) -> set[str]:
        """Neighbouring concepts regardless of edge direction."""
        self._require(name)
        return set(self._g.successors(name)) | set(self._g.predecessors(name))

    # -- views for the math layer ---------------------------------------------

    def undirected_view(self) -> nx.Graph:
        """Simple undirected graph (parallel edges collapsed) for structure
        heuristics like common neighbours and Adamic-Adar."""
        return nx.Graph(self._g)

    def traversal_view(self) -> nx.MultiDiGraph:
        """Graph for path-following: asymmetric edges go one way only;
        symmetric edges are traversable in both directions."""
        t = nx.MultiDiGraph()
        t.add_nodes_from(self._g.nodes)
        for u, v, k, data in self._g.edges(keys=True, data=True):
            t.add_edge(u, v, key=k, directed=data["directed"])
            if not data["directed"]:
                t.add_edge(v, u, key=k, directed=False)
        return t

    # -- (de)serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": [n.model_dump() for n in self.nodes()],
            "edges": [e.model_dump() for e in self.edges()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        kg = cls()
        for nd in data.get("nodes", []):
            node = Node(**nd)
            kg._g.add_node(
                node.name,
                description=node.description,
                position=tuple(node.position) if node.position else None,
            )
        for ed in data.get("edges", []):
            kg.add_edge(Edge(**ed))
        return kg

    # -- internal ---------------------------------------------------------------

    def _require(self, name: str) -> None:
        if name not in self._g:
            raise KeyError(f"Unknown concept: {name!r}")

    def _find(self, source: str, target: str, predicate: str) -> tuple[str, str] | None:
        """Locate an edge by wording; symmetric edges match either orientation.
        Returns the stored (u, v) or None."""
        if self._g.has_edge(source, target, key=predicate):
            return (source, target)
        if self._g.has_edge(target, source, key=predicate):
            data = self._g.get_edge_data(target, source, key=predicate)
            if not data["directed"]:
                return (target, source)
        return None
