"""Answer a plain-English question from the map (graph RAG).

Retrieval is pure math: the query's embedding picks out the concepts it is
about (cosine similarity), the graph supplies their neighbourhood, and the
relations inside that neighbourhood become the facts. Only then does the
generative model get involved — to word an answer grounded in those facts
alone (app.llm.answer_question).

Whether the question is on-map at all is decided by geometry too: if no
concept is close enough to the query, `retrieve` reports that and the app
says so, instead of letting the model improvise an answer from thin air.
(The model provides a second, softer gate: even when concepts are close, it
is told to say when the retrieved facts don't actually answer the question.)

Membership (the abstraction DAG) is structure, not knowledge, and stays out
of retrieval like it stays out of every other edge consumer.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from app.embeddings import EmbeddingStore
from app.graph import Edge, KnowledgeGraph

# Cosine floor for "this concept is what the question is about". Loose on
# purpose — borderline questions still retrieve, and the model's answerable
# flag catches the ones the facts can't actually answer.
MIN_RELEVANCE = 0.35
MAX_SEEDS = 6     # at most this many query-matched concepts seed the subgraph
MAX_FACTS = 30    # cap on relations handed to the model


class SeedConcept(BaseModel):
    name: str
    score: float  # cosine similarity to the query


class Retrieval(BaseModel):
    """The slice of the map a question is about."""

    related: bool                    # False: nothing on the map matches the query
    seeds: list[SeedConcept] = Field(default_factory=list)
    nodes: list[str] = Field(default_factory=list)   # seeds + their neighbours
    facts: list[Edge] = Field(default_factory=list)  # most relevant first
    definitions: dict[str, str] = Field(default_factory=dict)


def fact_text(edge: Edge) -> str:
    """One relation as a readable line, matching how edges are worded to the
    model elsewhere."""
    tail = "" if edge.directed else " (and vice versa)"
    return f'"{edge.source}" {edge.predicate} "{edge.target}"{tail}'


def retrieve(
    kg: KnowledgeGraph,
    store: EmbeddingStore,
    question: str,
    min_relevance: float = MIN_RELEVANCE,
) -> Retrieval:
    """The subgraph relevant to `question`, or related=False if the map has
    nothing close enough to what it asks about."""
    names = kg.node_names()
    if not names:
        return Retrieval(related=False)

    vectors = store.embed([kg.node_text(n) for n in names])
    query = store.embed_one(question)
    scores = np.asarray(vectors) @ np.asarray(query)

    ranked = sorted(zip(names, scores), key=lambda p: -p[1])
    seeds = [
        SeedConcept(name=n, score=float(s))
        for n, s in ranked[:MAX_SEEDS]
        if s >= min_relevance
    ]
    if not seeds:
        return Retrieval(related=False)

    seed_score = {s.name: s.score for s in seeds}
    included = set(seed_score)
    for seed in seeds:
        included |= kg.neighbors(seed.name)

    # Every relation among the included nodes, most relevant first: a fact is
    # as relevant as the best-matching seed it touches.
    facts = [
        e for e in kg.edges() if e.source in included and e.target in included
    ]
    facts.sort(
        key=lambda e: -max(seed_score.get(e.source, 0.0), seed_score.get(e.target, 0.0))
    )
    facts = facts[:MAX_FACTS]

    # Only nodes that actually appear in the retrieved slice matter downstream.
    mentioned = {e.source for e in facts} | {e.target for e in facts} | set(seed_score)
    definitions = {
        node.name: node.description
        for node in kg.nodes()
        if node.name in mentioned and node.description
    }
    return Retrieval(
        related=True,
        seeds=seeds,
        nodes=sorted(mentioned),
        facts=facts,
        definitions=definitions,
    )
