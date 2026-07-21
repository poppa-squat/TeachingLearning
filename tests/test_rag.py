"""Unit tests for graph RAG retrieval (app/rag.py). Pure math — the model
never runs here; embeddings are stubbed with fixed vectors."""

from app import rag
from app.graph import Edge, KnowledgeGraph

from tests.test_core import StubStore

# Orthogonal axes: queries about "spectra" match a/b, everything else is far.
_ON_TOPIC = [1, 0, 0, 0, 0, 0, 0, 0]
_NEARBY = [0.9, 0.3, 0, 0, 0, 0, 0, 0]
_OFF_TOPIC = [0, 0, 0, 1, 0, 0, 0, 0]


def spectral_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for n in ["eigenvalues", "spectrum", "stability", "cooking"]:
        kg.add_node(n)
    kg.add_edge(Edge(source="eigenvalues", target="spectrum",
                     predicate="make up", directed=True))
    kg.add_edge(Edge(source="eigenvalues", target="stability",
                     predicate="determine the stability of", directed=True))
    kg.add_edge(Edge(source="stability", target="cooking",
                     predicate="has nothing on", directed=False))
    return kg


def spectral_store() -> StubStore:
    return StubStore({
        "eigenvalues": _ON_TOPIC,
        "spectrum": _NEARBY,
        "stability": _OFF_TOPIC,
        "cooking": _OFF_TOPIC,
        "what are eigenvalues?": _ON_TOPIC,
        "how do I roast a duck?": [0, 0, 0, 0, 0, 0, 1, 0],
    })


def test_off_map_question_is_flagged_not_answered():
    result = rag.retrieve(spectral_graph(), spectral_store(), "how do I roast a duck?")
    assert result.related is False
    assert result.facts == []


def test_empty_graph_is_never_related():
    assert rag.retrieve(KnowledgeGraph(), spectral_store(), "anything").related is False


def test_retrieval_seeds_neighbourhood_and_ranking():
    result = rag.retrieve(spectral_graph(), spectral_store(), "what are eigenvalues?")
    assert result.related
    seed_names = [s.name for s in result.seeds]
    assert seed_names[0] == "eigenvalues"          # best match first
    assert "cooking" not in seed_names             # below the floor
    # One hop from the seeds pulls in their neighbours...
    assert {"eigenvalues", "spectrum", "stability"} <= set(result.nodes)
    facts = {(e.source, e.target) for e in result.facts}
    assert ("eigenvalues", "spectrum") in facts
    assert ("eigenvalues", "stability") in facts
    # ...but not relations that touch no seed-relevant part of the query:
    # stability–cooking is included only if both endpoints were pulled in.
    # Facts are ranked by their best seed's similarity.
    assert result.facts[0].source == "eigenvalues"


def test_membership_is_invisible_to_retrieval():
    kg = spectral_graph()
    kg.add_member("eigenvalues", "spectrum")
    result = rag.retrieve(kg, spectral_store(), "what are eigenvalues?")
    # Membership adds no fact: only the two real edges around the seeds show up
    # (plus stability–cooking if both endpoints entered the neighbourhood).
    for e in result.facts:
        assert e.predicate != ""  # every fact is a real predicate edge
    assert len([e for e in result.facts
                if {e.source, e.target} == {"eigenvalues", "spectrum"}]) == 1


def test_isolated_matching_concept_yields_no_facts():
    kg = KnowledgeGraph()
    kg.add_node("eigenvalues")
    result = rag.retrieve(kg, spectral_store(), "what are eigenvalues?")
    assert result.related is True
    assert result.seeds and result.facts == []


def test_definitions_only_for_mentioned_nodes():
    kg = spectral_graph()
    kg.set_description("eigenvalues", "scaling factors of a linear map")
    kg.set_description("cooking", "making food")
    store = StubStore({
        "eigenvalues: scaling factors of a linear map": _ON_TOPIC,
        "spectrum": _NEARBY,
        "stability": _OFF_TOPIC,
        "cooking: making food": _OFF_TOPIC,
        "what are eigenvalues?": _ON_TOPIC,
    })
    result = rag.retrieve(kg, store, "what are eigenvalues?")
    assert "eigenvalues" in result.definitions
    # "cooking" only enters if a fact mentions it; its definition must not
    # appear unless it does.
    mentioned = {e.source for e in result.facts} | {e.target for e in result.facts}
    assert ("cooking" in result.definitions) == ("cooking" in mentioned)


def test_fact_cap_keeps_most_relevant():
    kg = KnowledgeGraph()
    kg.add_node("hub")
    presets = {"hub": _ON_TOPIC, "q": _ON_TOPIC}
    for i in range(rag.MAX_FACTS + 10):
        name = f"leaf{i}"
        kg.add_node(name)
        presets[name] = _OFF_TOPIC
        kg.add_edge(Edge(source="hub", target=name,
                         predicate=f"points at {i}", directed=True))
    result = rag.retrieve(kg, StubStore(presets), "q")
    assert len(result.facts) == rag.MAX_FACTS
    assert all(e.source == "hub" for e in result.facts)


def test_fact_text_marks_symmetric_edges():
    directed = Edge(source="a", target="b", predicate="causes", directed=True)
    symmetric = Edge(source="a", target="b", predicate="mirrors", directed=False)
    assert rag.fact_text(directed) == '"a" causes "b"'
    assert rag.fact_text(symmetric) == '"a" mirrors "b" (and vice versa)'
