"""Unit tests for the math/storage core. No GUI, no Ollama, no model download:
embeddings are stubbed with fixed vectors."""

import numpy as np
import pytest

from app import predict, reason, storage
from app.graph import Edge, KnowledgeGraph
from app.layout import meaning_positions


class StubStore:
    """Stands in for EmbeddingStore: known texts get preset (normalised)
    vectors; unknown texts get a deterministic pseudo-random one."""

    def __init__(self, presets: dict[str, list[float]] | None = None):
        self.presets = {
            k: np.asarray(v, dtype=np.float32) / np.linalg.norm(v)
            for k, v in (presets or {}).items()
        }

    def embed(self, texts):
        return np.stack([self._one(t) for t in texts])

    def embed_one(self, text):
        return self._one(text)

    @staticmethod
    def similarity(a, b):
        return float(np.dot(a, b))

    def _one(self, text):
        if text in self.presets:
            return self.presets[text]
        rng = np.random.default_rng(abs(hash(text)) % 2**32)
        v = rng.normal(size=8)
        return (v / np.linalg.norm(v)).astype(np.float32)


def calculus_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for n in ["limit", "derivative", "integral", "tangent line", "area"]:
        kg.add_node(n)
    kg.add_edge(Edge(source="limit", target="derivative",
                     predicate="is the foundation of", directed=True))
    kg.add_edge(Edge(source="limit", target="integral",
                     predicate="is the foundation of", directed=True))
    kg.add_edge(Edge(source="derivative", target="tangent line",
                     predicate="gives the slope of", directed=True))
    kg.add_edge(Edge(source="integral", target="area",
                     predicate="computes", directed=True))
    kg.add_edge(Edge(source="derivative", target="integral",
                     predicate="is the inverse of", directed=False))
    return kg


# ---- graph -------------------------------------------------------------------

def test_add_node_idempotent_and_validated():
    kg = KnowledgeGraph()
    assert kg.add_node("limit") is True
    assert kg.add_node("limit") is False
    with pytest.raises(ValueError):
        kg.add_node("   ")


def test_duplicate_edge_rejected_and_parallel_allowed():
    kg = calculus_graph()
    dup = Edge(source="limit", target="derivative",
               predicate="is the foundation of", directed=True)
    assert kg.add_edge(dup) is False
    parallel = Edge(source="limit", target="derivative",
                    predicate="lets you define", directed=True)
    assert kg.add_edge(parallel) is True
    assert len(kg.edges_between("limit", "derivative")) == 2


def test_symmetric_edge_matches_either_orientation():
    kg = calculus_graph()
    flipped = Edge(source="integral", target="derivative",
                   predicate="is the inverse of", directed=False)
    assert kg.add_edge(flipped) is False  # already there, other way round
    kg.remove_edge("integral", "derivative", "is the inverse of")
    assert not kg.has_connection("derivative", "integral")


def test_self_loop_rejected():
    kg = calculus_graph()
    with pytest.raises(ValueError):
        kg.add_edge(Edge(source="limit", target="limit",
                         predicate="is itself", directed=False))


def test_traversal_view_directions():
    t = calculus_graph().traversal_view()
    assert t.has_edge("limit", "derivative")
    assert not t.has_edge("derivative", "limit")       # asymmetric: one-way
    assert t.has_edge("derivative", "integral")
    assert t.has_edge("integral", "derivative")        # symmetric: both ways


def test_remove_node_drops_its_edges():
    kg = calculus_graph()
    kg.remove_node("derivative")
    assert all("derivative" not in (e.source, e.target) for e in kg.edges())


def test_description_stored_and_node_text_enriched():
    kg = KnowledgeGraph()
    kg.add_node("matrix", "a rectangular grid of numbers")
    kg.add_node("limit")
    assert kg.node_text("matrix") == "matrix: a rectangular grid of numbers"
    assert kg.node_text("limit") == "limit"
    kg.set_description("limit", "  the value a function approaches  ")
    assert kg.node_text("limit") == "limit: the value a function approaches"


def test_old_save_without_descriptions_still_loads():
    kg = KnowledgeGraph.from_dict(
        {"nodes": [{"name": "limit", "position": None}], "edges": []}
    )
    assert kg.node_text("limit") == "limit"
    assert kg.nodes()[0].description == ""


# ---- storage -----------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    kg = calculus_graph()
    kg.set_position("limit", (1.0, 2.0, 3.0))
    kg.set_description("limit", "the value a function approaches")
    storage.save(kg, tmp_path / "g.json", tmp_path / "snaps")
    loaded = storage.load(tmp_path / "g.json")
    assert loaded.to_dict() == kg.to_dict()
    assert loaded.node_text("limit") == "limit: the value a function approaches"


def test_snapshot_dedupe_and_optout(tmp_path):
    kg = calculus_graph()
    assert storage.save(kg, tmp_path / "g.json", tmp_path / "snaps") is not None
    assert storage.save(kg, tmp_path / "g.json", tmp_path / "snaps") is None
    kg.add_node("matrix")
    assert storage.save(kg, tmp_path / "g.json", tmp_path / "snaps",
                        snapshot=False) is None
    assert len(storage.list_snapshots(tmp_path / "snaps")) == 1


def test_restore_keeps_current_state_recoverable(tmp_path):
    g, s = tmp_path / "g.json", tmp_path / "snaps"
    kg = calculus_graph()
    snap = storage.save(kg, g, s)
    kg.add_node("matrix")
    storage.save(kg, g, s, snapshot=False)
    restored = storage.restore(snap.stem, g, s)
    assert not restored.has_node("matrix")
    # the pre-rollback state (with "matrix") was snapshotted during restore
    assert any(
        KnowledgeGraph.from_dict(
            __import__("json").loads((s / f"{name}.json").read_text())
        ).has_node("matrix")
        for name in storage.list_snapshots(s)
    )


def test_load_missing_file_gives_empty_graph(tmp_path):
    kg = storage.load(tmp_path / "nope.json")
    assert kg.node_names() == []


# ---- predict -----------------------------------------------------------------

def test_suggestions_exclude_connected_pairs():
    kg = calculus_graph()
    suggestions = predict.suggest_edges(kg, StubStore(), top_k=100)
    for s in suggestions:
        assert not kg.has_connection(s.a, s.b)
        assert 0.0 <= s.score <= 1.0


def test_meaning_signal_ranks_similar_pair_first():
    kg = KnowledgeGraph()
    for n in ["a", "b", "c"]:
        kg.add_node(n)
    store = StubStore({
        "a": [1, 0, 0, 0, 0, 0, 0, 0],
        "b": [0.99, 0.1, 0, 0, 0, 0, 0, 0],   # nearly identical to a
        "c": [-1, 0, 0, 0, 0, 0, 0, 0],       # opposite
    })
    top = predict.suggest_edges(kg, store, top_k=3)[0]
    assert {top.a, top.b} == {"a", "b"}


def test_descriptions_feed_the_meaning_signal():
    # The description changes what gets embedded ("name: description"), so a
    # pair that only looks similar through their definitions ranks first.
    kg = KnowledgeGraph()
    kg.add_node("a", "spectral values of an operator")
    kg.add_node("b")
    kg.add_node("c")
    store = StubStore({
        "a: spectral values of an operator": [1, 0, 0, 0, 0, 0, 0, 0],
        "b": [0.99, 0.1, 0, 0, 0, 0, 0, 0],
        "c": [-1, 0, 0, 0, 0, 0, 0, 0],
    })
    top = predict.suggest_edges(kg, store, top_k=3)[0]
    assert {top.a, top.b} == {"a", "b"}


# ---- reason ------------------------------------------------------------------

def test_paths_respect_direction_by_default():
    kg = calculus_graph()
    assert reason.find_paths(kg, "limit", "tangent line")      # forward chain
    assert not reason.find_paths(kg, "tangent line", "area")   # arrows block it


def test_reverse_fallback_finds_and_marks_paths():
    kg = calculus_graph()
    analysis = reason.analyse_connection(kg, StubStore(), "tangent line", "area")
    assert analysis.used_reverse
    assert analysis.paths
    first_steps = analysis.paths[0]
    assert any(not s.forward for s in first_steps)
    text = reason.path_text(first_steps)
    assert "<- (" in text  # backwards step is rendered against the arrow


def test_grouping_separates_dissimilar_paths():
    # Two orthogonal path-vectors must land in different facets.
    vectors = np.asarray([[1, 0], [0, 1]], dtype=float)
    groups = reason._group_by_similarity(vectors, threshold=0.6)
    assert len(groups) == 2
    close = np.asarray([[1, 0], [0.9, 0.4]], dtype=float)
    close /= np.linalg.norm(close, axis=1, keepdims=True)
    assert len(reason._group_by_similarity(close, threshold=0.6)) == 1


def test_linchpin_is_the_bridge_edge():
    kg = KnowledgeGraph()
    for n in ["a1", "a2", "bridge1", "bridge2", "b1", "b2"]:
        kg.add_node(n)
    cluster = [("a1", "a2"), ("a1", "bridge1"), ("a2", "bridge1"),
               ("bridge2", "b1"), ("bridge2", "b2"), ("b1", "b2")]
    for s, t in cluster:
        kg.add_edge(Edge(source=s, target=t, predicate="relates to", directed=False))
    kg.add_edge(Edge(source="bridge1", target="bridge2",
                     predicate="is the only link to", directed=True))
    top_edge, score = reason.linchpins(kg, top_k=1)[0]
    assert top_edge.predicate == "is the only link to"
    assert score > 0


# ---- layout ------------------------------------------------------------------

def test_small_graph_layout_uses_pca_and_plain_floats():
    positions = meaning_positions(["a", "b", "c"], StubStore())
    assert set(positions) == {"a", "b", "c"}
    for pos in positions.values():
        assert len(pos) == 3
        assert all(type(c) is float for c in pos)


def test_layout_degenerate_sizes():
    assert meaning_positions([], StubStore()) == {}
    assert meaning_positions(["only"], StubStore()) == {"only": (0.0, 0.0, 0.0)}


def test_layout_with_enriched_texts_stays_keyed_by_name():
    positions = meaning_positions(
        ["a", "b", "c"], StubStore(), texts=["a: one", "b: two", "c: three"]
    )
    assert set(positions) == {"a", "b", "c"}


# ---- embeddings cache --------------------------------------------------------

class _FakeModel:
    """Deterministic stand-in for the sentence-transformers model, so the cache
    can be exercised without downloading or loading a real embedding model."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        rows = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % 2**32)
            v = rng.normal(size=self.dim)
            rows.append((v / np.linalg.norm(v)).astype(np.float32))
        return np.stack(rows)


def _store_with_fake_model(cache_file):
    from app.embeddings import EmbeddingStore

    store = EmbeddingStore(cache_file=cache_file)
    store._model = _FakeModel()  # pre-set so no real model is ever loaded
    return store


def test_cache_survives_reload_without_reloading_model(tmp_path):
    cache = tmp_path / "cache.npz"
    texts = ["matrix: a grid of numbers", "real number", "model"]  # "model" is a
    #   deliberate collision with the metadata field name.
    first = _store_with_fake_model(cache).embed(texts)

    reopened = _store_with_fake_model(cache)
    reopened._model = None  # a genuine cache hit must not need the model at all
    again = reopened.embed(texts)

    assert np.allclose(first, again)
    assert reopened._model is None


def test_cache_discarded_when_model_changes(tmp_path, monkeypatch):
    import app.embeddings as emb

    cache = tmp_path / "cache.npz"
    _store_with_fake_model(cache).embed(["vector"])

    monkeypatch.setattr(emb, "MODEL_NAME", "Some/DifferentModel")
    assert emb.EmbeddingStore(cache_file=cache)._cache == {}


def test_legacy_cache_format_is_discarded(tmp_path):
    from app.embeddings import EmbeddingStore

    cache = tmp_path / "cache.npz"
    # Old scheme: each text was its own archive key, with no model tag.
    np.savez(cache, **{"eigenvalue": np.ones(4, dtype=np.float32)})
    assert EmbeddingStore(cache_file=cache)._cache == {}
