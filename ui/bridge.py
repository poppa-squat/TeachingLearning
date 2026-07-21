"""pywebview bridge: the API the web page calls.

Every public method here becomes `window.pywebview.api.<method>(...)` in the
page, returning a promise. pywebview runs each call on its own thread, so slow
work (LLM calls, UMAP) doesn't freeze the window — but we serialise access to
the graph with a lock.

Data policy: every mutation is written straight to the active map's graph.json
(crash safety); timestamped snapshots are only taken on explicit save, on
restore, when a document import creates a map, and when the window closes.
That keeps history meaningful rather than one entry per click.

Only the active map is held in memory; switching tabs saves it and loads the
other. All graph access is serialised by the lock, so a slow import can't
interleave with clicks on the current map.
"""

from __future__ import annotations

import logging
import threading

from pathlib import Path

from app import ingest, llm, predict, reason, storage, views
from app.embeddings import EmbeddingStore
from app.graph import Edge, KnowledgeGraph
from app.layout import meaning_positions
from app.reason import PathAnalysis, path_text
from app.workspace import Workspace

log = logging.getLogger(__name__)


class Api:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ws = Workspace()
        self._map_id = self._ws.active_id()
        self._kg: KnowledgeGraph = self._ws.load(self._map_id)
        self._store = EmbeddingStore()
        self._llm_available: bool | None = None
        # Warm the embedding model and Ollama check off the startup path.
        threading.Thread(target=self._warm_up, daemon=True).start()

    # -- state ----------------------------------------------------------------

    def get_state(self) -> dict:
        """Everything the page needs to (re)draw."""
        with self._lock:
            return {
                "nodes": [n.model_dump() for n in self._kg.nodes()],
                "edges": [e.model_dump() for e in self._kg.edges()],
                "llm_available": self._llm_available,
                "model": llm.MODEL,
                "maps": [m.model_dump() for m in self._ws.maps()],
                "active_map": self._map_id,
            }

    def log_js_error(self, message: str) -> None:
        """Called by the page so front-end failures land in the Python log."""
        log.error("JS error: %s", message)

    def llm_status(self) -> bool | None:
        """True/False once checked; None while the check is still running."""
        return self._llm_available

    # -- concepts ---------------------------------------------------------------

    def add_concept(self, name: str, description: str = "") -> dict:
        with self._lock:
            created = self._kg.add_node(name.strip(), description)
            if created:
                self._autosave()
            return self.get_state() | {"created": created}

    def set_description(self, name: str, description: str) -> dict:
        with self._lock:
            self._kg.set_description(name, description)
            self._autosave()
            return self.get_state()

    def remove_concept(self, name: str) -> dict:
        with self._lock:
            self._kg.remove_node(name)
            self._autosave()
            return self.get_state()

    def set_position(self, name: str, x: float, y: float, z: float) -> None:
        """Called when the user drops a node in manual layout mode."""
        with self._lock:
            self._kg.set_position(name, (x, y, z))
            self._autosave()

    # -- relationships ------------------------------------------------------------

    def add_relationship(self, concept_a: str, concept_b: str, sentence: str) -> dict:
        """Parse the user's plain-English sentence into a tidy edge and add it.
        If the local model is unavailable, the sentence itself becomes the
        predicate (symmetric, source order as selected) — nothing is lost."""
        sentence = sentence.strip()
        if not sentence:
            raise ValueError("Describe the relationship first")
        if self._llm_available:
            try:
                edge = llm.parse_relationship(concept_a, concept_b, sentence)
            except Exception:
                log.exception("LLM parse failed; storing sentence verbatim")
                edge = Edge(
                    source=concept_a, target=concept_b,
                    predicate=sentence, directed=False,
                )
        else:
            edge = Edge(
                source=concept_a, target=concept_b,
                predicate=sentence, directed=False,
            )
        with self._lock:
            added = self._kg.add_edge(edge)
            if added:
                self._autosave()
            return self.get_state() | {
                "added": added,
                "edge": edge.model_dump(),
            }

    def add_edge_direct(
        self, source: str, target: str, predicate: str, directed: bool
    ) -> dict:
        """Add an already-tidy edge (e.g. an accepted suggestion)."""
        edge = Edge(
            source=source, target=target, predicate=predicate, directed=directed
        )
        with self._lock:
            added = self._kg.add_edge(edge)
            if added:
                self._autosave()
            return self.get_state() | {"added": added}

    def remove_relationship(self, source: str, target: str, predicate: str) -> dict:
        with self._lock:
            self._kg.remove_edge(source, target, predicate)
            self._autosave()
            return self.get_state()

    # -- membership (abstraction levels) ----------------------------------------

    def add_member(self, child: str, parent: str) -> dict:
        """Record that child is part of parent. Cycles raise, and the
        rejection reaches the page as an error message."""
        with self._lock:
            added = self._kg.add_member(child, parent)
            if added:
                self._autosave()
            return self.get_state() | {"added": added}

    def remove_member(self, child: str, parent: str) -> dict:
        with self._lock:
            self._kg.remove_member(child, parent)
            self._autosave()
            return self.get_state()

    def suggest_members(self, top_k: int = 8) -> list[dict]:
        """Concepts that probably belong inside a container (math only)."""
        with self._lock:
            suggestions = predict.suggest_members(self._kg, self._store, top_k=top_k)
        return [s.model_dump() for s in suggestions]

    def get_focus_view(self, focused: list[str]) -> dict:
        """Everything the page needs to draw a zoom into these containers."""
        with self._lock:
            return views.focus_view(self._kg, focused)

    # -- the smart features ----------------------------------------------------

    def suggest(self, top_k: int = 8) -> list[dict]:
        """Unconnected pairs that probably relate (math only, fast)."""
        with self._lock:
            suggestions = predict.suggest_edges(self._kg, self._store, top_k=top_k)
        return [s.model_dump() for s in suggestions]

    def verbalise(self, concept_a: str, concept_b: str) -> dict:
        """Have the model word the likely relationship for a suggested pair."""
        if not self._llm_available:
            raise RuntimeError("The local model isn't available")
        with self._lock:
            context = [
                e
                for e in self._kg.edges()
                if {concept_a, concept_b} & {e.source, e.target}
            ]
        edge = llm.verbalise_suggestion(concept_a, concept_b, context)
        return edge.model_dump()

    def explain(self, concept_a: str, concept_b: str) -> dict:
        """Explain how two concepts connect: paths, facet groups, and (when
        the model is up) a sentence per facet."""
        with self._lock:
            analysis: PathAnalysis = reason.analyse_connection(
                self._kg, self._store, concept_a, concept_b
            )
        result = analysis.model_dump()
        result["path_texts"] = [path_text(p) for p in analysis.paths]
        if analysis.paths and self._llm_available:
            try:
                result["facet_sentences"] = llm.summarise_connection(analysis).facets
            except Exception:
                log.exception("LLM summary failed; returning raw chains only")
                result["facet_sentences"] = None
        else:
            result["facet_sentences"] = None
        return result

    def linchpins(self, top_k: int = 5) -> list[dict]:
        """The relationships most chains of reasoning depend on."""
        with self._lock:
            scored = reason.linchpins(self._kg, top_k=top_k)
        return [
            {"edge": edge.model_dump(), "score": score}
            for edge, score in scored
            if score > 0
        ]

    def get_meaning_positions(self) -> dict:
        """Derived (x, y, z) per concept for meaning-based layout mode."""
        with self._lock:
            names = self._kg.node_names()
            texts = [self._kg.node_text(n) for n in names]
        positions = meaning_positions(names, self._store, texts)
        return {name: list(pos) for name, pos in positions.items()}

    # -- maps (tabs) -------------------------------------------------------------

    def switch_map(self, map_id: str) -> dict:
        with self._lock:
            if map_id != self._map_id:
                self._ws.save(self._kg, self._map_id, snapshot=False)
                self._ws.set_active(map_id)
                self._map_id = map_id
                self._kg = self._ws.load(map_id)
            return self.get_state()

    def new_map(self, title: str = "") -> dict:
        with self._lock:
            self._ws.save(self._kg, self._map_id, snapshot=False)
            info = self._ws.create(title)
            self._map_id = info.id
            self._kg = KnowledgeGraph()
            self._autosave()
            return self.get_state()

    def rename_map(self, map_id: str, title: str) -> dict:
        with self._lock:
            self._ws.rename(map_id, title)
            return self.get_state()

    def close_map(self, map_id: str) -> dict:
        """Delete a map (its folder is kept in maps/.trash on disk). The
        workspace guarantees at least one map remains."""
        with self._lock:
            self._ws.delete(map_id)
            self._map_id = self._ws.active_id()
            self._kg = self._ws.load(self._map_id)
            return self.get_state()

    # -- importing a document ----------------------------------------------------

    def import_document(self) -> dict:
        """Pick a PDF/text file, distill it with the model, and open the
        resulting graph in a new tab. Returns {"cancelled": True} if the user
        dismissed the file dialog."""
        import webview

        picked = webview.windows[0].create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=("Documents (*.pdf;*.txt;*.md)", "All files (*.*)"),
        )
        if not picked:
            return {"cancelled": True}
        path = Path(picked[0])
        return self._import(ingest.extract_text(path), title=path.stem)

    def import_text(self, text: str, title: str = "") -> dict:
        """Distill pasted text into a graph in a new tab."""
        if not text.strip():
            raise ValueError("Paste some text first")
        return self._import(text.strip(), title=title.strip() or "Imported text")

    def _import(self, text: str, title: str) -> dict:
        if not self._llm_available:
            raise RuntimeError(
                "Importing needs the AI model, which isn't available right now"
            )
        clipped, truncated = ingest.clip(text, llm.PROVIDER)
        kg = ingest.build_graph(llm.extract_graph(clipped))
        with self._lock:
            self._ws.save(self._kg, self._map_id, snapshot=False)
            info = self._ws.create(title)
            self._map_id = info.id
            self._kg = kg
            self._ws.save(kg, self._map_id)  # first snapshot of the new map
            state = self.get_state()
        # Warm the new texts into the embedding cache in the background so
        # suggestions/layout are snappy on the fresh map.
        threading.Thread(target=self._warm_embeddings, daemon=True).start()
        return state | {
            "report": {
                "concepts": len(kg.node_names()),
                "relations": len(kg.edges()),
                "truncated": truncated,
            }
        }

    # -- snapshots -----------------------------------------------------------------

    def save_snapshot(self) -> dict:
        with self._lock:
            snap = self._ws.save(self._kg, self._map_id)
            return {"snapshot": snap.stem if snap else None,
                    "snapshots": self.list_snapshots()}

    def list_snapshots(self) -> list[str]:
        with self._lock:
            return storage.list_snapshots(self._ws.snapshot_dir(self._map_id))

    def restore_snapshot(self, name: str) -> dict:
        with self._lock:
            self._kg = storage.restore(
                name,
                self._ws.graph_file(self._map_id),
                self._ws.snapshot_dir(self._map_id),
            )
            return self.get_state()

    # -- lifecycle --------------------------------------------------------------

    def on_closing(self) -> None:
        """Final snapshot when the window closes."""
        with self._lock:
            self._ws.save(self._kg, self._map_id)

    # -- internal ---------------------------------------------------------------

    def _autosave(self) -> None:
        self._ws.save(self._kg, self._map_id, snapshot=False)

    def _warm_up(self) -> None:
        self._llm_available = llm.is_available()
        if not self._llm_available:
            log.warning(
                "Ollama/%s not reachable — relationship sentences will be "
                "stored verbatim until it is.", llm.MODEL,
            )
        self._warm_embeddings()

    def _warm_embeddings(self) -> None:
        try:
            with self._lock:
                node_texts = [self._kg.node_text(n) for n in self._kg.node_names()]
                predicates = [e.predicate for e in self._kg.edges()]
            texts = node_texts + predicates
            if texts:
                self._store.embed(texts)  # loads the model + fills the cache
        except Exception:
            log.exception("Embedding warm-up failed")
