"""Save/load the graph as plain JSON, with full-snapshot version history.

Every save also writes a timestamped copy into the snapshots directory (unless
nothing changed since the last snapshot), so the user can always roll back to
an earlier state of their map. Restoring first snapshots the current state, so
a rollback is itself undoable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app import paths
from app.graph import KnowledgeGraph

_SNAPSHOT_FMT = "%Y%m%d-%H%M%S"


def _graph_file(path: Path | None) -> Path:
    """The save file — the caller's override, or the per-user default."""
    return Path(path) if path is not None else paths.data_path("graph.json")


def _snapshot_dir(snapshot_dir: Path | None) -> Path:
    """The snapshot directory — the caller's override, or the per-user default."""
    return Path(snapshot_dir) if snapshot_dir is not None else paths.data_path("snapshots")


def save(
    kg: KnowledgeGraph,
    path: Path | None = None,
    snapshot_dir: Path | None = None,
    snapshot: bool = True,
) -> Path | None:
    """Write the graph to `path`, and (unless snapshot=False) keep a
    timestamped copy. Returns the snapshot path, or None if no snapshot was
    written (disabled, or nothing changed since the latest one).

    `path`/`snapshot_dir` default to the per-user data directory (see
    app.paths); pass explicit paths to override (e.g. in tests)."""
    path = _graph_file(path)
    payload = json.dumps(kg.to_dict(), indent=2, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    if not snapshot:
        return None

    snapshot_dir = _snapshot_dir(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_snapshot(snapshot_dir)
    if latest is not None and latest.read_text(encoding="utf-8") == payload:
        return None

    name = f"graph-{datetime.now().strftime(_SNAPSHOT_FMT)}"
    snap = snapshot_dir / f"{name}.json"
    # Same-second saves would collide; suffix until unique.
    counter = 1
    while snap.exists():
        snap = snapshot_dir / f"{name}-{counter}.json"
        counter += 1
    snap.write_text(payload, encoding="utf-8")
    return snap


def load(path: Path | None = None) -> KnowledgeGraph:
    """Load the graph, or return an empty one if no save exists yet."""
    path = _graph_file(path)
    if not path.exists():
        return KnowledgeGraph()
    data = json.loads(path.read_text(encoding="utf-8"))
    return KnowledgeGraph.from_dict(data)


def list_snapshots(snapshot_dir: Path | None = None) -> list[str]:
    """Snapshot names, newest first."""
    snapshot_dir = _snapshot_dir(snapshot_dir)
    if not snapshot_dir.exists():
        return []
    return sorted(
        (p.stem for p in snapshot_dir.glob("graph-*.json")), reverse=True
    )


def restore(
    name: str,
    path: Path | None = None,
    snapshot_dir: Path | None = None,
) -> KnowledgeGraph:
    """Roll back to a named snapshot. The current state is snapshotted first."""
    path = _graph_file(path)
    snapshot_dir = _snapshot_dir(snapshot_dir)
    snap = snapshot_dir / f"{name}.json"
    if not snap.exists():
        raise FileNotFoundError(f"No snapshot named {name!r}")
    save(load(path), path, snapshot_dir)  # preserve current state before rollback
    kg = KnowledgeGraph.from_dict(json.loads(snap.read_text(encoding="utf-8")))
    save(kg, path, snapshot_dir)
    return kg


def _latest_snapshot(snapshot_dir: Path) -> Path | None:
    names = list_snapshots(snapshot_dir)
    return snapshot_dir / f"{names[0]}.json" if names else None
