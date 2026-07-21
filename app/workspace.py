"""The workspace: several maps, each shown as a tab in the UI.

Each map is a self-contained folder in the per-user data directory:

    maps/<id>/graph.json     the map itself
    maps/<id>/snapshots/     its version history

A single `workspace.json` at the data-dir root lists the maps (id + title, in
tab order) and remembers which one is active. Older installs kept exactly one
map as `graph.json` + `snapshots/` at the root; on first run those are moved
into `maps/` as the first tab, so nothing is lost.

Closing a tab deletes the map from the workspace, but the folder is moved into
`maps/.trash/` rather than erased — cheap insurance against a mis-click.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from app import paths, storage
from app.graph import KnowledgeGraph

_WORKSPACE_FILE = "workspace.json"
_MAPS_DIR = "maps"
_TRASH_DIR = ".trash"
_DEFAULT_TITLE = "My Map"


class MapInfo(BaseModel):
    id: str
    title: str


class Workspace:
    """Owns workspace.json and the maps/ directory tree."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else paths.data_dir()
        self._maps: list[MapInfo] = []
        self._active = ""
        self._load_or_init()

    # -- queries ---------------------------------------------------------------

    def maps(self) -> list[MapInfo]:
        return list(self._maps)

    def active_id(self) -> str:
        return self._active

    def title(self, map_id: str) -> str:
        return self._info(map_id).title

    def graph_file(self, map_id: str) -> Path:
        self._info(map_id)
        return self._root / _MAPS_DIR / map_id / "graph.json"

    def snapshot_dir(self, map_id: str) -> Path:
        self._info(map_id)
        return self._root / _MAPS_DIR / map_id / "snapshots"

    # -- graph I/O (thin wrappers over storage with per-map paths) --------------

    def load(self, map_id: str) -> KnowledgeGraph:
        return storage.load(self.graph_file(map_id))

    def save(self, kg: KnowledgeGraph, map_id: str, snapshot: bool = True):
        return storage.save(
            kg, self.graph_file(map_id), self.snapshot_dir(map_id), snapshot
        )

    # -- mutations --------------------------------------------------------------

    def set_active(self, map_id: str) -> None:
        self._info(map_id)
        self._active = map_id
        self._write()

    def create(self, title: str = "") -> MapInfo:
        title = title.strip() or "Untitled"
        info = MapInfo(id=self._new_id(title), title=title)
        (self._root / _MAPS_DIR / info.id).mkdir(parents=True)
        self._maps.append(info)
        self._active = info.id
        self._write()
        return info

    def rename(self, map_id: str, title: str) -> None:
        title = title.strip()
        if not title:
            raise ValueError("A map needs a title")
        self._info(map_id).title = title
        self._write()

    def delete(self, map_id: str) -> None:
        """Drop the map from the workspace; its folder goes to maps/.trash/.
        Deleting the last map leaves a fresh empty one, so there is always a
        tab to show."""
        info = self._info(map_id)
        folder = self._root / _MAPS_DIR / map_id
        if folder.exists():
            trash = self._root / _MAPS_DIR / _TRASH_DIR
            trash.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.move(str(folder), str(trash / f"{map_id}-{stamp}"))
        self._maps.remove(info)
        if not self._maps:
            self.create(_DEFAULT_TITLE)  # writes workspace.json itself
            return
        if self._active == map_id:
            self._active = self._maps[0].id
        self._write()

    # -- internal ---------------------------------------------------------------

    def _info(self, map_id: str) -> MapInfo:
        for info in self._maps:
            if info.id == map_id:
                return info
        raise KeyError(f"Unknown map: {map_id!r}")

    def _new_id(self, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "map"
        taken = {info.id for info in self._maps}
        # The trash may hold a folder with this id from an earlier life too.
        trash = self._root / _MAPS_DIR / _TRASH_DIR
        candidate, counter = slug, 2
        while candidate in taken or (self._root / _MAPS_DIR / candidate).exists():
            candidate = f"{slug}-{counter}"
            counter += 1
        return candidate

    def _load_or_init(self) -> None:
        ws_file = self._root / _WORKSPACE_FILE
        if ws_file.exists():
            data = json.loads(ws_file.read_text(encoding="utf-8"))
            self._maps = [MapInfo(**m) for m in data.get("maps", [])]
            self._active = data.get("active", "")
            if not self._maps:
                self.create(_DEFAULT_TITLE)
            elif not any(m.id == self._active for m in self._maps):
                self._active = self._maps[0].id
                self._write()
            return
        self._migrate_legacy() or self.create(_DEFAULT_TITLE)

    def _migrate_legacy(self) -> bool:
        """Move a pre-workspace `graph.json` (+ `snapshots/`) into maps/ as the
        first tab. Returns True if there was anything to migrate."""
        legacy_graph = self._root / "graph.json"
        if not legacy_graph.exists():
            return False
        info = MapInfo(id=self._new_id(_DEFAULT_TITLE), title=_DEFAULT_TITLE)
        folder = self._root / _MAPS_DIR / info.id
        folder.mkdir(parents=True)
        shutil.move(str(legacy_graph), str(folder / "graph.json"))
        legacy_snaps = self._root / "snapshots"
        if legacy_snaps.exists():
            shutil.move(str(legacy_snaps), str(folder / "snapshots"))
        self._maps = [info]
        self._active = info.id
        self._write()
        return True

    def _write(self) -> None:
        payload = {
            "maps": [info.model_dump() for info in self._maps],
            "active": self._active,
        }
        (self._root / _WORKSPACE_FILE).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
