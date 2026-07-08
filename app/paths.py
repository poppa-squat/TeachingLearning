"""Where the app keeps its per-user runtime files.

The saved graph, its snapshot history, the embeddings cache, and the LLM cost
log are the *user's* data, not part of the program. They belong in the
operating system's standard per-user data location — not the current working
directory, which is merely wherever the app happened to be launched from and
which disappears entirely once the app is packaged as a binary rather than run
from the repo root.

`platformdirs` resolves the right place on each OS:

    Linux    ~/.local/share/TeachingLearning
    macOS    ~/Library/Application Support/TeachingLearning
    Windows  %LOCALAPPDATA%\\TeachingLearning

Set TEACHINGLEARNING_DATA_DIR to override the location (handy for development
and tests, which point it at a throwaway directory).
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

APP_NAME = "TeachingLearning"
DATA_DIR_ENV = "TEACHINGLEARNING_DATA_DIR"


def data_dir() -> Path:
    """The per-user directory for this app's runtime files.

    Resolved fresh on each call — so the override env var is honoured even if it
    is set after import — and created on demand.
    """
    override = os.environ.get(DATA_DIR_ENV)
    base = Path(override) if override else Path(platformdirs.user_data_dir(APP_NAME))
    base.mkdir(parents=True, exist_ok=True)
    return base


def data_path(name: str) -> Path:
    """Path to `name` inside the per-user data directory."""
    return data_dir() / name
