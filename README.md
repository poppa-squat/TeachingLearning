# TeachingLearning

This repository contains a desktop app for mapping your study of STEM topics as
a 3D **knowledge graph**: concepts are nodes, and relationships between them —
written in your own plain English — are edges. The app suggests connections you
haven't drawn yet, explains how far-apart concepts link up, and shows which
relationships your understanding hinges on. Everything runs locally; see
`CLAUDE.md` for the full design.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and, for the plain-English
features, [Ollama](https://ollama.com). The desktop window uses each platform's
built-in web view — WebKit2GTK on Linux, WebView2 on Windows, Cocoa WebKit on
macOS — so no browser is bundled.

```bash
# 1. Linux only: system libraries for the desktop window (PyGObject build
#    headers). Not needed on macOS or Windows — uv skips PyGObject there.
sudo apt-get install -y libgirepository1.0-dev libgirepository-2.0-dev python3-dev

# 2. Ollama + the local language model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# 3. Python environment
uv sync
```

The first run downloads the small embedding model (~90 MB) automatically.
If Ollama isn't running, the app still works: your relationship sentences are
stored verbatim and the AI-assisted features switch off until it's back.

## Run

```bash
uv run main.py
```

- **Add concepts** in the side panel; click two nodes to select them, then
  describe how they relate in plain English.
- **Layout toggle**: arrange nodes by hand (positions are saved), or let
  meaning decide — similar concepts cluster together.
- **Suggestions / Explain / Linchpins**: see connections you didn't draw.
- **History**: save a snapshot any time; roll back fearlessly (rolling back
  itself snapshots your current map first).

Your map and its history live in a per-user data directory, not the project
folder — `~/.local/share/TeachingLearning` on Linux, `~/Library/Application
Support/TeachingLearning` on macOS, `%LOCALAPPDATA%\TeachingLearning` on
Windows (`graph.json`, `snapshots/`, plus the embedding cache and cost log).
Set `TEACHINGLEARNING_DATA_DIR` to point it elsewhere.

## Tests

```bash
uv run pytest
```
