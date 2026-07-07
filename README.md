# TeachingLearning

This repository contains a desktop app for mapping your study of STEM topics as
a 3D **knowledge graph**: concepts are nodes, and relationships between them —
written in your own plain English — are edges. The app suggests connections you
haven't drawn yet, explains how far-apart concepts link up, and shows which
relationships your understanding hinges on. Everything runs locally; see
`CLAUDE.md` for the full design.

## Setup

Requirements: Linux with WebKit2GTK (preinstalled on most GNOME systems),
[uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) for the
plain-English features.

```bash
# 1. System libraries for the desktop window (PyGObject build headers)
sudo apt-get install -y libgirepository1.0-dev libgirepository-2.0-dev python3-dev

# 2. Ollama + the local language model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# 3. Python environment (also builds PyGObject)
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

Your map lives in `graph.json`; snapshots in `snapshots/`.

## Tests

```bash
uv run pytest
```
