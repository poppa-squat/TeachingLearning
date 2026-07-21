# TeachingLearning

This repository contains a desktop app for mapping your study of STEM topics (or whatever you want) as
a 3D **knowledge graph**: concepts are nodes, and relationships between them are edges. The app suggests connections you
haven't drawn yet, explains how far-apart concepts link up, and shows which
relationships your understanding hinges on. Everything can run locally; see
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

## Configuration

By default the plain-English features run locally through Ollama — no setup
beyond the steps above. To use the DeepSeek cloud model instead (faster, richer
suggestions; paid, prepaid credits), create a `.env` file in the project root:

```bash
cp .env.example .env
# then edit .env and set your DEEPSEEK_API_KEY
```

`.env` is loaded automatically at startup and is git-ignored, so your key stays
out of version control. The keys it accepts:

| Variable            | Default                  | Notes                                    |
| ------------------- | ------------------------ | ---------------------------------------- |
| `LLM_PROVIDER`      | `ollama`                 | Set to `deepseek` to use the cloud API.  |
| `DEEPSEEK_API_KEY`  | —                        | Required when `LLM_PROVIDER=deepseek`.   |
| `DEEPSEEK_MODEL`    | `deepseek-v4-pro`        | Optional override.                       |
| `OLLAMA_URL`        | `http://localhost:11434` | Optional override.                       |
| `OLLAMA_MODEL`      | `qwen3:4b`               | Optional override.                       |

Already-exported shell variables take precedence over `.env`, so you can still
override any of these for a single run:

```bash
LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... uv run main.py
```

DeepSeek spend is logged to `llm_costs.log` (one line per API call).

## Run

```bash
uv run main.py
```

- **Add concepts** in the side panel; click two nodes to select them, then
  describe how they relate in plain English.
- **Import a document**: pick a PDF or text file (or paste text) and the AI
  model distills it into its key concepts and connections, opened as a new
  map. Text-based PDFs only — scanned pages have no text to read.
- **Tabs**: keep several maps open at once; `+` starts an empty one,
  double-click a tab to rename it.
- **Layout toggle**: arrange nodes by hand (positions are saved), or let
  meaning decide — similar concepts cluster together.
- **Suggestions / Explain / Linchpins**: see connections you didn't draw.
- **Ask your map**: ask a question in plain English; the answer is built only
  from the concepts and relationships on the active map. If the map doesn't
  cover the question, the app says so instead of making something up.
- **History**: save a snapshot any time; roll back fearlessly (rolling back
  itself snapshots your current map first).

Your maps and their history live in a per-user data directory, not the project
folder — `~/.local/share/TeachingLearning` on Linux, `~/Library/Application
Support/TeachingLearning` on macOS, `%LOCALAPPDATA%\TeachingLearning` on
Windows. Each map is a folder under `maps/` holding its `graph.json` and
`snapshots/`; `workspace.json` lists the open tabs, and closed maps are kept
in `maps/.trash/`. The embedding cache and cost log live alongside. Set
`TEACHINGLEARNING_DATA_DIR` to point it elsewhere. (An older single
`graph.json` from a previous version is migrated into the first tab
automatically.)

## Tests

```bash
uv run pytest
```
