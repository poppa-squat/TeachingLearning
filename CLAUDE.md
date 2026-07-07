# CLAUDE.md

Guidance for any AI assistant (and humans) working on this project. Read this
first before writing code.

---

## 1. What this project is (in plain terms)

A desktop app for mapping a person's study of STEM topics as a **knowledge
graph**: a picture made of **nodes** (concepts, e.g. "eigenvalues") and
**edges** (relationships between concepts, e.g. "gives the geometric intuition
behind").

The user builds the map themselves:
- They add a node by naming a concept.
- They connect two nodes by selecting them and **typing, in plain English, how
  they relate**. They are not forced to pick from a menu of relationship types.

The app then helps them see connections they didn't draw themselves — including
suggesting how two *unconnected* concepts might relate — and lets them see the
whole thing laid out visually so the structure of their understanding is
literally something they can look at.

The graph is expected to stay **small** (tens to low hundreds of nodes per
user). This is a foundational assumption — see §5.

---

## 2. Key words used in this project

Short glossary so nothing here is mysterious:

- **Node** — one concept on the map.
- **Edge** — a relationship between two concepts.
- **Predicate** — the words describing an edge ("is a special case of"). It is
  **free text**, not a fixed category.
- **Embedding** — a piece of text turned into a list of numbers, arranged so
  that texts with *similar meaning* get *similar numbers*. This lets the program
  do math on meaning.
- **Persistence** — saving the graph to disk so it survives closing the app.
- **Local model** — an AI model that runs on the user's own computer, no
  internet or cloud account needed. Used here as the fallback when the
  network isn't available.

---

## 3. How it works (the pipeline)

1. **User adds a concept** → stored as a node.
2. **User describes a relationship in plain English** → an AI model reads
   the sentence and turns it into a tidy record: `(source, target, predicate,
   directed?)`. The predicate keeps the user's actual wording.
3. **Every node and predicate gets an embedding** (its meaning as numbers),
   computed once and cached.
4. **Suggesting missing edges** — for two concepts the user has *not* connected,
   the app scores how likely they relate using two signals:
   - *Meaning*: are their embeddings close? (cosine similarity)
   - *Structure*: do they share neighbours? (common-neighbour / Adamic-Adar
     heuristics — standard graph measures, no AI training involved)
   When a pair looks related, the AI model then **writes** the likely
   relationship in words.
5. **Reasoning about relationships** — the app can look at the relationships
   *themselves* (not just the concepts) to find, e.g., which relationship is a
   "linchpin" whose removal would break many chains of reasoning.
6. **Explaining a connection between two far-apart concepts** — the app finds
   the paths between them, combines the relationships along each path, checks
   whether the different paths *agree* or *disagree*, and has the AI model
   summarise the result (one relationship if they agree, several distinct
   "facets" if they disagree).

**Core principle — geometry decides, the model verbalises:** all the
*decisions* about what relates to what are made by math on the embeddings (plain,
fast, repeatable). The generative AI model is only used to turn plain English
*into* tidy records, and tidy records back *into* plain English. It never makes
the reasoning decisions. Keep it that way.

---

## 4. Tech stack

A desktop app. The generative model is a cloud API; everything else (graph,
embeddings, math, UI) runs on the user's machine, and a local model serves as
the fallback when the network isn't available.

**Language:** Python.

**The "brains" (all Python):**
- **NetworkX** — holds the graph and does all graph operations (paths,
  shared-neighbour scores, the "relationships-as-nodes" view, importance
  scores). Correct choice specifically because the graph is small.
- **sentence-transformers** — makes the embeddings. Uses `Qwen3-Embedding-0.6B`
  (1024 dims): still small and CPU-fast at this graph size, but sharper on STEM
  vocabulary and on relational phrasing than the original `all-MiniLM-L6-v2`
  starter model. It must be an *embedding* model, not a chat model. The on-disk
  cache is tagged with the model name, so swapping models discards the stale
  cache and recomputes rather than mixing incompatible vector spaces.
- **DeepSeek API** — runs the generative AI model (V4 Pro) used for the two
  "translation" jobs in §3. **Ollama** is the local fallback when the network
  (or an API key) isn't available; both are spoken to through the same
  OpenAI-compatible endpoint, so swapping is configuration, not code (§8).
- **Instructor + Pydantic** — forces the AI model's answer into the exact record
  shape we need. Pydantic defines the shape; Instructor makes the model obey it
  and retries if it doesn't. (See §6 for the shape.)

**Saving (persistence + history):** the graph is saved as a plain JSON file. On
top of that, the app keeps a **history of past versions** so a user can roll back
to an earlier state of their map. Because the graph is small, the simple, correct
approach is to save a **full snapshot** on each save (a numbered/dated copy) and
let the user pick an old snapshot to restore — no clever change-tracking needed.
This exists for a specific reason: users should feel free to experiment without
fear of wrecking their map.

**Similarity math:** plain NumPy. At this graph size, comparing every pair
directly is instant — we deliberately do **not** need heavyweight
fast-search libraries (FAISS etc.). Revisit only past ~10,000 items.

**The window + the drawing:**
- **pywebview** — opens a normal desktop window that shows a small web page, and
  lets Python and that page talk to each other. Keeps the whole app Python-first.
- **3d-force-graph** — draws the graph in **3D** inside that window: nodes float
  in space and the user can rotate, spin, and drag them in all directions, so the
  map is not stuck flat on a plane. Runs on the browser's built-in 3D graphics
  (WebGL, via a library called Three.js — both come bundled, nothing extra to
  install by hand). We borrow this web library purely for the picture; all
  thinking stays in Python. (Note: the earlier 2D option, Cytoscape.js, cannot do
  3D, which is why we use this instead.)
- **umap-learn** — used only for the "meaning-based" layout mode below. It
  squashes each node's long embedding (1024 numbers) down to a 3D position
  (x, y, z) while keeping similar concepts close.

**Node layout — a user toggle between two modes:**
- **Manual mode:** the user drags nodes wherever they like; we save each node's
  (x, y, z) position and reuse it next time.
- **Meaning-based clustering mode:** node positions are computed from the
  embeddings by UMAP, so concepts with similar meaning sit near each other
  automatically. These positions are *derived*, not saved — they're recomputed
  (e.g. when nodes are added).
- The toggle simply chooses which set of positions the 3D view uses. Both are
  kept available; switching does not lose the user's manual arrangement.

---

## 5. Design rules that must not be broken

- **The graph is small.** Every choice above assumes this. Do not add
  infrastructure that only pays off at large scale (approximate nearest-neighbour
  indexes, graph databases for speed, etc.) unless the small-graph assumption
  actually changes.
- **Do NOT use graph machine-learning models** (KGE models like TransE, or graph
  neural networks like R-GCN). They need lots of data to train and a fixed list
  of relationship types. This project has neither — it's small, and predicates
  are free text. The embedding-plus-heuristics approach in §3 is the correct
  substitute and needs no training.
- **Predicates are free text.** Never force the relationship into a fixed
  category list. Meaning lives in the predicate's embedding, not in it matching a
  label.
- **Geometry decides, the model verbalises** (§3). Don't move reasoning into the
  generative model.
- **A local fallback must keep working.** The generative model lives in the
  cloud (DeepSeek), but the app must stay usable without a network: Ollama
  covers the translation jobs locally, and if no model is available at all
  the app degrades gracefully (§8). No *other* feature may require the
  network — the graph, embeddings, math, and UI all stay on the user's
  machine.

---

## 6. The core data shape

A node and an edge, as Pydantic models:

```python
class Node(BaseModel):
    name: str                          # the concept, e.g. "eigenvalues"
    description: str = ""              # the user's own definition; optional
    # ^ when present, the node is embedded as "name: description" (not the bare
    #   name), so the meaning signals use what the user means by the name.
    position: tuple[float, float, float] | None = None
    # ^ saved manual (x, y, z) for manual layout mode; None if never placed by hand.
    #   Meaning-based positions are NOT stored here — they're recomputed by UMAP.

class Edge(BaseModel):
    source: str        # the concept the relationship starts from
    target: str        # the concept it connects to
    predicate: str     # free text, the user's own wording — NOT a category
    directed: bool     # True = asymmetric (one-way); False = symmetric (two-way)
```

Notes:
- `directed` records **whether the relationship is symmetric**, and nothing else.
  `True` = asymmetric: it reads differently each way ("A is a prerequisite for
  B"), and the **source → target order gives the direction** — that's the only
  place direction is stored. `False` = symmetric: it holds both ways ("A is
  analogous to B"), so path-following code may travel it in either direction and
  the source/target order is not meaningful.
- Near-duplicate wordings ("is a prerequisite for" vs "you need before") are
  tidied up *after* storage by clustering their embeddings — not by restricting
  what the user can type.

---

## 7. Project layout (as implemented)

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`), not a
requirements.txt. Run the app with `uv run main.py`; tests with `uv run pytest`.

```
/                     project root
  CLAUDE.md           this file
  README.md           human-facing setup notes
  pyproject.toml      Python dependencies (managed with uv)
  /app
    graph.py          NetworkX graph in memory + the Node/Edge models (§6)
    storage.py        save/load JSON + snapshot history (roll back to old versions)
    embeddings.py     make + cache embeddings (sentence-transformers)
    llm.py            model calls (DeepSeek, or the Ollama fallback), wrapped
                      with Instructor/Pydantic; writes the cost log
    predict.py        missing-edge scoring (meaning + structure)
    reason.py         path-finding, path comparison, agree/disagree logic
    layout.py         meaning-based 3D positions via UMAP (clustering mode)
  /ui
    index.html        the page the graph is drawn into (side panel + 3D canvas)
    main.js           3D drawing (3d-force-graph) + user interaction
    bridge.py         pywebview: connects Python (/app) to the page
    /vendor           committed JS bundle (3d-force-graph + three) so the UI
                      needs no internet at runtime
  /tests              pytest unit tests for the /app core (embeddings stubbed)
  /snapshots          saved graph versions (created at runtime, not committed)
  graph.json          the user's map (created at runtime, not committed)
  llm_costs.log       accrued DeepSeek spend, one line per API call (created
                      at runtime, not committed)
  main.py             starts the app window
```

---

## 8. Decisions made, and what's still open

**Settled:**
- **Persistence:** single JSON save file, plus full-snapshot version history so
  users can roll back (§4). Embedded databases (Kùzu, LanceDB) are a possible
  future upgrade, not needed now.
- **Desktop window:** pywebview window + 3d-force-graph for 3D drawing (§4).
- **Direction:** keep the `directed` flag; it marks symmetric vs asymmetric only
  (§6).
- **Standalone project** for now. If it later plugs into the learning-journal
  website, this layout stays valid and we'd add an export path.

- **Generative model (July 2026):** **DeepSeek V4 Pro** via the cloud API is
  the primary model for the plain-English translation jobs ($0.435/M input,
  $0.87/M output — roughly $0.30 per thousand app calls), chosen for speed
  and for stronger, more *descriptive* predicates when suggesting edges.
  Selected with `LLM_PROVIDER=deepseek` plus `DEEPSEEK_API_KEY`; model
  overridable via `DEEPSEEK_MODEL`. It is a paid, prepaid-credit API: every
  call (retries included) is appended to `llm_costs.log` with its token
  counts, dollar cost, and running total, and `llm.accrued_cost()` exposes
  the total programmatically.
- **Local fallback:** Ollama running **qwen3:4b** (small enough for an 8 GB
  GPU, good at structured output; `OLLAMA_MODEL` / `OLLAMA_URL` to override)
  covers the same jobs when there's no network, no API key, or no credit.
  If no model is available at all the app degrades gracefully: relationship
  sentences are stored verbatim as symmetric edges and AI-assisted features
  switch off.

**Still open:**
- *Automatic fallback isn't wired yet.* The code picks its provider once at
  startup from `LLM_PROVIDER` (defaulting to Ollama) and does not switch to
  the local model on a network failure — today the "fallback" is manual (set
  the env var) or the graceful no-model degradation. Bringing the code in
  line with the design above means: default to DeepSeek when a key is
  present, and fall back to Ollama automatically when the API is
  unreachable.

---

## 9. How the AI assistant should help on this project

- Respect §5 at all times — especially "no graph ML" and "graph is small."
- Prefer plain, direct explanations; define any unfamiliar term on first use.
- When adding a feature, keep the split clean: decisions in the math layer,
  wording in the model layer.
- Flag it explicitly if a request would break a design rule, rather than quietly
  working around it.
