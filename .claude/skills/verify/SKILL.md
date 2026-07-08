---
name: verify
description: Launch and drive the Knowledge Map desktop app end-to-end to verify a change at its real surface (pywebview window + 3d-force-graph page).
---

# Verifying the Knowledge Map app

The surface is a pywebview (WebKitGTK) window showing `ui/index.html`. Unit
tests (`uv run pytest`) stub embeddings and never open the window — real
verification means driving the window.

## Launch, isolated from the user's data

`storage`/`embeddings` paths are **cwd-relative** (`graph.json`,
`snapshots/`, `embeddings_cache.npz`). Never launch a test run from the repo
root or you'll mutate the user's real map. Instead: `os.chdir` to a scratch
dir, seed a `graph.json` there, `sys.path.insert(0, <repo root>)`, set
`WEBKIT_DISABLE_DMABUF_RENDERER=1`, then create the window exactly like
`main.py` does (`webview.create_window(..., js_api=ui.bridge.Api())`) and
pass a driver function to `webview.start(driver, window)` — it runs on a
thread once the window is up and can call `window.evaluate_js(...)`;
`window.destroy()` ends the run.

A working harness from a past verification: seed graph → click nodes/buttons
→ assert DOM + on-disk `graph.json`. Rebuild from the notes below; the
model (`Qwen3-Embedding-0.6B`) loads from the HF cache in ~seconds, and the
first embedding-backed call may take ~30 s cold.

## Driving the page (hard-won gotchas)

- `window.__graph` (set in `initGraph`) is the ForceGraph3D instance;
  `window.__mainLoaded` signals the module ran. Wait for both, then for
  `__graph.graphData().nodes.length` to reach the seeded count.
- **Clicking a 3D node**: project with
  `__graph.graph2ScreenCoords(n.x, n.y, n.z)` (canvas-relative; add the
  canvas `getBoundingClientRect()` offsets), then dispatch on the canvas:
  `pointermove`, pause ~250 ms (hover raycast is throttled), then
  `pointerdown` + `pointerup` (+ `click`). The event init **must** include
  `view: window` (WebKit derives `pageX/pageY` from it — the library
  raycasts with pageX; without it every event lands at 0,0) **and**
  `pointerType: 'mouse', button: 0, buttons: 1` (otherwise the library's
  click path won't fire). `bubbles: true` so the container hears it.
- Panel buttons are plain DOM — find by text and `.click()`.
- `evaluate_js` does **not** await Promises (a pending Promise serializes to
  `{}`). For async results, write to a `window.__x` global in `.then/.catch`
  and poll it.
- Screenshots: `gnome-screenshot -f out.png` (full screen; no ImageMagick on
  this machine). Pause ~1 s first for the frame to settle.

## Flows worth driving

Overview render + container tint → click container → panel "Focus on" →
member/ghost split → select member container → "Add to focus" (Venn legend,
signatures) → ghost "Add to <container>" (check `graph.json` parents on
disk) → cycle rejection via `api.add_member` (expect ValueError naming the
cycle) → breadcrumb back to Overview (saved positions on disk must be
byte-identical) → "Find likely members of containers" → accept a card
(parents count on disk +1).
