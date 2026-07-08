"""Starts the app window.

Run with:  uv run main.py
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing anything from /app: llm.py reads LLM_PROVIDER and
# the DeepSeek settings at import time, so the values must be in the environment
# first. Real (already-exported) env vars win over the file.
load_dotenv(Path(__file__).parent / ".env")

# WebKitGTK's DMA-BUF renderer breaks WebGL on NVIDIA drivers; must be set
# before WebKit is loaded. Linux/WebKitGTK-only — on macOS (Cocoa WebKit) and
# Windows (WebView2) this variable is meaningless, so scope it to Linux.
if sys.platform.startswith("linux"):
    os.environ.setdefault("WEBKIT_DISABLE_DMABUF_RENDERER", "1")

import webview

from ui.bridge import Api


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    api = Api()
    window = webview.create_window(
        "Knowledge Map",
        str(Path(__file__).parent / "ui" / "index.html"),
        js_api=api,
        width=1320,
        height=860,
        min_size=(900, 600),
    )
    window.events.closing += api.on_closing
    webview.start()


if __name__ == "__main__":
    main()
