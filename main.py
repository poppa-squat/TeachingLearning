"""Starts the app window.

Run with:  uv run main.py
"""

import logging
import os
from pathlib import Path

# WebKitGTK's DMA-BUF renderer breaks WebGL on NVIDIA drivers; must be set
# before WebKit is loaded.
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
