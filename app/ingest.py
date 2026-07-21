"""Turn a document (PDF or plain text) into a fresh KnowledgeGraph.

The pipeline is deliberately thin:

1. `extract_text` pulls the raw text out of the file (pypdf for PDFs — text
   layer only, no OCR; anything else is read as plain text).
2. `clip` caps the text so it fits the model's context (the local model has a
   far smaller window than the cloud API).
3. `llm.extract_graph` — the model's third translation job — distills the text
   into tidy concept/relation records.
4. `build_graph` turns those records into a KnowledgeGraph, deterministically:
   dedupe concepts, match relation endpoints to concept names case-insensitively,
   and drop anything that doesn't resolve. No model calls here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.graph import Edge, KnowledgeGraph

if TYPE_CHECKING:  # avoid importing the LLM stack just for a type name
    from app.llm import ExtractedGraph

_TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown"}

# Character caps before the text goes to the model. DeepSeek's window takes a
# whole paper comfortably; the local qwen3:4b runs with a much smaller context,
# so its cap is tight enough not to overflow it.
CHAR_LIMITS = {"deepseek": 120_000, "ollama": 12_000}


def extract_text(path: Path | str) -> str:
    """The document's text. Raises ValueError for unsupported or empty files
    (an empty PDF usually means a scanned/image-only one)."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"No such file: {path}")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError(
                "No extractable text in this PDF — it may be scanned images. "
                "Only text-based PDFs are supported."
            )
        return text.strip()
    if path.suffix.lower() in _TEXT_SUFFIXES or not path.suffix:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            raise ValueError("The file is empty")
        return text
    raise ValueError(
        f"Unsupported file type {path.suffix!r} — use a PDF or a plain-text file"
    )


def clip(text: str, provider: str) -> tuple[str, bool]:
    """Cap the text for the given provider. Returns (text, was_truncated)."""
    limit = CHAR_LIMITS.get(provider, CHAR_LIMITS["ollama"])
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def build_graph(extracted: "ExtractedGraph") -> KnowledgeGraph:
    """Tidy records -> KnowledgeGraph. Deterministic cleanup only: duplicate
    concepts collapse (first description wins), relation endpoints are matched
    to concept names ignoring case/whitespace, and relations that don't resolve
    to two distinct extracted concepts are dropped."""
    kg = KnowledgeGraph()
    canonical: dict[str, str] = {}  # casefolded name -> stored name
    for concept in extracted.concepts:
        name = concept.name.strip()
        if not name or name.casefold() in canonical:
            continue  # blank, or the same concept in a different case
        canonical[name.casefold()] = name
        kg.add_node(name, concept.description.strip())

    for relation in extracted.relations:
        source = canonical.get(relation.source.strip().casefold())
        target = canonical.get(relation.target.strip().casefold())
        predicate = relation.predicate.strip()
        if not source or not target or source == target or not predicate:
            continue
        kg.add_edge(
            Edge(
                source=source,
                target=target,
                predicate=predicate,
                directed=relation.directed,
            )
        )
    return kg
