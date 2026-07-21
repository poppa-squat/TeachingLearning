"""Unit tests for the workspace (tabs) and the document-import pipeline.
No GUI and no model: the LLM step is exercised only through its Pydantic
response shapes, and PDFs are tiny handcrafted files."""

import pytest

from app import ingest
from app.graph import Edge, KnowledgeGraph
from app.llm import ExtractedConcept, ExtractedGraph, ExtractedRelation
from app.workspace import Workspace


# ---- workspace ---------------------------------------------------------------

def test_fresh_workspace_has_one_active_map(tmp_path):
    ws = Workspace(tmp_path)
    assert len(ws.maps()) == 1
    assert ws.active_id() == ws.maps()[0].id
    assert ws.load(ws.active_id()).node_names() == []
    assert (tmp_path / "workspace.json").exists()


def test_create_switch_rename_and_persistence(tmp_path):
    ws = Workspace(tmp_path)
    first = ws.maps()[0]
    second = ws.create("Quantum Notes")
    assert ws.active_id() == second.id  # creating a map activates it
    ws.rename(second.id, "Quantum Mechanics")
    ws.set_active(first.id)

    reopened = Workspace(tmp_path)  # everything survives a restart
    assert [m.title for m in reopened.maps()] == [first.title, "Quantum Mechanics"]
    assert reopened.active_id() == first.id


def test_maps_are_isolated(tmp_path):
    ws = Workspace(tmp_path)
    a, b = ws.active_id(), ws.create("Other").id
    kg = KnowledgeGraph()
    kg.add_node("eigenvalues")
    ws.save(kg, a)
    assert ws.load(a).has_node("eigenvalues")
    assert not ws.load(b).has_node("eigenvalues")


def test_duplicate_titles_get_distinct_ids(tmp_path):
    ws = Workspace(tmp_path)
    a = ws.create("Untitled")
    b = ws.create("Untitled")
    assert a.id != b.id


def test_delete_moves_folder_to_trash(tmp_path):
    ws = Workspace(tmp_path)
    keep = ws.active_id()
    doomed = ws.create("Doomed")
    kg = KnowledgeGraph()
    kg.add_node("saved by the trash")
    ws.save(kg, doomed.id)
    ws.delete(doomed.id)
    assert [m.id for m in ws.maps()] == [keep]
    assert ws.active_id() == keep
    trashed = list((tmp_path / "maps" / ".trash").glob(f"{doomed.id}-*/graph.json"))
    assert len(trashed) == 1
    with pytest.raises(KeyError):
        ws.graph_file(doomed.id)


def test_deleting_the_last_map_leaves_a_fresh_one(tmp_path):
    ws = Workspace(tmp_path)
    only = ws.active_id()
    ws.delete(only)
    assert len(ws.maps()) == 1
    assert ws.active_id() != ""
    assert ws.load(ws.active_id()).node_names() == []


def test_legacy_single_graph_migrates_into_first_tab(tmp_path):
    import json

    kg = KnowledgeGraph()
    kg.add_node("limit", "the value a function approaches")
    (tmp_path / "graph.json").write_text(json.dumps(kg.to_dict()))
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    (snaps / "graph-20260101-000000.json").write_text(json.dumps(kg.to_dict()))

    ws = Workspace(tmp_path)
    assert len(ws.maps()) == 1
    migrated = ws.load(ws.active_id())
    assert migrated.has_node("limit")
    assert not (tmp_path / "graph.json").exists()      # moved, not copied
    assert not (tmp_path / "snapshots").exists()
    assert (ws.snapshot_dir(ws.active_id()) / "graph-20260101-000000.json").exists()


# ---- ingest: text extraction --------------------------------------------------

def _write_pdf(path, text: str) -> None:
    """A minimal one-page PDF with `text` in its content stream."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref_pos,
    )
    path.write_bytes(bytes(out))


def test_extract_text_from_plain_text_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("  Eigenvalues scale eigenvectors.  ")
    assert ingest.extract_text(f) == "Eigenvalues scale eigenvectors."


def test_extract_text_from_pdf(tmp_path):
    f = tmp_path / "paper.pdf"
    _write_pdf(f, "Eigenvalues scale eigenvectors")
    assert "Eigenvalues scale eigenvectors" in ingest.extract_text(f)


def test_extract_text_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        ingest.extract_text(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n ")
    with pytest.raises(ValueError):
        ingest.extract_text(empty)
    weird = tmp_path / "image.png"
    weird.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError):
        ingest.extract_text(weird)


def test_extract_text_rejects_textless_pdf(tmp_path):
    from pypdf import PdfWriter

    f = tmp_path / "scanned.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with f.open("wb") as fh:
        writer.write(fh)
    with pytest.raises(ValueError, match="scanned"):
        ingest.extract_text(f)


def test_clip_caps_per_provider():
    text = "x" * (ingest.CHAR_LIMITS["ollama"] + 1)
    clipped, truncated = ingest.clip(text, "ollama")
    assert truncated and len(clipped) == ingest.CHAR_LIMITS["ollama"]
    same, truncated = ingest.clip(text, "deepseek")
    assert not truncated and same == text
    _, truncated = ingest.clip("short", "ollama")
    assert not truncated


# ---- ingest: records -> graph --------------------------------------------------

def _extraction() -> ExtractedGraph:
    return ExtractedGraph(
        concepts=[
            ExtractedConcept(name="Eigenvalues", description="scaling factors"),
            ExtractedConcept(name="eigenvalues", description="dupe, other case"),
            ExtractedConcept(name="Determinant", description="volume scaling"),
            ExtractedConcept(name="  ", description="blank name, dropped"),
        ],
        relations=[
            ExtractedRelation(
                source="determinant", target="EIGENVALUES",  # sloppy case: resolved
                predicate="is the product of", directed=True,
            ),
            ExtractedRelation(
                source="Eigenvalues", target="Trace",  # unknown endpoint: dropped
                predicate="sum to", directed=True,
            ),
            ExtractedRelation(
                source="Eigenvalues", target="eigenvalues",  # self-loop: dropped
                predicate="equals", directed=False,
            ),
            ExtractedRelation(
                source="Determinant", target="Eigenvalues",  # blank predicate: dropped
                predicate="   ", directed=False,
            ),
        ],
    )


def test_build_graph_cleans_records_deterministically():
    kg = ingest.build_graph(_extraction())
    assert set(kg.node_names()) == {"Eigenvalues", "Determinant"}
    # First description wins for the case-duplicate concept.
    assert kg.node_text("Eigenvalues") == "Eigenvalues: scaling factors"
    assert kg.edges() == [
        Edge(source="Determinant", target="Eigenvalues",
             predicate="is the product of", directed=True)
    ]


def test_build_graph_keeps_symmetric_flag_and_dedupes_edges():
    extracted = ExtractedGraph(
        concepts=[
            ExtractedConcept(name="a", description=""),
            ExtractedConcept(name="b", description=""),
        ],
        relations=[
            ExtractedRelation(source="a", target="b",
                              predicate="is analogous to", directed=False),
            ExtractedRelation(source="b", target="a",  # same symmetric edge again
                              predicate="is analogous to", directed=False),
        ],
    )
    kg = ingest.build_graph(extracted)
    assert len(kg.edges()) == 1
    assert kg.edges()[0].directed is False
