"""The generative model's two translation jobs (and nothing more):

1. Plain English -> tidy record: the user's sentence about how two concepts
   relate becomes (source, target, predicate, directed).
2. Tidy records -> plain English: word a suggested relationship, or summarise
   the facets of a multi-path connection.

All reasoning decisions are made elsewhere by math on embeddings; this module
only translates. It talks to one of two providers, both through the
OpenAI-compatible chat endpoint, with Instructor + Pydantic forcing the answer
into shape (retrying when the model strays):

- "ollama" (default): the local server, free, fully offline.
- "deepseek": the DeepSeek cloud API (paid, prepaid credits). Opt-in only —
  this is a deliberate exception to the everything-local rule, and the app
  still degrades gracefully without it.

Provider selection (env vars):
    LLM_PROVIDER        "ollama" (default) or "deepseek"
    OLLAMA_URL          default "http://localhost:11434"
    OLLAMA_MODEL        default "qwen3:4b"
    DEEPSEEK_API_KEY    required when LLM_PROVIDER=deepseek
    DEEPSEEK_MODEL      default "deepseek-v4-pro"
    LLM_COST_LOG        default <per-user data dir>/llm_costs.log (see app.paths)

Cost tracking: when the provider is DeepSeek, every API response (including
Instructor's validation retries — they cost money too) appends one line to the
cost log with its token counts, dollar cost, and the running total.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from app import paths
from app.graph import Edge
from app.reason import PathAnalysis, path_text

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

DEEPSEEK_URL = os.environ.get("DEEPSEEK_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

MODEL = DEEPSEEK_MODEL if PROVIDER == "deepseek" else OLLAMA_MODEL
_MAX_RETRIES = 3

# qwen3's soft switch for skipping its "thinking" preamble. Only sent to
# Ollama; cloud models would just see it as noise (billed noise, at that).
_NUDGE = "" if PROVIDER == "deepseek" else " /no_think"


class LLMUnavailable(Exception):
    """The configured provider can't be used: Ollama isn't running / the model
    isn't pulled, or DEEPSEEK_API_KEY is missing."""


# --------------------------------------------------------------------------
# Cost tracking (DeepSeek only — Ollama is free)
# --------------------------------------------------------------------------

# $ per 1M tokens: (input cache-hit, input cache-miss, output).
# Source: https://api-docs.deepseek.com/quick_start/pricing (July 2026).
_PRICES = {
    "deepseek-v4-pro": (0.003625, 0.435, 0.87),
    "deepseek-v4-flash": (0.0028, 0.14, 0.28),
}

COST_LOG = Path(os.environ.get("LLM_COST_LOG") or paths.data_path("llm_costs.log"))

_current_job = "?"  # which public function made the call; set before each call
_total: float | None = None  # running all-time total, seeded from the log file


def accrued_cost() -> float:
    """All-time spend in dollars, as recorded in the cost log."""
    global _total
    if _total is None:
        _total = 0.0
        if COST_LOG.exists():
            for line in COST_LOG.read_text().splitlines():
                match = re.search(r"cost=\$([0-9.]+)", line)
                if match:
                    _total += float(match.group(1))
    return _total


def _log_cost(response) -> None:
    """Instructor hook: fires on every raw API response, retries included."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    model = getattr(response, "model", MODEL)

    # DeepSeek splits prompt tokens into cache hits (cheap) and misses (full
    # price); fall back to counting everything as misses if the fields are
    # absent.
    hit = getattr(usage, "prompt_cache_hit_tokens", None) or 0
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if miss is None:
        miss = usage.prompt_tokens - hit
    out = usage.completion_tokens

    hit_p, miss_p, out_p = (
        _PRICES.get(model) or _PRICES.get(DEEPSEEK_MODEL) or _PRICES["deepseek-v4-pro"]
    )
    cost = (hit * hit_p + miss * miss_p + out * out_p) / 1_000_000

    global _total
    _total = accrued_cost() + cost

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = (
        f"{stamp}  {model}  {_current_job}  "
        f"in={hit + miss}(hit={hit},miss={miss})  out={out}  "
        f"cost=${cost:.6f}  total=${_total:.4f}\n"
    )
    try:
        with COST_LOG.open("a") as f:
            f.write(line)
    except OSError as exc:  # don't let a logging failure break the app
        print(f"warning: could not write cost log: {exc}")


# --------------------------------------------------------------------------
# Response shapes
# --------------------------------------------------------------------------


class ParsedRelation(BaseModel):
    """The tidy record extracted from one plain-English sentence."""

    source: str = Field(description="The concept the relationship starts from")
    target: str = Field(description="The concept it points to")
    predicate: str = Field(
        description=(
            "The relationship phrase, keeping the user's own wording, e.g. "
            "'is a special case of'. Just the phrase — not a full sentence, "
            "and not the concept names."
        )
    )
    directed: bool = Field(
        description=(
            "True if the relationship is asymmetric (reads differently each "
            "way, like 'is a prerequisite for'); False if symmetric (holds "
            "both ways, like 'is analogous to')."
        )
    )

    @field_validator("source", "target")
    @classmethod
    def _must_be_a_selected_concept(cls, value: str, info: ValidationInfo) -> str:
        allowed = (info.context or {}).get("concepts")
        if allowed and value not in allowed:
            raise ValueError(f"must be exactly one of {sorted(allowed)!r}")
        return value


class SuggestedRelation(ParsedRelation):
    """Like ParsedRelation, but the model is writing the predicate itself, so
    the field invites a rich description instead of preserving user wording."""

    predicate: str = Field(
        description=(
            "A specific, informative relationship phrase saying HOW the "
            "concepts relate — the mechanism, intuition, or dependency. "
            "Descriptive multi-word phrasing is encouraged, e.g. 'provides "
            "the geometric intuition behind' or 'determines the stability "
            "of solutions to'. Avoid generic labels like 'is related to' "
            "or 'is a special case of' unless nothing more precise is true. "
            "It must read naturally as: source <predicate> target."
        )
    )


class ConnectionSummary(BaseModel):
    """Plain-English wording of a path analysis, one sentence per facet."""

    facets: list[str] = Field(
        description="One clear sentence per distinct facet of the connection"
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def is_available() -> bool:
    """True when the configured provider is reachable and usable."""
    if PROVIDER == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            return False
        try:
            # Listing models is free — no tokens billed.
            OpenAI(
                base_url=DEEPSEEK_URL, api_key=os.environ["DEEPSEEK_API_KEY"]
            ).models.list()
            return True
        except Exception:
            return False
    try:
        import ollama

        client = ollama.Client(host=OLLAMA_URL)
        models = [m.model for m in client.list().models]
        return any(name.startswith(MODEL) for name in models)
    except Exception:
        return False


def parse_relationship(concept_a: str, concept_b: str, sentence: str) -> Edge:
    """Turn the user's sentence about how two selected concepts relate into an
    Edge. The user's wording is kept as the predicate."""
    global _current_job
    _current_job = "parse_relationship"
    result = _client().chat.completions.create(
        model=MODEL,
        max_retries=_MAX_RETRIES,
        response_model=ParsedRelation,
        context={"concepts": {concept_a, concept_b}},
        messages=[
            {
                "role": "system",
                "content": (
                    "You turn one sentence describing how two concepts relate "
                    "into a tidy record. The source and target must each be "
                    "exactly one of the two given concepts (copy them "
                    "verbatim, one each). Preserve the user's own relationship "
                    "wording as the predicate; do not rephrase it into a "
                    "category, and do not shorten it — a long, descriptive "
                    "predicate is fine." + _NUDGE
                ),
            },
            {
                "role": "user",
                "content": (
                    f'The two concepts: "{concept_a}" and "{concept_b}".\n'
                    f"The user wrote: {sentence!r}"
                ),
            },
        ],
    )
    if result.source == result.target:
        raise ValueError("Model failed to distinguish the two concepts")
    return Edge(**result.model_dump())


def verbalise_suggestion(
    concept_a: str,
    concept_b: str,
    context_edges: list[Edge],
) -> Edge:
    """Write the likely relationship between two concepts the math layer has
    already decided are probably related. Context is each concept's existing
    relationships, so the wording fits the user's map."""
    global _current_job
    _current_job = "verbalise_suggestion"
    context = "\n".join(
        f'- "{e.source}" {e.predicate} "{e.target}"'
        f"{'' if e.directed else ' (and vice versa)'}"
        for e in context_edges
    ) or "(no existing relationships)"
    result = _client().chat.completions.create(
        model=MODEL,
        max_retries=_MAX_RETRIES,
        response_model=SuggestedRelation,
        context={"concepts": {concept_a, concept_b}},
        messages=[
            {
                "role": "system",
                "content": (
                    "Two STEM concepts on a learner's knowledge map appear to "
                    "be related. Propose the most likely relationship between "
                    "them as a predicate phrase. Be specific about HOW they "
                    "relate — the mechanism, intuition, or dependency — not "
                    "just THAT they relate. A descriptive multi-word predicate "
                    "is better than a terse category label. The phrase must "
                    "read naturally as: source <predicate> target. The source "
                    "and target must each be exactly one of the two given "
                    "concepts." + _NUDGE
                ),
            },
            {
                "role": "user",
                "content": (
                    f'The two concepts: "{concept_a}" and "{concept_b}".\n'
                    f"Their existing relationships on the map:\n{context}"
                ),
            },
        ],
    )
    if result.source == result.target:
        raise ValueError("Model failed to distinguish the two concepts")
    return Edge(**result.model_dump())


def summarise_connection(analysis: PathAnalysis) -> ConnectionSummary:
    """Word the result of a path analysis: one sentence if the paths agree,
    one per facet if they don't. The grouping was already decided by geometry;
    the model only writes it out."""
    if not analysis.paths:
        return ConnectionSummary(facets=[])

    global _current_job
    _current_job = "summarise_connection"

    blocks = []
    for i, group in enumerate(analysis.groups, start=1):
        chains = "\n".join(f"  {path_text(analysis.paths[j])}" for j in group)
        blocks.append(f"Facet {i}:\n{chains}")
    body = "\n".join(blocks)

    return _client().chat.completions.create(
        model=MODEL,
        max_retries=_MAX_RETRIES,
        response_model=ConnectionSummary,
        messages=[
            {
                "role": "system",
                "content": (
                    "A learner's knowledge map connects two concepts through "
                    "chains of relationships, already grouped into facets "
                    "(groups of chains that say roughly the same thing). "
                    "Write exactly one clear sentence per facet describing "
                    f'how "{analysis.start}" relates to "{analysis.end}" '
                    "according to that facet's chains. Do not add facts that "
                    "are not in the chains." + _NUDGE
                ),
            },
            {"role": "user", "content": body},
        ],
    )


@lru_cache(maxsize=1)
def _client() -> instructor.Instructor:
    if PROVIDER == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMUnavailable("DEEPSEEK_API_KEY is not set")
        client = instructor.from_openai(
            OpenAI(base_url=DEEPSEEK_URL, api_key=api_key),
            mode=instructor.Mode.JSON,
        )
        # Log the cost of every raw response — retries included, since
        # DeepSeek bills them too.
        client.on("completion:response", _log_cost)
        return client
    try:
        return instructor.from_openai(
            OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )
    except Exception as exc:  # pragma: no cover
        raise LLMUnavailable(str(exc)) from exc
