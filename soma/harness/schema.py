"""Pydantic schema for ``harness.map.json``.

Why Pydantic instead of stdlib ``dataclass`` + manual validation?

1. Harness Maps come from disk (often hand-written JSON).  Pydantic gives
   us free, structured error messages instead of cryptic ``KeyError``s when
   somebody forgets a field.
2. We want extras to be rejected by default — typos like ``afects`` (one f)
   should not silently produce a node with no downstream edges.  Pydantic's
   ``model_config = ConfigDict(extra="forbid")`` gives us that for free.
3. The schema is the *only* place we need to define defaults; everywhere
   else in the codebase can assume a fully-validated ``HarnessSchema``.

These models are pure data — no traversal logic.  That lives in
``soma.harness.map`` so the schema layer can be tested in isolation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from soma.types import Impact

# ---------------------------------------------------------------------------
# Conflict resolution / scoring config
# ---------------------------------------------------------------------------


class ScoringWeights(BaseModel):
    """Weights used by the Semantic Arbiter to score conflict proposals.

    Must sum to (approximately) 1.0 — we validate this so a typo in the JSON
    cannot quietly skew every arbitration.
    """

    model_config = ConfigDict(extra="forbid")

    structural_importance: float = Field(ge=0.0, le=1.0, default=0.35)
    causal_depth: float = Field(ge=0.0, le=1.0, default=0.25)
    downstream_impact: float = Field(ge=0.0, le=1.0, default=0.25)
    confidence: float = Field(ge=0.0, le=1.0, default=0.15)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ScoringWeights:
        total = (
            self.structural_importance
            + self.causal_depth
            + self.downstream_impact
            + self.confidence
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Arbiter scoring weights must sum to 1.0 (got {total:.4f}). "
                "Check structural_importance + causal_depth + downstream_impact + confidence."
            )
        return self


class ConflictResolutionConfig(BaseModel):
    """How the runtime should react when a conflict is detected."""

    model_config = ConfigDict(extra="forbid")

    default_strategy: str = "arbiter"
    """``arbiter`` = call the LLM, ``last_write_wins`` = simple override,
    ``reject`` = bounce the second writer."""

    auto_resolve_threshold: float = Field(ge=0.0, le=1.0, default=0.85)
    """If the Arbiter is *very* confident the runtime will skip
    the human-in-the-loop step.  Below this threshold the resolution is
    queued for review."""

    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)


# ---------------------------------------------------------------------------
# Node + map schema
# ---------------------------------------------------------------------------


class HarnessNodeSchema(BaseModel):
    """Schema for one entry under ``nodes`` in ``harness.map.json``."""

    model_config = ConfigDict(extra="forbid")

    impact: Impact
    affects: list[str] = Field(default_factory=list)
    """Downstream nodes — when this node changes, these need to react."""

    order: int = Field(ge=1, default=1)
    """Topological order hint.  ``order=1`` is a root, ``order=2`` depends
    on roots, etc.  We *re-derive* the real topological order from the
    edges; ``order`` exists only as documentation / a sanity check."""

    description: str = ""


class HarnessSchema(BaseModel):
    """Top-level schema for the entire ``harness.map.json`` file."""

    model_config = ConfigDict(extra="forbid")

    version: str
    domain: str
    nodes: dict[str, HarnessNodeSchema]
    hard_constraints: list[str] = Field(default_factory=list)
    conflict_resolution: ConflictResolutionConfig = Field(default_factory=ConflictResolutionConfig)

    # ------------------------------------------------------------------
    # Cross-field validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _affects_must_reference_known_nodes(self) -> HarnessSchema:
        """Catch the most common authoring mistake — typos in ``affects``."""
        known = set(self.nodes.keys())
        broken: list[str] = []
        for name, node in self.nodes.items():
            for target in node.affects:
                if target not in known:
                    broken.append(f"{name} -> {target}")
                if target == name:
                    broken.append(f"{name} -> {name} (self-loop)")
        if broken:
            raise ValueError(
                "Harness Map references unknown / invalid nodes in `affects`: " + ", ".join(broken)
            )
        return self
