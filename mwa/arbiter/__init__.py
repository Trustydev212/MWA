"""Conflict detection and resolution.

This package owns the *decision* of what to do when two agents disagree
about the value of a node.  It does **not** own the storage layer (that's
``mwa.world``) and it does **not** own the dependency graph (that's
``mwa.harness``); it just consumes both.

Two resolution strategies live here:

- :class:`RuleBasedResolver` (this milestone) — fast, deterministic, no LLM.
  Handles the easy cases (same value, dominant confidence).  Escalates
  ambiguous cases.
- ``SemanticArbiter`` (next milestone) — LLM-backed, handles escalations.

Splitting them lets the cheap path stay cheap and lets us measure how often
we actually need the LLM in real workloads — a number we very much want
to know before paying production token bills.
"""

from mwa.arbiter.detector import ConflictDetector
from mwa.arbiter.resolution import Resolution, ResolutionDecision
from mwa.arbiter.rules import RuleBasedResolver

__all__ = [
    "ConflictDetector",
    "Resolution",
    "ResolutionDecision",
    "RuleBasedResolver",
]
