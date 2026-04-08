"""World Model layer — temporal storage of facts written by agents.

This package defines the storage abstraction (:class:`WorldModelProtocol`)
and a single concrete backend (:class:`InMemoryWorldModel`).  Future
backends — Graphiti, Neo4j, FalkorDB — will live alongside ``memory.py``
and implement the same protocol so callers don't change.

Design philosophy
-----------------
- Storage is *append-only*.  Facts are never mutated; superseded facts
  get a ``valid_to`` timestamp recorded next to them.
- The World Model owns hard-constraint enforcement.  It will never let
  a write produce a state that violates the Harness Map's constraints.
- The World Model does *not* know about the Semantic Arbiter.  When a
  write conflicts with an existing fact it surfaces a :class:`Conflict`
  to the caller, which decides how to resolve it.  This keeps the
  storage layer free of LLM dependencies.
"""

from soma.world.base import WorldModelProtocol
from soma.world.memory import InMemoryWorldModel

__all__ = [
    "InMemoryWorldModel",
    "WorldModelProtocol",
]
