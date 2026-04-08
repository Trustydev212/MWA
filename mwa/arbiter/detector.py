"""Conflict detection — *finds* contradictions, doesn't resolve them.

The detector is a thin layer over the World Model.  Its only job is to
ask the storage backend "is this proposal compatible with the current
state?" and return a structured :class:`Conflict` if not.

Why have it as a separate class instead of just calling
``world.detect_conflict()`` directly?

1. **Convenience**: real workflows are "given a stream of proposals,
   bucket them into clean writes vs conflicts".  The detector exposes
   that bulk API once and every caller benefits.
2. **Extensibility**: future detectors might do more than ask the
   backend — e.g. cross-node causal-violation checks, or ordering
   conflicts when an upstream node was updated mid-write.  The class
   gives those checks somewhere to live.
3. **Testability**: callers can swap in a fake detector that always
   returns no conflicts (or always one) without touching storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mwa.types import Conflict, WriteProposal

if TYPE_CHECKING:
    from mwa.world import WorldModelProtocol


class ConflictDetector:
    """Detects per-node value contradictions against a World Model."""

    def __init__(self, world: WorldModelProtocol) -> None:
        self._world = world

    async def check(self, proposal: WriteProposal) -> Conflict | None:
        """Return a :class:`Conflict` iff ``proposal`` would contradict
        the current state.  ``None`` means the proposal is safe to apply
        directly (no existing fact, or value matches existing).
        """
        return await self._world.detect_conflict(proposal)

    async def partition(
        self, proposals: list[WriteProposal]
    ) -> tuple[list[WriteProposal], list[Conflict]]:
        """Bulk variant: split a batch into clean writes and conflicts.

        Useful when an agent submits several writes at once and we want
        to fast-path the clean ones to storage while routing the rest
        through the resolver.

        Note: this method evaluates each proposal independently against
        the current world state — it does NOT account for proposals
        within the same batch interacting.  If two proposals in the same
        batch both target the same node, both will be classified by what
        they conflict with at call time.  Callers that need batch-internal
        coherence should serialise the batch.
        """
        clean: list[WriteProposal] = []
        conflicts: list[Conflict] = []
        for p in proposals:
            conflict = await self.check(p)
            if conflict is None:
                clean.append(p)
            else:
                conflicts.append(conflict)
        return clean, conflicts
