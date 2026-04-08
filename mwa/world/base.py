"""Protocol definition for World Model backends.

Every storage backend (in-memory, Graphiti, Neo4j, ...) must implement
this protocol so the rest of MWA never touches a backend-specific API.

Why a Protocol instead of an ABC?
---------------------------------
- Protocols give us *structural* typing — a backend doesn't need to inherit
  from anything, it just needs to implement the methods.  That makes
  testing with fakes trivial and lets third-party backends drop in without
  touching MWA imports.
- mypy still enforces the contract at type-check time.

Async-by-default
----------------
The protocol is async because every realistic backend (Graphiti, Neo4j,
FalkorDB) is async.  In-memory backends just don't ``await`` anything
inside their methods, but they still expose the async signature so
swapping backends never breaks callers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mwa.types import Conflict, Episode, Fact, WriteProposal


@runtime_checkable
class WorldModelProtocol(Protocol):
    """The minimal contract a World Model backend must satisfy."""

    @property
    def version(self) -> int:
        """Monotonically increasing version of the world.

        Incremented on every successful ``apply``; not incremented on
        ``reject`` (because rejected writes do not change observable state).
        Callers can compare versions to detect "did the world move under me
        while I was thinking?".
        """
        ...

    async def read(self, node: str) -> Fact | None:
        """Return the currently-valid fact for ``node``, or ``None``."""
        ...

    async def history(
        self,
        node: str,
        *,
        include_rejected: bool = False,
    ) -> list[Fact]:
        """Return every fact ever recorded for ``node`` in chronological order.

        ``include_rejected=False`` (default) hides facts that lost a conflict.
        Set to ``True`` for audit / debugging.
        """
        ...

    async def detect_conflict(self, proposal: WriteProposal) -> Conflict | None:
        """Return a :class:`Conflict` iff applying ``proposal`` would
        contradict the current fact for the same node.

        Pure inspection — does NOT mutate the world.  Returns ``None`` if
        the node has no current fact, or if the proposed value matches
        the existing one (a no-op write).
        """
        ...

    async def apply(self, proposal: WriteProposal) -> Episode:
        """Persist ``proposal`` as the new current fact for its node.

        Side effects:
        - increments :attr:`version`
        - supersedes any previous current fact for the same node by
          stamping its ``valid_to``
        - returns the persisted :class:`Episode`

        Raises:
            HardConstraintViolation: if applying the proposal would
                violate a hard constraint declared in the Harness Map.
        """
        ...

    async def reject(self, proposal: WriteProposal, *, reason: str) -> Episode:
        """Record ``proposal`` as a rejected write for audit.

        Does NOT touch the current fact, does NOT increment version.
        Returns the persisted :class:`Episode` so callers can reference it
        in resolution logs.
        """
        ...

    async def episodes_since(self, version: int) -> list[Episode]:
        """Return every episode that was applied with version > ``version``.

        Used by reconnecting agents to replay missed updates.
        """
        ...
