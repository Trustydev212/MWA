"""Pure-Python in-memory implementation of :class:`WorldModelProtocol`.

This backend exists for three reasons:

1. **Tests.**  Every other layer in SOMA needs *something* to write/read
   from in unit tests; spinning up Neo4j just to test the Arbiter is
   absurd.  In-memory storage gives us deterministic, fast, no-infra
   tests for the entire stack.

2. **Local development.**  Quickstart examples should run with
   ``python quickstart.py`` and zero setup.  In-memory makes that work.

3. **Reference implementation.**  Anyone porting SOMA to a new graph
   database can read this file in 30 minutes and understand exactly
   what semantics they need to preserve.

The implementation is intentionally boring — dicts and lists, no
optimisation tricks.  Performance for >10k nodes is not a goal here;
that's what graph DBs are for.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from soma.errors import HardConstraintViolation, WorldModelError
from soma.types import Conflict, Episode, Fact, WriteProposal, _utcnow

if TYPE_CHECKING:
    from soma.harness import HarnessMap


class InMemoryWorldModel:
    """Append-only, single-process World Model.

    Thread/coroutine-safe via a single :class:`asyncio.Lock`.  This is
    fine for the in-memory case because the critical sections are tiny
    (dict updates, no I/O).  Real backends will use DB-level locking.

    Parameters
    ----------
    harness:
        Optional :class:`HarnessMap`.  If provided, every ``apply`` runs
        the projected world state through hard-constraint validation
        before persisting.  Recommended for production; tests can omit
        it when they want to focus on storage semantics in isolation.
    """

    def __init__(self, harness: HarnessMap | None = None) -> None:
        self._harness = harness

        # Append-only log of every write attempt — applied OR rejected.
        # Indexed by episode id so we can reference past writes from
        # rejections, conflicts, audit reports, etc.
        self._episodes: dict[str, Episode] = {}

        # Per-node ordered list of episode ids (chronological).
        self._history: dict[str, list[str]] = defaultdict(list)

        # node -> currently-valid episode id (or absent if no current fact).
        self._current: dict[str, str] = {}

        # Lifecycle metadata kept *outside* Episode so Episode can stay frozen.
        # episode_id -> (valid_from, valid_to_or_None, status, rejection_reason)
        self._lifecycle: dict[str, _Lifecycle] = {}

        self._version: int = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    async def read(self, node: str) -> Fact | None:
        async with self._lock:
            return self._current_fact(node)

    async def history(
        self,
        node: str,
        *,
        include_rejected: bool = False,
    ) -> list[Fact]:
        async with self._lock:
            facts: list[Fact] = []
            for ep_id in self._history.get(node, []):
                lc = self._lifecycle[ep_id]
                if lc.status == "rejected" and not include_rejected:
                    continue
                facts.append(self._build_fact(ep_id))
            return facts

    async def episodes_since(self, version: int) -> list[Episode]:
        """Return every applied episode with version_at_apply > ``version``.

        We track this via :attr:`_lifecycle.applied_at_version`.
        """
        async with self._lock:
            return [
                self._episodes[ep_id]
                for ep_id, lc in self._lifecycle.items()
                if lc.status == "applied" and lc.applied_at_version > version
            ]

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    async def detect_conflict(self, proposal: WriteProposal) -> Conflict | None:
        async with self._lock:
            current = self._current_episode(proposal.node)
            if current is None:
                return None
            if current.value == proposal.value:
                # No-op write; not a conflict.  Caller should still be able
                # to apply it (it'll be a touch / refresh).
                return None
            return Conflict(node=proposal.node, existing=current, proposed=proposal)

    async def apply(self, proposal: WriteProposal) -> Episode:
        async with self._lock:
            self._enforce_hard_constraints(proposal)

            now = _utcnow()
            episode = self._episode_from(proposal, now)

            # Supersede previous current, if any.
            previous_id = self._current.get(proposal.node)
            if previous_id is not None:
                self._lifecycle[previous_id] = self._lifecycle[previous_id].superseded(now)

            self._version += 1
            self._episodes[episode.id] = episode
            self._history[proposal.node].append(episode.id)
            self._current[proposal.node] = episode.id
            self._lifecycle[episode.id] = _Lifecycle(
                valid_from=now,
                valid_to=None,
                status="applied",
                rejection_reason=None,
                applied_at_version=self._version,
            )
            return episode

    async def reject(self, proposal: WriteProposal, *, reason: str) -> Episode:
        if not reason:
            raise WorldModelError("reject() requires a non-empty reason")
        async with self._lock:
            now = _utcnow()
            episode = self._episode_from(proposal, now)

            self._episodes[episode.id] = episode
            self._history[proposal.node].append(episode.id)
            self._lifecycle[episode.id] = _Lifecycle(
                valid_from=now,
                valid_to=now,  # never became current
                status="rejected",
                rejection_reason=reason,
                applied_at_version=self._version,  # unchanged
            )
            return episode

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _episode_from(self, proposal: WriteProposal, now: datetime) -> Episode:
        return Episode(
            agent_id=proposal.agent_id,
            node=proposal.node,
            value=proposal.value,
            confidence=proposal.confidence,
            causal_parents=proposal.causal_parents,
            timestamp=now,
        )

    def _current_episode(self, node: str) -> Episode | None:
        ep_id = self._current.get(node)
        return self._episodes[ep_id] if ep_id is not None else None

    def _current_fact(self, node: str) -> Fact | None:
        ep_id = self._current.get(node)
        return self._build_fact(ep_id) if ep_id is not None else None

    def _build_fact(self, episode_id: str) -> Fact:
        episode = self._episodes[episode_id]
        lc = self._lifecycle[episode_id]
        return Fact(
            episode=episode,
            valid_from=lc.valid_from,
            valid_to=lc.valid_to,
            is_current=(self._current.get(episode.node) == episode_id),
            is_rejected=(lc.status == "rejected"),
            rejection_reason=lc.rejection_reason,
        )

    def _enforce_hard_constraints(self, proposal: WriteProposal) -> None:
        """Project the proposal onto the current state and validate.

        Skipped silently if no Harness Map was provided to the constructor —
        useful in unit tests that focus on storage semantics, but you should
        always pass one in production.
        """
        if self._harness is None:
            return
        projected = self._project_state(proposal)
        violations = self._harness.validate_state(projected)
        if violations:
            first = violations[0]
            raise HardConstraintViolation(
                f"Write to `{proposal.node}` violates hard constraint: {first.reason}",
                constraint=first.constraint,
                node=proposal.node,
                value=proposal.value,
            )

    def _project_state(self, proposal: WriteProposal) -> dict[str, Any]:
        """Build a snapshot of the world *as if* ``proposal`` were applied.

        Used purely for hard-constraint evaluation.  We start from the
        current state and overlay the proposal.
        """
        state: dict[str, Any] = {}
        for node, ep_id in self._current.items():
            state[node] = self._episodes[ep_id].value
        state[proposal.node] = proposal.value
        return state


# ---------------------------------------------------------------------------
# Internal lifecycle record (intentionally not a public type)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lifecycle:
    """Per-episode metadata that does *not* belong on the immutable Episode.

    We deliberately keep this private — callers see :class:`Fact` instead.
    """

    valid_from: datetime
    valid_to: datetime | None
    status: str  # "applied" | "rejected"
    rejection_reason: str | None
    applied_at_version: int

    def superseded(self, when: datetime) -> _Lifecycle:
        return replace(self, valid_to=when)
