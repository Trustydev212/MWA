"""Exception hierarchy for SOMA.

Every error raised by SOMA inherits from ``SomaError`` so callers can catch
the whole package with a single ``except`` clause.  More specific subclasses
exist when callers might reasonably want to react differently — e.g. a
hard-constraint violation should never trigger an Arbiter call, while a
value contradiction should.
"""

from __future__ import annotations


class SomaError(Exception):
    """Base class for every exception raised inside SOMA."""


class HarnessMapError(SomaError):
    """Raised when a Harness Map is structurally invalid.

    Examples: unknown node referenced in ``affects``, cycle detected in the
    dependency graph, malformed JSON, schema validation failure.
    """


class HardConstraintViolation(SomaError):
    """A write would violate a hard constraint declared in the Harness Map.

    These are non-recoverable at the runtime level — the write is rejected
    immediately, the Semantic Arbiter is *not* invoked.
    """

    def __init__(self, message: str, *, constraint: str, node: str, value: object) -> None:
        super().__init__(message)
        self.constraint = constraint
        self.node = node
        self.value = value


class ConflictError(SomaError):
    """Two or more agents proposed contradicting values for the same node.

    This is the *recoverable* sibling of :class:`HardConstraintViolation`.
    The runtime catches this and routes it through the Semantic Arbiter.
    """


class LLMProviderError(SomaError):
    """An LLM provider call failed (rate limit, network, schema, ...)."""


class WorldModelError(SomaError):
    """Generic World Model failure (storage, version mismatch, ...)."""
