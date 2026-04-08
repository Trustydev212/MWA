"""Hard constraint evaluation for the Harness Map.

Hard constraints are *non-negotiable* invariants the World Model must
maintain.  They are evaluated **before** any write hits the World Model and
**before** the Semantic Arbiter is consulted.  A constraint violation
short-circuits the entire pipeline with :class:`HardConstraintViolation`.

Constraint DSL (v0)
-------------------
We deliberately start with a tiny DSL — just enough to express the examples
in the README — and grow it only when a real harness map needs more.  The
v0 syntax recognises three forms:

1. **Mutual exclusion (value)**::

       "duration không thể đồng thời là 15s và 60s"
       "tone cannot be both formal and casual"

   Form: ``<node> ... <value_a> (và|and) <value_b>`` — asserts that the
   node may hold *one* of those values but never both at once.  In a
   single-valued world model that's automatic, but the constraint is still
   useful as documentation and for the future multi-valued case.

2. **Mutual exclusion (combination across nodes)**::

       "visual_style cartoon không compatible với tone corporate"
       "visual_style cartoon incompatible with tone corporate"

   Form: ``<node_a> <val_a> ... <node_b> <val_b>`` — asserts the two
   ``(node, value)`` pairs cannot be true simultaneously.

3. **Free-form (passthrough)**: anything that doesn't match the patterns
   above is stored verbatim and surfaces only if a future evaluator
   recognises it.  We never silently drop a constraint.

We picked Vietnamese keywords as well as English because the example
harness map in README.md is bilingual.  The DSL is documented as v0 — we
expect it to evolve once we have ~10 real maps to learn from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConstraintViolation:
    """One reason why a candidate world state violates a hard constraint."""

    constraint: str
    node: str
    value: Any
    reason: str


# ---------------------------------------------------------------------------
# DSL parser (v0)
# ---------------------------------------------------------------------------

# Tokens that mean "and" — used to find the two operands in a mutex constraint.
_AND_TOKENS = (" và ", " and ")

# Tokens that mean "incompatible / cannot / không thể / không compatible".
_INCOMPAT_TOKENS = (
    "không thể đồng thời",
    "cannot be both",
    "không compatible với",
    "incompatible with",
    "not compatible with",
)


@dataclass(frozen=True)
class _ParsedConstraint:
    """Internal representation of a parsed constraint string."""

    raw: str
    kind: str  # "value_mutex" | "pair_mutex" | "freeform"
    node_a: str | None = None
    value_a: Any = None
    node_b: str | None = None
    value_b: Any = None
    values: tuple[Any, ...] = field(default_factory=tuple)


def parse_constraint(raw: str) -> _ParsedConstraint:
    """Parse one constraint string into a structured form.

    Public mostly so tests can poke at it directly; production code goes
    through :class:`HardConstraintEvaluator`.
    """
    text = raw.strip()
    lowered = text.lower()

    # ---- pair-mutex (cross-node) ------------------------------------------------
    # "<node_a> <val_a> ... incompatible ... <node_b> <val_b>"
    for marker in (
        "không compatible với",
        "incompatible with",
        "not compatible with",
    ):
        if marker in lowered:
            left_raw, right_raw = _split_once(text, marker)
            left_tokens = _tokens(left_raw)
            right_tokens = _tokens(right_raw)
            if len(left_tokens) >= 2 and len(right_tokens) >= 2:
                return _ParsedConstraint(
                    raw=text,
                    kind="pair_mutex",
                    node_a=left_tokens[0],
                    value_a=_coerce_value(" ".join(left_tokens[1:])),
                    node_b=right_tokens[0],
                    value_b=_coerce_value(" ".join(right_tokens[1:])),
                )

    # ---- value-mutex (single node, two forbidden simultaneous values) ----------
    for marker in ("không thể đồng thời", "cannot be both"):
        if marker in lowered:
            left_raw, right_raw = _split_once(text, marker)
            node_tokens = _tokens(left_raw)
            if not node_tokens:
                continue
            node = node_tokens[0]
            and_split = _split_on_any(right_raw, _AND_TOKENS)
            if and_split is None:
                continue
            val_a, val_b = and_split
            return _ParsedConstraint(
                raw=text,
                kind="value_mutex",
                node_a=node,
                values=(
                    _coerce_value(_strip_filler(val_a)),
                    _coerce_value(_strip_filler(val_b)),
                ),
            )

    return _ParsedConstraint(raw=text, kind="freeform")


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[\s,]+", s.strip()) if t]


def _split_once(text: str, marker: str) -> tuple[str, str]:
    idx = text.lower().find(marker)
    return text[:idx].strip(), text[idx + len(marker) :].strip()


def _split_on_any(text: str, markers: tuple[str, ...]) -> tuple[str, str] | None:
    lowered = text.lower()
    for marker in markers:
        idx = lowered.find(marker)
        if idx != -1:
            return text[:idx].strip(), text[idx + len(marker) :].strip()
    return None


_FILLER_WORDS = ("là", "is", "be", "are", "được")
"""Tiny lexicon of copulas / filler words to strip from value fragments.

Both Vietnamese ``"là 15s"`` and English ``"be 15s"`` mean *the value 15s*;
we want the parsed value to be ``"15s"``.  Kept intentionally small — we'd
rather miss a phrasing than swallow part of a real value.
"""


def _strip_filler(s: str) -> str:
    s = s.strip().rstrip(".").strip()
    tokens = s.split()
    while tokens and tokens[0].lower() in _FILLER_WORDS:
        tokens.pop(0)
    return " ".join(tokens)


def _coerce_value(s: str) -> Any:
    """Best-effort coerce a string fragment to int / bool / str.

    Used so a constraint like ``"duration không thể đồng thời là 15s và 60s"``
    parses ``15s`` as the literal string ``"15s"`` (we keep the unit) but
    a bare ``"15"`` becomes the int ``15``.
    """
    s = s.strip().strip("\"'")
    if s.isdigit():
        return int(s)
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class HardConstraintEvaluator:
    """Check candidate world states against the parsed hard constraints."""

    def __init__(self, raw_constraints: list[str]) -> None:
        self._raw = list(raw_constraints)
        self._parsed = [parse_constraint(c) for c in raw_constraints]

    @property
    def constraints(self) -> list[str]:
        return list(self._raw)

    def evaluate(self, state: dict[str, Any]) -> list[ConstraintViolation]:
        """Return every violation in ``state``.

        Empty list = candidate state is valid.  We return *all* violations
        (not just the first) so the runtime can surface them together.
        """
        violations: list[ConstraintViolation] = []

        for parsed in self._parsed:
            if parsed.kind == "value_mutex":
                violations.extend(self._check_value_mutex(parsed, state))
            elif parsed.kind == "pair_mutex":
                violations.extend(self._check_pair_mutex(parsed, state))
            # freeform constraints have no programmatic check (yet)

        return violations

    # ------------------------------------------------------------------
    # checkers
    # ------------------------------------------------------------------

    def _check_value_mutex(
        self, parsed: _ParsedConstraint, state: dict[str, Any]
    ) -> list[ConstraintViolation]:
        node = parsed.node_a
        if node is None or node not in state:
            return []
        current = state[node]
        # In a single-valued world the constraint is *always* satisfied
        # (a single field can never hold both values), but if the proposed
        # value is *not one of the allowed alternatives* that's also a
        # violation worth flagging.  We only flag the obvious case here:
        # both forbidden values appear in a list-valued node.
        if isinstance(current, list | tuple | set):
            present = [v for v in parsed.values if v in current]
            if len(present) >= 2:
                return [
                    ConstraintViolation(
                        constraint=parsed.raw,
                        node=node,
                        value=current,
                        reason=f"node `{node}` holds mutually-exclusive values {present}",
                    )
                ]
        return []

    def _check_pair_mutex(
        self, parsed: _ParsedConstraint, state: dict[str, Any]
    ) -> list[ConstraintViolation]:
        a, b = parsed.node_a, parsed.node_b
        if a is None or b is None:
            return []
        if a not in state or b not in state:
            return []
        if self._holds(state[a], parsed.value_a) and self._holds(state[b], parsed.value_b):
            return [
                ConstraintViolation(
                    constraint=parsed.raw,
                    node=a,
                    value=state[a],
                    reason=(f"`{a}={parsed.value_a}` is incompatible with `{b}={parsed.value_b}`"),
                )
            ]
        return []

    @staticmethod
    def _holds(actual: Any, expected: Any) -> bool:
        """True iff ``actual`` "contains" ``expected``.

        Works for both scalar nodes (``actual == expected``) and multi-valued
        nodes (``expected in actual``).  This lets pair-mutex constraints
        fire when one of the nodes is a list/set.
        """
        if isinstance(actual, list | tuple | set | frozenset):
            return expected in actual
        return bool(actual == expected)
