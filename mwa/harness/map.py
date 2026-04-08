"""Loaded, traversable Harness Map.

This module turns a validated :class:`HarnessSchema` into something
queryable: BFS / DFS traversals, topological order, subgraph extraction,
and constraint evaluation.

Why a separate class instead of doing this on ``HarnessSchema`` directly?

- The schema layer is *pure data* — no derived state, no caches, no
  algorithms.  That keeps it cheap to construct in tests.
- The map layer pre-computes the *reverse* edges (``affected_by``) and the
  topological order once at load time.  Pre-computing is fine because the
  Harness Map is immutable at runtime.
- Pre-computing also lets us catch cycles at *load* time (in production
  startup) instead of at *traverse* time (deep in a hot path).
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mwa.errors import HarnessMapError
from mwa.harness.constraints import ConstraintViolation, HardConstraintEvaluator
from mwa.harness.schema import HarnessNodeSchema, HarnessSchema
from mwa.types import Impact


class HarnessMap:
    """The loaded, traversable Harness Map.

    Construct via :meth:`from_dict`, :meth:`from_json`, or :meth:`load`.
    Direct ``__init__`` is supported for callers that already hold a
    :class:`HarnessSchema`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, schema: HarnessSchema) -> None:
        self._schema = schema
        # Forward edges = `affects` (already in the schema).
        # Reverse edges = `affected_by` — derived once here.
        self._affected_by: dict[str, list[str]] = defaultdict(list)
        for name, node in schema.nodes.items():
            for target in node.affects:
                self._affected_by[target].append(name)

        # Pre-compute the topological order.  This *also* serves as our
        # cycle check — Kahn's algorithm fails fast on a cycle.
        self._topo_order: list[str] = self._compute_topological_order()

        # Constraint evaluator built once.
        self._evaluator = HardConstraintEvaluator(schema.hard_constraints)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HarnessMap:
        """Build a HarnessMap from an already-parsed dict (e.g. JSON / YAML)."""
        try:
            schema = HarnessSchema.model_validate(data)
        except ValidationError as exc:
            raise HarnessMapError(f"Invalid Harness Map: {exc}") from exc
        return cls(schema)

    @classmethod
    def from_json(cls, text: str) -> HarnessMap:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HarnessMapError(f"Harness Map JSON is malformed: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> HarnessMap:
        """Load a Harness Map from a JSON file on disk."""
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise HarnessMapError(f"Cannot read Harness Map at {p}: {exc}") from exc
        return cls.from_json(text)

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._schema.version

    @property
    def domain(self) -> str:
        return self._schema.domain

    @property
    def nodes(self) -> dict[str, HarnessNodeSchema]:
        return dict(self._schema.nodes)

    def __contains__(self, node: str) -> bool:
        return node in self._schema.nodes

    def __iter__(self) -> Iterable[str]:
        return iter(self._schema.nodes)

    def __len__(self) -> int:
        return len(self._schema.nodes)

    def get(self, node: str) -> HarnessNodeSchema:
        try:
            return self._schema.nodes[node]
        except KeyError as exc:
            raise HarnessMapError(f"Unknown node: {node}") from exc

    def impact_of(self, node: str) -> Impact:
        return self.get(node).impact

    def affects(self, node: str) -> list[str]:
        """Direct downstream nodes (one hop)."""
        return list(self.get(node).affects)

    def affected_by(self, node: str) -> list[str]:
        """Direct upstream nodes (one hop)."""
        if node not in self._schema.nodes:
            raise HarnessMapError(f"Unknown node: {node}")
        return list(self._affected_by.get(node, []))

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def traverse_downstream(self, node: str, depth: int = -1) -> list[str]:
        """Return every node reachable from ``node`` via ``affects`` edges.

        ``depth=-1`` means unbounded (the default).  ``depth=0`` returns the
        empty list (no hops).  ``depth=1`` is equivalent to :meth:`affects`.

        The result is in BFS order so callers that care about *how far*
        a node is can rely on the ordering.  The starting node is excluded.
        """
        return self._bfs(node, lambda n: self.get(n).affects, depth)

    def traverse_upstream(self, node: str, depth: int = -1) -> list[str]:
        """Return every node that transitively affects ``node``."""
        return self._bfs(node, lambda n: self._affected_by.get(n, []), depth)

    def _bfs(
        self,
        start: str,
        neighbours: Any,
        depth: int,
    ) -> list[str]:
        if start not in self._schema.nodes:
            raise HarnessMapError(f"Unknown node: {start}")
        if depth == 0:
            return []

        seen: set[str] = {start}
        order: list[str] = []
        queue: deque[tuple[str, int]] = deque([(start, 0)])

        while queue:
            current, current_depth = queue.popleft()
            if depth != -1 and current_depth >= depth:
                continue
            for nxt in neighbours(current):
                if nxt in seen:
                    continue
                seen.add(nxt)
                order.append(nxt)
                queue.append((nxt, current_depth + 1))

        return order

    def topological_order(self) -> list[str]:
        """Return a safe update order for the entire map.

        Updating nodes in this order guarantees no node is updated before
        any of its upstream dependencies.
        """
        return list(self._topo_order)

    def _compute_topological_order(self) -> list[str]:
        # Kahn's algorithm.  Edges are ``upstream -> downstream`` (i.e.
        # ``affects``).  We process roots first.
        in_degree: dict[str, int] = dict.fromkeys(self._schema.nodes, 0)
        for node in self._schema.nodes.values():
            for target in node.affects:
                in_degree[target] += 1

        # Tie-break by ``order`` field then by name so the result is stable
        # across Python versions / dict orderings.
        def sort_key(n: str) -> tuple[int, str]:
            return (self._schema.nodes[n].order, n)

        ready = sorted([n for n, d in in_degree.items() if d == 0], key=sort_key)
        order: list[str] = []

        while ready:
            current = ready.pop(0)
            order.append(current)
            for target in self._schema.nodes[current].affects:
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    # Insert in sorted position so the overall result is stable.
                    ready.append(target)
                    ready.sort(key=sort_key)

        if len(order) != len(self._schema.nodes):
            cyclic = sorted(n for n, d in in_degree.items() if d > 0)
            raise HarnessMapError(
                f"Harness Map has a dependency cycle involving: {', '.join(cyclic)}"
            )
        return order

    # ------------------------------------------------------------------
    # Subgraph extraction (used by the Arbiter — keeps prompts small)
    # ------------------------------------------------------------------

    def get_subgraph(self, nodes: Iterable[str]) -> dict[str, list[str]]:
        """Return the minimal subgraph that contains ``nodes`` and their edges.

        Returned as ``{node: [downstream...]}``.  Edges that point to nodes
        outside the subgraph are dropped.  This is exactly the shape the
        Arbiter wants to put into its prompt.
        """
        keep = set(nodes)
        for n in list(keep):
            if n not in self._schema.nodes:
                raise HarnessMapError(f"Unknown node in subgraph request: {n}")
        subgraph: dict[str, list[str]] = {}
        for n in keep:
            subgraph[n] = [t for t in self._schema.nodes[n].affects if t in keep]
        return subgraph

    # ------------------------------------------------------------------
    # Constraint evaluation
    # ------------------------------------------------------------------

    def validate_state(self, state: dict[str, Any]) -> list[ConstraintViolation]:
        """Run every hard constraint against ``state`` and return violations."""
        return self._evaluator.evaluate(state)

    @property
    def hard_constraints(self) -> list[str]:
        return self._evaluator.constraints
