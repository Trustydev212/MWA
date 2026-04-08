"""Harness Map — the static dependency graph of the World.

The Harness Map is the *constitution* of the World Model.  It is defined
once by the architect (in ``harness.map.json``) and never changes at
runtime.  Every other layer in MWA reads from it:

- the World Model uses it to know which nodes are downstream of a write
- the Conflict Detector uses it to enforce hard constraints before any DB write
- the Semantic Arbiter uses it to score conflicts (structural importance)
- the Transport layer uses it to figure out which agents to push updates to

Why static?  Because if the dependency graph could change at runtime, every
single arbitration would need to re-discover topology — which is exactly
the O(n) cost we are trying to avoid.
"""

from mwa.harness.constraints import (
    ConstraintViolation,
    HardConstraintEvaluator,
    parse_constraint,
)
from mwa.harness.map import HarnessMap
from mwa.harness.schema import (
    ConflictResolutionConfig,
    HarnessNodeSchema,
    HarnessSchema,
    ScoringWeights,
)

__all__ = [
    "ConflictResolutionConfig",
    "ConstraintViolation",
    "HardConstraintEvaluator",
    "HarnessMap",
    "HarnessNodeSchema",
    "HarnessSchema",
    "ScoringWeights",
    "parse_constraint",
]
