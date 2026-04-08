"""MWA — Multi World Agent.

A runtime layer that lets multiple LLM agents observe and act on a single
shared World Model instead of message-passing each other.

The package is intentionally split into independent layers so each one can be
tested and reasoned about in isolation:

    mwa.types       - shared dataclasses (Episode, Conflict, ...)
    mwa.errors      - exception hierarchy
    mwa.harness     - Harness Map: static dependency graph of the world
    mwa.llm         - provider-agnostic LLM interface (Anthropic/OpenAI/...)
    mwa.world       - World Model (in-memory + graph backends)
    mwa.arbiter     - conflict detection + semantic arbitration
    mwa.transport   - WebSocket hub + targeted push
    mwa.sdk         - WorldAgent class for end users
"""

from mwa.errors import (
    ConflictError,
    HardConstraintViolation,
    HarnessMapError,
    MWAError,
)

__version__ = "0.0.1"

__all__ = [
    "ConflictError",
    "HardConstraintViolation",
    "HarnessMapError",
    "MWAError",
    "__version__",
]
