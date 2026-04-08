"""SOMA — Shared Observer Multi Agent.

A runtime layer that lets multiple LLM agents observe and act on a single
shared World Model instead of message-passing each other.

The package is intentionally split into independent layers so each one can be
tested and reasoned about in isolation:

    soma.types       - shared dataclasses (Episode, Conflict, ...)
    soma.errors      - exception hierarchy
    soma.harness     - Harness Map: static dependency graph of the world
    soma.llm         - provider-agnostic LLM interface (Anthropic/OpenAI/...)
    soma.world       - World Model (in-memory + graph backends)
    soma.arbiter     - conflict detection + semantic arbitration
    soma.transport   - WebSocket hub + targeted push
    soma.sdk         - WorldAgent class for end users
"""

from soma.errors import (
    ConflictError,
    HardConstraintViolation,
    HarnessMapError,
    SomaError,
)

__version__ = "0.0.1"

__all__ = [
    "ConflictError",
    "HardConstraintViolation",
    "HarnessMapError",
    "SomaError",
    "__version__",
]
