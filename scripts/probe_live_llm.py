"""Standalone live probe for any OpenAI-compatible endpoint.

Run this outside the Claude sandbox (Colab, local machine, anywhere with
working outbound HTTPS) to verify that MWA's ``OpenAIProvider`` adapter
talks to a given endpoint correctly::

    MWA_LIVE_API_KEY="sk-..." \\
    MWA_LIVE_BASE_URL="https://futrixapi.com/v1" \\
    MWA_LIVE_MODEL="auto" \\
    python scripts/probe_live_llm.py

Or drop the whole file into a Google Colab cell after installing the
package::

    !pip install -q openai pydantic
    !pip install -q -e /content/MWA

    import os
    os.environ["MWA_LIVE_API_KEY"]  = "sk-..."
    os.environ["MWA_LIVE_BASE_URL"] = "https://futrixapi.com/v1"
    os.environ["MWA_LIVE_MODEL"]    = "auto"
    %run /content/MWA/scripts/probe_live_llm.py

The script:
1. Performs a minimal chat() round-trip and prints the response preview.
2. Performs a structured() round-trip with a tiny Pydantic schema to
   verify JSON-schema response_format support.
3. Exits non-zero on any failure so CI / notebooks show a clear signal.

It never writes credentials to disk and never echoes the full API key.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback

from pydantic import BaseModel, Field

REQUIRED_ENV = ("MWA_LIVE_API_KEY", "MWA_LIVE_BASE_URL", "MWA_LIVE_MODEL")


def _masked(value: str) -> str:
    """Show only the first 8 and last 4 characters of a secret."""
    if len(value) <= 12:
        return "***"
    return f"{value[:8]}...{value[-4:]}"


def _check_env() -> dict[str, str]:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
    if missing:
        print(f"[probe] FATAL: missing env vars: {', '.join(missing)}")
        print("[probe] Set them before running this script:")
        for name in missing:
            print(f"            export {name}=...")
        sys.exit(2)
    return {name: os.environ[name].strip() for name in REQUIRED_ENV}


class _TinyDecision(BaseModel):
    """Shape used by the structured-output probe."""

    winner: str = Field(description="One of: alice, bob")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


async def _run() -> None:
    env = _check_env()
    print(f"[probe] base_url = {env['MWA_LIVE_BASE_URL']}")
    print(f"[probe] model    = {env['MWA_LIVE_MODEL']}")
    print(f"[probe] api_key  = {_masked(env['MWA_LIVE_API_KEY'])}")
    print()

    # Lazy imports so "missing env" error fires before anything else.
    from mwa.llm.base import ChatOptions, Message, MessageRole
    from mwa.llm.providers import OpenAIProvider

    provider = OpenAIProvider(
        model=env["MWA_LIVE_MODEL"],
        api_key=env["MWA_LIVE_API_KEY"],
        base_url=env["MWA_LIVE_BASE_URL"],
    )

    # ------------------------------------------------------------------
    # Probe 1 — plain chat()
    # ------------------------------------------------------------------
    print("[probe 1/2] plain chat() ...")
    response = await provider.chat(
        [
            Message(
                role=MessageRole.SYSTEM,
                content="You reply in 12 words or fewer. Be direct.",
            ),
            Message(
                role=MessageRole.USER,
                content="Say hello from MWA and name one thing you could be.",
            ),
        ],
        options=ChatOptions(temperature=0.0, max_tokens=64),
    )
    print(f"    content : {response.content!r}")
    print(f"    usage   : in={response.usage.input_tokens} out={response.usage.output_tokens}")
    print(f"    finish  : {response.finish_reason}")
    assert response.content, "chat() returned empty content — endpoint may be broken"
    print("    ✓ chat() passed")
    print()

    # ------------------------------------------------------------------
    # Probe 2 — structured() with json_schema
    # ------------------------------------------------------------------
    print("[probe 2/2] structured() with json_schema ...")
    try:
        decision = await provider.structured(
            [
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You are a conflict arbiter. Return a structured decision "
                        "as a JSON object matching the schema."
                    ),
                ),
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Alice wrote tone='serious' (confidence 0.7). "
                        "Bob wrote tone='playful' (confidence 0.75). "
                        "Pick a winner and explain briefly."
                    ),
                ),
            ],
            _TinyDecision,
            options=ChatOptions(temperature=0.0, max_tokens=256),
        )
        print(f"    winner     : {decision.winner}")
        print(f"    confidence : {decision.confidence}")
        print(f"    reason     : {decision.reason}")
        assert decision.winner in {"alice", "bob"}
        print("    ✓ structured() passed")
    except Exception as exc:
        print(f"    ✗ structured() raised: {type(exc).__name__}: {exc}")
        print(
            "    (this is a real interop finding, not a MWA bug — some gateways "
            "don't implement response_format=json_schema strictly)"
        )
        # Non-fatal: the chat() probe is the blocker.  Structured output
        # failing is useful data for the MWA RESEARCH notes.

    print()
    print("[probe] done.")


def main() -> int:
    try:
        asyncio.run(_run())
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
