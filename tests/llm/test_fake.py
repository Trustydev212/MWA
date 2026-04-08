"""Tests for FakeProvider — confirms it honours LLMProvider protocol."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from mwa.llm import (
    ChatOptions,
    LLMProvider,
    Message,
    MessageRole,
    ResponseSchemaError,
    TransientProviderError,
)
from mwa.llm.providers import FakeProvider


class _Decision(BaseModel):
    winner: str
    confidence: float


def _user(text: str) -> Message:
    return Message(role=MessageRole.USER, content=text)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_fake_satisfies_protocol() -> None:
    fake = FakeProvider()
    assert isinstance(fake, LLMProvider)
    assert fake.name == "fake"
    assert fake.model == "fake-1"


# ---------------------------------------------------------------------------
# Text chat
# ---------------------------------------------------------------------------


async def test_chat_returns_queued_text() -> None:
    fake = FakeProvider().enqueue_text("hello", input_tokens=5, output_tokens=1)
    response = await fake.chat([_user("hi")])
    assert response.content == "hello"
    assert response.provider == "fake"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 1


async def test_chat_records_calls_for_inspection() -> None:
    fake = FakeProvider().enqueue_text("a").enqueue_text("b")
    await fake.chat([_user("first")])
    await fake.chat([_user("second")], options=ChatOptions(temperature=0.7))

    assert len(fake.calls) == 2
    # Second call should have temperature 0.7
    assert fake.calls[1][1].temperature == 0.7


async def test_empty_queue_raises() -> None:
    fake = FakeProvider()
    with pytest.raises(RuntimeError, match="no responses are queued"):
        await fake.chat([_user("hi")])


async def test_enqueue_error_is_raised() -> None:
    fake = FakeProvider().enqueue_error(TransientProviderError("boom"))
    with pytest.raises(TransientProviderError, match="boom"):
        await fake.chat([_user("hi")])


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_emits_full_content_as_one_chunk() -> None:
    fake = FakeProvider().enqueue_text("hello world")
    chunks = [c async for c in fake.stream([_user("hi")])]
    assert len(chunks) == 1
    assert chunks[0].delta == "hello world"


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


async def test_structured_returns_schema_instance() -> None:
    fake = FakeProvider().enqueue_json({"winner": "alice", "confidence": 0.9})
    decision = await fake.structured([_user("resolve")], _Decision)
    assert decision.winner == "alice"
    assert decision.confidence == 0.9


async def test_structured_rejects_invalid_json() -> None:
    fake = FakeProvider().enqueue_text("not valid json")
    with pytest.raises(ResponseSchemaError):
        await fake.structured([_user("resolve")], _Decision)


async def test_structured_rejects_schema_mismatch() -> None:
    fake = FakeProvider().enqueue_json({"wrong_key": "alice"})
    with pytest.raises(ResponseSchemaError):
        await fake.structured([_user("resolve")], _Decision)


# ---------------------------------------------------------------------------
# Chained enqueues
# ---------------------------------------------------------------------------


async def test_multiple_enqueues_consumed_in_order() -> None:
    fake = FakeProvider().enqueue_text("first").enqueue_text("second").enqueue_text("third")
    results = [
        (await fake.chat([_user("_")])).content,
        (await fake.chat([_user("_")])).content,
        (await fake.chat([_user("_")])).content,
    ]
    assert results == ["first", "second", "third"]
