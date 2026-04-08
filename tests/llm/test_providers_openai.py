"""Tests for OpenAIProvider — mocked SDK client, no network."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from mwa.llm import (
    ChatOptions,
    Message,
    MessageRole,
    PermanentProviderError,
    RateLimitError,
    ResponseSchemaError,
    TransientProviderError,
)
from mwa.llm.providers import OpenAIProvider

# ---------------------------------------------------------------------------
# Fake openai client
# ---------------------------------------------------------------------------


@dataclass
class _FakeDelta:
    content: str | None = None


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str | None = "stop"
    delta: _FakeDelta = field(default_factory=_FakeDelta)


@dataclass
class _FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


class _FakeCompletions:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self._next: _FakeResponse | BaseException | None = None

    def will_return(self, response: _FakeResponse) -> None:
        self._next = response

    def will_raise(self, error: BaseException) -> None:
        self._next = error

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.create_calls.append(kwargs)
        if isinstance(self._next, BaseException):
            raise self._next
        if self._next is None:
            raise RuntimeError("_FakeCompletions.create called without a queued response")
        return self._next


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


@pytest.fixture
def fake_client() -> _FakeOpenAIClient:
    return _FakeOpenAIClient()


@pytest.fixture
def provider(fake_client: _FakeOpenAIClient) -> OpenAIProvider:
    return OpenAIProvider(model="gpt-test", client=fake_client)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def test_chat_translates_messages_and_parses_response(
    provider: OpenAIProvider,
    fake_client: _FakeOpenAIClient,
) -> None:
    fake_client.chat.completions.will_return(
        _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content="Hi!"), finish_reason="stop")],
            usage=_FakeUsage(prompt_tokens=4, completion_tokens=2),
        )
    )
    response = await provider.chat(
        [
            Message(role=MessageRole.SYSTEM, content="Be brief."),
            Message(role=MessageRole.USER, content="Hello"),
        ],
        options=ChatOptions(temperature=0.2, max_tokens=50),
    )
    assert response.content == "Hi!"
    assert response.provider == "openai"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 2
    assert response.finish_reason == "stop"

    call = fake_client.chat.completions.create_calls[0]
    assert call["model"] == "gpt-test"
    assert call["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
    ]
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 50


async def test_chat_omits_none_optional_fields_from_request(
    provider: OpenAIProvider,
    fake_client: _FakeOpenAIClient,
) -> None:
    """Strict OpenAI-compatible gateways (e.g. futrixapi) reject explicit
    ``null`` values for ``top_p`` / ``max_tokens`` / ``stop``.  We must
    omit them entirely when the caller didn't set them, not send nulls.
    """
    fake_client.chat.completions.will_return(
        _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content="ok"))])
    )
    await provider.chat(
        [Message(role=MessageRole.USER, content="hi")],
        options=ChatOptions(),  # all optional fields at their default None / empty
    )
    call = fake_client.chat.completions.create_calls[0]
    assert "top_p" not in call
    assert "max_tokens" not in call
    assert "stop" not in call
    # But required / explicitly-set fields are always present:
    assert call["model"] == "gpt-test"
    assert "messages" in call
    assert call["temperature"] == 0.0  # default


async def test_chat_includes_explicit_optional_fields(
    provider: OpenAIProvider,
    fake_client: _FakeOpenAIClient,
) -> None:
    fake_client.chat.completions.will_return(
        _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content="ok"))])
    )
    await provider.chat(
        [Message(role=MessageRole.USER, content="hi")],
        options=ChatOptions(max_tokens=128, top_p=0.9, stop=("STOP",)),
    )
    call = fake_client.chat.completions.create_calls[0]
    assert call["max_tokens"] == 128
    assert call["top_p"] == 0.9
    assert call["stop"] == ["STOP"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _FakeRateLimit(Exception):
    pass


_FakeRateLimit.__name__ = "RateLimitError"


class _FakeAuth(Exception):
    pass


_FakeAuth.__name__ = "AuthenticationError"


class _FakeConn(Exception):
    pass


_FakeConn.__name__ = "APIConnectionError"


class _FakeBadRequest(Exception):
    pass


_FakeBadRequest.__name__ = "BadRequestError"


async def test_rate_limit_translation(
    provider: OpenAIProvider, fake_client: _FakeOpenAIClient
) -> None:
    fake_client.chat.completions.will_raise(_FakeRateLimit("429"))
    with pytest.raises(RateLimitError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


async def test_auth_is_permanent(provider: OpenAIProvider, fake_client: _FakeOpenAIClient) -> None:
    fake_client.chat.completions.will_raise(_FakeAuth("bad"))
    with pytest.raises(PermanentProviderError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


async def test_bad_request_is_permanent(
    provider: OpenAIProvider, fake_client: _FakeOpenAIClient
) -> None:
    """HTTP 400 means the payload is malformed — retry will fail identically."""
    fake_client.chat.completions.will_raise(
        _FakeBadRequest("Error code: 400 - top_p: Invalid input: expected number, received null")
    )
    with pytest.raises(PermanentProviderError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


async def test_connection_error_is_transient(
    provider: OpenAIProvider, fake_client: _FakeOpenAIClient
) -> None:
    fake_client.chat.completions.will_raise(_FakeConn("down"))
    with pytest.raises(TransientProviderError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


# ---------------------------------------------------------------------------
# Structured output — json_schema strategy
# ---------------------------------------------------------------------------


class _Decision(BaseModel):
    winner: str
    confidence: float


async def test_structured_parses_json(
    provider: OpenAIProvider, fake_client: _FakeOpenAIClient
) -> None:
    """Default ``auto`` strategy tries json_schema first.  A conforming
    response should be returned as-is without triggering the fallback."""
    fake_client.chat.completions.will_return(
        _FakeResponse(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(content=json.dumps({"winner": "alice", "confidence": 0.9}))
                )
            ]
        )
    )
    result = await provider.structured(
        [Message(role=MessageRole.USER, content="decide")], _Decision
    )
    assert result.winner == "alice"
    assert result.confidence == 0.9

    call = fake_client.chat.completions.create_calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == "_Decision"
    assert call["response_format"]["json_schema"]["strict"] is True


async def test_structured_json_schema_strategy_does_not_fallback(
    fake_client: _FakeOpenAIClient,
) -> None:
    """With ``json_schema`` strategy explicit, a bad response raises
    immediately — no retry with json_object."""
    strict = OpenAIProvider(
        model="gpt-test",
        client=fake_client,  # type: ignore[arg-type]
        structured_strategy="json_schema",
    )
    fake_client.chat.completions.will_return(
        _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content='{"nope": true}'))])
    )
    with pytest.raises(ResponseSchemaError, match="json_schema mode"):
        await strict.structured([Message(role=MessageRole.USER, content="x")], _Decision)
    # Exactly one API call — no retry with the fallback path.
    assert len(fake_client.chat.completions.create_calls) == 1


# ---------------------------------------------------------------------------
# Structured output — json_object strategy (portable fallback)
# ---------------------------------------------------------------------------


async def test_structured_json_object_strategy_injects_schema_hint(
    fake_client: _FakeOpenAIClient,
) -> None:
    portable = OpenAIProvider(
        model="gpt-test",
        client=fake_client,  # type: ignore[arg-type]
        structured_strategy="json_object",
    )
    fake_client.chat.completions.will_return(
        _FakeResponse(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(content=json.dumps({"winner": "bob", "confidence": 0.7}))
                )
            ]
        )
    )
    result = await portable.structured(
        [Message(role=MessageRole.USER, content="decide")], _Decision
    )
    assert result.winner == "bob"

    call = fake_client.chat.completions.create_calls[0]
    assert call["response_format"] == {"type": "json_object"}
    # Schema hint must be prepended as the FIRST message — trailing system
    # messages get ignored by some gateways, and leading system guidance
    # is the most influential signal.
    first_msg = call["messages"][0]
    assert first_msg["role"] == "system"
    assert "CRITICAL OUTPUT CONSTRAINT" in first_msg["content"]
    # The field name list must appear verbatim so the model sees exact strings
    assert "'winner'" in first_msg["content"]
    assert "'confidence'" in first_msg["content"]


async def test_structured_json_object_strips_markdown_fences(
    fake_client: _FakeOpenAIClient,
) -> None:
    """Weak gateway models love to wrap JSON in ```json ... ``` blocks
    even when told not to.  We must strip the decoration before parsing."""
    portable = OpenAIProvider(
        model="gpt-test",
        client=fake_client,  # type: ignore[arg-type]
        structured_strategy="json_object",
    )
    # Simulate a model that returned JSON wrapped in markdown fences
    wrapped = '```json\n{"winner": "carol", "confidence": 0.6}\n```'
    fake_client.chat.completions.will_return(
        _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=wrapped))])
    )
    result = await portable.structured(
        [Message(role=MessageRole.USER, content="decide")], _Decision
    )
    assert result.winner == "carol"
    assert result.confidence == 0.6


async def test_structured_json_object_strips_preamble(
    fake_client: _FakeOpenAIClient,
) -> None:
    """Some models prefix the JSON with prose like 'Here is the decision:'.
    We should extract the first {...} block rather than erroring out."""
    portable = OpenAIProvider(
        model="gpt-test",
        client=fake_client,  # type: ignore[arg-type]
        structured_strategy="json_object",
    )
    with_preamble = 'Sure! Here is the decision:\n{"winner": "dave", "confidence": 0.5}\nHope this helps!'
    fake_client.chat.completions.will_return(
        _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=with_preamble))])
    )
    result = await portable.structured(
        [Message(role=MessageRole.USER, content="decide")], _Decision
    )
    assert result.winner == "dave"


# ---------------------------------------------------------------------------
# Structured output — auto strategy (tries schema, falls back on failure)
# ---------------------------------------------------------------------------


async def test_structured_auto_falls_back_on_schema_mismatch(
    fake_client: _FakeOpenAIClient,
) -> None:
    """When json_schema mode returns schema-mismatched JSON (futrixapi
    behaviour), auto strategy should retry once with json_object and
    return the second response."""
    auto = OpenAIProvider(
        model="gpt-test",
        client=fake_client,  # type: ignore[arg-type]
        structured_strategy="auto",
    )
    # First call: json_schema returns junk that doesn't match _Decision.
    # Second call: json_object returns valid JSON.
    fake_client.chat.completions.will_return(
        _FakeResponse(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(
                        content=json.dumps({"arbiter": "Futrix", "confidence_gap": 0.05})
                    )
                )
            ]
        )
    )
    # _FakeCompletions only stores one _next at a time — we need a queue.
    # Easiest: monkey-patch the create() method with a counter-driven one.

    call_count = {"n": 0}

    async def create_multi(**kwargs: Any) -> _FakeResponse:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: schema-mismatched JSON
            return _FakeResponse(
                choices=[
                    _FakeChoice(
                        message=_FakeMessage(
                            content=json.dumps({"arbiter": "Futrix", "confidence_gap": 0.05})
                        )
                    )
                ]
            )
        # Second call: valid JSON
        return _FakeResponse(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(
                        content=json.dumps({"winner": "alice", "confidence": 0.88})
                    )
                )
            ]
        )

    fake_client.chat.completions.create = create_multi  # type: ignore[method-assign]

    result = await auto.structured([Message(role=MessageRole.USER, content="decide")], _Decision)
    assert result.winner == "alice"
    assert result.confidence == 0.88
    assert call_count["n"] == 2

    # After the latch, a second structured() call should skip json_schema
    # entirely — only one more API call.
    call_count_after_latch = call_count["n"]
    await auto.structured([Message(role=MessageRole.USER, content="decide again")], _Decision)
    assert call_count["n"] == call_count_after_latch + 1  # exactly one more call


async def test_structured_rejects_invalid_json(
    provider: OpenAIProvider, fake_client: _FakeOpenAIClient
) -> None:
    """Even auto strategy gives up after the fallback also fails."""
    # First call (json_schema): not even valid JSON → ResponseSchemaError
    # Second call (json_object fallback): also garbage → ResponseSchemaError raised to caller
    call_count = {"n": 0}

    async def create_broken(**kwargs: Any) -> _FakeResponse:
        call_count["n"] += 1
        return _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content="not json at all"))]
        )

    fake_client.chat.completions.create = create_broken  # type: ignore[method-assign]

    with pytest.raises(ResponseSchemaError):
        await provider.structured([Message(role=MessageRole.USER, content="decide")], _Decision)
    # auto strategy → tried json_schema then json_object, both failed → 2 calls
    assert call_count["n"] == 2
