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
# Structured output via json_schema
# ---------------------------------------------------------------------------


class _Decision(BaseModel):
    winner: str
    confidence: float


async def test_structured_parses_json(
    provider: OpenAIProvider, fake_client: _FakeOpenAIClient
) -> None:
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


async def test_structured_rejects_invalid_json(
    provider: OpenAIProvider, fake_client: _FakeOpenAIClient
) -> None:
    fake_client.chat.completions.will_return(
        _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content="not json at all"))])
    )
    with pytest.raises(ResponseSchemaError):
        await provider.structured([Message(role=MessageRole.USER, content="decide")], _Decision)
