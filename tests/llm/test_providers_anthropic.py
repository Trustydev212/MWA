"""Tests for AnthropicProvider — mocked SDK client, no network.

We never import the real ``anthropic`` package here.  Instead we pass a
hand-rolled fake client into the provider constructor.  That lets us
test the translation logic (MWA Message -> Anthropic payload -> MWA
ChatResponse) without requiring the SDK to be installed at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mwa.llm import (
    ChatOptions,
    Message,
    MessageRole,
    PermanentProviderError,
    RateLimitError,
    ResponseSchemaError,
    TransientProviderError,
)
from mwa.llm.providers import AnthropicProvider

# ---------------------------------------------------------------------------
# Fake anthropic client
# ---------------------------------------------------------------------------


@dataclass
class _FakeBlock:
    type: str
    text: str = ""
    name: str | None = None
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]
    usage: _FakeUsage = field(default_factory=_FakeUsage)
    stop_reason: str = "end_turn"


class _FakeMessages:
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
            raise RuntimeError("_FakeMessages.create called without a queued response")
        return self._next


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


@pytest.fixture
def fake_client() -> _FakeAnthropicClient:
    return _FakeAnthropicClient()


@pytest.fixture
def provider(fake_client: _FakeAnthropicClient) -> AnthropicProvider:
    return AnthropicProvider(model="claude-test", client=fake_client)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Basic chat
# ---------------------------------------------------------------------------


async def test_chat_builds_proper_payload_and_parses_response(
    provider: AnthropicProvider,
    fake_client: _FakeAnthropicClient,
) -> None:
    fake_client.messages.will_return(
        _FakeResponse(
            content=[_FakeBlock(type="text", text="Hello!")],
            usage=_FakeUsage(input_tokens=7, output_tokens=3),
            stop_reason="end_turn",
        )
    )

    response = await provider.chat(
        [
            Message(role=MessageRole.SYSTEM, content="You are a test bot."),
            Message(role=MessageRole.USER, content="Hi"),
        ],
        options=ChatOptions(temperature=0.1, max_tokens=100),
    )

    assert response.content == "Hello!"
    assert response.provider == "anthropic"
    assert response.model == "claude-test"
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3
    assert response.finish_reason == "end_turn"

    call = fake_client.messages.create_calls[0]
    assert call["model"] == "claude-test"
    assert call["system"] == "You are a test bot."
    assert call["messages"] == [{"role": "user", "content": "Hi"}]
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 100


async def test_system_messages_are_concatenated(
    provider: AnthropicProvider,
    fake_client: _FakeAnthropicClient,
) -> None:
    fake_client.messages.will_return(_FakeResponse(content=[_FakeBlock(type="text", text="ok")]))
    await provider.chat(
        [
            Message(role=MessageRole.SYSTEM, content="Rule 1"),
            Message(role=MessageRole.SYSTEM, content="Rule 2"),
            Message(role=MessageRole.USER, content="go"),
        ]
    )
    assert fake_client.messages.create_calls[0]["system"] == "Rule 1\n\nRule 2"


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


class _FakeRateLimitException(Exception):
    """Exception name must contain 'RateLimit' for the translator."""


_FakeRateLimitException.__name__ = "RateLimitError"


class _FakeAuthException(Exception):
    pass


_FakeAuthException.__name__ = "AuthenticationError"


class _FakeServerException(Exception):
    pass


_FakeServerException.__name__ = "InternalServerError"


async def test_rate_limit_is_translated(
    provider: AnthropicProvider,
    fake_client: _FakeAnthropicClient,
) -> None:
    fake_client.messages.will_raise(_FakeRateLimitException("429 slow down"))
    with pytest.raises(RateLimitError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


async def test_auth_error_is_permanent(
    provider: AnthropicProvider,
    fake_client: _FakeAnthropicClient,
) -> None:
    fake_client.messages.will_raise(_FakeAuthException("bad key"))
    with pytest.raises(PermanentProviderError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


async def test_server_error_is_transient(
    provider: AnthropicProvider,
    fake_client: _FakeAnthropicClient,
) -> None:
    fake_client.messages.will_raise(_FakeServerException("500"))
    with pytest.raises(TransientProviderError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


# ---------------------------------------------------------------------------
# Structured output via tool_use
# ---------------------------------------------------------------------------


from pydantic import BaseModel  # noqa: E402 (declared late so tests above stay self-contained)


class _Decision(BaseModel):
    winner: str
    confidence: float


async def test_structured_returns_validated_schema(
    provider: AnthropicProvider,
    fake_client: _FakeAnthropicClient,
) -> None:
    fake_client.messages.will_return(
        _FakeResponse(
            content=[
                _FakeBlock(
                    type="tool_use",
                    name="_Decision",
                    input={"winner": "alice", "confidence": 0.9},
                )
            ]
        )
    )

    result = await provider.structured(
        [Message(role=MessageRole.USER, content="resolve")],
        _Decision,
    )
    assert isinstance(result, _Decision)
    assert result.winner == "alice"
    assert result.confidence == 0.9

    call = fake_client.messages.create_calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "_Decision"}
    assert len(call["tools"]) == 1
    assert call["tools"][0]["name"] == "_Decision"


async def test_structured_missing_tool_raises(
    provider: AnthropicProvider,
    fake_client: _FakeAnthropicClient,
) -> None:
    fake_client.messages.will_return(
        _FakeResponse(content=[_FakeBlock(type="text", text="I don't know")])
    )
    with pytest.raises(ResponseSchemaError, match="did not return a tool_use"):
        await provider.structured([Message(role=MessageRole.USER, content="resolve")], _Decision)


async def test_structured_invalid_tool_args_raises(
    provider: AnthropicProvider,
    fake_client: _FakeAnthropicClient,
) -> None:
    fake_client.messages.will_return(
        _FakeResponse(
            content=[
                _FakeBlock(
                    type="tool_use",
                    name="_Decision",
                    input={"winner": "alice"},  # missing confidence
                )
            ]
        )
    )
    with pytest.raises(ResponseSchemaError):
        await provider.structured([Message(role=MessageRole.USER, content="resolve")], _Decision)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_provider_identity(provider: AnthropicProvider) -> None:
    assert provider.name == "anthropic"
    assert provider.model == "claude-test"


def test_count_tokens_is_a_heuristic(provider: AnthropicProvider) -> None:
    assert provider.count_tokens("hello world") >= 1
