"""Tests for OllamaProvider — mocked httpx client, no network."""

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
from mwa.llm.providers import OllamaProvider

# ---------------------------------------------------------------------------
# Fake httpx
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    status_code: int = 200
    _json: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = _FakeHTTPStatusError(f"HTTP {self.status_code}")
            err.response = self  # type: ignore[attr-defined]
            raise err


class _FakeHTTPStatusError(Exception):
    response: _FakeResponse | None = None


class _FakeHttpxClient:
    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self._next: _FakeResponse | BaseException | None = None

    def will_return(self, response: _FakeResponse) -> None:
        self._next = response

    def will_raise(self, error: BaseException) -> None:
        self._next = error

    async def post(self, url: str, *, json: dict[str, Any], timeout: float) -> _FakeResponse:
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        if isinstance(self._next, BaseException):
            raise self._next
        if self._next is None:
            raise RuntimeError("_FakeHttpxClient.post called without a queued response")
        return self._next


@pytest.fixture
def fake_client() -> _FakeHttpxClient:
    return _FakeHttpxClient()


@pytest.fixture
def provider(fake_client: _FakeHttpxClient) -> OllamaProvider:
    return OllamaProvider(
        model="llama3-test",
        client=fake_client,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def test_chat_posts_to_api_chat_and_parses_response(
    provider: OllamaProvider,
    fake_client: _FakeHttpxClient,
) -> None:
    fake_client.will_return(
        _FakeResponse(
            _json={
                "model": "llama3-test",
                "message": {"role": "assistant", "content": "hello from llama"},
                "done": True,
                "prompt_eval_count": 8,
                "eval_count": 4,
            }
        )
    )
    response = await provider.chat(
        [
            Message(role=MessageRole.SYSTEM, content="be nice"),
            Message(role=MessageRole.USER, content="hi"),
        ],
        options=ChatOptions(temperature=0.3, max_tokens=64),
    )
    assert response.content == "hello from llama"
    assert response.provider == "ollama"
    assert response.usage.input_tokens == 8
    assert response.usage.output_tokens == 4

    call = fake_client.post_calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    body = call["json"]
    assert body["model"] == "llama3-test"
    assert body["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.3
    assert body["options"]["num_predict"] == 64


# ---------------------------------------------------------------------------
# HTTP status → error class
# ---------------------------------------------------------------------------


async def test_http_429_is_rate_limit(
    provider: OllamaProvider, fake_client: _FakeHttpxClient
) -> None:
    fake_client.will_return(_FakeResponse(status_code=429))
    with pytest.raises(RateLimitError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


async def test_http_500_is_transient(
    provider: OllamaProvider, fake_client: _FakeHttpxClient
) -> None:
    fake_client.will_return(_FakeResponse(status_code=502))
    with pytest.raises(TransientProviderError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


async def test_http_404_is_permanent(
    provider: OllamaProvider, fake_client: _FakeHttpxClient
) -> None:
    fake_client.will_return(_FakeResponse(status_code=404))
    with pytest.raises(PermanentProviderError):
        await provider.chat([Message(role=MessageRole.USER, content="hi")])


# ---------------------------------------------------------------------------
# Structured output (format=json + schema injection)
# ---------------------------------------------------------------------------


class _Decision(BaseModel):
    winner: str
    confidence: float


async def test_structured_injects_schema_and_parses(
    provider: OllamaProvider, fake_client: _FakeHttpxClient
) -> None:
    fake_client.will_return(
        _FakeResponse(
            _json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"winner": "bob", "confidence": 0.72}),
                },
                "done": True,
            }
        )
    )
    result = await provider.structured(
        [Message(role=MessageRole.USER, content="decide")], _Decision
    )
    assert result.winner == "bob"
    assert result.confidence == 0.72

    body = fake_client.post_calls[0]["json"]
    assert body["format"] == "json"
    # schema hint should be in the last message
    last_msg = body["messages"][-1]
    assert last_msg["role"] == "system"
    assert "JSON schema" in last_msg["content"]


async def test_structured_rejects_invalid_content(
    provider: OllamaProvider, fake_client: _FakeHttpxClient
) -> None:
    fake_client.will_return(
        _FakeResponse(
            _json={
                "message": {"role": "assistant", "content": "I refuse to comply"},
                "done": True,
            }
        )
    )
    with pytest.raises(ResponseSchemaError):
        await provider.structured([Message(role=MessageRole.USER, content="decide")], _Decision)


def test_metadata(provider: OllamaProvider) -> None:
    assert provider.name == "ollama"
    assert provider.model == "llama3-test"
