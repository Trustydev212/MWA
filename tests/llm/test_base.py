"""Tests for the pure base types + error hierarchy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mwa.errors import LLMProviderError, MWAError
from mwa.llm import (
    ChatOptions,
    ChatResponse,
    Message,
    MessageRole,
    PermanentProviderError,
    RateLimitError,
    ResponseSchemaError,
    TransientProviderError,
    Usage,
)


def test_error_hierarchy_rooted_at_mwa_error() -> None:
    for cls in (
        RateLimitError,
        TransientProviderError,
        PermanentProviderError,
        ResponseSchemaError,
    ):
        assert issubclass(cls, LLMProviderError)
        assert issubclass(cls, MWAError)


def test_message_role_values_are_stable() -> None:
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.TOOL.value == "tool"


def test_message_is_frozen() -> None:
    msg = Message(role=MessageRole.USER, content="hi")
    with pytest.raises(ValidationError):
        msg.content = "bye"  # type: ignore[misc]


def test_usage_addition() -> None:
    a = Usage(input_tokens=10, output_tokens=5)
    b = Usage(input_tokens=3, output_tokens=2)
    total = a + b
    assert total.input_tokens == 13
    assert total.output_tokens == 7
    assert total.total_tokens == 20


def test_usage_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1, output_tokens=0)


def test_chat_options_validates_temperature_range() -> None:
    ChatOptions(temperature=0.0)
    ChatOptions(temperature=2.0)
    with pytest.raises(ValidationError):
        ChatOptions(temperature=-0.1)
    with pytest.raises(ValidationError):
        ChatOptions(temperature=2.1)


def test_chat_options_validates_max_tokens_positive() -> None:
    with pytest.raises(ValidationError):
        ChatOptions(max_tokens=0)


def test_chat_response_minimum_fields() -> None:
    r = ChatResponse(content="hello", model="m", provider="p")
    assert r.content == "hello"
    assert r.usage.input_tokens == 0
    assert r.finish_reason is None
