"""Concrete :class:`~mwa.llm.base.LLMProvider` adapters.

Each adapter lazy-imports its vendor SDK inside ``__init__`` so that
importing ``mwa.llm.providers`` is cheap and doesn't require any
particular SDK to be installed.  Example::

    from mwa.llm.providers import FakeProvider          # always works
    from mwa.llm.providers import AnthropicProvider     # needs anthropic SDK
    from mwa.llm.providers import OpenAIProvider        # needs openai SDK
    from mwa.llm.providers import OllamaProvider        # needs httpx

Installing the extra for the provider you want is the supported path:

    uv pip install mwa[anthropic]
    uv pip install mwa[openai]
    uv pip install mwa[ollama]
    uv pip install mwa[all-llm]

:class:`FakeProvider` has no SDK dependency and is safe to use in tests
and examples.
"""

from mwa.llm.providers.anthropic import AnthropicProvider
from mwa.llm.providers.fake import FakeProvider
from mwa.llm.providers.ollama import OllamaProvider
from mwa.llm.providers.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "FakeProvider",
    "OllamaProvider",
    "OpenAIProvider",
]
