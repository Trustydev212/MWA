"""Concrete :class:`~soma.llm.base.LLMProvider` adapters.

Each adapter lazy-imports its vendor SDK inside ``__init__`` so that
importing ``soma.llm.providers`` is cheap and doesn't require any
particular SDK to be installed.  Example::

    from soma.llm.providers import FakeProvider          # always works
    from soma.llm.providers import AnthropicProvider     # needs anthropic SDK
    from soma.llm.providers import OpenAIProvider        # needs openai SDK
    from soma.llm.providers import OllamaProvider        # needs httpx

Installing the extra for the provider you want is the supported path:

    uv pip install soma[anthropic]
    uv pip install soma[openai]
    uv pip install soma[ollama]
    uv pip install soma[all-llm]

:class:`FakeProvider` has no SDK dependency and is safe to use in tests
and examples.
"""

from soma.llm.providers.anthropic import AnthropicProvider
from soma.llm.providers.fake import FakeProvider
from soma.llm.providers.ollama import OllamaProvider
from soma.llm.providers.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "FakeProvider",
    "OllamaProvider",
    "OpenAIProvider",
]
