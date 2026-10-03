from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_PROVIDER_ALIASES = {
    "openai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    def is_configured(self) -> bool:
        """True when this provider has enough settings to make a live call."""

        if self.provider == "ollama":
            return bool(self.model_name)
        if self.provider == "custom":
            return bool(self.model_name and self.base_url)
        return bool(self.model_name and self.api_key)


def normalize_provider(value: str) -> str:
    """Map aliases / common typos (e.g. `anthorpic`) to a canonical provider name."""

    key = (value or "").strip().lower()
    if key in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[key]
    raise ValueError(f"Unsupported provider {value!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")


def invoke_with_usage(agent, inputs: dict, **kwargs) -> tuple[dict, int, int]:
    """Invoke a LangChain agent and return (result, input_tokens, output_tokens).

    Usage is summed over every model call in the turn (tool loops, summarization),
    so it reflects what the provider actually billed.
    """

    from langchain_core.callbacks import get_usage_metadata_callback

    with get_usage_metadata_callback() as cb:
        result = agent.invoke(inputs, **kwargs)
    input_tokens = sum(u.get("input_tokens", 0) for u in cb.usage_metadata.values())
    output_tokens = sum(u.get("output_tokens", 0) for u in cb.usage_metadata.values())
    return result, input_tokens, output_tokens


def message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def build_chat_model(config: ProviderConfig):
    """Instantiate the LangChain chat model for the selected provider.

    Imports are lazy so the offline benchmark/tests never need provider SDKs.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        elif provider == "custom":
            raise ValueError("Provider `custom` requires CUSTOM_BASE_URL.")
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError:
            # OpenRouter is OpenAI-compatible, so fall back to ChatOpenAI.
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )

    raise ValueError(f"Unsupported provider: {config.provider}")
