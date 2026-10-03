from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "qwen2.5:7b",
    "openrouter": "openai/gpt-4o-mini",
}

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}

BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "custom": "CUSTOM_BASE_URL",
    "gemini": None,
    "anthropic": None,
    "ollama": "OLLAMA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab: paths, compact-memory knobs, and models."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # "offline" (deterministic, default) or "live" (LangChain agent with a real LLM).
    mode: str = "offline"
    # Bonus: only facts with confidence >= threshold are written into User.md.
    profile_confidence_threshold: float = 0.6
    # Bonus: memory decay for interests (score = mentions * decay ** age_in_turns).
    interest_decay: float = 0.97
    max_interests: int = 8

    @property
    def live_enabled(self) -> bool:
        return self.mode == "live" and self.model.is_configured()


def _load_dotenv(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        # Minimal fallback parser so `.env` still works without python-dotenv.
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    raw_provider = os.getenv(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai")
    provider = normalize_provider(raw_provider)
    model_name = os.getenv(f"{prefix}_MODEL") or (
        fallback.model_name if fallback and fallback.provider == provider else DEFAULT_MODELS[provider]
    )
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0" if prefix == "JUDGE" else "0.2"))
    key_env = API_KEY_ENV[provider]
    url_env = BASE_URL_ENV[provider]
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=os.getenv(key_env) if key_env else None,
        base_url=os.getenv(url_env) if url_env else None,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` + environment variables and return a populated LabConfig.

    Env knobs: LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE, JUDGE_PROVIDER / JUDGE_MODEL,
    <PROVIDER>_API_KEY, CUSTOM_BASE_URL, OLLAMA_BASE_URL, LAB_MODE (offline|live),
    COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES, PROFILE_CONFIDENCE_THRESHOLD.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = Path(os.getenv("LAB_STATE_DIR") or root / "state")
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        mode=os.getenv("LAB_MODE", "offline").strip().lower(),
        profile_confidence_threshold=float(os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.6")),
    )
