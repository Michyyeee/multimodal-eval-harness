"""Unified model factory and registry for multi-provider competitor evaluations (Zero dependencies)."""

from typing import Any, Dict, List

from src.models.anthropic_model import AnthropicVisionModel
from src.models.base import BaseVisionModel
from src.models.gemini_model import GeminiVisionModel
from src.models.mock_model import MockVisionModel
from src.models.openai_model import OpenAIVisionModel


def get_model(model_identifier: str, **kwargs: Any) -> BaseVisionModel:
    """Factory function resolving model identifiers into concrete provider instances.
    
    Supports formats:
      - Explicit provider prefix: "gemini:gemini-1.5-flash", "openai:gpt-4o", "anthropic:claude-3-5-sonnet-20241022"
      - Implicit model name: "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-flash"
      - Hermetic mocks: "mock", "mock_adversarial"
    """
    clean_id = model_identifier.strip()

    # 1. Check for provider prefix
    if ":" in clean_id:
        provider, model_name = clean_id.split(":", 1)
        provider = provider.lower()
        if provider == "gemini":
            return GeminiVisionModel(model_name=model_name, **kwargs)
        elif provider in ("openai", "gpt"):
            return OpenAIVisionModel(model_name=model_name, **kwargs)
        elif provider in ("anthropic", "claude"):
            return AnthropicVisionModel(model_name=model_name, **kwargs)
        elif provider == "mock":
            is_adv = "adversarial" in model_name.lower()
            return MockVisionModel(model_name=model_name, simulate_adversarial=is_adv)
        else:
            raise ValueError(f"Unsupported provider prefix '{provider}'. Supported: gemini, openai, anthropic, mock")

    # 2. Check for mock
    if clean_id.lower().startswith("mock"):
        is_adv = "adversarial" in clean_id.lower()
        return MockVisionModel(model_name=clean_id, simulate_adversarial=is_adv)

    # 3. Auto-detect provider based on model family naming
    lower_id = clean_id.lower()
    if lower_id.startswith("gemini"):
        return GeminiVisionModel(model_name=clean_id, **kwargs)
    elif lower_id.startswith("gpt") or lower_id.startswith("o1") or lower_id.startswith("o3"):
        return OpenAIVisionModel(model_name=clean_id, **kwargs)
    elif lower_id.startswith("claude"):
        return AnthropicVisionModel(model_name=clean_id, **kwargs)

    # Default fallback to Gemini
    return GeminiVisionModel(model_name=clean_id, **kwargs)


def list_supported_providers() -> Dict[str, List[str]]:
    """Lists standard supported models across providers."""
    return {
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        "openai": ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "mock": ["mock", "mock_adversarial"],
    }
