"""Single-node AI agent built on python-ai-sdk.

This agent accepts one prompt and returns one model response.
Provider and model are runtime-configurable so you can compare vendors.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import ai_sdk


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model: str
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    reasoning_effort: Optional[str] = None


class SingleNodeAgent:
    """Minimal one-node agent that performs a single prompt -> response call."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._model = self._build_model()

    def _build_model(self) -> Any:
        provider_name = self.config.provider.strip().lower().replace("-", "_")
        provider_factory = getattr(ai_sdk, provider_name, None)

        if provider_factory is None or not callable(provider_factory):
            raise ValueError(
                f"Provider '{self.config.provider}' is not available in ai_sdk. "
                "Install/enable that provider and use its factory name "
                "(for example: 'openai', 'anthropic')."
            )

        if self.config.api_key:
            return provider_factory(self.config.model, api_key=self.config.api_key)
        return provider_factory(self.config.model)

    def run(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        generate_kwargs = {
            "model": self._model,
            "system": self.config.system_prompt,
            "prompt": prompt,
        }
        if self.config.reasoning_effort:
            try:
                signature = inspect.signature(ai_sdk.generate_text)
            except (TypeError, ValueError):
                signature = None

            if signature is not None:
                parameters = signature.parameters.values()
                supports_reasoning_effort = (
                    "reasoning_effort" in signature.parameters
                    or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
                )
                if not supports_reasoning_effort:
                    raise ValueError(
                        "The installed ai_sdk.generate_text() does not support reasoning_effort, "
                        f"but reasoning_effort={self.config.reasoning_effort!r} was requested."
                    )

            generate_kwargs["reasoning_effort"] = self.config.reasoning_effort

        result = ai_sdk.generate_text(**generate_kwargs)
        return result.text
