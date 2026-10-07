from mini_hermes.config import AppConfig
from mini_hermes.providers.anthropic import AnthropicProvider
from mini_hermes.providers.base import LLMProvider
from mini_hermes.providers.openai_compatible import OpenAICompatibleProvider

PROVIDERS :dict[str,type[LLMProvider]] ={
    "openai_compatible":OpenAICompatibleProvider,
    "anthropic":AnthropicProvider,
}


def create_provider(
    config:AppConfig,
) -> LLMProvider:
    provider_class = PROVIDERS.get(config.provider)

    if provider_class is None:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"不支持的 provider：{config.provider}。"
            f"支持的 provider 有：{', '.join(PROVIDERS)}"
        )


    return provider_class(config)
