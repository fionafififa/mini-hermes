from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mini_hermes.messages import Message, ModelResponse


class LLMProvider(ABC):
    """所有模型provider都遵守的接口。"""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        #本阶段不跑通tools,先跑通文本回答
        tools: list[dict[str,Any]] | None = None,
    ) -> ModelResponse:
        """根据消息生成模型响应。"""
        raise NotImplementedError