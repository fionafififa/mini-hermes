from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from mini_hermes.messages import Message

logger = logging.getLogger(__name__)


def estimate_input_tokens(
    messages: list[Message], tools: list[dict[str, Any]],
) -> int:
    """教学用粗估：计算消息、工具调用、工具结果和工具定义的大小。"""
    text = json.dumps(
        {"messages": [asdict(m) for m in messages], "tools": tools},
        ensure_ascii=False,
    )
    ascii_chars = sum(ord(char) < 128 for char in text)
    other_chars = len(text) - ascii_chars
    return (ascii_chars + 3) // 4 + other_chars * 2 + len(messages) * 8


@dataclass(frozen=True)
class ContextBudget:
    context_window: int
    reserved_output_tokens: int

    def __post_init__(self) -> None:
        if type(self.context_window) is not int or self.context_window < 1:
            raise ValueError("context_window 必须是正整数。")
        if (
            type(self.reserved_output_tokens) is not int
            or not 0 <= self.reserved_output_tokens < self.context_window
        ):
            raise ValueError("输出预留必须是非负整数，且小于上下文窗口。")

    @property
    def input_limit(self) -> int:
        # 先预留输出，再留出 20% 输入余量；粗估不等于真实 tokenizer。
        return int((self.context_window - self.reserved_output_tokens) * 0.8)

    def prepare(
        self, messages: list[Message], tools: list[dict[str, Any]],
    ) -> list[Message]:
        if not messages or messages[0].role != "system":
            raise ValueError("上下文必须以 system 消息开头。")
        if any(m.role == "system" for m in messages[1:]):
            raise ValueError("system 消息只能出现在开头。")

        turn_starts = [i for i, m in enumerate(messages) if m.role == "user"]
        if not turn_starts or turn_starts[0] != 1:
            raise ValueError("system 后必须从一个 user 回合开始。")

        request_messages = list(messages)
        before_tokens = estimate_input_tokens(request_messages, tools)
        tokens = before_tokens
        dropped_turns = 0

        # 只移除完整旧回合，最后一个 user 开始的当前回合始终保留。
        while tokens > self.input_limit and dropped_turns < len(turn_starts) - 1:
            dropped_turns += 1
            start = turn_starts[dropped_turns]
            request_messages = [messages[0], *messages[start:]]
            tokens = estimate_input_tokens(request_messages, tools)

        if tokens > self.input_limit:
            raise RuntimeError(
                f"系统提示、工具定义与当前回合约需 {tokens} 个输入 token，"
                f"超过预算 {self.input_limit}。请缩短当前输入或工具输出，"
                "或使用更大的上下文窗口；当前回合不会被拆开。"
            )

        logger.info(
            "context: estimated=%d -> %d; input_limit=%d; dropped_turns=%d",
            before_tokens, tokens, self.input_limit, dropped_turns,
        )
        return request_messages