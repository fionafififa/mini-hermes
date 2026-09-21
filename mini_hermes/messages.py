from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """模型请求执行的一次工具调用"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("工具调用ID必须是一个非空字符串。")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("工具调用名称必须是一个非空字符串。")
        if not isinstance(self.arguments, dict):
            raise ValueError("工具调用参数必须是一个字典。")


@dataclass
class Message:
    """会话中的一条消息"""

    role: Role
    content:str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"无效的角色: {self.role}. 角色必须是 'system', 'user', 'assistant' 或 'tool'.")
        if not isinstance(self.content, str):
            raise ValueError("当前版本的content必须是一个字符串。")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("只有角色为 'assistant' 的消息才可以包含工具调用。")
        if self.role == "tool":
            if(
                not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip()
            ):
                raise ValueError("角色为 'tool' 的消息必须包含有效的工具调用ID。")
        elif self.tool_call_id is not None:
            raise ValueError("只有角色为 'tool' 的消息才可以包含工具调用ID。")


@dataclass
class TokenUsage:
    """一次模型请求报告的 token 用量"""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens



@dataclass
class ModelResponse:
    """模型请求的响应"""

    message: Message
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if self.message.role != "assistant":
            raise ValueError("模型响应的消息角色必须是 'assistant'.")       
