from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from mini_hermes.messages import ToolCall


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]



@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

class ToolRegistry:
    """工具注册表，负责管理所有可用的工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self,spec: ToolSpec) -> None:
        """注册一个新的工具。"""
        if spec.name in self._tools:
            raise ValueError(f"工具 '{spec.name}' 已经注册。")
        self._tools[spec.name] = spec

    def schemas(self)-> list[dict[str, Any]]:
        """返回给模型所有注册工具的 JSON Schema 列表。排序保证每次请求的顺序稳定。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for _, spec in sorted(self._tools.items())
        ]

    def execute(self, call: ToolCall) ->  str:
        """执行工具调用，并返回结果JSON字符串。"""
        if call.parse_error:
            return self._error("invalid_arguments", f"参数解析失败: {call.parse_error}")
        
        spec = self._tools.get(call.name)
        if spec is None:
            return self._error("tool_not_found", f"未找到工具: {call.name}")

        try:
            data = spec.handler(call.arguments)
            return json.dumps(
                {"ok": True, "data": data}, ensure_ascii=False
            )
        except (ValueError, FileNotFoundError) as exc:
            return self._error("invalid_arguments", str(exc))
        except Exception as exc:
            return self._error(
                "execution_error", f"{type(exc).__name__}: {str(exc)}"
            )


    @staticmethod
    def _error(code: str, message: str) -> str:
        """返回一个标准化的错误JSON字符串。"""
        return json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )
