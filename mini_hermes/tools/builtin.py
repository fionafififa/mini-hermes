from __future__ import annotations

from pathlib import Path
from typing import Any

from mini_hermes.tools.registry import ToolRegistry, ToolSpec


def build_default_registry(workspace: Path) -> ToolRegistry:
    root = workspace.resolve()
    registry = ToolRegistry()

    def add_numbers(args: dict[str, Any]) -> dict[str, Any]:
        a = args.get("a")
        b = args.get("b")

        # bool 在 Python 中也是 int 的子类，因此这里检查精确类型。
        if type(a) is not int or type(b) is not int:
            raise ValueError("a 和 b 必须是整数")

        return {"sum": a + b}

    registry.register(
        ToolSpec(
            name="add_numbers",
            description="计算两个整数的和。需要准确计算整数加法时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "第一个整数"},
                    "b": {"type": "integer", "description": "第二个整数"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            handler=add_numbers,
        )
    )

    def read_file(args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path 必须是非空文件路径")

        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError("只能读取项目目录内的文件")
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")

        # 最多读取 12001 个字符，用最后一个字符判断是否截断。
        with target.open("r", encoding="utf-8") as file:
            text = file.read(12001)

        return {
            "path": target.relative_to(root).as_posix(),
            "content": text[:12000],
            "truncated": len(text) > 12000,
        }

    registry.register(
        ToolSpec(
            name="read_file",
            description=(
                "读取当前项目目录内的 UTF-8 文本文件，"
                "最多返回前 12000 个字符。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "相对于项目根目录的路径，"
                            "例如 mini_hermes/messages.py"
                        ),
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=read_file,
        )
    )

    return registry