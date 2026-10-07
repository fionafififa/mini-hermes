from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    """程序启动时读取的配置。"""

    provider: str
    model: str
    base_url: str
    api_key_env: str = "MINI_HERMES_API_KEY"
    max_iterations: int = 8
    max_output_tokens: int = 1024
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("provider", "model", "base_url", "api_key_env"):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"配置项 {name} 必须是非空字符串")

        for name in ("max_iterations", "max_output_tokens"):
            value = getattr(self, name)

            if type(value) is not int or value <= 0:
                 # bool 是 int 的子类，因此这里使用 type(...) is int
                raise ValueError(f"配置项 {name} 必须是正整数")

        if not isinstance(self.extra_body, dict):
            raise ValueError("配置项 extra_body 必须是对象（TOML table）")

        # Provider 负责核心消息和调用上限；扩展参数不能覆盖它们。
        reserved = {"model", "messages", "tools", "system", "max_tokens", "stream"}
        conflicts = reserved.intersection(self.extra_body)
        if conflicts:
            raise ValueError(
                f"extra_body 不能覆盖核心请求字段：{', '.join(sorted(conflicts))}"
            )


def load_config(path: str | Path = "config.toml") -> AppConfig:
    """从 TOML 文件读取配置；默认路径相对于当前工作目录。"""

    config_path = Path(path)

    # utf-8-sig 同时支持普通 UTF-8 和带 BOM 的 UTF-8 文件。
    text = config_path.read_text(encoding="utf-8-sig")
    values = tomllib.loads(text)

    return AppConfig(**values)


def get_api_key(config: AppConfig) -> str:
    """在实际调用模型时，从环境变量读取密钥。"""

    api_key = os.environ.get(config.api_key_env, "").strip()

    if not api_key:
        raise ValueError(
            f"缺少 API Key，请设置环境变量：{config.api_key_env}"
        )

    return api_key

