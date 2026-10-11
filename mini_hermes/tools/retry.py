from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

class TemporaryToolError(Exception):
    """由工具明确标识：本次失败可能通过重试恢复。"""


@dataclass(frozen = True)
class RetryPolicy:
    #总尝试次数，包含第一次执行；不是额外重试次数。
    max_attempts: int = 1
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if type(self.max_attempts) != int or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer.")
        for delay in (self.base_delay_seconds, self.max_delay_seconds):
            if type(delay) not in (int, float) or not math.isfinite(delay):
                raise ValueError("base_delay_seconds and max_delay_seconds must be finite numbers.")
        if not 0<=self.base_delay_seconds<=self.max_delay_seconds:
            raise ValueError("base_delay_seconds must be between 0 and max_delay_seconds.")


def call_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy,
    *,
    tool_name: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    delay = policy.base_delay_seconds
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except TemporaryToolError:
            if attempt == policy.max_attempts:
                raise
            # 不记录工具参数和原始异常，避免日志泄漏敏感信息。
            logger.warning(
                "tool_retry: tool=%s attempt=%d/%d delay=%.2fs",
                tool_name, attempt, policy.max_attempts, delay,
            )
            sleep(delay)
            delay = min(delay * 2, policy.max_delay_seconds)