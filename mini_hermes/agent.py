from __future__ import annotations

from mini_hermes.messages import Message
from mini_hermes.providers.base import LLMProvider
from mini_hermes.tools.registry import ToolRegistry


class Agent:
    def __init__(self,
        provider: LLMProvider,
        tools: ToolRegistry,
        system_prompt: str,
        max_iterations:int =8,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations 必须大于等于 1。")

        self.provider = provider
        self.tools = tools
        self.max_iterations = max_iterations
        self.messages: list[Message] = [
            Message(
                role="system",
                content=system_prompt,
            )
        ]

    def run_turn(self, user_input: str) -> str:
        """运行一次交互回合，返回模型的最终文本回复。"""
        if not user_input.strip():
            raise ValueError("用户输入不能为空。")

        self.messages.append(
            Message(role="user", content=user_input)
        )

        #一次迭代就是一次模型请求，一次回复可以包含多个工具调用。
        for _ in range(self.max_iterations):
            response = self.provider.generate(
                messages=self.messages,
                tools=self.tools.schemas(),
            )
            assistant_message = response.message

            if(
                not assistant_message.tool_calls
                and not assistant_message.content.strip()
            ):
                raise RuntimeError("模型返回了空内容，也没有请求调用工具")

            #先记录模型发出的工具，再追加对应的工具结果。
            self.messages.append(assistant_message)

            if not assistant_message.tool_calls:
                #模型没有请求调用工具，直接返回文本回复。
                return assistant_message.content

            for call in assistant_message.tool_calls:
                result = self.tools.execute(call)
                self.messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=call.id,
                    )
                )

        answer = (
                f"本轮已达到{self.max_iterations}次模型调用上限，"
                "工具结果已记录，但模型尚未给出最终回答。"
            )
        self.messages.append(
                Message(
                    role="assistant",
                    content=answer)
                )
        return answer

            