from __future__ import annotations

from mini_hermes.context import ContextBudget
from mini_hermes.memory import MemoryStore
from mini_hermes.messages import Message
from mini_hermes.providers.base import LLMProvider
from mini_hermes.session_store import SessionStore
from mini_hermes.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 8,
        *,
        store: SessionStore | None = None,
        session_id: str | None = None,
        memory: MemoryStore | None = None,
        context: ContextBudget | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations 必须大于等于 1。")
        if session_id is not None and store is None:
            raise ValueError("提供 session_id 时必须同时提供 store。")

        self.provider = provider
        self.tools = tools
        self.max_iterations = max_iterations
        self.store = store
        self.session_id = session_id
        self.memory = memory
        self.context = context

        if store is not None and session_id is not None:
            saved = store.load_session(session_id)
            self.messages = [
                Message(role="system", content=saved.system_prompt),
                *saved.messages,
            ]
        else:
            self.messages = [Message(role="system", content=system_prompt)]
            if store is not None:
                self.session_id = store.create_session(system_prompt)

    def _commit_turn(self, messages: list[Message]) -> None:
        """保存完整新增回合，不保存裁剪后的请求视图。"""
        if self.store is not None:
            assert self.session_id is not None, "会话ID不能为空。"
            new_messages = messages[len(self.messages):]
            self.store.append_message(self.session_id, new_messages)
        self.messages = messages

    def run_turn(self, user_input: str) -> str:
        if not user_input.strip():
            raise ValueError("用户输入不能为空。")

        # 每个用户回合只召回一次，工具循环复用同一份记忆快照。
        recalled = self.memory.recall(user_input) if self.memory is not None else ""
        user_content = (
            f"{recalled}\n\n【当前用户输入】\n{user_input}"
            if recalled
            else user_input
        )
        messages = [
            *self.messages,
            Message(role="user", content=user_content),
        ]
        tools = self.tools.schemas()

        for _ in range(self.max_iterations):
            # 工具结果追加后，下次模型请求前必须重新检查预算。
            request_messages = (
                self.context.prepare(messages, tools)
                if self.context is not None
                else messages
            )
            response = self.provider.generate(
                messages=request_messages,
                tools=tools,
            )
            assistant_message = response.message
            if (
                not assistant_message.tool_calls
                and not assistant_message.content.strip()
            ):
                raise RuntimeError("模型返回了空内容，也没有请求调用工具")

            # 始终向完整的工作历史追加，而不是向请求视图追加。
            messages.append(assistant_message)
            if not assistant_message.tool_calls:
                self._commit_turn(messages)
                return assistant_message.content

            for call in assistant_message.tool_calls:
                result = self.tools.execute(call)
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=call.id,
                ))

        answer = (
            f"本轮已达到{self.max_iterations}次模型调用上限，"
            "工具结果已记录，但模型尚未给出最终回答。"
        )
        messages.append(Message(role="assistant", content=answer))
        self._commit_turn(messages)
        return answer