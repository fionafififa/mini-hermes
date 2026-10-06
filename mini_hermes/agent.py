from __future__ import annotations

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
        max_iterations:int =8,
        *,
        store: SessionStore | None = None,
        session_id: str | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations 必须大于等于 1。")

        if session_id is not None and store is None:
            raise ValueError(
                "如果提供了 session_id(恢复历史会话)，则必须同时提供 store。"
            )

        self.provider = provider
        self.tools = tools
        self.max_iterations = max_iterations
        self.store = store
        self.session_id = session_id

        if store is not None and session_id is not None:
            #恢复历史会话时使用数据库中的原始system_prompt
            saved = store.load_session(session_id)
            
            self.messages = [
                Message(
                    role="system",
                    content=saved.system_prompt,
            ),
            *saved.messages
            ] 
        else:
            self.messages = [
                Message(
                    role="system",
                    content=system_prompt,
                )
            ] 

            if store is not None:
                #新会话时，保存system_prompt到数据库
                self.session_id = store.create_session(system_prompt)

    def _commit_turn(self,messages:list[Message]) -> None:
        """提交当前回合的消息到数据库，再更新内存里的会话。"""

        if self.store is not None:
            assert self.session_id is not None, "会话ID不能为空。"

            #self.messages 仍是上一轮结束时的历史
            #超出其长度的部分，就是本轮新增消息
            new_messages = messages[len(self.messages):]

            self.store.append_message(
                self.session_id, 
                new_messages
            )

        #数据库提交成功后，才更新内存里的会话。
        self.messages = messages



 
    def run_turn(self, user_input: str) -> str:
        """运行一次交互回合，返回模型的最终文本回复。"""
        if not user_input.strip():
            raise ValueError("用户输入不能为空。")

        #本轮现在副本上运行；失败时，已有会话保持完整。
        messages = [
            *self.messages,
            Message(role="user", content=user_input),  
        ]

        #一次迭代就是一次模型请求，一次回复可以包含多个工具调用。
        for _ in range(self.max_iterations):
            response = self.provider.generate(
                messages=messages,
                tools=self.tools.schemas(),
            )
            assistant_message = response.message

            if(
                not assistant_message.tool_calls
                and not assistant_message.content.strip()
            ):
                raise RuntimeError("模型返回了空内容，也没有请求调用工具")

            #先记录模型发出的工具，再追加对应的工具结果。
            messages.append(assistant_message)

            if not assistant_message.tool_calls:
                #模型没有请求调用工具，提交到数据库。
                self._commit_turn(messages)
                return assistant_message.content

            for call in assistant_message.tool_calls:
                result = self.tools.execute(call)
                messages.append(
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
        messages.append(
                Message(
                    role="assistant",
                    content=answer)
                )
        self._commit_turn(messages)
        return answer

            