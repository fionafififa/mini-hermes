import json
from dataclasses import asdict

from mini_hermes.config import load_config
from mini_hermes.messages import Message, ModelResponse, ToolCall


def main() -> None:
    config = load_config()

    print("配置加载成功:")
    print(json.dumps(asdict(config), indent=2, ensure_ascii=False))

    #以下都是手工构造的实例数据，不调用模型或者执行工具
    tool_call = ToolCall(
        id="tool_call_1",
        name="add_tool",
        arguments={"a": 1, "b": 2}
    )

    tool_request = ModelResponse(
        message=Message(
            role="assistant",
            content="我来计算",
            tool_calls=[tool_call]
        )
    )

    tool_result = Message(
        role="tool",
        content=json.dumps({"result": 3}),
        tool_call_id=tool_call.id
    )

    final_response = ModelResponse(
        message=Message(
            role="assistant",
            content="1+2=3",
        )
    )

    messages = [
        Message(role="system", content="你是一个计算器助手。"),
        Message(role="user", content="请帮我计算 1 + 2。"),
        tool_request.message,
        tool_result,
        final_response.message,
    ]

    print("\n模拟的消息流:")
    print(
        json.dumps(
            [asdict(message) for message in messages],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n第一阶段演示完成：配置和消息结构均已创建。")


if __name__ == "__main__":
    main()
