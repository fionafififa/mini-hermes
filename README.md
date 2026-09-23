# Mini Hermes

一个用于学习大模型 Agent 开发的 Python 项目。

学习参考：[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)。

## 当前进度

- [x] 定义消息、工具调用和模型响应
- [x] 读取和校验配置
- [x] 接入第一个模型 provider，跑通一次非流式文本调用
- [ ] 实现工具调用循环
- [ ] 保存与恢复会话
- [ ] 接入第二个 provider
- [ ] 增加长期记忆
- [ ] 增加上下文压缩
- [ ] 增加工具重试

## 环境

- Python 3.11+
- 开发环境使用 Python 3.13
- 使用 uv 管理依赖和虚拟环境
- 模型客户端依赖：`openai>=2.0,<3`，具体版本记录在 `uv.lock`

## 项目结构

```text
mini_hermes/
├── config.py                     # TOML 配置校验、从环境变量读取密钥
├── messages.py                   # Message、ToolCall、TokenUsage、ModelResponse
├── main.py                       # 第一阶段：模拟消息流，不调用模型
├── stage2.py                     # 第二阶段：输入问题，调用模型，打印响应
└── providers/
    ├── __init__.py
    ├── base.py                   # LLMProvider 抽象接口
    └── openai_compatible.py      # 消息转换、Chat Completions 请求和响应转换
```

## Windows PowerShell 运行步骤

### 1. 安装依赖并准备配置

在项目根目录执行。仅在本地配置不存在时复制模板，避免覆盖已有设置：

```powershell
uv sync --locked
if (-not (Test-Path -LiteralPath config.toml)) {
    Copy-Item -LiteralPath config.example.toml -Destination config.toml
}
```

编辑 `config.toml`，填写同一服务商对应的模型 ID 和 API 基础地址：

```toml
provider = "openai_compatible"
model = "填写实际模型ID"
base_url = "填写服务商提供的兼容API基础地址"
api_key_env = "MINI_HERMES_API_KEY"
max_iterations = 8
```

`base_url` 按服务商文档填写，通常包含 `/v1`；不要额外拼接
`/chat/completions`。`max_iterations` 为后续 agent 循环预留，本阶段尚未使用。

### 2. 设置密钥

在运行程序的同一个 PowerShell 窗口执行。以下输入方式兼容 Windows
PowerShell 5.1 和 PowerShell 7，输入时不显示密钥正文：

```powershell
$stage2Secret = Read-Host '请输入 API Key' -AsSecureString
$env:MINI_HERMES_API_KEY = [System.Net.NetworkCredential]::new('', $stage2Secret).Password
```

环境变量名必须与 `config.toml` 的 `api_key_env` 一致。关闭窗口后需重新设置。
当前配置模块读取进程环境变量，不会自动加载 `.env` 文件。
不要把真实密钥或包含密钥的终端截图、输出提交到仓库。

### 3. 运行第二阶段

```powershell
.\.venv\Scripts\python.exe -m mini_hermes.stage2
```

示例输入：

```text
用两句话解释 Python 的列表和元组有什么区别。
```

程序输出模型回答、服务端提供的 token 用量（如果有），以及内部
`ModelResponse` 的 JSON 展示，然后退出。
命令中的 `_` 是普通下划线，不要在它前面添加反斜杠。

第一阶段的模拟演示仍可单独运行：

```powershell
.\.venv\Scripts\python.exe -m mini_hermes.main
```

## 第二阶段调用流程

```text
stage2.main()
  → load_config() 读取 config.toml
  → input() 获取问题
  → 构造 system 和 user 两条 Message
  → OpenAICompatibleProvider.generate(messages)
      → to_api_message() 转换为 API 消息字典
      → get_api_key() 读取环境变量
      → SDK 发送 POST <base_url>/chat/completions
      → 等待完整响应，SDK 将响应解析为 completion 对象
      → 提取正文、工具调用、token 用量
      → 转换为 ModelResponse(message=Message(...), usage=TokenUsage(...))
  → 打印正文、用量和内部数据结构
```

### 请求发出后发生什么

`OpenAICompatibleProvider.__init__()` 只保存配置；真正触发模型请求的是
`generate()` 中的 `client.chat.completions.create(**request)`。

1. `to_api_message()` 把内部 dataclass 转成接口需要的字典。普通消息只带
   `role` 和 `content`，不附带空的工具字段。
2. SDK 将 `model`、`messages`、`stream` 等参数编码成 JSON 请求体，使用
   配置的服务地址，并把 API Key 放在认证请求头中。
3. 服务端根据模型 ID 和消息生成回答。`stream=False` 表示本地调用等待完整响应，
   然后继续执行后面的 Python 代码；客户端没有逐段打印生成内容。
4. SDK 将响应解析为对象。`completion.choices[0].message` 取出第一条候选回复；
   `completion.usage` 提供服务端报告的用量。
5. Provider 将 SDK 对象映射为项目自己的 `Message`、`ToolCall`、`TokenUsage`
   和 `ModelResponse`。`with OpenAI(...)` 结束时关闭客户端连接。
6. 入口通过 `response.message.content` 打印正文。`asdict(response)` 和
   `json.dumps(...)` 只负责展示内部结构，不会再调用模型。

本阶段设置了 `max_retries=0`，SDK 不自动重试失败请求。
网络、认证和无效工具参数等错误目前直接抛出，尚未实现应用层恢复。

### 为什么 tool_calls 是空列表

入口调用的是：

```python
response = provider.generate(messages)
```

没有传入可用工具说明，因此 `tools` 使用默认值 `None`，下面的分支不会执行：

```python
if tools:
    request["tools"] = tools
```

本次请求没有向模型声明可调用工具，正常返回的是文本答案。Provider 先创建
`tool_calls = []`，再遍历 `raw_message.tool_calls or []`；没有返回调用时，
循环执行零次，内部列表保持为空。

定义 `ToolCall` 类只是建立数据结构；声明可用工具需要发送名称、用途和参数 schema，
执行工具还需要本地 handler 和调用循环。这些将在第三阶段实现。
即使以后提供了工具，模型也可能直接回答；空列表本身不是故障。

`tool_call_id: null` 是内部 `Message` 的默认字段值。当前消息角色是 `assistant`；
这个字段在后续 `role="tool"` 的结果消息中，用来关联对应的工具调用。
它出现在 `asdict()` 的输出中，不代表本次请求把这个空字段发给了模型。

### 内部结构与 API 结构的对应

| API 返回字段 | 项目内部字段 |
|---|---|
| `choices[0].message.content` | `ModelResponse.message.content` |
| `choices[0].message.tool_calls` | `ModelResponse.message.tool_calls` |
| 工具调用的 `function.arguments`（JSON 字符串） | `ToolCall.arguments`（字典） |
| `usage.prompt_tokens` | `TokenUsage.input_tokens` |
| `usage.completion_tokens` | `TokenUsage.output_tokens` |

`TokenUsage.total_tokens` 是输入与输出之和的计算属性，不是 dataclass 字段，
所以会单独打印，但不会出现在 `asdict(response)` 导出的 `usage` 中。
Token 是模型的文本计量单位，不等同于字数；当前代码展示的是服务端报告值，
并未在本地重新计算用量。服务端未提供用量时，`usage` 保持 `None`。

## 第二阶段验收与范围

- 手动验收：开发者于 2026-09-23 提供的真实调用输出显示，程序成功回答了
  “Python 的列表和元组有什么区别”，并返回输入 56、输出 158、合计 214 token。
  这是一次运行记录，不是固定的预期输出或测试断言。
- 已完成：单次非流式文本请求、内部消息与 API 消息转换、工具调用信息解析和用量映射。
- 当前仍是一次问答；没有工具注册/执行循环、跨次对话历史、持久化、长期记忆、
  上下文压缩或工具重试。工具调用解析分支尚未通过这次纯文本调用验证。
- 下一阶段：声明工具 schema，执行模型返回的调用，追加对应的 tool 结果，
  再次请求模型形成完整的 agent 循环。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

现有 2 个基础测试覆盖消息关联与角色约束、配置加载与校验、环境变量密钥读取，
均已通过。这些测试不访问真实模型，也尚未覆盖 provider 的协议转换与请求行为。

## 配置

- `config.example.toml`：可提交的配置模板。
- `config.toml`：本地配置，不提交。
- API Key 通过配置指定的环境变量读取，不写入配置文件。
- `pyproject.toml` 与 `uv.lock` 一起提交，便于重现依赖环境。
