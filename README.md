# Mini Hermes

参考 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 的大模型 Agent 学习项目，按阶段实现最小可运行版本。

## 进度

- [x] 阶段一：消息结构与配置校验
- [x] 阶段二：OpenAI 兼容 Provider，完成单次模型调用
- [x] 阶段三：工具注册、执行与 Agent 多轮调用循环
- [x] 阶段四：SQLite 会话持久化、恢复历史、整轮提交
- [x] 阶段五：Provider 工厂、Anthropic 协议适配、恢复会话后切换 Provider
- [ ] 后续：长期记忆、上下文压缩、工具重试

## 第五阶段的主流程

```text
stage5.py：读取配置、选择 Provider、创建或恢复会话
    ↓
Agent.run_turn：user → Provider.generate → assistant
                         ↑                    ↓ 有工具调用
                         └── tool 结果 ← ToolRegistry.execute
    ↓ 无工具调用，得到最终回答
SessionStore：先提交本轮到 SQLite，再更新 Agent 的内存历史
```

`Agent` 只使用统一的 `Message`、`ToolCall` 和 `ModelResponse`，不判断服务商名称。
协议差异由 Provider 处理，因此切换 Provider 不需要改工具循环或重写数据库里的历史。

关键文件：

- `mini_hermes/stage5.py`：命令行入口，处理 `--config`、`--db`、`--session`、`--list`。
- `mini_hermes/providers/factory.py`：按 `config.provider` 创建适配器。
- `mini_hermes/providers/openai_compatible.py`：转换 OpenAI 兼容协议的消息和工具调用。
- `mini_hermes/providers/anthropic.py`：转换 Anthropic 协议的消息、工具 schema 和响应。
- `mini_hermes/agent.py`：模型调用、工具执行、继续调用模型的主循环。
- `mini_hermes/session_store.py`：保存、列出和恢复会话。

Anthropic 的消息体不使用 `role="tool"`：模型请求工具时返回 assistant 的 `tool_use` 块；
工具执行结果则变成 user 消息中的 `tool_result` 块，通过 `tool_use_id` 关联原调用。
同批多个工具结果合并进同一条 user 消息。转换只生成 API 请求数据，不改动内部历史。
开头的 system 消息移到请求顶层 `system` 字段。

`provider="anthropic"` 表示使用 Anthropic **协议**，不代表必须调用 Anthropic **公司**的服务。
本项目的示例用这套协议访问 DeepSeek，仍然使用 DeepSeek 的 Key。

## 运行（Windows PowerShell）

以下命令都在项目根目录执行。需要 Python 3.11+ 和 uv。

### 1. 安装依赖并创建本地配置

```powershell
uv sync --locked

if (-not (Test-Path -LiteralPath config.toml)) {
    Copy-Item -LiteralPath config.example.toml -Destination config.toml
}

if (-not (Test-Path -LiteralPath config.anthropic.toml)) {
    Copy-Item -LiteralPath config.anthropic.example.toml -Destination config.anthropic.toml
}
```

这些命令不会覆盖已有配置。`config.example.toml` 中的模型和地址是占位值，必须按实际服务修改。
如果使用 DeepSeek，`config.toml` 可配置为：

```toml
provider = "openai_compatible"
model = "deepseek-flash"
base_url = "https://api.deepseek.com"
api_key_env = "MINI_HERMES_API_KEY"
max_iterations = 8
max_output_tokens = 1024

[extra_body.thinking]
type = "disabled"
```

`config.anthropic.example.toml` 已提供 DeepSeek 的 Anthropic 协议配置：

```toml
provider = "anthropic"
model = "deepseek-flash"
base_url = "https://api.deepseek.com/anthropic"
api_key_env = "ANTHROPIC_API_KEY"
max_iterations = 8
max_output_tokens = 1024

[extra_body.thinking]
type = "disabled"
```

`extra_body` 由适配器交给 SDK，合并到实际 JSON 请求中；服务端不会收到名为 `extra_body` 的外层字段。
它用于服务特有参数，不能覆盖 `model`、`messages`、`tools`、`system`、`max_tokens` 或 `stream`。
不填写时默认为空对象，保留原配置的兼容性。

### 2. 设置 Key

下面两份配置都访问 DeepSeek，所以可共用同一把 Key。先单独执行第一行：

```powershell
$miniHermesSecret = Read-Host '请输入完整 DeepSeek API Key' -AsSecureString
```

出现输入提示后，仅粘贴完整 Key 并回车，不要加引号、`Bearer `，也不要粘贴网址或整条命令。
随后执行：

```powershell
$env:MINI_HERMES_API_KEY = ([System.Net.NetworkCredential]::new('', $miniHermesSecret).Password).Trim()
$env:ANTHROPIC_API_KEY = $env:MINI_HERMES_API_KEY
```

`ANTHROPIC_API_KEY` 只是本地变量名，这个 DeepSeek 示例中它保存的仍是 DeepSeek Key。
若改用 Anthropic 官方服务，必须同时换成该服务的模型、地址和对应 Key。
环境变量只对当前窗口及其子进程生效；新开 PowerShell 后需要重新设置。

### 3. 启动第五阶段

通过 OpenAI 兼容协议运行：

```powershell
.\.venv\Scripts\python.exe -m mini_hermes.stage5 --config config.toml
```

通过 Anthropic 协议运行：

```powershell
.\.venv\Scripts\python.exe -m mini_hermes.stage5 --config config.anthropic.toml
```

在程序的 `你：` 提示符中输入：

```text
请调用 add_numbers 计算 123 + 456，并根据工具返回值给出答案。
```

应能观察到 `add_numbers` 调用、包含 `"sum": 579` 的工具结果和最终回答。
模型自己决定是否调用工具，因此验证工具循环时，不要只检查最终文本，还要查看工具调用记录。
真实请求会产生 API 用量；`max_output_tokens` 限制单次请求输出，`max_iterations` 限制单个用户回合的请求次数。

程序内的命令：

- `/history`：显示当前会话历史。
- `/exit`：退出程序，返回 PowerShell。

PowerShell 命令必须在退出程序后执行，不能粘贴到 `你：` 提示符中。

### 4. 恢复会话并切换 Provider

默认数据库为 `data/sessions.db`，启动时会打印当前会话 ID。退出后可以列出会话：

```powershell
.\.venv\Scripts\python.exe -m mini_hermes.stage5 --list
```

把下方占位字符串替换为实际 ID，恢复会话并选择另一份配置：

```powershell
.\.venv\Scripts\python.exe -m mini_hermes.stage5 --config config.anthropic.toml --session '替换为实际会话ID'
```

恢复后可输入“请调用 add_numbers，把上一轮结果加 1”。若上一轮结果是 579，工具结果应为 580。
继续运行时可以再换回 `--config config.toml`，使用相同会话 ID。
可通过 `--db '其他数据库路径'` 指定独立数据库；恢复时必须使用保存该会话的同一数据库。

恢复使用数据库中最初保存的 system prompt，不会用当前代码中的新提示覆盖它。
本阶段支持文本和工具历史切换；包含无法解析的原始工具参数时，Anthropic 适配器会明确报错，
应继续使用原 Provider 或新建会话。

早期入口仍保留：`stage2` 单次调用，`stage3` 进程内工具循环，`stage4` OpenAI 兼容协议的会话持久化。

## 当前边界与排错

- 第五阶段尚不支持思考模式。DeepSeek 默认开启思考，示例通过 `thinking.type="disabled"` 显式关闭。
  未关闭时 Anthropic 响应可能包含 `thinking` 块，当前适配器会报“未知类型：thinking”。
  不能只丢弃思考信息后声称完整支持：思考模式下的工具历史还可能要求回传相应信息。
  参考 [DeepSeek 思考模式文档](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)。
- Provider 异常不会提交本轮消息；Anthropic 输出达到 `max_output_tokens` 时也不会提交或执行其中的工具调用。
  可以增大输出上限后重试。会话事务只能保护消息记录，不能撤销已经发生的外部工具副作用。
- `read_file` 只能读取当前项目目录内的 UTF-8 文件，最多返回前 12000 个字符。
- `AuthenticationError` 或空正文 400 不能靠猜测参数定位。先核对当前配置的 `api_key_env`，
  再查询模型列表，最后验证最小消息请求；两个变量“都有值”不代表其中内容相同。
  API 地址与 Key 的服务来源必须匹配。
- 若保留了本地诊断脚本，`check_deepseek_access.py` 只查询模型列表；
  `check_deepseek_protocols.py --infer` 仅在模型列表成功后发起最多两次短生成请求，会产生 API 用量。
  诊断脚本不是运行第五阶段的必要依赖。

DeepSeek 的 Anthropic 协议地址为 `https://api.deepseek.com/anthropic`，
SDK 自动请求 `/v1/messages`，不要在配置地址上重复追加该路径。
参考 [DeepSeek Anthropic API 文档](https://api-docs.deepseek.com/zh-cn/guides/anthropic_api/)。

## 测试

### 离线回归

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

2026-10-07 验证：14 个测试通过，不需要真实 API Key，也不调用收费模型服务。
覆盖消息和配置校验、错误工具参数、工具结果回传、历史保留、SQLite 恢复与失败时整轮不提交。
第五阶段还通过真实 SDK 访问本地 HTTP 服务，验证两种协议的转换、跨协议恢复、额外请求参数透传、
输出上限透传和截断响应处理；不只是替换 Provider 返回值的 mock 测试。

### 真实接口验证

2026-10-07 使用 DeepSeek `deepseek-flash`、真实 SDK 和 `stage5` CLI 实测：

- Anthropic 协议：`add_numbers(123, 456)` 返回 579，模型据此回答，消息成功写入 SQLite。
- 同一临时 SQLite 会话中，OpenAI 兼容 → Anthropic → OpenAI 兼容依次得到 579、580、581。
  每次恢复后都实际调用工具，旧历史和最初的 system prompt 保持不变。

实测使用临时数据库，没有改动已有 `data/sessions.db`。真实服务可用性仍需按运行时结果判断，
离线测试通过不等于远程服务始终可用。

## 提交与密钥安全

`config.toml`、`config.anthropic.toml`、运行数据库及 API Key 不提交到仓库。
提交无密钥的 `config.example.toml` 和 `config.anthropic.example.toml`；依赖记录在 `pyproject.toml` 与 `uv.lock`。
本地诊断脚本可以保留用于排错，提交前单独审查，不要直接用 `git add .` 将所有临时文件加入提交。
如果 Key 已出现在聊天、截图、日志或 Git 中，应在服务商平台撤销并更换；删除显示内容不能撤销 Key。

第五阶段可使用的提交说明：

```text
feat(stage5): 支持双协议 Provider 切换与 DeepSeek 非思考模式

- 新增 Provider 工厂和 Anthropic 消息、工具及响应适配
- 保留统一 Agent 循环与 SQLite 历史，支持恢复后跨协议切换
- 增加 extra_body 配置透传和输出上限，示例显式关闭思考模式
- 更新配置模板与 README，14 个离线测试及真实 CLI 工具链路通过
```
