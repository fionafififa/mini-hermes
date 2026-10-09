# Mini Hermes

参考 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 的大模型 Agent 学习项目，按阶段实现最小可运行版本。

## 进度

- [x] 阶段一：消息结构与配置校验
- [x] 阶段二：OpenAI 兼容 Provider，完成单次模型调用
- [x] 阶段三：工具注册、执行与 Agent 多轮调用循环
- [x] 阶段四：SQLite 会话持久化、恢复历史、整轮提交
- [x] 阶段五：Provider 工厂、Anthropic 协议适配、恢复会话后切换 Provider
- [x] 阶段六 P0：显式保存、更新和删除长期记忆；跨会话关键词召回与长度限制
- [x] 阶段六 P1：用户／项目记忆分区、来源与 UTC 时间、范围内更新、召回诊断、P0 数据迁移
- [ ] 阶段七：上下文压缩；结合压缩流程接入压缩前记忆保存检查点
- [ ] 阶段八：工具重试与失败处理
- [ ] 记忆后续：自动提炼、语义检索、版本历史与会话归属校验

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

## 第六阶段的记忆流程（P0 + P1）

```text
/remember 或 /remember-project → MemoryStore → data/memory.db
新问题 → 当前用户／项目过滤 → 项目同名记忆覆盖用户记忆
       → 关键词评分与排序 → 条数／字符预算 → 召回诊断日志（可选）
       → 与本轮问题组成一条 user 消息 → 原有 Agent / Provider / 工具循环
       → SessionStore 保存本轮实际使用的消息快照
```

`data/sessions.db` 保存每个会话的消息历史；`data/memory.db` 保存跨会话使用的记忆。
记忆命令由 CLI 直接执行，不调用模型。只有普通问题才会按关键词召回相关记忆。

### 记忆归属与更新

P1 的有效记录保存在 `memory_records` 表，主键为 `(user_id, project_id, key)`：

| 范围 | project_id | 使用规则 |
|---|---|---|
| 用户记忆 | 空字符串 | 当前用户跨项目可用 |
| 项目记忆 | 指定项目标识 | 仅当前用户、指定项目可用；不是团队共享记忆 |

召回只能读取当前用户的用户记忆和当前项目记忆。同名项目记录先覆盖用户记录，再计算关键词匹配；
即使项目记录未命中关键词，也不会回退召回被覆盖的用户记录。删除项目覆盖后，用户同名记录重新生效。

同一用户、同一范围、同一 `key` 再次保存会更新原记录，保留 `created_at`，更新 `updated_at`。
两者使用带时区的 UTC 时间。`source` 和 `source_session_id` 记录最近一次写入的来源和所在会话，
例如 `cli:/remember-project`；它们不是完整版本历史，也不保证会话历史中保存了原始记忆命令。

### 召回与历史快照

关键词按完整字符串在问题中进行不区分大小写的子串匹配，不做分词、同义词或语义检索。
按命中关键词数排序，同分时项目记录优先，再按 `key` 排序；默认最多 3 条，整个记忆参考块最多 1000 个字符。
字符预算包含包装文字和换行，超出预算的记录整条跳过。它不是 token 预算，也不控制累积的会话历史长度。

每轮仅召回一次，工具循环复用本轮记忆快照。更新或删除记忆只影响后续召回，
不会重写旧 user 消息或 system prompt；历史中已经保存的记忆仍然存在。

启用 `--memory-log` 后，每轮输出一条 `memory_recall` 诊断：

| 字段 | 含义 |
|---|---|
| `visible_count` | 归属过滤后可见的记录数 |
| `effective_count` | 同名覆盖后的记录数 |
| `candidate_count` | 关键词匹配上的记录数 |
| `selected` | 真正注入请求的记录范围、key、分数和命中关键词 |
| `skipped` | 被跳过的候选，原因是 `char_budget` 或 `item_limit` |
| `chars` | 最终记忆参考块字符数；无召回时为 0 |

匹配成功不等于最终注入。日志不包含完整原始问题和记忆值，但包含用户标识、项目路径和关键词，
仍可能涉及隐私；目前仅输出到终端，没有专用日志文件。

### P0 数据迁移

首次使用 P1 打开 P0 数据库，会在一个事务内创建新表、复制旧记录并设置 `PRAGMA user_version = 1`。
旧记录固定归属 `user_id="default"`、`project_id=""`，来源为 `legacy_p0`，不随首次启动的 `--user` 改变。
P0 未保存原始时间，因此迁移记录的创建／更新时间仅代表迁移时间。

旧 `memories` 表保留为迁移快照，不再参与 P1 读写；后续启动不重复导入，删除的新记录不会被旧表复活。
升级前请退出程序并备份记忆数据库；升级后继续使用 P1，不要用旧 P0 代码写入旧表并期待自动同步。
`/forget` 删除的是新表有效记录，不会清除旧表、数据库备份或会话历史，不是隐私彻底擦除。

关键文件：`mini_hermes/memory.py` 负责 SQLite 记忆读写和召回，
`mini_hermes/agent.py` 在每轮模型调用前召回一次，`mini_hermes/stage6.py` 提供记忆命令和启动入口。

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

### 5. 使用第六阶段长期记忆（P0 + P1）

沿用上一节的配置和 API Key，启动阶段六并开启召回诊断：

```powershell
.\.venv\Scripts\python.exe -m mini_hermes.stage6 --config config.toml --user default --project . --memory-log
```

也可以把配置换成 `config.anthropic.toml`。默认会话数据库是 `data/sessions.db`，
记忆数据库是 `data/memory.db`；可分别通过 `--db` 和 `--memory-db` 指定其他路径。
`--user` 默认是 `default`；`--project` 默认是当前目录，经绝对路径解析和系统大小写规范化后作为项目标识。
它只选择记忆分区，不改变工具工作目录；省略 `--memory-log` 则不显示召回诊断。
`--list` 列出会话，`--session '会话ID'` 恢复指定会话；恢复时必须保持原用户和项目不变。

**切换用户或项目请退出后新建会话，不带 `--session`。** 当前会话库没有归属校验，
这些参数仅用于本地记忆分区，不是登录认证、权限系统或完整的多用户会话隔离。

在程序的 `你：` 提示符中输入：

```text
/remember example_language=我偏好 Python 代码示例 | 代码,编程,示例
/memories
```

`/memories` 应显示 `[user] example_language=我偏好 Python 代码示例 | 代码,编程,示例`，
以及来源、来源会话、创建时间和更新时间。它列出当前可见的用户／项目记录，包括被同名覆盖的用户记录。
命令格式中的 `=`、`|`、`,` 使用英文半角符号，关键词直接写成逗号分隔的文本，无需加列表括号或引号。

输入 `/exit`，使用相同用户和项目重新运行阶段六且不加 `--session`，再问“给我一个简单的代码示例。”。
新会话虽然没有旧聊天历史，仍能从 `memory.db` 召回这条记忆；输入 `/history`，
查看本轮 user 消息中是否包含记忆参考块。应以请求内容为验收依据，不只看模型是否恰好回答了 Python。

保存项目同名覆盖，再更新这条项目记忆：

```text
/remember-project example_language=这个项目使用 TypeScript 代码示例 | 代码,编程,示例
/memories
给我一个简单的代码示例。
/remember-project example_language=这个项目改用 Go 代码示例 | 代码,编程,示例
/memories
再给我一个代码示例。
```

应观察到第一次召回使用 TypeScript，更新后下一轮使用 Go；项目记录仍只有一条，创建时间不变、更新时间刷新。
用户范围的 Python 记录不变，过去已保存的 TypeScript 请求快照也不变。
`selected` 应包含 `scope="project"` 的 `example_language`，实际请求内容通过 `/history` 查看。

记忆命令及删除范围：

| 命令 | 操作范围 |
|---|---|
| `/remember key=value \| 关键词1,关键词2` | 保存／更新当前用户的跨项目记忆 |
| `/remember-project key=value \| 关键词1,关键词2` | 保存／更新当前用户、当前项目的记忆 |
| `/memories` | 查看当前用户和项目可见的记录及元数据 |
| `/forget key` | 仅删除用户范围；不会误删项目同名记录 |
| `/forget-project key` | 仅删除当前项目范围；用户同名记录重新生效 |

测试删除效果时请新建会话，因为旧历史中的记忆快照不会被删除。
验证用户分区时，先 `/exit`，再在 PowerShell 中启动新会话：

```powershell
.\.venv\Scripts\python.exe -m mini_hermes.stage6 --config config.toml --user bob --project . --memory-log
```

Bob 的 `/memories` 不应出现 `default` 用户的条目。测试项目分区可以保持 `--user default`，
将 `--project` 换成另一个固定路径；新项目仍能使用用户范围记忆，但不能使用原项目范围的记录。

## 当前边界与排错

- 当前适配器尚不支持思考模式。DeepSeek 默认开启思考，示例通过 `thinking.type="disabled"` 显式关闭。
  未关闭时 Anthropic 响应可能包含 `thinking` 块，当前适配器会报“未知类型：thinking”。
  不能只丢弃思考信息后声称完整支持：思考模式下的工具历史还可能要求回传相应信息。
  参考 [DeepSeek 思考模式文档](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)。
- Provider 异常不会提交本轮消息；Anthropic 输出达到 `max_output_tokens` 时也不会提交或执行其中的工具调用。
  可以增大输出上限后重试。会话事务只能保护消息记录，不能撤销已经发生的外部工具副作用。
- `read_file` 只能读取当前项目目录内的 UTF-8 文件，最多返回前 12000 个字符。
- 第六阶段的关键词按完整字符串匹配，不做同义词或语义匹配；未命中时直接发送原问题。
  如果 `/memories` 把关键词显示成逐字逗号分隔，请检查读取数据库时是否用 `json.loads(keywords)` 还原列表。
- P1 已隔离记忆记录，但尚未给 `SessionStore` 增加用户／项目归属字段，`--list` 也不是按用户筛选的会话列表。
  切换分区必须新建会话；恢复历史只能沿用原用户和项目。项目路径变化也会改变项目记忆分区标识。
- 记忆来源仅记录最近一次写入，不包含完整版本历史；自动提炼、语义检索和压缩前保存检查点尚未实现。
  `/forget` 不清除历史消息、P0 迁移快照或备份；终端召回日志也可能含隐私。
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

2026-10-09 验证：16 个测试通过，不需要真实 API Key，也不调用收费模型服务。
覆盖消息和配置校验、错误工具参数、工具结果回传、历史保留、SQLite 恢复与失败时整轮不提交。
第五阶段还通过真实 SDK 访问本地 HTTP 服务，验证两种协议的转换、跨协议恢复、额外请求参数透传、
输出上限透传和截断响应处理；不只是替换 Provider 返回值的 mock 测试。
第六阶段另验证了关键词从 SQLite 恢复为列表、完整关键词匹配、记忆进入模型请求，
以及有无召回时的消息保存和会话恢复。
P1 新增 `tests/test_stage6_p1.py`，验证真实旧表迁移的固定归属、删除后重启不复活，
以及同数据库用户／项目切换、项目同名覆盖、范围内更新与删除、来源与时间、完整字符预算和召回日志。
通过实际 Agent 调用验证更新记忆后只改变后续 user 消息，已提交前缀、system prompt 和恢复后的历史保持不变。
这些是离线回归结果，不代表本轮重新验证了远程模型服务。

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
- 更新配置模板与 README，离线测试及真实 CLI 工具链路通过
```

第六阶段变更的简短英文提交说明：

```text
feat(stage6): add persistent keyword-based memory and usage docs
```

第六阶段 P1 变更可使用：

```text
feat(memory): add scoped provenance, migration and recall diagnostics

- Scope memory records by user, project and key; project records override user defaults
- Track latest source/session and UTC timestamps while preserving creation time on updates
- Migrate P0 records once and add whole-record recall diagnostics
- Cover scoped memory behavior and immutable conversation history with offline tests
```
