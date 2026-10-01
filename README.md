# Mini Hermes

参考 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 的大模型 Agent 学习项目，按阶段实现最小可运行版本。

## 进度

- [x] 阶段一：消息结构与配置校验
- [x] 阶段二：OpenAI 兼容 Provider，完成单次模型调用
- [x] 阶段三：工具注册、执行与 Agent 多轮调用循环
- [ ] 后续：会话持久化、第二个 Provider、长期记忆、上下文压缩、工具重试

## 阶段三做了什么

`Agent` 将工具 schema 交给模型，执行模型请求的 `add_numbers` 或 `read_file`，再把带有 `tool_call_id` 的工具结果发回模型，直到得到最终回答或达到 `max_iterations` 上限。同一进程中的后续提问可使用已有对话历史；退出后暂不保存。

工具参数解析失败时，Provider 保留原始参数和错误信息，由工具注册表返回错误结果供模型处理。`read_file` 仅能读取当前项目目录内的 UTF-8 文件，最多返回前 12000 个字符。

## 运行（Windows PowerShell）

在项目根目录执行，首次运行时安装依赖并从模板创建本地配置：

```powershell
uv sync --locked
if (-not (Test-Path -LiteralPath config.toml)) {
    Copy-Item -LiteralPath config.example.toml -Destination config.toml
}
```

在 `config.toml` 中填写实际 `model` 和 `base_url`。每次打开新的 PowerShell 窗口，都要重新设置 API Key 环境变量（名称须与 `api_key_env` 一致）：

```powershell
$miniHermesSecret = Read-Host '请输入 API Key' -AsSecureString
$env:MINI_HERMES_API_KEY = [System.Net.NetworkCredential]::new('', $miniHermesSecret).Password
.\.venv\Scripts\python.exe -m mini_hermes.stage3
```

例如输入“请调用 add_numbers 计算 19 和 23 的和，再根据工具结果回答”，可观察工具调用、工具结果和最终回复；输入 `/exit` 退出。第二阶段的单次调用仍可通过 `.\.venv\Scripts\python.exe -m mini_hermes.stage2` 运行。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前 5 个离线测试通过，覆盖基础消息/配置、工具结果回传、对话历史和错误参数处理。测试使用预设的模型返回值，不需要 API Key；真实模型及 SDK 的端到端调用需用上面的交互命令另行验证。

`config.toml` 和 API Key 不提交到仓库；依赖版本记录在 `pyproject.toml` 与 `uv.lock` 中。
