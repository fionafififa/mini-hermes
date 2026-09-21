# Mini Hermes

一个用于学习大模型 Agent 开发的 Python 项目。

学习参考：[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)。

## 当前进度

- [x] 定义消息、工具调用和模型响应
- [x] 读取和校验配置
- [ ] 接入第一个模型 provider
- [ ] 实现工具调用循环
- [ ] 保存与恢复会话
- [ ] 接入第二个 provider
- [ ] 增加长期记忆
- [ ] 增加上下文压缩
- [ ] 增加工具重试

## 环境

- Python 3.11+
- 开发环境使用 Python 3.13
- 当前阶段无第三方依赖

## Windows PowerShell 运行步骤

在项目根目录执行：

```powershell
uv venv --python 3.13
Copy-Item config.example.toml config.toml
.\.venv\Scripts\python.exe -m mini_hermes.main
```

当前程序只展示配置和消息结构，不会调用模型或执行工具。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 配置

- `config.example.toml`：可提交的配置模板。
- `config.toml`：本地配置，不提交。
- API Key 通过配置指定的环境变量读取，不写入配置文件。