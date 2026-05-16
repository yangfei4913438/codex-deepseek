# codex-deepseek

[ccswitch-deepseek](https://github.com/liuzhengming/ccswitch-deepseek) 的 Python 移植版 — 一个协议翻译代理，将 OpenAI Responses API 转换为 Chat Completions API，配合 [cc-switch](https://github.com/farion1231/cc-switch) 让 Codex 可以使用 DeepSeek 或任何 OpenAI 兼容模型。

零外部依赖 — 仅使用 Python 标准库。

感谢原版 [ccswitch-deepseek](https://github.com/liuzhengming/ccswitch-deepseek) 项目的设计与协议研究。

## 快速开始

### 1. 配置环境

```bash
cp .env.example .env
```

编辑 `.env`，填入 API key 和可选配置（见[配置项](#配置项)）。

### 2. 启动

```bash
./start.sh
```

或使用 uv 直接运行：

```bash
uv run python -m src.main
```

代理监听 `http://127.0.0.1:11435`。

## 配置项

| 变量 | 默认值 | 说明 |
|----------|---------|-------------|
| `api_key` | — | API 密钥（必填） |
| `base_url` | `https://api.deepseek.com` | API 基础地址 |
| `model` | `deepseek-v4-pro` | 模型名称 |
| `port` | `11435` | 服务监听端口 |
| `is_deepseek` | `true` | 设为 `false` 如果不是 DeepSeek 模型 |
| `multimodal` | `false` | 设为 `true` 如果模型支持图片输入 |

## 支持的模型提供商

本代理支持任何提供 **OpenAI 兼容 Chat Completions API** 的模型服务。只需配置对应的 `base_url`、`model` 和 `api_key` 即可。

> **提示：** 部分第三方模型也兼容 DeepSeek 格式的 `thinking` 参数。如果模型支持 `thinking: {type: "enabled"}`，可以保持 `is_deepseek=true`。

## 工作原理

Codex 使用 **OpenAI Responses API** 协议，大多数 AI 模型提供商只提供 **Chat Completions API**。
本代理在两者之间做实时的协议转换。

### 请求链路

```
Codex (app/cli) ──▶  cc-switch  ──▶  代理 :11435  ──▶  上游 API
```

1. Codex 发送请求到 cc-switch（配置的 provider 端点）
2. cc-switch 将请求路由到本代理的 `/responses`
3. 代理将 Responses API 的 `input` 列表翻译为 Chat Completions 的 `messages` 数组
4. 翻译后的请求转发到 `{base_url}/chat/completions`
5. 上游 API 返回的 SSE 流式响应被翻译回 Responses API 事件并返回

### 协议翻译覆盖

**输入方向（Responses → Chat Completions）**

| 源格式 | 目标格式 |
|--------|--------|
| `input_text` / `output_text` / `reasoning_text` | 消息文本内容 |
| `function_call` 条目 | assistant `tool_calls` |
| `function_call_output` 条目 | `tool` 角色消息 |
| `reasoning` 条目 | 跳过；`reasoning_content` 保留到相邻消息 |
| `developer` 角色 | `system` 角色 |
| `input_image` / `input_file` / `input_audio` | 跳过并统计 |
| `instructions` | 前置 system 消息 |
| `temperature` / `top_p` / `max_output_tokens` | 透传 |
| `tools` / `tool_choice` | 翻译为 Chat Completions 格式 |
| `thinking` / `reasoning` | 思考模式控制（DeepSeek 格式） |

**输出方向（Chat Completions SSE → Responses SSE）**

| Chat Completions SSE 事件 | Responses API 事件 |
|--------------------|---------------------|
| 首个 delta | `response.created` + `response.in_progress` |
| `delta.content` | `response.output_text.delta` / `done` |
| `delta.reasoning_content` | `response.reasoning_text.delta` / `done` |
| `delta.tool_calls` | `response.function_call_arguments.delta` / `done` |
| 流结束 | `response.output_item.done` × N + `response.completed`（含 usage） |

### reasoning_content 自动恢复

在多轮对话中，DeepSeek 会在携带 tool_calls 的 assistant 消息上省略 `reasoning_content`。
代理会自动记住前一轮的推理内容并在下一轮恢复，确保推理链在函数调用过程中不会中断。

### 模型身份注入

每次请求前，代理会插入一条 system 消息告知模型它的真实身份，
防止 Codex 或其他工具注入冲突的身份声明。

## 配合 [cc-switch](https://github.com/farion1231/cc-switch) 使用

**cc-switch** 是一个跨平台 AI CLI 管理工具，负责 provider 配置和请求路由。
本代理作为独立服务运行；cc-switch 将 Codex 请求路由到它。

### 配置步骤

**1. 启动代理服务：**

```bash
./start.sh
```

**2. 在 cc-switch 中添加 Codex provider：**

cc-switch 会管理 Codex 的配置文件（`~/.codex/config.toml` 和 `~/.codex/auth.json`）。
添加 provider 时填写以下字段：

| 字段 | 值 |
|-------|-------|
| name | `codex-deepseek` |
| base_url | `http://127.0.0.1:11435` |
| wire_api | `responses` |
| requires_openai_auth | `true` |

生成的 `~/.codex/config.toml` 内容如下：

```toml
model_provider = "custom"
model = "deepseek-v4-pro"
model_reasoning_effort = "high"

[model_providers.custom]
name = "codex-deepseek"
base_url = "http://127.0.0.1:11435"
wire_api = "responses"
requires_openai_auth = true
```

> **注意：** Codex 要求 `~/.codex/auth.json` 中的 `OPENAI_API_KEY` 非空才能通过客户端校验，但实际的上游 API 认证由本代理的 `.env` 处理 — 所以 `auth.json` 中填写任意占位值即可。

**3. 重启终端**使配置生效。

## 文件结构

| 文件 | 说明 |
|------|-------------|
| `src/main.py` | HTTP 服务（标准库 `http.server`） |
| `src/log.py` | 彩色 ANSI 日志 |
| `src/translate.py` | 输入翻译（Responses → Chat） |
| `src/sse.py` | SSE 事件翻译（Chat → Responses） |
| `src/recover.py` | reasoning_content 自动记忆与恢复 |
| `tests/test_translate.py` | 28 个单元测试 |

## 脚本

```bash
./start.sh   # 启动代理服务
./test.sh    # 运行单元测试
```

## License

MIT
