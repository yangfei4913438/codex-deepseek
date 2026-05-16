# codex-deepseek

[中文](README.zh-CN.md)

Python port of [ccswitch-deepseek](https://github.com/liuzhengming/ccswitch-deepseek) — a protocol translation proxy that converts OpenAI Responses API to Chat Completions API, enabling Codex to use DeepSeek or any OpenAI-compatible model via [cc-switch](https://github.com/farion1231/cc-switch).

Zero external dependencies — Python standard library only.

Thanks to the original [ccswitch-deepseek](https://github.com/liuzhengming/ccswitch-deepseek) project for the design and protocol research.

## Quick Start

### 1. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API key and optional settings (see [Configuration](#configuration)).

### 2. Start

```bash
./start.sh
```

Or with uv directly:

```bash
uv run python -m src.main
```

The proxy listens at `http://127.0.0.1:11435`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `api_key` | — | API key (required) |
| `base_url` | `https://api.deepseek.com` | API base URL |
| `model` | `deepseek-v4-pro` | Model name |
| `port` | `11435` | Server listen port |
| `is_deepseek` | `true` | Set to `false` if not using a DeepSeek model |
| `multimodal` | `false` | Set to `true` if the model supports image inputs |

## Supported Providers

This proxy works with any model provider that offers an **OpenAI-compatible Chat Completions API**. Just configure `base_url`, `model`, and `api_key` accordingly.

> **Tip:** Some third-party model providers also support the DeepSeek-format `thinking` parameter. If the model recognizes `thinking: {type: "enabled"}`, you can keep `is_deepseek=true`.

## How It Works

Codex speaks **OpenAI Responses API**. Most AI model providers speak **Chat Completions API**.
This proxy translates between the two protocols in real time.

### Request chain

```
Codex (app or CLI) ──▶  cc-switch  ──▶  proxy :11435  ──▶  Upstream API
```

1. Codex sends a request to cc-switch (its configured provider endpoint)
2. cc-switch routes the request to this proxy at `/responses`
3. The proxy translates Responses API `input` items into Chat Completions `messages`
4. The translated request is forwarded to `{base_url}/chat/completions`
5. The upstream API's SSE streaming response is translated back into Responses API events and returned

### Translation coverage

**Input (Responses → Chat Completions)**

| Source | Target |
|--------|--------|
| `input_text` / `output_text` / `reasoning_text` | message text content |
| `function_call` item | assistant `tool_calls` |
| `function_call_output` item | `tool` role message |
| `reasoning` item | skipped; `reasoning_content` retained on adjacent message |
| `developer` role | `system` role |
| `input_image` / `input_file` / `input_audio` | skipped with stats |
| `instructions` | prepended system message |
| `temperature` / `top_p` / `max_output_tokens` | passthrough |
| `tools` / `tool_choice` | translated to Chat Completions format |
| `thinking` / `reasoning` | thinking mode control (DeepSeek format) |

**Output (Chat Completions SSE → Responses SSE)**

| Chat Completions SSE event | Responses API event |
|--------------------|---------------------|
| first delta | `response.created` + `response.in_progress` |
| `delta.content` | `response.output_text.delta` / `done` |
| `delta.reasoning_content` | `response.reasoning_text.delta` / `done` |
| `delta.tool_calls` | `response.function_call_arguments.delta` / `done` |
| stream end | `response.output_item.done` × N + `response.completed` (with usage) |

### reasoning_content recovery

DeepSeek omits `reasoning_content` on tool-call assistant messages in multi-turn conversations.
The proxy automatically remembers reasoning from the previous turn and restores it on the next,
so the reasoning chain stays intact across function calls.

### Identity injection

A system message is prepended to every request telling the model its true identity,
preventing conflicting identity claims from Codex or other tools.

## Integration with [cc-switch](https://github.com/farion1231/cc-switch)

**cc-switch** is a cross-platform AI CLI management tool that handles provider configuration and request routing.
This proxy is an independent service; cc-switch routes Codex requests to it.

### Setup

**1. Start the proxy:**

```bash
./start.sh
```

**2. Add a new Codex provider in cc-switch:**

cc-switch manages Codex's config files (`~/.codex/config.toml` and `~/.codex/auth.json`).
Fill in these fields when adding the provider:

| Field | Value |
|-------|-------|
| name | `codex-deepseek` |
| base_url | `http://127.0.0.1:11435` |
| wire_api | `responses` |
| requires_openai_auth | `true` |

The resulting `~/.codex/config.toml` will look like:

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

> **Note:** Codex requires a non-empty `OPENAI_API_KEY` in `~/.codex/auth.json` to pass its client-side check, but the actual upstream authentication is handled by this proxy's own `.env` — so any placeholder value works in `auth.json`.

**3. Restart your terminal** for changes to take effect.

## Files

| File | Description |
|------|-------------|
| `src/main.py` | HTTP server (stdlib `http.server`) |
| `src/log.py` | Colored ANSI logging |
| `src/translate.py` | Input translation (Responses → Chat) |
| `src/sse.py` | SSE event translation (Chat → Responses) |
| `src/recover.py` | reasoning_content auto-restore |
| `tests/test_translate.py` | 28 unit tests |

## Scripts

```bash
./start.sh   # Start the proxy server
./test.sh    # Run unit tests
```

## License

MIT
