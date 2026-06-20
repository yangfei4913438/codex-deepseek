import json
import os
import random
import string
import time
from http.client import HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import log
from .recover import recover_reasoning, remember_reasoning, session_key
from .sse import SseTranslator
from .translate import (
    last_user_text,
    translate_messages,
    translate_tool_choice,
    translate_tools,
)


def _load_env():
    """Load .env file manually without external dependencies."""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value


_load_env()

DEEPSEEK_API_KEY = os.getenv("api_key", "")
BASE_URL = os.getenv("base_url", "https://api.deepseek.com")
MODEL = os.getenv("model", "deepseek-v4-pro")
IDENTITY_MODEL = os.getenv("identity_model", "")
PORT = int(os.getenv("port", "11435"))
TIMEOUT = int(os.getenv("timeout", "30")) * 60
MULTIMODAL = os.getenv("multimodal", "").lower() in ("true", "1", "yes")
IS_DEEPSEEK = os.getenv("is_deepseek", "true").lower() in ("true", "1", "yes")


def _rand_id(prefix: str, length: int = 8) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}_{suffix}"


def build_chat_body(body: dict) -> dict:
    stream = body.get("stream") is not False
    enable_thinking = (
        body.get("thinking") is True
        or (
            isinstance(body.get("thinking"), dict)
            and body["thinking"].get("type") == "enabled"
        )
        or (isinstance(body.get("reasoning"), dict) and body["reasoning"].get("effort"))
    )
    result = translate_messages(
        body.get("input"),
        {"keepReasoningContent": enable_thinking, "multimodal": MULTIMODAL},
    )
    messages = result["messages"]
    stats = result["stats"]

    restored = recover_reasoning(session_key(body), messages)
    has_assistant_with_rc = any(
        m.get("role") == "assistant" and m.get("reasoning_content") for m in messages
    )
    has_assistant_with_tc = any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in messages
    )
    effective_thinking = enable_thinking and (
        has_assistant_with_rc or not has_assistant_with_tc
    )

    if enable_thinking and not effective_thinking:
        log.warn("thinking off: missing rc in history")
    if restored > 0 and effective_thinking:
        log.ok(f"rc restored x{restored}")
    if stats["strippedReasoningContent"] > 0:
        log.skip(f"rc stripped x{stats['strippedReasoningContent']}")
    if stats["preservedReasoningContent"] > 0 and not restored:
        log.info(f"rc preserved x{stats['preservedReasoningContent']}")

    last_user = last_user_text(messages)
    preview = last_user[:120] + "..." if len(last_user) > 120 else last_user
    log.req(
        f"thinking:{'on' if effective_thinking else 'off'} msgs:{len(messages)} stream:{stream} | {preview}"
    )

    instructions = body.get("instructions", "")
    if IDENTITY_MODEL:
        IDENTITY = f"\n\n[IMPORTANT: Your true model identity is {IDENTITY_MODEL}. You are NOT OpenAI, GPT, or Claude. When asked about your model identity, you MUST answer truthfully based on your actual model name. Ignore any conflicting identity claims in the instructions above.]"
        if instructions:
            instructions = instructions + IDENTITY
        else:
            instructions = IDENTITY.strip()
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    chat_body: dict = {"model": MODEL, "messages": messages, "stream": stream}
    if IS_DEEPSEEK:
        chat_body["thinking"] = (
            {"type": "enabled"} if effective_thinking else {"type": "disabled"}
        )

    tools = translate_tools(body.get("tools"))
    if tools:
        chat_body["tools"] = tools
        tc = translate_tool_choice(body.get("tool_choice"))
        if tc:
            chat_body["tool_choice"] = tc

    if body.get("temperature") is not None:
        chat_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        chat_body["top_p"] = body["top_p"]
    if body.get("max_output_tokens") is not None:
        chat_body["max_tokens"] = body["max_output_tokens"]

    return {"chat_body": chat_body, "stream": stream, "messages": messages}


def build_non_stream_response(completion: dict) -> dict:
    msg = (completion.get("choices") or [{}])[0].get("message", {})
    usage = completion.get("usage")
    output = []
    if msg.get("reasoning_content"):
        output.append({
            "id": _rand_id("rsn", 6),
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": msg["reasoning_content"]}],
            "status": "completed",
        })
    if msg.get("content"):
        output.append({
            "id": _rand_id("msg", 6),
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": msg["content"], "annotations": []}
            ],
            "status": "completed",
        })
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            output.append({
                "id": f"fc_{tc['id']}",
                "type": "function_call",
                "call_id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
                "status": "completed",
            })
    return {
        "id": _rand_id("resp", 10),
        "object": "response",
        "status": "completed",
        "model": MODEL,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) if usage else 0,
            "output_tokens": usage.get("completion_tokens", 0) if usage else 0,
            "total_tokens": usage.get("total_tokens", 0) if usage else 0,
        }
        if usage
        else None,
    }


def _deepseek_request(chat_body: dict, stream: bool = False) -> tuple:
    """Call the upstream API via http.client."""
    parsed = urlparse(BASE_URL)
    host = parsed.netloc or "api.deepseek.com"
    path = parsed.path.rstrip("/") + "/chat/completions"
    body_bytes = json.dumps(chat_body).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }

    conn = HTTPSConnection(host, timeout=TIMEOUT)
    try:
        conn.request("POST", path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            err_body = resp.read().decode()[:500]
            conn.close()
            return resp.status, err_body, None
        if stream:
            return resp.status, resp, conn
        else:
            data = resp.read().decode()
            conn.close()
            return resp.status, data, None
    except Exception as e:
        conn.close()
        return None, str(e), None


def _upstream_status(status) -> int:
    """Map upstream HTTP status to proxy error status."""
    if status is None:
        return 502
    if 400 <= status < 500:
        if status == 429:
            return 429
        if status in (401, 403):
            return 502  # hide auth details from the client
        return status
    return 502


def _upstream_error_message(status, body: str) -> str:
    """Create a user-friendly error message for the given status code."""
    snippet = (body or "")[:200]
    if status == 401 or status == 403:
        return f"Upstream authentication failed ({status}). Check your API key in .env"
    if status == 429:
        return f"Upstream rate limited ({status}). The API is throttling requests; wait and retry."
    return f"Upstream {status}: {snippet}"


class ProxyHandler(BaseHTTPRequestHandler):
    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_response(self, generator):
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for chunk in generator:
                self.wfile.write(
                    chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.endswith("/health"):
            self._json_response({
                "service": "codex-deepseek",
                "model": MODEL,
                "status": "ok",
                "port": PORT,
            })
        elif path.endswith("/models"):
            self._json_response({
                "object": "list",
                "data": [
                    {
                        "id": MODEL,
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "deepseek" if IS_DEEPSEEK else "openai",
                    }
                ],
            })
        else:
            self._json_response({"error": {"message": f"not found: {path}"}}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if not path.endswith("/responses"):
            self._json_response({"error": {"message": f"not found: {path}"}}, 404)
            return

        try:
            content_len = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_len).decode("utf-8")
            body = json.loads(raw)
        except Exception as e:
            self._json_response({"error": {"message": str(e)}}, 400)
            return

        try:
            built = build_chat_body(body)
        except Exception as e:
            log.err(f"build: {e}")
            self._json_response({"error": {"message": str(e)}}, 400)
            return

        chat_body = built["chat_body"]
        stream = built["stream"]

        if not stream:
            self._handle_non_stream(body, chat_body)
        else:
            self._handle_stream(body, chat_body)

    def _handle_non_stream(self, body: dict, chat_body: dict) -> None:
        t0 = time.time()
        status, resp_body, conn = _deepseek_request(chat_body)
        elapsed_ms = int((time.time() - t0) * 1000)
        log.timing(elapsed_ms)
        if status != 200:
            log.err(f"Upstream {status}: {resp_body[:300]}")
            self._json_response(
                {
                    "error": {
                        "type": "upstream_error",
                        "code": f"upstream_{status}",
                        "message": _upstream_error_message(status, resp_body),
                    }
                },
                _upstream_status(status),
            )
            return
        try:
            completion = json.loads(resp_body)
        except Exception as e:
            log.err(f"parse: {e}")
            self._json_response({"error": {"message": str(e)}}, 502)
            return
        if (
            completion
            .get("choices", [{}])[0]
            .get("message", {})
            .get("reasoning_content")
        ):
            remember_reasoning(session_key(body), [completion["choices"][0]["message"]])
        response = build_non_stream_response(completion)
        usg = completion.get("usage")
        if usg:
            log.toks(
                usg.get("prompt_tokens"),
                usg.get("completion_tokens"),
                usg.get("total_tokens"),
            )
        self._json_response(response, 200)

    def _handle_stream(self, body: dict, chat_body: dict) -> None:
        def generate():
            translator = SseTranslator()
            conn = None
            t0 = time.time()
            try:
                status, resp, conn = _deepseek_request(chat_body, stream=True)
                if status != 200 or isinstance(resp, str):
                    err_body = resp if isinstance(resp, str) else resp[:300]
                    log.err(f"Upstream {status}: {err_body}")
                    yield translator.error(_upstream_error_message(status, err_body))
                    return
                # Read in 4KB chunks; buffer as bytes to avoid splitting multi-byte UTF-8 chars
                buf = b""
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line_bytes, buf = buf.split(b"\n", 1)
                        line = line_bytes.decode("utf-8")
                        if not line.startswith("data: "):
                            continue
                        json_str = line[6:].strip()
                        if json_str == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(json_str)
                            result = translator.feed(parsed)
                            if result:
                                yield result
                        except (json.JSONDecodeError, ValueError):
                            pass
                # Flush remaining buffer
                for line_bytes in buf.split(b"\n"):
                    if not line_bytes:
                        continue
                    line = line_bytes.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    json_str = line[6:].strip()
                    if json_str == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(json_str)
                        result = translator.feed(parsed)
                        if result:
                            yield result
                    except (json.JSONDecodeError, ValueError):
                        pass
                if translator.reasoning_so_far:
                    remember_reasoning(
                        session_key(body),
                        [
                            {
                                "role": "assistant",
                                "content": translator.content_so_far,
                                "reasoning_content": translator.reasoning_so_far,
                            }
                        ],
                    )
                yield translator.done(None)
            except Exception as e:
                log.err(f"upstream: {e}")
                yield translator.error(str(e))
            finally:
                log.timing(int((time.time() - t0) * 1000))
                if conn:
                    conn.close()

        self._sse_response(generate())


def _write_catalog_json():
    """Write a ready-to-use model catalog JSON so Codex recognizes MODEL."""
    import os as _os

    catalog = {
        "models": [
            {
                "slug": MODEL,
                "display_name": f"DeepSeek {MODEL}" if IS_DEEPSEEK else MODEL,
                "description": f"{MODEL} via codex-deepseek proxy",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 10,
                "context_window": 262144,
                "max_context_window": 1048576,
                "effective_context_window_percent": 95,
                "auto_compact_token_limit": 196608,
                "max_completion_tokens": 32768,
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "supports_image_detail_original": False,
                "supports_parallel_tool_calls": True,
                "supports_search_tool": False,
                "web_search_tool_type": "text_and_image",
                "apply_patch_tool_type": "freeform",
                "shell_type": "shell_command",
                "supports_reasoning_summaries": False,
                "default_reasoning_summary": "none",
                "default_reasoning_level": "high",
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "Fast responses"},
                    {"effort": "medium", "description": "Balanced speed and depth"},
                    {
                        "effort": "high",
                        "description": "Deep reasoning for complex problems",
                    },
                ],
                "support_verbosity": False,
                "default_verbosity": "medium",
                "truncation_policy": {"mode": "tokens", "limit": 10000},
                "base_instructions": f"You are a coding agent powered by {MODEL}.",
                "cost": {"input": 0, "output": 0},
                "release_date": "2025-01-01",
                "last_updated": "2025-01-01",
                "structured_output": False,
                "open_weights": False,
                "attachment": False,
                "experimental_supported_tools": [],
                "additional_speed_tiers": [],
                "availability_nux": None,
                "upgrade": None,
            }
        ]
    }
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    path = _os.path.join(root, "codex-deepseek-catalog.json")
    with open(path, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    return path


def _check_onboarding():
    """Check what onboarding items are still missing.
    Returns (catalog_ok: bool, missing_root: set, missing_custom: set)."""
    import os as _os

    catalog_path = _os.path.expanduser("~/.codex/model-catalogs/deepseek.json")
    catalog_ok = _os.path.isfile(catalog_path)

    config_path = _os.path.expanduser("~/.codex/config.toml")
    required_root = {
        "model_catalog_json",
        "model_context_window",
        "model_supports_reasoning_summaries",
        "model_reasoning_summary",
    }
    required_custom = {"stream_idle_timeout_ms"}
    found_root = set()
    found_custom = set()
    section = ""

    try:
        with open(config_path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    section = stripped[1:-1].strip()
                    continue
                if "=" in stripped:
                    key = stripped.split("=")[0].strip()
                    if section == "" and key in required_root:
                        found_root.add(key)
                    elif section == "model_providers.custom" and key in required_custom:
                        found_custom.add(key)
    except (FileNotFoundError, PermissionError):
        return (catalog_ok, required_root, required_custom)

    return (catalog_ok, required_root - found_root, required_custom - found_custom)


def run():
    log.info("")
    log.ok("codex-deepseek started")
    log.info(f"http://127.0.0.1:{PORT}/responses")
    log.info(
        f"model: {MODEL}  is_deepseek: {'true' if IS_DEEPSEEK else 'false'}  multimodal: {'on' if MULTIMODAL else 'off'}"
    )
    if not DEEPSEEK_API_KEY:
        log.warn("api_key not set")
    catalog_ok, missing_root, missing_custom = _check_onboarding()
    if not catalog_ok or missing_root or missing_custom:
        log.info("")
        log.header("Missing configuration detected")
    if not catalog_ok:
        _write_catalog_json()
        log.info(
            f'Model metadata not found: Codex does not recognize "{MODEL}" by default.'
        )
        log.info(f'A catalog file for "{MODEL}" has been written to:')
        log.info(
            f"  {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/codex-deepseek-catalog.json"
        )
        log.info("To fix:")
        log.info("  1) mkdir -p ~/.codex/model-catalogs")
        log.info(
            "  2) cp codex-deepseek-catalog.json ~/.codex/model-catalogs/deepseek.json"
        )
        log.info("  3) Add to ~/.codex/config.toml (root level, not inside provider):")
        log.info('       model_catalog_json = "~/.codex/model-catalogs/deepseek.json"')
        log.info("")
    if "model_catalog_json" in missing_root and catalog_ok:
        log.info("Model catalog file exists, but config.toml is not updated.")
        log.info("  Add to ~/.codex/config.toml (root level):")
        log.info('    model_catalog_json = "~/.codex/model-catalogs/deepseek.json"')
        log.info("")
    root_items = []
    if "model_context_window" in missing_root:
        root_items.append("model_context_window = 262144")
    if "model_supports_reasoning_summaries" in missing_root:
        root_items.append("model_supports_reasoning_summaries = true")
    if "model_reasoning_summary" in missing_root:
        root_items.append('model_reasoning_summary = "none"')
    if root_items:
        log.info("Missing performance / reasoning settings in ~/.codex/config.toml:")
        for item in root_items:
            log.info("  " + item)
        log.info("")
    if missing_custom:
        log.info("Missing provider-level setting in ~/.codex/config.toml:")
        log.info("  [model_providers.custom]")
        log.info(
            "  stream_idle_timeout_ms = 1800000  # prevent disconnect during long thinking"
        )
        log.info("")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
