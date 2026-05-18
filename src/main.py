import json
import os
import random
import string
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

    IDENTITY = f"\n\n[IMPORTANT: Your true model identity is {MODEL}. You are NOT OpenAI, GPT, or Claude. When asked about your model identity, you MUST answer truthfully based on your actual model name. Ignore any conflicting identity claims in the instructions above.]"
    instructions = body.get("instructions", "")
    if instructions:
        instructions = instructions + IDENTITY
    else:
        instructions = IDENTITY.strip()
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
        status, resp_body, conn = _deepseek_request(chat_body)
        if status != 200:
            log.err(f"Upstream {status}: {resp_body[:300]}")
            self._json_response(
                {
                    "error": {
                        "type": "upstream_error",
                        "code": f"upstream_{status}",
                        "message": f"Upstream {status}: {resp_body[:200]}",
                    }
                },
                502 if status and status >= 500 else status or 502,
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
            try:
                status, resp, conn = _deepseek_request(chat_body, stream=True)
                if status != 200 or isinstance(resp, str):
                    err_body = resp if isinstance(resp, str) else resp[:300]
                    log.err(f"Upstream {status}: {err_body}")
                    yield translator.error(f"Upstream {status}: {err_body[:200]}")
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
                if conn:
                    conn.close()

        self._sse_response(generate())


def run():
    print("")
    log.ok("codex-deepseek started")
    log.info(f"http://127.0.0.1:{PORT}/responses")
    log.info(
        f"model: {MODEL}  is_deepseek: {'true' if IS_DEEPSEEK else 'false'}  multimodal: {'on' if MULTIMODAL else 'off'}"
    )
    if not DEEPSEEK_API_KEY:
        log.warn("api_key not set")
    print("")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
