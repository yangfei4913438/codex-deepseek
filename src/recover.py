import hashlib
import json
import threading
from collections import deque
from typing import Any

_lock = threading.Lock()

# Session-key → reasoning-content queue, capped per session and globally
_sessions: dict[str, deque[str]] = {}
MAX_SESSIONS = 64
MAX_PER_SESSION = 32


def _first_user_text(input_data: Any) -> str:
    """Extract the text of the first user message from raw Responses-API input."""
    if not isinstance(input_data, list):
        return ""
    for item in input_data:
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") in (
                    "input_text",
                    "output_text",
                    "text",
                    "reasoning_text",
                ):
                    t = p.get("text", "")
                    if t:
                        parts.append(t)
            text = "".join(parts).strip()
            if text:
                return text
    return ""


def session_key(body: dict) -> str:
    """Derive a stable session key from the first user message.

    The first user message is stable across all turns of a conversation,
    so a single codex session gets the same key for every request.
    """
    first = _first_user_text(body.get("input"))
    if not first:
        return "default"
    digest = hashlib.sha256(first.encode("utf-8")).hexdigest()[:16]
    return digest


def _evict_one() -> None:
    """Remove the oldest session entry (FIFO)."""
    if _sessions:
        _sessions.pop(next(iter(_sessions)))


def remember_reasoning(key: str, messages: list[dict]) -> None:
    """Persist reasoning_content so it can be restored in the next turn."""
    with _lock:
        if key not in _sessions:
            while len(_sessions) >= MAX_SESSIONS:
                _evict_one()
            _sessions[key] = deque(maxlen=MAX_PER_SESSION)
        q = _sessions[key]
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and msg.get("reasoning_content")
            ):
                q.append(msg["reasoning_content"])


def recover_reasoning(key: str, messages: list[dict]) -> int:
    """Restore reasoning_content that DeepSeek stripped from tool-call messages."""
    with _lock:
        q = _sessions.get(key)
        if not q:
            return 0
        recovered = 0
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and "reasoning_content" not in msg
            ):
                if recovered < len(q):
                    msg["reasoning_content"] = q[recovered]
                    recovered += 1
        return recovered
