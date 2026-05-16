import json
import os
import random
import string
from typing import Optional

from . import log

MODEL = os.getenv("model", "deepseek-v4-pro")


def _rand_id(prefix: str, length: int = 8) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}_{suffix}"


class SseTranslator:
    def __init__(self):
        self.response_id = _rand_id("resp")
        self.message_item_id = _rand_id("item")
        self.text_started = False
        self.content_so_far = ""
        self.reasoning_started = False
        self.reasoning_so_far = ""
        self.reasoning_item_id: Optional[str] = None
        self.tool_calls: dict[int, dict] = {}
        self.started = False
        self.output_item_count = 0
        self.output_items: list[dict] = []
        self._last_usage = None

    def _emit(self, event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def feed(self, chunk: dict) -> str:
        """Process one chunk from DeepSeek and return SSE event string(s)."""
        delta = (chunk.get("choices") or [{}])[0].get("delta")
        if not delta:
            return ""
        if chunk.get("usage"):
            self._last_usage = chunk["usage"]

        output = ""
        if delta.get("content"):
            output += self._ensure_started()
            self.content_so_far += delta["content"]
            if not self.text_started:
                self.text_started = True
                oi = self.output_item_count
                self.output_item_count += 1
                self.output_items.append({
                    "index": oi,
                    "type": "message",
                    "itemId": self.message_item_id,
                })
                output += self._emit(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": self.response_id,
                        "output_index": oi,
                        "item": {
                            "id": self.message_item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    },
                )
                output += self._emit(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "response_id": self.response_id,
                        "item_id": self.message_item_id,
                        "output_index": oi,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
            output += self._emit(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": self._msg_index(),
                    "content_index": 0,
                    "delta": delta["content"],
                },
            )

        if delta.get("reasoning_content"):
            output += self._ensure_started()
            self.reasoning_so_far += delta["reasoning_content"]
            if not self.reasoning_started:
                self.reasoning_started = True
                self.reasoning_item_id = _rand_id("rsn")
                oi = self.output_item_count
                self.output_item_count += 1
                self.output_items.append({
                    "index": oi,
                    "type": "reasoning",
                    "itemId": self.reasoning_item_id,
                })
                output += self._emit(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": self.response_id,
                        "output_index": oi,
                        "item": {
                            "id": self.reasoning_item_id,
                            "type": "reasoning",
                            "status": "in_progress",
                            "summary": [],
                        },
                    },
                )
                output += self._emit(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "response_id": self.response_id,
                        "item_id": self.reasoning_item_id,
                        "output_index": oi,
                        "content_index": 0,
                        "part": {"type": "reasoning_text", "text": ""},
                    },
                )
            r_idx = self._rsn_index()
            if r_idx >= 0:
                output += self._emit(
                    "response.reasoning_text.delta",
                    {
                        "type": "response.reasoning_text.delta",
                        "response_id": self.response_id,
                        "item_id": self.reasoning_item_id,
                        "output_index": r_idx,
                        "content_index": 0,
                        "delta": delta["reasoning_content"],
                    },
                )

        if delta.get("tool_calls"):
            output += self._ensure_started()
            for tc in delta["tool_calls"]:
                idx = tc["index"]
                if idx not in self.tool_calls:
                    call = {
                        "id": tc.get("id", f"call_{idx}"),
                        "name": (tc.get("function") or {}).get("name", ""),
                        "arguments": "",
                    }
                    self.tool_calls[idx] = call
                    oi = self.output_item_count
                    self.output_item_count += 1
                    self.output_items.append({
                        "index": oi,
                        "type": "function_call",
                        "itemId": f"fc_{call['id']}",
                    })
                    output += self._emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "response_id": self.response_id,
                            "output_index": oi,
                            "item": {
                                "id": f"fc_{call['id']}",
                                "type": "function_call",
                                "call_id": call["id"],
                                "name": call["name"],
                                "status": "in_progress",
                            },
                        },
                    )
                    log.info("tool: " + call["name"] + " (" + call["id"] + ")")
                call = self.tool_calls[idx]
                if (tc.get("function") or {}).get("name"):
                    call["name"] = tc["function"]["name"]
                d = (tc.get("function") or {}).get("arguments", "")
                call["arguments"] += d
                oi = self._item_index(f"fc_{call['id']}")
                if oi >= 0:
                    output += self._emit(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "response_id": self.response_id,
                            "item_id": f"fc_{call['id']}",
                            "output_index": oi,
                            "delta": d,
                        },
                    )

        return output

    def done(self, usage_override: Optional[dict] = None) -> str:
        output = self._ensure_started()
        usage = usage_override or self._last_usage or None

        if self.text_started:
            oi = self._msg_index()
            output += self._emit(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": oi,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": self.content_so_far,
                        "annotations": [],
                    },
                },
            )
            output += self._emit(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "response_id": self.response_id,
                    "item_id": self.message_item_id,
                    "output_index": oi,
                    "content_index": 0,
                    "text": self.content_so_far,
                },
            )
            output += self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "output_index": oi,
                    "item": {
                        "id": self.message_item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": self.content_so_far,
                                "annotations": [],
                            }
                        ],
                        "status": "completed",
                    },
                },
            )
            log.resp(f"text output: {len(self.content_so_far)} chars")

        if self.reasoning_started:
            oi = self._rsn_index()
            output += self._emit(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "response_id": self.response_id,
                    "item_id": self.reasoning_item_id,
                    "output_index": oi,
                    "content_index": 0,
                    "part": {"type": "reasoning_text", "text": self.reasoning_so_far},
                },
            )
            output += self._emit(
                "response.reasoning_text.done",
                {
                    "type": "response.reasoning_text.done",
                    "response_id": self.response_id,
                    "item_id": self.reasoning_item_id,
                    "output_index": oi,
                    "content_index": 0,
                    "text": self.reasoning_so_far,
                },
            )
            output += self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "output_index": oi,
                    "item": {
                        "id": self.reasoning_item_id,
                        "type": "reasoning",
                        "content": [
                            {"type": "reasoning_text", "text": self.reasoning_so_far}
                        ],
                        "status": "completed",
                    },
                },
            )
            log.resp(f"reasoning output: {len(self.reasoning_so_far)} chars")

        for idx, call in self.tool_calls.items():
            oi = self._item_index(f"fc_{call['id']}")
            out_idx = oi if oi >= 0 else idx + 1
            output += self._emit(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": self.response_id,
                    "item_id": f"fc_{call['id']}",
                    "output_index": out_idx,
                    "arguments": call["arguments"],
                    "name": call["name"],
                    "call_id": call["id"],
                },
            )
            output += self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "output_index": out_idx,
                    "item": {
                        "id": f"fc_{call['id']}",
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                        "status": "completed",
                    },
                },
            )
            log.resp("tool done: " + call["name"])

        resp_usage = None
        if usage:
            resp_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        out_snapshot = []
        for o in self.output_items:
            if o["type"] == "message":
                out_snapshot.append({
                    "id": o["itemId"],
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": self.content_so_far,
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                })
            elif o["type"] == "reasoning":
                out_snapshot.append({
                    "id": o["itemId"],
                    "type": "reasoning",
                    "content": [
                        {"type": "reasoning_text", "text": self.reasoning_so_far}
                    ],
                    "status": "completed",
                })
            elif o["type"] == "function_call":
                for c in self.tool_calls.values():
                    if f"fc_{c['id']}" == o["itemId"]:
                        out_snapshot.append({
                            "id": o["itemId"],
                            "type": "function_call",
                            "call_id": c["id"],
                            "name": c["name"],
                            "arguments": c["arguments"],
                            "status": "completed",
                        })

        output += self._emit(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "completed",
                    "model": MODEL,
                    "output": out_snapshot,
                    "usage": resp_usage,
                },
            },
        )

        if usage:
            log.toks(
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
            )
        log.ok("SSE done: " + self.response_id)

        output += "event: done\ndata: [DONE]\n\n"
        return output

    def error(self, msg: str) -> str:
        out = self._emit(
            "error", {"type": "error", "code": "proxy_error", "message": msg}
        )
        log.err("SSE error: " + msg)
        out += "event: done\ndata: [DONE]\n\n"
        return out

    def _ensure_started(self) -> str:
        if self.started:
            return ""
        self.started = True
        out = ""
        out += self._emit(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": MODEL,
                    "output": [],
                },
            },
        )
        out += self._emit(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response_id": self.response_id,
            },
        )
        log.info("SSE start: " + self.response_id)
        return out

    def _msg_index(self) -> int:
        for o in self.output_items:
            if o["type"] == "message":
                return o["index"]
        return 0

    def _rsn_index(self) -> int:
        for o in self.output_items:
            if o["type"] == "reasoning":
                return o["index"]
        return -1

    def _item_index(self, item_id: str) -> int:
        for o in self.output_items:
            if o["itemId"] == item_id:
                return o["index"]
        return -1
