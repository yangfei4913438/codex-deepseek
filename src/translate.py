from typing import Any, Optional

from . import log


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not content:
        return ""
    if not isinstance(content, list):
        if (
            isinstance(content, dict)
            and "type" in content
            and content.get("text") is not None
        ):
            return content["text"]
        return ""
    return "".join(
        p.get("text", "")
        for p in content
        if isinstance(p, dict)
        and p.get("type") in ("input_text", "output_text", "text", "reasoning_text")
    )


def translate_messages(input_data: Any, options: Optional[dict] = None) -> dict:
    options = options or {}
    keep_reasoning_content = options.get("keepReasoningContent", False)
    messages: list[dict] = []
    stats = {
        "skipped": {"reasoning": 0, "image": 0, "file": 0, "audio": 0, "other": 0},
        "strippedReasoningContent": 0,
        "preservedReasoningContent": 0,
    }

    if not isinstance(input_data, list):
        if isinstance(input_data, str) and input_data.strip():
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, dict):
            text = extract_text(input_data.get("content"))
            if text:
                messages.append({"role": "user", "content": text})
        return {"messages": messages, "stats": stats}

    for item in input_data:
        if not item:
            continue

        if item.get("type") == "function_call":
            last = (
                messages[-1]
                if messages and messages[-1].get("role") == "assistant"
                else None
            )
            target = (
                last if last is not None else {"role": "assistant", "tool_calls": []}
            )
            if last is None:
                messages.append(target)
            if "tool_calls" not in target:
                target["tool_calls"] = []
            target["tool_calls"].append({
                "id": item.get("call_id") or item.get("id"),
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                },
            })
            if item.get("status") == "incomplete":
                log.warn(
                    "function_call status incomplete: "
                    + (item.get("call_id") or item.get("id"))
                )
            if item.get("reasoning_content") and "reasoning_content" not in target:
                target["reasoning_content"] = item["reasoning_content"]
            continue

        if item.get("type") == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id"),
                "content": extract_text(item.get("output")),
            })
            if item.get("status") == "incomplete":
                log.warn(
                    "function_call_output status incomplete: "
                    + (item.get("call_id") or item.get("id"))
                )
            continue

        if item.get("type") == "reasoning":
            stats["skipped"]["reasoning"] += 1
            if item.get("reasoning_content"):
                last = messages[-1] if messages else None
                if last and "reasoning_content" not in last:
                    last["reasoning_content"] = item["reasoning_content"]
            continue

        if "role" in item:
            role = "system" if item["role"] == "developer" else item["role"]
            text_content = extract_text(item.get("content"))

            # Track skipped content types and inject hints
            skipped_images = 0
            if isinstance(item.get("content"), list):
                for p in item["content"]:
                    t = p.get("type") if isinstance(p, dict) else None
                    if t == "input_image":
                        skipped_images += 1
                        stats["skipped"]["image"] += 1
                    elif t == "input_file":
                        stats["skipped"]["file"] += 1
                    elif t == "input_audio":
                        stats["skipped"]["audio"] += 1

            if skipped_images > 0 and role == "user" and text_content:
                hint = "image" if skipped_images == 1 else f"{skipped_images} images"
                log.warn(f"{hint} skipped, model does not support vision input")
                text_content += f"\n\n[Note: The user attached {hint} which could not be displayed. Do NOT describe or speculate about the image content — just let the user know you cannot view images and ask them to describe it in text if needed.]"
            elif not text_content and skipped_images > 0:
                log.warn("image-only message skipped, model does not support vision input")

            if text_content:
                msg: dict = {"role": role, "content": text_content}
                if item.get("reasoning_content"):
                    msg["reasoning_content"] = item["reasoning_content"]
                if item.get("tool_calls"):
                    msg["tool_calls"] = item["tool_calls"]
                if item.get("tool_call_id"):
                    msg["tool_call_id"] = item["tool_call_id"]
                messages.append(msg)

            continue

        if item.get("type") == "message":
            text_content = extract_text(item.get("content"))
            if text_content:
                messages.append({"role": "user", "content": text_content})
            continue

        stats["skipped"]["other"] += 1

    if keep_reasoning_content:
        stats["preservedReasoningContent"] = sum(
            1 for m in messages if m.get("reasoning_content")
        )
    else:
        for m in messages:
            if "reasoning_content" in m:
                del m["reasoning_content"]
                stats["strippedReasoningContent"] += 1

    return {"messages": messages, "stats": stats}


def last_user_text(messages: list[dict]) -> str:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return extract_text(messages[i].get("content"))
    return ""


def translate_tools(raw_tools: Any) -> list[dict]:
    if not isinstance(raw_tools, list):
        return []
    result = []
    for t in raw_tools:
        name = t.get("name") or t.get("function", {}).get("name")
        if not name:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description")
                or t.get("function", {}).get("description", ""),
                "parameters": t.get("parameters")
                or t.get("function", {}).get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            },
        })
    return result


def translate_tool_choice(tool_choice: Any) -> Any:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and tool_choice.get("name")
    ):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice
