_queue: list[str] = []


def remember_reasoning(key: str, messages: list[dict]) -> None:
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("reasoning_content")
        ):
            _queue.append(msg["reasoning_content"])


def recover_reasoning(key: str, messages: list[dict]) -> int:
    if not _queue:
        return 0
    recovered = 0
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and "reasoning_content" not in msg
        ):
            msg["reasoning_content"] = _queue[min(recovered, len(_queue) - 1)]
            recovered += 1
    return recovered


def session_key(body: dict) -> str:
    return "g"
