import sys
from typing import Optional

C = {
    "reset": "[0m",
    "cyan": "[36m",
    "green": "[32m",
    "yellow": "[33m",
    "red": "[31m",
    "magenta": "[35m",
    "gray": "[90m",
    "bold": "[1m",
}


def info(msg: str, *args) -> None:
    print(f"{C['cyan']}[INFO]{C['reset']} {msg}", *args, file=sys.stderr)


def ok(msg: str, *args) -> None:
    print(f"{C['green']}[ OK ]{C['reset']} {msg}", *args, file=sys.stderr)


def warn(msg: str, *args) -> None:
    print(f"{C['yellow']}[WARN]{C['reset']} {msg}", *args, file=sys.stderr)


def err(msg: str, *args) -> None:
    print(f"{C['red']}[ERR ]{C['reset']} {msg}", *args, file=sys.stderr)


def req(msg: str, *args) -> None:
    print(f"{C['magenta']}[REQ ]{C['reset']} {msg}", *args, file=sys.stderr)


def resp(msg: str, *args) -> None:
    print(f"{C['green']}[RESP]{C['reset']} {msg}", *args, file=sys.stderr)


def skip(msg: str, *args) -> None:
    print(f"{C['gray']}[SKIP]{C['reset']} {msg}", *args, file=sys.stderr)


def toks(
    prompt: Optional[int] = None,
    completion: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    parts = []
    if prompt is not None:
        parts.append(f"in:{prompt}")
    if completion is not None:
        parts.append(f"out:{completion}")
    if total is not None:
        parts.append(f"total:{total}")
    print(f"{C['gray']}[TOKS]{C['reset']} {' '.join(parts)}", file=sys.stderr)


def header(msg: str) -> None:
    print(f"\n{C['bold']}{C['cyan']}=== {msg} ==={C['reset']}", file=sys.stderr)
