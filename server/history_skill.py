from __future__ import annotations

import json
import os
import tempfile
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Callable


_HISTORY_ROOT = Path(__file__).resolve().parent / "users_data"
_TAG_WRITE_LOCK = threading.RLock()

# Shared by automatic summaries and offline history re-indexing.
CONCEPT_TAG_RULES = (
    'tags 是单层概念标签字符串数组，例如 ["日语","宫保鸡丁","兔子"]，不再输出二元标签。'
    "先逐项检查已有标签表，能准确表达时必须原样复用；同义表达复用同一名称，"
    "只有已有标签不能准确表达本段内容时才新增，不为复用而牺牲含义。"
    "标签可以是具体对象、话题、行为或处境，如日语、兔子、熬夜、被催促、独处；"
    "不按词性分类，不使用东西、事情、生活相关、喜欢、讨厌等泛化标签。"
    "喜欢或讨厌、肯定或否定等态度保留在原文，不拼成喜欢兔子或讨厌兔子这样的标签。"
    "名称简洁明确，优先最具体且可复用的概念，避免同义词和无必要的上下位标签堆叠。"
    "熬夜不等于失眠，独处不等于孤独，不能合并含义不同的概念。"
    "每段最多六个标签，去重，不强行凑数。纯问候、心跳或无实质内容可返回 []。"
    "不得把密钥、口令、账号凭据或整句私人信息作为标签。"
)


def validate_concept_tags(value, *, limit=None):
    if not isinstance(value, list) or (limit is not None and len(value) > limit):
        raise ValueError("tags must be a flat string array within the tag limit")
    tags = []
    for tag in value:
        if (not isinstance(tag, str) or not tag.strip() or len(tag.strip()) > 100
                or any(ord(char) < 32 for char in tag)):
            raise ValueError("Each concept tag must be a non-empty short string")
        tag = tag.strip()
        if tag not in tags:
            tags.append(tag)
    return tags
_history_identity_var: ContextVar[tuple[str, str] | None] = ContextVar(
    "history_identity",
    default=None,
)
_history_search_request_var: ContextVar[Callable | None] = ContextVar(
    "history_search_request", default=None,
)


def dialogue_tag_lists(server_dir=None):
    """Read the one growing concept registry locally, including an empty new registry."""
    server_dir = Path(server_dir) if server_dir is not None else Path(__file__).resolve().parent
    path = server_dir / "data" / "dialogue_tags"
    with _TAG_WRITE_LOCK:
        if not path.exists():
            return []
        return validate_concept_tags([line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()])


def register_dialogue_tags(tags, server_dir=None):
    """Merge additions under a lock and replace atomically; concurrent saves cannot lose tags."""
    tags = validate_concept_tags(tags)
    server_dir = Path(server_dir) if server_dir is not None else Path(__file__).resolve().parent
    path = server_dir / "data" / "dialogue_tags"
    with _TAG_WRITE_LOCK:
        labels = dialogue_tag_lists(server_dir)
        merged = list(dict.fromkeys(labels + tags))
        if merged != labels or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                    temporary = Path(output.name)
                    output.write(("\n".join(merged) + ("\n" if merged else "")).encode("utf-8"))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return merged


def get_dialogue_tags() -> str:
    """Return the current flat concept registry from the server."""
    return json.dumps(dialogue_tag_lists(), ensure_ascii=False)


def set_history_search_request(requester):
    return _history_search_request_var.set(requester)


def reset_history_search_request(token):
    _history_search_request_var.reset(token)


def search_dialogue_history(tags) -> str:
    """Fetch only the matching messages from this request's originating client."""
    requester = _history_search_request_var.get()
    if requester is None:
        return json.dumps({"error": "Client dialogue history search is unavailable."})
    tags = validate_concept_tags(tags, limit=30)
    if not tags or any(tag not in dialogue_tag_lists() for tag in tags):
        raise ValueError("Select existing concept labels from get_dialogue_tags")
    result = requester({"tags": tags})
    if not isinstance(result, dict):
        raise ValueError("Client dialogue history returned an invalid result")
    if result.get("error"):
        return json.dumps({"error": str(result["error"])}, ensure_ascii=False)
    messages = result.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError("Client dialogue history must return a message list")
    if result.get("matched") is False:
        messages = []
    content = json.dumps({"messages": messages, "matched": bool(messages)}, ensure_ascii=False)
    if len(content.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError("History result exceeds 2 MiB")
    return content


def set_history_identity(username: str, pet_id: str):
    return _history_identity_var.set((str(username or ""), str(pet_id or "")))


def reset_history_identity(token) -> None:
    _history_identity_var.reset(token)


def _safe_component(value: str, label: str) -> str:
    value = str(value or "").strip()
    name = Path(value).name
    if not value or name != value or name in {".", ".."}:
        raise ValueError(f"Invalid {label} for dialogue history.")
    return name


def get_all_dialogue_history() -> str:
    """Return every persisted dialogue event for the current user and pet."""
    identity = _history_identity_var.get()
    if identity is None:
        return json.dumps(
            {"error": "Dialogue history is unavailable outside an authenticated chat request."},
            ensure_ascii=False,
        )

    try:
        username = _safe_component(identity[0], "username")
        pet_id = _safe_component(identity[1], "pet_id")
    except ValueError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    history_path = _HISTORY_ROOT / username / pet_id / "chat_history.txt"
    if not history_path.exists():
        return json.dumps(
            {"event_count": 0, "events": []},
            ensure_ascii=False,
        )

    events = []
    invalid_lines = []
    try:
        with history_path.open("r", encoding="utf-8-sig") as history_file:
            for line_number, raw_line in enumerate(history_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines.append(line_number)
                    continue
                events.append(event)
    except OSError as exc:
        return json.dumps(
            {"error": f"Failed to read dialogue history: {exc}"},
            ensure_ascii=False,
        )

    result = {
        "event_count": len(events),
        "events": events,
    }
    if invalid_lines:
        result["skipped_invalid_lines"] = invalid_lines
    return json.dumps(result, ensure_ascii=False)
