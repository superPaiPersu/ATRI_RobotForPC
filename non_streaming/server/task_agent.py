from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from server import chat_ai
    from server.file_skill import reset_file_access_audit, set_file_access_audit
except ImportError:
    import chat_ai
    from file_skill import reset_file_access_audit, set_file_access_audit


DIR_PATH = Path(__file__).resolve().parent
TASK_AGENT_MAX_STEPS = int(os.getenv("AI_PET_TASK_AGENT_MAX_STEPS", "8"))
TASK_TURNS_TO_KEEP = 8

CONTINUE_WORDS = (
    "continue",
    "go on",
    "keep going",
    "resume",
    "继续",
    "接着",
    "刚才",
    "上面",
    "上一轮",
)

TASK_WORDS = (
    "analyze",
    "inspect",
    "debug",
    "fix",
    "implement",
    "build",
    "refactor",
    "review",
    "summarize",
    "search",
    "read",
    "find",
    "explain",
    "trace",
    "分析",
    "读一下",
    "读取",
    "查一下",
    "找一下",
    "排查",
    "修",
    "实现",
    "改",
    "重构",
    "总结",
    "解释",
    "梳理",
)

CONTEXT_WORDS = (
    "file",
    "folder",
    "directory",
    "project",
    "code",
    "repo",
    "log",
    "error",
    "traceback",
    "文件",
    "目录",
    "项目",
    "代码",
    "仓库",
    "日志",
    "报错",
    "实现",
    "架构",
)


def run_task_turn(
    model: str,
    api_url: str,
    api_key: str,
    messages,
    username: str = "",
    pet_id: str = "",
    on_chunk=None,
):
    if not _task_agent_enabled():
        return chat_ai.get_answer(model, api_url, api_key, messages)

    normalized = _normalize_messages(messages)
    latest_user_text = _latest_user_text(normalized)
    state = _load_task_state(username, pet_id)
    should_handle = _is_task_request(latest_user_text, state)

    if not should_handle:
        return chat_ai.get_answer(model, api_url, api_key, messages)

    now = datetime.now().isoformat(timespec="seconds")
    if not _is_active_state(state) or not _is_continue_request(latest_user_text):
        state = {
            "task_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "goal": _compact(latest_user_text, 240),
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "turns": [],
            "tool_audit": [],
        }
    else:
        state["updated_at"] = now

    task_messages = _with_task_context(messages, state)
    file_audit: list[dict[str, Any]] = []
    audit_token = set_file_access_audit(file_audit)
    step_token = chat_ai.set_agent_step_limit(TASK_AGENT_MAX_STEPS)
    try:
        answer = chat_ai.get_answer(model, api_url, api_key, task_messages)
    finally:
        chat_ai.reset_agent_step_limit(step_token)
        reset_file_access_audit(audit_token)

    _update_task_state(state, latest_user_text, answer, file_audit)
    _save_task_state(username, pet_id, state)
    return answer


def _task_agent_enabled() -> bool:
    value = os.getenv("AI_PET_TASK_AGENT_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _normalize_messages(messages) -> list[dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    normalized = []
    for message in messages or []:
        if isinstance(message, str):
            normalized.append({"role": "user", "content": message})
            continue
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").lower()
        content = message.get("content")
        if content is None:
            content = message.get("text") or message.get("parts")
        content = _content_to_text(content).strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _latest_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def _is_task_request(text: str, state: dict[str, Any] | None) -> bool:
    clean = (text or "").strip().lower()
    if not clean or clean == "heart_beat":
        return False
    if _is_active_state(state) and _is_continue_request(clean):
        return True
    has_task_word = any(word in clean for word in TASK_WORDS)
    path_like = bool(re.search(r"[a-z]:\\|/|\\|\.\w{1,8}\b", clean))
    has_context_word = any(word in clean for word in CONTEXT_WORDS)
    return has_task_word or has_context_word or path_like


def _is_continue_request(text: str) -> bool:
    clean = (text or "").strip().lower()
    return any(word in clean for word in CONTINUE_WORDS)


def _is_active_state(state: dict[str, Any] | None) -> bool:
    return isinstance(state, dict) and state.get("status") == "running"


def _with_task_context(messages, state: dict[str, Any]):
    context = {
        "task_id": state.get("task_id"),
        "goal": state.get("goal"),
        "status": state.get("status"),
        "recent_turns": state.get("turns", [])[-3:],
        "recent_tool_audit": state.get("tool_audit", [])[-12:],
    }
    instruction = (
        "Task mode is active. Work toward the user's goal across turns. "
        "The available agent tools include screen context and local file access: "
        "inspect_screen_activity, list_authorized_dirs, search_user_files, "
        "list_user_files, read_user_file, bocha_web_search, and get_current_time. "
        "Prefer search_user_files before read_user_file unless the user gave an exact path. "
        "read_user_file can read text files, PDF files, and .docx Word documents. "
        "Do not expose raw tool logs or full file dumps unless the user asks. "
        "If evidence is insufficient, ask a focused follow-up. "
        "Keep the final answer in the existing pet reply JSON format. "
        "Current task state JSON: "
        + json.dumps(context, ensure_ascii=False)
    )
    if isinstance(messages, list):
        return list(messages) + [{"role": "developer", "content": instruction}]
    return [
        {"role": "user", "content": str(messages or "")},
        {"role": "developer", "content": instruction},
    ]


def _update_task_state(
    state: dict[str, Any],
    latest_user_text: str,
    answer,
    file_audit: list[dict[str, Any]],
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    reply_text = _answer_text(answer)
    state["updated_at"] = now
    state["last_user"] = _compact(latest_user_text, 500)
    state["last_reply"] = _compact(reply_text, 800)

    turns = state.setdefault("turns", [])
    turns.append({
        "at": now,
        "user": _compact(latest_user_text, 300),
        "reply": _compact(reply_text, 500),
    })
    del turns[:-TASK_TURNS_TO_KEEP]

    if file_audit:
        tool_audit = state.setdefault("tool_audit", [])
        tool_audit.extend(_dedupe_audit(file_audit))
        del tool_audit[:-40]

def _answer_text(answer) -> str:
    content = ""
    if isinstance(answer, dict):
        content = str(answer.get("content") or "")
    else:
        content = str(answer or "")
    try:
        payload = json.loads(content)
    except ValueError:
        return content
    if isinstance(payload, dict):
        return str(payload.get("cn") or payload.get("content") or payload.get("jp") or content)
    return content


def _dedupe_audit(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for entry in entries:
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return result


def _task_state_path(username: str, pet_id: str) -> Path:
    safe_username = Path(str(username or "unknown_user")).name
    safe_pet_id = Path(str(pet_id or "default")).name
    task_dir = DIR_PATH / "users_data" / safe_username / safe_pet_id / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir / "current_task.json"


def _load_task_state(username: str, pet_id: str) -> dict[str, Any] | None:
    path = _task_state_path(username, pet_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _save_task_state(username: str, pet_id: str, state: dict[str, Any]) -> None:
    path = _task_state_path(username, pet_id)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _compact(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
