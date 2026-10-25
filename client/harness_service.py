from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

try:
    from deepseek_harness import DeepSeekHarness
    from deepseek_harness_runtime import resolve_bundled_launch_args
except Exception as exc:  # Keep the desktop client usable when the optional SDK is absent.
    DeepSeekHarness = None
    resolve_bundled_launch_args = None
    _SDK_IMPORT_ERROR = exc
else:
    _SDK_IMPORT_ERROR = None


DEFAULT_HARNESS_MODEL = "deepseek-v4-pro"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 840.0


class HarnessService:
    """Own a reusable DeepSeek Harness runtime on the desktop client."""

    def __init__(
        self,
        base_dir: str | Path,
        username: str,
        server_url: str = "",
        auth_token: str = "",
    ):
        self.base_dir = Path(base_dir).resolve()
        self.username = str(username or "user").strip() or "user"
        self.server_url = str(server_url or "").strip()
        self.auth_token = str(auth_token or "").strip()
        self.data_root = self.base_dir / "client_data" / "harness"
        self.patch_path = self.base_dir / "harness.patch.yml"
        self._run_lock = threading.RLock()
        self._harness: Any = None
        self._runtime_key: tuple[str, ...] | None = None
        self._bootstrapped_sessions: set[str] = set()
        self._availability_error = self._check_availability()

    @property
    def available(self) -> bool:
        return not self._availability_error

    @property
    def availability_error(self) -> str:
        return self._availability_error

    def run(self, payload: dict | None) -> dict:
        if not self.available:
            return {"ok": False, "error": self._availability_error}

        data = payload if isinstance(payload, dict) else {}
        api_url = str(data.get("api_url") or "").strip()
        api_key = str(data.get("api_key") or "").strip()
        model = str(data.get("model") or DEFAULT_HARNESS_MODEL).strip()
        pet_id = str(data.get("pet_id") or "default").strip() or "default"
        conversation_id = str(data.get("conversation_id") or "default").strip() or "default"
        if not api_url:
            return {"ok": False, "error": "DeepSeek Harness requires api_url"}
        if not api_key:
            return {"ok": False, "error": "DeepSeek Harness requires api_key"}
        if not model:
            return {"ok": False, "error": "DeepSeek Harness requires a model"}

        messages = _normalize_messages(data.get("messages"))
        if not messages:
            return {"ok": False, "error": "DeepSeek Harness received no conversation messages"}

        with self._run_lock:
            try:
                harness, workspace = self._ensure_harness(pet_id, api_url, api_key, model)
                session_id = _session_id(self.username, pet_id, conversation_id)
                full_context = session_id not in self._bootstrapped_sessions
                prompt = _build_prompt(messages, full_context=full_context)
                result = harness.run(prompt, session_id=session_id)
                response = str(result.final_response or "").strip()
                if not response:
                    reason = result.finish_reason or "unknown"
                    raise RuntimeError(f"Harness returned an empty response (finish_reason={reason})")
                self._bootstrapped_sessions.add(session_id)
                return {
                    "ok": True,
                    "content": response,
                    "finish_reason": result.finish_reason,
                    "session_id": result.session_id,
                }
            except Exception as exc:
                self._close_runtime()
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    def close(self) -> None:
        with self._run_lock:
            self._close_runtime()

    def _check_availability(self) -> str:
        if DeepSeekHarness is None or resolve_bundled_launch_args is None:
            detail = f": {_SDK_IMPORT_ERROR}" if _SDK_IMPORT_ERROR else ""
            return "deepseek-harness-sdk is unavailable" + detail
        if not self.patch_path.is_file():
            return f"Harness patch file is missing: {self.patch_path}"
        try:
            launch_args = resolve_bundled_launch_args()
        except Exception as exc:
            return f"DeepSeek Harness runtime is unavailable: {exc}"
        if not launch_args or not Path(launch_args[0]).is_file():
            return "DeepSeek Harness runtime executable is missing"
        return ""

    def _ensure_harness(self, pet_id: str, api_url: str, api_key: str, model: str):
        pet_component = _safe_component(pet_id, "pet")
        user_component = _safe_component(self.username, "user")
        pet_root = self.data_root / user_component / pet_component
        dsh_home = pet_root / "home"
        workspace = pet_root / "workspace"
        dsh_home.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)

        # The sdk profile already mounts skill-filesystem and the skill tool.
        # Its user-dsh root is <dsh_home>/skills, independent of workspace cwd.
        skill_source = self.base_dir / "skills" / "dialogue-memory" / "SKILL.md"
        skill_target = dsh_home / "skills" / "dialogue-memory" / "SKILL.md"
        skill_content = skill_source.read_bytes()
        if not skill_target.is_file() or skill_target.read_bytes() != skill_content:
            skill_target.parent.mkdir(parents=True, exist_ok=True)
            skill_target.write_bytes(skill_content)

        key_digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        token_digest = hashlib.sha256(self.auth_token.encode("utf-8")).hexdigest()
        runtime_key = (str(pet_root), api_url, key_digest, model, self.server_url, token_digest)
        if self._harness is not None and self._runtime_key == runtime_key:
            return self._harness, workspace

        self._close_runtime()
        if not getattr(sys, "frozen", False):
            # The MCP process is launched with this workspace as cwd, so make
            # the bridge importable without relying on the parent's PYTHONPATH.
            shutil.copyfile(self.base_dir / "bocha_mcp_server.py", workspace / "bocha_mcp_server.py")
            shutil.copyfile(self.base_dir / "pet_template" / "harness_bridge.py", workspace / "harness_bridge.py")

        bocha_config_path = workspace / "bocha_mcp_config.json"
        bocha_config_path.write_text(
            json.dumps(
                {
                    "server_url": self.server_url,
                    "username": self.username,
                    "auth_token": self.auth_token,
                    "pet_id": pet_id,
                    "pet_bridge_registry": str(self.base_dir / "client_data" / "harness_bridge"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if getattr(sys, "frozen", False):
            bocha_arguments = [
                "--bocha-mcp-server",
                "--bocha-config",
                str(bocha_config_path),
            ]
        else:
            bocha_arguments = [
                "-m",
                "bocha_mcp_server",
                "--config",
                str(bocha_config_path),
            ]

        # Keep MCP launch settings in a workspace-local patch instead of the
        # Harness environment. The MCP client registers its child process
        # environment during startup; passing these values through env can
        # trigger recursive environment registration in some runtime builds.
        bocha_patch_path = workspace / "bocha_mcp.patch.yml"
        bocha_patch_path.write_text(
            _build_bocha_patch(
                command=sys.executable,
                arguments=bocha_arguments,
                cwd=str(workspace),
            ),
            encoding="utf-8",
        )

        stream_plugin_path = workspace / "harness_stream_plugin.mjs"
        stream_plugin_path.write_text(_HARNESS_STREAM_PLUGIN, encoding="utf-8")
        stream_config_path = workspace / "harness_stream_config.json"
        stream_config_path.write_text(
            json.dumps(
                {
                    "endpoint": self.server_url.rstrip("/") + "/api/harness-stream",
                    "username": self.username,
                    "auth_token": self.auth_token,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        stream_patch_path = workspace / "harness_stream.patch.yml"
        stream_patch_path.write_text(
            _build_stream_patch(stream_plugin_path),
            encoding="utf-8",
        )

        timeout = _request_timeout_seconds()
        self._harness = DeepSeekHarness(
            dsh_home=str(dsh_home),
            cwd=str(workspace),
            runtime_cwd=str(workspace),
            profile="sdk",
            patches=(str(self.patch_path), str(bocha_patch_path), str(stream_patch_path)),
            provider="deepseek-official",
            model=model,
            base_url=api_url,
            api_key=api_key,
            initialize_timeout_seconds=45.0,
            request_timeout_seconds=timeout,
            shutdown_timeout_seconds=3.0,
        )
        self._runtime_key = runtime_key
        return self._harness, workspace

    def _close_runtime(self) -> None:
        harness = self._harness
        self._harness = None
        self._runtime_key = None
        if harness is not None:
            try:
                harness.close()
            except Exception as exc:
                print(f"DeepSeek Harness shutdown failed: {type(exc).__name__}: {exc}")


def _build_bocha_patch(command: str, arguments: list[str], cwd: str) -> str:
    """Build a literal MCP patch for one Harness workspace."""
    return "\n".join(
        (
            "- insert:",
            "    - id: mcp-bocha",
            "      name: '@deepseek-ai/dsh-mcp-client'",
            "      config:",
            "        serverName: bocha",
            "        transport: stdio",
            f"        command: {json.dumps(command, ensure_ascii=False)}",
            f"        args: {json.dumps(arguments, ensure_ascii=False)}",
            f"        cwd: {json.dumps(cwd, ensure_ascii=False)}",
            "        failOnStartupError: true",
            "",
        )
    )


def _build_stream_patch(plugin_path: Path) -> str:
    return "\n".join(
        (
            "- insert:",
            "    - id: desktop-pet-harness-stream",
            f"      name: {json.dumps(plugin_path.as_uri(), ensure_ascii=False)}",
            "",
        )
    )


_HARNESS_STREAM_PLUGIN = r'''import { readFileSync } from "node:fs"

let config = {}
try {
  config = JSON.parse(readFileSync(new URL("./harness_stream_config.json", import.meta.url), "utf8"))
} catch {}

const endpoint = String(config.endpoint || "")
const username = String(config.username || "")
const authToken = String(config.auth_token || "")
const attempts = new Map()
let postChain = Promise.resolve()

function post(sessionId, frame) {
  if (!endpoint || !username || !authToken || !sessionId) return
  const body = JSON.stringify({
    username,
    auth_token: authToken,
    session_id: sessionId,
    frame,
  })
  postChain = postChain
    .then(() => fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
    }))
    .then(response => response.arrayBuffer())
    .catch(() => undefined)
}

function flush(attempt) {
  if (attempt.timer !== undefined) {
    clearTimeout(attempt.timer)
    attempt.timer = undefined
  }
  if (!attempt.text) return
  const text = attempt.text
  attempt.text = ""
  post(attempt.sessionId, {
    type: "chunk",
    attempt_id: attempt.attemptId,
    chunk: { type: "text-delta", text },
  })
}

export function apply(ctx) {
  ctx.on("agent/assistant-stream", ({ agent, frame }) => {
    const sessionId = String(agent?.id || "")
    const attemptId = String(frame?.attemptId || "")
    if (!sessionId || !attemptId) return

    if (frame.type === "start") {
      const attempt = { sessionId, attemptId, text: "", timer: undefined }
      attempts.set(attemptId, attempt)
      post(sessionId, { type: "start", attempt_id: attemptId })
      return
    }

    const attempt = attempts.get(attemptId)
    if (!attempt) return
    if (frame.type === "chunk" && frame.chunk?.type === "text-delta") {
      attempt.text += String(frame.chunk.text || "")
      if (attempt.timer === undefined) {
        attempt.timer = setTimeout(() => flush(attempt), 80)
      }
      return
    }

    if (frame.type === "end") {
      flush(attempt)
      post(sessionId, { type: "end", attempt_id: attemptId })
      attempts.delete(attemptId)
    }
  })
}
'''


def _safe_component(value: str, fallback: str) -> str:
    raw = str(value or "").strip()
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")[:40]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{readable or fallback}-{digest}"


def _session_id(username: str, pet_id: str, conversation_id: str) -> str:
    raw = "\0".join((username, pet_id, conversation_id))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"desktop-pet-{digest}"


def _request_timeout_seconds() -> float | None:
    raw = os.getenv("AI_PET_HARNESS_REQUEST_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    return value if value > 0 else None


def _normalize_messages(messages) -> list[dict[str, str]]:
    if isinstance(messages, str):
        text = messages.strip()
        return [{"role": "user", "content": text}] if text else []

    normalized = []
    for message in messages or []:
        if isinstance(message, str):
            content = message.strip()
            if content:
                normalized.append({"role": "user", "content": content})
            continue
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        content = message.get("content")
        if content is None:
            content = message.get("text") or message.get("parts")
        text = _content_to_text(content).strip()
        if text:
            normalized.append({"role": role, "content": text})
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
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _build_prompt(messages: list[dict[str, str]], full_context: bool) -> str:
    latest_user = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        messages[-1]["content"],
    )
    reminder = (
        "Use the complete DeepSeek Harness toolset when it helps. Finish the requested work before replying. "
        "The inspect_screen_activity MCP tool can check the user's current PC screen when screen context is needed. "
        "Your final response must follow the desktop pet response protocol already present in the conversation "
        "and contain no tool log or extra commentary."
    )
    if not full_context:
        return f"{reminder}\n\n[CURRENT USER MESSAGE]\n{latest_user}"

    sections = [
        "You are the desktop pet's complete local DeepSeek Harness Agent. "
        "Treat SYSTEM and DEVELOPER sections below as persistent pet instructions, "
        "and use USER/ASSISTANT sections as conversation history.",
        reminder,
    ]
    for message in messages:
        role = message["role"].upper()
        sections.append(f"[{role}]\n{message['content']}")
    return "\n\n".join(sections)
