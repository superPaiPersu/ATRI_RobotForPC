"""Local-only RPC between the live desktop pet and Harness MCP.

No Qt imports and no server round trip. Callbacks execute on an RPC worker
thread, so they must only read Python data and must not manipulate widgets.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def bridge_registry_dir():
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    return root / "client_data" / "harness_bridge"


def _descriptor_path(registry_dir, username, pet_id):
    if not username or not pet_id:
        raise ValueError("username and pet_id are required for the local Harness bridge")
    identity = json.dumps([str(username), str(pet_id)], ensure_ascii=False)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path(registry_dir) / (digest + ".json")


def _concept_tags(value):
    if not isinstance(value, list) or not value or len(value) > 30:
        raise ValueError("tags must contain 1 to 30 concept strings")
    if any(not isinstance(tag, str) or not tag.strip() or len(tag) > 100 or
           any(ord(char) < 32 for char in tag) for tag in value):
        raise ValueError("each tag must be a non-empty short concept string")
    return list(dict.fromkeys(value))


def live_dialogue_tags(callback, fallback):
    """Use the owner's current registry and indexes, rather than a bridge-start snapshot."""
    owner = getattr(callback, "__self__", None)
    labels = list(getattr(owner, "tags_lt", fallback))
    for tags in list(getattr(owner, "tags_dict", {}).values()):
        labels.extend(tags)
    if any(not isinstance(label, str) for label in labels):
        raise ValueError("Migrate dialogue tags to the flat concept protocol first")
    return list(dict.fromkeys(labels))


def merge_saved_history(callback, text, vocabulary):
    """Update only live retrieval indexes after an acknowledged save; leave prompts untouched."""
    owner = getattr(callback, "__self__", None)
    if not isinstance(getattr(owner, "tags_dict", None), dict) or not isinstance(getattr(owner, "tags_index_dict", None), dict):
        return
    sessions, messages, tags = [], None, []
    for line in text.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind == "dialogue_session_start":
            if messages is not None:
                raise ValueError("Nested saved history session")
            messages = [{"role": "developer", "content": "【历史会话信息】一段历史对话开始于 {}，记录标记：{}。".format(
                event.get("started_at"), event.get("marker") or event.get("source") or "",
            )}]
        elif messages is None:
            raise ValueError("Saved history session start missing")
        elif kind == "dialogue_message":
            for message in event.get("messages") or []:
                content = message.get("content")
                if content is None:
                    content = message.get("cn") or message.get("text")
                if str(content or "").strip():
                    role = message.get("role") or "user"
                    messages.append({"role": "assistant" if role == "model" else role, "content": str(content).strip()})
        elif kind == "dialogue_session_end":
            messages.append({"role": "developer", "content": "【历史会话信息】上一段历史对话结束于 {}。".format(event.get("ended_at"))})
            tags = event.get("tags") or []
            if tags:
                _concept_tags(tags)
            sessions.append((messages, tags))
            messages = None
        else:
            raise ValueError("Unsupported saved history event")
    if messages is not None:
        raise ValueError("Incomplete saved history session")
    if not isinstance(vocabulary, list) or any(not isinstance(tag, str) for tag in vocabulary):
        raise ValueError("Invalid concept vocabulary")
    for messages, tags in sessions:
        index = next((index for index, old in list(owner.tags_index_dict.items())
                      if [item for item in old if item is not None] == messages), None)
        if index is None:
            index = max((i for i in owner.tags_index_dict if type(i) is int), default=-1) + 1
        owner.tags_index_dict[index] = messages
        owner.tags_dict[index] = tags
    owner.tags_lt = list(vocabulary)


def run_history_search(callback, tags):
    """Resolve live tags_match session IDs to at most five positive matches.

    Message-list callbacks remain supported. Ranking is owned by the callback;
    this adapter only resolves its selected IDs and removes zero-score entries.
    """
    tags = _concept_tags(tags)
    owner = getattr(callback, "__self__", None)
    tag_index = getattr(owner, "tags_dict", None)
    history_index = getattr(owner, "tags_index_dict", None)
    indexed = isinstance(tag_index, dict) and isinstance(history_index, dict)
    scores = {}
    if indexed:
        targets = set(tags)
        for index, labels in list(tag_index.items()):
            scores[index] = len(targets.intersection(labels))

    try:
        result = callback(tags)
    except KeyError as exc:
        if exc.args != (None,):
            raise
        result = []

    matched = None
    if isinstance(result, dict):
        if result.get("error"):
            raise RuntimeError("History search callback failed")
        result, matched = result.get("messages"), result.get("matched")
    if result is None:
        result = []
    if not isinstance(result, list):
        raise ValueError("History search callback must return messages or session IDs")

    if result and all(type(index) in (int, str) for index in result):
        if not indexed:
            raise ValueError("Session IDs require a bound callback with history indexes")
        messages, seen = [], set()
        for index in result:
            if index in seen:
                continue
            if index not in scores:
                raise ValueError("Unknown history session ID")
            if scores[index] <= 0:
                continue
            seen.add(index)
            if len(seen) > 5:
                break
            session = history_index[index]
            if not isinstance(session, list):
                raise ValueError("History session must contain a message list")
            # The template parser may leave None for an empty legacy event.
            messages.extend(message for message in list(session) if message is not None)
    else:
        messages = result
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("History search callback must return valid message objects")
    if matched is False or not any(message.get("role") in ("user", "assistant") for message in messages):
        messages = []
    return {"messages": messages, "matched": bool(messages)}


class PetHarnessBridge:
    def __init__(self, search_callback, tags_callback, *, username, pet_id, registry_dir=None):
        self.search_callback = search_callback
        self.tags_callback = tags_callback
        self.path = _descriptor_path(registry_dir or bridge_registry_dir(), username, pet_id)
        self._token = secrets.token_urlsafe(32)
        self._callback_lock = threading.RLock()
        self._server = None
        self._thread = None

    def start(self):
        if self._server is not None:
            return self
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.connection.settimeout(15)
                if self.path != "/rpc":
                    self.send_error(404)
                    return
                credential = self.headers.get("Authorization", "").encode("utf-8")
                if not hmac.compare_digest(credential, ("Bearer " + bridge._token).encode("utf-8")):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > MAX_REQUEST_BYTES:
                        raise ValueError("Invalid request length")
                    request = json.loads(self.rfile.read(length))
                    if not isinstance(request, dict):
                        raise ValueError("RPC request must be an object")
                    with bridge._callback_lock:
                        if request.get("method") == "get_dialogue_tags":
                            result = live_dialogue_tags(bridge.search_callback, bridge.tags_callback())
                        elif request.get("method") == "search_dialogue_history":
                            tags = _concept_tags(request.get("tags"))
                            result = run_history_search(bridge.search_callback, tags)
                        else:
                            raise ValueError("Unknown local RPC method")
                    response = {"ok": True, "result": result}
                    body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                except Exception as exc:
                    # Do not leak exception bodies containing local data.
                    response = {"ok": False, "error": "Local history callback failed: " + type(exc).__name__}
                    body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                if len(body) > MAX_RESPONSE_BYTES:
                    body = b'{"ok":false,"error":"History result exceeds 2 MiB"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        temporary_path = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as output:
                temporary_path = Path(output.name)
                json.dump({"port": server.server_port, "token": self._token}, output)
            temporary_path.replace(self.path)
            self._server = server
            self._thread = threading.Thread(target=server.serve_forever, daemon=True, name="pet-harness-bridge")
            self._thread.start()
        except Exception:
            server.server_close()
            self._server = None
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return self

    def close(self):
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
            self._thread.join(timeout=2)
        try:
            descriptor = json.loads(self.path.read_text(encoding="utf-8"))
            if descriptor.get("token") == self._token:
                self.path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def call_pet_bridge(registry_dir, username, pet_id, method, *, tags=None, timeout=15):
    """Read discovery on each call so a restarted pet is found automatically."""
    path = _descriptor_path(registry_dir, username, pet_id)
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        port = descriptor["port"]
        token = descriptor["token"]
        if type(port) is not int or not 0 < port < 65536 or not isinstance(token, str):
            raise ValueError("Invalid descriptor")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Desktop pet history bridge is not ready; start it in template main.py") from exc
    payload = {"method": method}
    if tags is not None:
        payload["tags"] = _concept_tags(tags)
    request = Request(
        "http://127.0.0.1:" + str(port) + "/rpc",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        # Loopback RPC must bypass system/VPN HTTP proxy environment settings.
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Local history response is too large")
        data = json.loads(raw)
    except (HTTPError, URLError, OSError, ValueError) as exc:
        raise RuntimeError("Cannot reach desktop pet history bridge") from exc
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "Local history request failed")
    return data["result"]
