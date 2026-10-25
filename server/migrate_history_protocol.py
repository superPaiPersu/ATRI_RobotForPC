"""Back up and migrate persisted JSONL history without model requests.

Run without arguments to inspect; use --apply to write the planned changes.
Old messages (including wait), their order, and existing summaries are preserved.
Missing semantic tags are left empty; unknown times are never invented.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


UNKNOWN_TIME = "未知（旧历史未记录）"
_LEGACY_START = re.compile(
    r"^(?:\[系统消息\][:：]|【历史会话信息】)\s*一段历史对话开始于 "
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})，记录标记[:：](?P<marker>.*?)[。.]?$"
)
_LEGACY_END = re.compile(
    r"^(?:\[系统消息\][:：]|【历史会话信息】)\s*上一段历史对话结束于 "
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[。.]?$"
)


def source_id(rows, legacy_summary=False):
    canonical = [{k: v for k, v in row.items() if k not in ("tags", "summary_source_id")} for row in rows]
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if legacy_summary:
        encoded = b"legacy-summary\0" + encoded
    return hashlib.sha256(encoded).hexdigest()


def read_rows(data):
    rows = []
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number}: expected a JSON object")
        rows.append(row)
    return rows


def convert_legacy_boundaries(rows, kind):
    """Recover explicit old timestamps, retaining each original marker as metadata."""
    converted = []
    legacy_open = False
    for original in rows:
        row = original
        messages = row.get("messages", [])
        if row.get("type") == f"{kind}_message" and len(messages) == 1 and isinstance(messages[0], dict):
            text = str(messages[0].get("content") or "")
            start, end = _LEGACY_START.fullmatch(text), _LEGACY_END.fullmatch(text)
            if start:
                row = {"type": f"{kind}_session_start", "started_at": start["time"],
                       "marker": start["marker"], "messages": [], "legacy_event": copy.deepcopy(original)}
            elif end:
                row = {"type": f"{kind}_session_end", "ended_at": end["time"],
                       "messages": [], "legacy_event": copy.deepcopy(original)}
        if row.get("type") == f"{kind}_session_start":
            if legacy_open:
                converted.append({"type": f"{kind}_session_end", "ended_at": UNKNOWN_TIME,
                                  "messages": [], "legacy_missing_boundary": True})
            legacy_open = "legacy_event" in row
        elif row.get("type") == f"{kind}_session_end":
            legacy_open = False
        converted.append(row)
    if legacy_open:
        converted.append({"type": f"{kind}_session_end", "ended_at": UNKNOWN_TIME,
                          "messages": [], "legacy_missing_boundary": True})
    return converted


def group_sessions(rows, kind):
    """Wrap consecutive unbounded legacy messages; reject broken boundaries."""
    sessions, loose, current = [], [], None

    def flush_loose():
        if loose:
            sessions.append([
                {"type": f"{kind}_session_start", "started_at": UNKNOWN_TIME,
                 "marker": "legacy-history-unknown-boundaries", "messages": []},
                *loose,
                {"type": f"{kind}_session_end", "ended_at": UNKNOWN_TIME, "messages": []},
            ])
            loose.clear()

    for original in convert_legacy_boundaries(rows, kind):
        row = copy.deepcopy(original)
        event_type = row.get("type")
        if event_type not in (f"{kind}_session_start", f"{kind}_message", f"{kind}_session_end"):
            raise ValueError(f"Unsupported {kind} event: {event_type}")
        if not isinstance(row.get("messages", []), list):
            raise ValueError("messages must be an array")
        row.setdefault("messages", [])
        if event_type == f"{kind}_session_start":
            if current is not None:
                raise ValueError("Nested session start")
            flush_loose()
            row["started_at"] = row.get("started_at") or UNKNOWN_TIME
            row.setdefault("marker", row.get("source") or "legacy-history")
            current = [row]
        elif event_type == f"{kind}_session_end":
            if current is None:
                raise ValueError("Session end without start")
            row["ended_at"] = row.get("ended_at") or UNKNOWN_TIME
            current.append(row)
            sessions.append(current)
            current = None
        else:
            if any(not isinstance(message, dict) for message in row["messages"]):
                raise ValueError("Each message must be an object")
            (current if current is not None else loose).append(row)
    if current is not None:
        raise ValueError("Unfinished session; refusing to invent an end")
    flush_loose()
    return sessions


def normalize_sessions(rows, kind, dialogue_sessions=()):
    sessions = group_sessions(rows, kind)
    by_start = defaultdict(list)
    for session in dialogue_sessions:
        if session[0]["started_at"] != UNKNOWN_TIME:
            by_start[session[0]["started_at"]].append(session)
    for session in sessions:
        end = session[-1]
        end.setdefault("tags", [])
        if not isinstance(end["tags"], list):
            raise ValueError("tags must be an array")
        if kind == "dialogue":
            expected = source_id(session)
            if end.get("summary_source_id") not in (None, "", expected):
                raise ValueError("Existing dialogue source ID differs from its content")
            end["summary_source_id"] = expected
        else:
            end.setdefault("action", "summary" if any(
                str(message.get("content") or "").strip()
                for row in session for message in row.get("messages", [])
            ) else "no_sense")
            if end["action"] not in ("summary", "no_sense"):
                raise ValueError("Invalid summary action")
            if not end.get("summary_source_id"):
                matches = by_start[session[0]["started_at"]]
                if len(matches) == 1:
                    end["summary_source_id"] = matches[0][-1]["summary_source_id"]
                else:
                    # Aggregated old summaries may span many dialogues. Do not
                    # falsely associate one with an arbitrary source session.
                    end["legacy_unlinked"] = True
                    end["summary_source_id"] = source_id(session, legacy_summary=True)
    return sessions


def encode_sessions(sessions):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for session in sessions for row in session).encode("utf-8")


def build_plan(root):
    paths = sorted((root / "server/users_data").rglob("chat_history.txt"))
    paths += sorted((root / "server/users_data").rglob("summary_history.txt"))
    paths += sorted((root / "client/client_data/pets").glob("*/config/dialogue_history.txt"))
    dialogues, plan = {}, []
    for path in paths:
        kind = "summary" if path.name == "summary_history.txt" else "dialogue"
        original = path.read_bytes()
        rows = read_rows(original)
        sessions = normalize_sessions(rows, kind, dialogues.get(path.parent, ()))
        if kind == "dialogue":
            dialogues[path.parent] = sessions
        changed_rows = [row for session in sessions for row in session]
        if changed_rows == rows:
            continue
        plan.append({"path": path, "before": original, "after": encode_sessions(sessions),
                     "sessions": len(sessions),
                     "unknown_time_sessions": sum(UNKNOWN_TIME in (s[0]["started_at"], s[-1]["ended_at"]) for s in sessions),
                     "unlinked_summaries": sum(s[-1].get("legacy_unlinked") is True for s in sessions)})
    return plan


def atomic_write(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_plan(root, plan):
    if not plan:
        return None
    for entry in plan:
        if entry["path"].read_bytes() != entry["before"]:
            raise RuntimeError(f"History changed during inspection: {entry['path']}")
    backup = root / ".history_protocol_backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True)
    manifest = []
    # Back up every original before changing any live file.
    for entry in plan:
        relative = entry["path"].relative_to(root)
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry["before"])
        manifest.append({"path": relative.as_posix(),
                         "before_sha256": hashlib.sha256(entry["before"]).hexdigest(),
                         "after_sha256": hashlib.sha256(entry["after"]).hexdigest()})
    (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for entry in plan:
        if entry["path"].read_bytes() != entry["before"]:
            raise RuntimeError(f"History changed before replacement; backup: {backup}")
        atomic_write(entry["path"], entry["after"])
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = build_plan(root)
    report = [{"file": str(e["path"].relative_to(root)), **{k: e[k] for k in
              ("sessions", "unknown_time_sessions", "unlinked_summaries")}} for e in plan]
    backup = apply_plan(root, plan) if args.apply else None
    print(json.dumps({"apply": args.apply, "files": report, "backup": str(backup) if backup else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
