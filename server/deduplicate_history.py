"""Remove repeated complete sessions, preserving distinct messages and wait.

Inspect with no arguments; --apply backs up originals before replacing files.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

try:
    from .migrate_history_protocol import apply_plan, encode_sessions, read_rows, source_id
except ImportError:
    from migrate_history_protocol import apply_plan, encode_sessions, read_rows, source_id


def split_sessions(rows, kind):
    sessions, current = [], None
    for row in rows:
        event_type = row.get("type")
        if event_type == f"{kind}_session_start":
            if current is not None:
                raise ValueError("Nested history session")
            current = [row]
        elif current is None:
            raise ValueError("History has messages outside a session")
        elif event_type == f"{kind}_session_end":
            current.append(row)
            sessions.append(current)
            current = None
        elif event_type == f"{kind}_message":
            current.append(row)
        else:
            raise ValueError("Unexpected history event")
        if not isinstance(row.get("messages"), list):
            raise ValueError("Expected history message array")
    if current is not None:
        raise ValueError("Unfinished history session")
    return sessions


def duplicate_key(session, kind):
    messages = [message for row in session for message in row["messages"]]
    started = session[0].get("started_at")
    # A missing timestamp cannot establish that two similar dialogues are the
    # same occurrence. Retain the complete boundary metadata in that case.
    if not isinstance(started, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", started):
        boundary = [session[0], session[-1]]
    else:
        boundary = started
    key = [boundary, messages]
    if kind == "summary":
        # Identical summaries can legitimately describe different sources.
        key.extend([session[-1].get("summary_source_id"), session[-1].get("action")])
    return json.dumps(key, ensure_ascii=False, sort_keys=True)


def deduplicate(sessions, kind):
    kept, positions, aliases = [], {}, {}
    removed = []
    for index, original in enumerate(sessions):
        session = copy.deepcopy(original)
        key = duplicate_key(session, kind)
        if key not in positions:
            positions[key] = len(kept)
            kept.append(session)
            continue
        retained = kept[positions[key]]
        # Keep the first occurrence and its exact boundaries. Prefer an
        # existing richer tag set, without inventing or accumulating labels.
        tags = session[-1].get("tags", [])
        if len(tags) > len(retained[-1].get("tags", [])):
            retained[-1]["tags"] = copy.deepcopy(tags)
        if kind == "dialogue":
            canonical = retained[-1].get("summary_source_id") or source_id(retained)
            for identifier in (session[-1].get("summary_source_id"), source_id(session)):
                if identifier and identifier != canonical:
                    aliases[identifier] = canonical
        removed.append({"session_index": index, "kept_index": positions[key],
                        "started_at": session[0].get("started_at"),
                        "source_id": session[-1].get("summary_source_id")})
    return kept, aliases, removed


def build_plan(root):
    paths = sorted((root / "server/users_data").rglob("chat_history.txt"))
    paths += sorted((root / "client/client_data/pets").glob("*/config/dialogue_history.txt"))
    paths += sorted((root / "server/users_data").rglob("summary_history.txt"))
    files, aliases_by_parent = [], {}
    for path in paths:
        kind = "summary" if path.name == "summary_history.txt" else "dialogue"
        before = path.read_bytes()
        rows = read_rows(before)
        sessions = split_sessions(rows, kind)
        relinked = 0
        if kind == "summary":
            aliases = aliases_by_parent.get(path.parent, {})
            for session in sessions:
                end = session[-1]
                identifier = end.get("summary_source_id")
                if identifier in aliases:
                    end["summary_source_id"] = aliases[identifier]
                    relinked += 1
        kept, aliases, removed = deduplicate(sessions, kind)
        if kind == "dialogue":
            aliases_by_parent[path.parent] = aliases
        files.append({"path": path, "before": before, "after": encode_sessions(kept),
                      "sessions_before": len(sessions), "sessions_after": len(kept),
                      "removed": removed, "relinked_summaries": relinked})
    plan = [file for file in files if read_rows(file["before"]) != read_rows(file["after"])]
    report = [{"file": str(file["path"].relative_to(root)),
               **{key: file[key] for key in ("sessions_before", "sessions_after", "relinked_summaries")},
               "removed": len(file["removed"])} for file in files]
    return plan, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan, report = build_plan(root)
    backup = apply_plan(root, plan) if args.apply else None
    result = {"apply": args.apply, "backup": str(backup) if backup else None, "files": report}
    if backup:
        (backup / "deduplication.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
