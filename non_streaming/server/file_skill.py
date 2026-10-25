from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Callable


_file_access_request_var: ContextVar[
    Callable[[dict[str, Any]], dict[str, Any]] | None
] = ContextVar("file_access_request", default=None)
_file_access_audit_var: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "file_access_audit",
    default=None,
)


def set_file_access_request(
    requester: Callable[[dict[str, Any]], dict[str, Any]] | None,
):
    return _file_access_request_var.set(requester)


def reset_file_access_request(token) -> None:
    _file_access_request_var.reset(token)


def set_file_access_audit(audit: list[dict[str, Any]] | None):
    return _file_access_audit_var.set(audit)


def reset_file_access_audit(token) -> None:
    _file_access_audit_var.reset(token)


def _record_file_access(operation: str, args: dict[str, Any] | None, response: dict[str, Any]) -> None:
    audit = _file_access_audit_var.get()
    if audit is None:
        return

    entry: dict[str, Any] = {"operation": operation}
    args = args or {}
    if operation in {"read_user_file", "list_user_files"}:
        entry["path"] = response.get("path") or args.get("path") or args.get("directory") or ""
    elif operation == "search_user_files":
        entry["query"] = args.get("query") or ""
        entry["count"] = response.get("count")
    elif operation == "list_authorized_dirs":
        dirs = response.get("authorized_dirs")
        entry["count"] = len(dirs) if isinstance(dirs, list) else None
    if response.get("error"):
        entry["error"] = str(response.get("error"))[:200]
    audit.append(entry)


def _request_client_file_access(operation: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    requester = _file_access_request_var.get()
    if requester is None:
        response = {
            "error": "No desktop client file source is available for this user.",
            "hint": "Start the desktop client and configure authorized file directories.",
        }
        _record_file_access(operation, args, response)
        return response

    response = requester({"operation": operation, "args": args or {}})
    if not isinstance(response, dict):
        response = {"error": "Client file source returned an invalid response."}
    _record_file_access(operation, args, response)
    return response


def list_authorized_dirs() -> str:
    return json.dumps(
        _request_client_file_access("list_authorized_dirs"),
        ensure_ascii=False,
    )


def list_user_files(
    directory: str = "",
    depth: int = 2,
    limit: int = 100,
    extensions: list[str] | None = None,
) -> str:
    return json.dumps(
        _request_client_file_access(
            "list_user_files",
            {
                "directory": directory,
                "depth": depth,
                "limit": limit,
                "extensions": extensions or [],
            },
        ),
        ensure_ascii=False,
    )


def search_user_files(
    query: str,
    directory: str = "",
    extensions: list[str] | None = None,
    limit: int = 20,
    max_files_scanned: int = 5000,
) -> str:
    return json.dumps(
        _request_client_file_access(
            "search_user_files",
            {
                "query": query,
                "directory": directory,
                "extensions": extensions or [],
                "limit": limit,
                "max_files_scanned": max_files_scanned,
            },
        ),
        ensure_ascii=False,
    )


def read_user_file(
    path: str,
    start: int = 0,
    max_chars: int = 12000,
) -> str:
    return json.dumps(
        _request_client_file_access(
            "read_user_file",
            {
                "path": path,
                "start": start,
                "max_chars": max_chars,
            },
        ),
        ensure_ascii=False,
    )
