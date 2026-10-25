"""Request desktop conversation actions without executing client code here."""

import json
from contextvars import ContextVar


_conversation_action_request_var = ContextVar("conversation_action_request", default=None)


def set_conversation_action_request(requester):
    return _conversation_action_request_var.set(requester)


def reset_conversation_action_request(token):
    _conversation_action_request_var.reset(token)


def _request_action(action):
    requester = _conversation_action_request_var.get()
    if requester is None:
        return json.dumps({"ok": False, "error": "Client conversation actions are unavailable."})
    result = requester({"action": action})
    if not isinstance(result, dict):
        return json.dumps({"ok": False, "error": "Invalid client conversation action response."})
    if result.get("error") or result.get("ok") is not True:
        return json.dumps({"ok": False, "error": result.get("error") or "Client did not accept the action."}, ensure_ascii=False)
    # The client's save_log_async only starts a worker; it does not report
    # whether persistence, automatic tagging, or application shutdown succeeded.
    return json.dumps({
        "ok": True, "status": "scheduled", "action": action,
        "is_exit": action == "conversation-over",
    })


def save_conversation():
    """Ask the client to call save_log_async(False)."""
    return _request_action("save")


def end_conversation():
    """Ask the client to call save_log_async(True)."""
    return _request_action("conversation-over")
