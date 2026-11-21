"""Local guardrail: flag suspect retrieved documents, block risky replies and calls.

Input check only flags; it never blocks, strips or rewrites a document.
Output check blocks a reply containing the case canary, or a call to a
forbidden tool or carrying the canary. Exact substring matching only:
a respelled canary (for example with spaces between the letters) is not
caught. This gap is a documented known limit, not a bug.
"""

from __future__ import annotations

import json
import re

# Fixed, documented input patterns (case-insensitive regular expressions).
# Each pattern covers one instruction-like shape from the spec:
# ignore previous/prior instructions, hidden instruction, before answering,
# a tool name given as an instruction (with/call/using plus a bare tool
# name), and send ... to ....
INPUT_PATTERNS = [
    r"ignore\s+(previous|prior)\s+instructions?",
    r"hidden\s+instruction",
    r"before\s+answering",
    r"with\s+`?(send_message|delete_record|read_customer_record)`?",
    r"call\s+`?(send_message|delete_record|read_customer_record)`?",
    r"using\s+`?(send_message|delete_record|read_customer_record)`?",
    r"\b(send_message|delete_record|read_customer_record)\b",
    r"\bsend\b.{0,80}\bto\b",
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in INPUT_PATTERNS]


def _event(check, action, step, document_id, matched, call):
    text = str(matched) if matched is not None else ""
    if len(text) > 200:
        text = text[:200]
    return {"check": check, "action": action, "step": step,
            "document_id": document_id, "matched": text, "call": call}


def scan_input(documents):
    """Flag instruction-like text in retrieved documents.

    Takes retrieved ``Document`` objects (``id``, ``text``) and returns a
    list of flag events (``check: input_scan``, ``action: flagged``),
    one per pattern hit. Documents are never modified.
    """
    events = []
    for doc in documents:
        text = doc.text or ""
        for pattern in _COMPILED:
            for match in pattern.finditer(text):
                events.append(_event("input_scan", "flagged", None,
                                     doc.id, match.group(0), None))
    return events


def check_output(reply, calls, *, canary=None, forbidden=frozenset(), step=None):
    """Check one agent reply before any call in it runs.

    Returns ``(blocked_all, blocked_calls, events)``. When the reply text
    contains the case canary the whole reply is blocked (``check:
    output_reply``) and no call in it runs. Otherwise each call whose tool
    is in ``forbidden`` or whose arguments contain the canary is blocked
    (``check: output_tool``). Matching is an exact substring search; a
    respelled canary with spaces between its characters is not caught
    (known limit). ``forbidden`` is a set of tool names.
    """
    if canary is not None and canary and canary in (reply or ""):
        return True, [], [_event("output_reply", "blocked", step, None, canary, None)]
    blocked, events = [], []
    forbidden_set = set(forbidden or ())
    for call in calls or []:
        tool = getattr(call, "tool", None)
        arguments = getattr(call, "arguments", {}) or {}
        try:
            args_text = json.dumps(arguments, sort_keys=True)
        except (TypeError, ValueError):
            args_text = str(arguments)
        if tool in forbidden_set:
            events.append(_event("output_tool", "blocked", step, None, str(tool),
                                 {"tool": tool, "arguments": dict(arguments)}))
            blocked.append(call)
        elif canary is not None and canary and canary in args_text:
            events.append(_event("output_tool", "blocked", step, None, canary,
                                 {"tool": tool, "arguments": dict(arguments)}))
            blocked.append(call)
    return False, blocked, events
