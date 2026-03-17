"""Batch conversion of persisted LangChain messages to AI SDK UI message dicts.

This is the persistence counterpart to ``to_ui_message_stream``: while the
stream adapter converts live LangGraph events, this module converts a list
of LangChain ``BaseMessage`` objects (as stored in LangGraph checkpoints)
into the UI message format expected by Vercel AI SDK frontends.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .utils import _extract_structured_content


def to_ui_messages(lc_messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert persisted LangChain messages to AI SDK UI message dicts.

    Groups consecutive ``AIMessage`` + ``ToolMessage`` sequences into single
    assistant messages, and converts content blocks (text, reasoning, tool
    invocations, files) into AI SDK v5 ``parts``.

    Returns a list of dicts, each with ``role`` and ``parts`` keys.
    """
    ui_messages: list[dict[str, Any]] = []
    i = 0

    while i < len(lc_messages):
        msg = lc_messages[i]

        if isinstance(msg, HumanMessage):
            parts = _convert_human_parts(msg)
            if parts:
                ui_messages.append({"role": "user", "parts": parts})
            i += 1

        elif isinstance(msg, AIMessage):
            parts: list[dict[str, Any]] = []

            # Group consecutive AIMessage + ToolMessage into one assistant message
            while i < len(lc_messages) and isinstance(lc_messages[i], (AIMessage, ToolMessage)):
                current = lc_messages[i]

                if isinstance(current, AIMessage):
                    parts.extend(_convert_ai_content_parts(current))

                    if current.tool_calls:
                        tool_results = _collect_tool_results(lc_messages, i + 1)
                        for tc in current.tool_calls:
                            tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                            tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                            tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            parts.append(_build_tool_invocation_part(tc_id, tc_name, tc_args, tool_results.get(tc_id)))

                # ToolMessages are consumed by the look-ahead above
                i += 1

            if parts:
                ui_messages.append({"role": "assistant", "parts": parts})

        elif isinstance(msg, SystemMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content:
                ui_messages.append({"role": "system", "parts": [{"type": "text", "text": content}]})
            i += 1

        else:
            i += 1

    return ui_messages


# ── Human message conversion ─────────────────────────────────────────────


def _convert_human_parts(msg: HumanMessage) -> list[dict[str, Any]]:
    """Convert HumanMessage content to UI parts."""
    parts: list[dict[str, Any]] = []

    if isinstance(msg.content, str):
        if msg.content:
            parts.append({"type": "text", "text": msg.content})
    elif isinstance(msg.content, list):
        file_parts: list[dict[str, Any]] = []
        text_parts: list[dict[str, Any]] = []

        for item in msg.content:
            if isinstance(item, str):
                text_parts.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                converted = _convert_human_content_block(item)
                if converted:
                    if converted["type"] == "text":
                        text_parts.append(converted)
                    else:
                        file_parts.append(converted)

        # Files first, then text (matches AI SDK streaming order)
        parts.extend(file_parts)
        parts.extend(text_parts)

    return parts


def _convert_human_content_block(item: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single content block from a HumanMessage to a UI part."""
    item_type = item.get("type", "")

    if item_type == "text":
        text = item.get("text", "")
        return {"type": "text", "text": text} if text else None

    if item_type == "image_url":
        url = item.get("image_url", {}).get("url", "")
        media_type = _infer_media_type_from_data_url(url, "image/png")
        return {"type": "file", "url": url, "mediaType": media_type}

    if item_type in ("image", "file"):
        source = item.get("source")
        if isinstance(source, dict):
            media_type = source.get("media_type", "")
            data = source.get("data", "")
            filename = item.get("filename", "")
            url = f"data:{media_type};base64,{data}" if source.get("type") == "base64" and data else data
            part: dict[str, Any] = {"type": "file", "url": url, "mediaType": media_type}
            if filename:
                part["filename"] = filename
            return part

        # LangChain file format: {"type": "file", "data": "...", "mimeType": "..."}
        data = item.get("data", "")
        mime_type = item.get("mimeType", "")
        filename = item.get("filename", "")
        url = item.get("url", "")
        if data and not url:
            if data.startswith(("http://", "https://", "data:")):
                url = data
            else:
                url = f"data:{mime_type};base64,{data}" if mime_type else data
        part = {"type": "file", "url": url, "mediaType": mime_type}
        if filename:
            part["filename"] = filename
        return part

    return None


# ── AI message conversion ────────────────────────────────────────────────


def _convert_ai_content_parts(msg: AIMessage) -> list[dict[str, Any]]:
    """Extract text and reasoning parts from an AIMessage."""
    parts: list[dict[str, Any]] = []

    if isinstance(msg.content, str):
        if msg.content:
            parts.append({"type": "text", "text": msg.content})
    elif isinstance(msg.content, list):
        for item in msg.content:
            if isinstance(item, dict):
                if item.get("type") == "thinking" and "thinking" in item:
                    parts.append({"type": "reasoning", "text": item["thinking"]})
                elif item.get("type") == "text" and item.get("text"):
                    parts.append({"type": "text", "text": item["text"]})
            elif isinstance(item, str):
                parts.append({"type": "text", "text": item})

    return parts


# ── Tool result collection ───────────────────────────────────────────────


def _collect_tool_results(messages: list[BaseMessage], start: int) -> dict[str, dict[str, Any]]:
    """Look ahead from *start* to collect ToolMessage results keyed by tool_call_id."""
    results: dict[str, dict[str, Any]] = {}
    j = start
    while j < len(messages) and isinstance(messages[j], ToolMessage):
        tool_msg: ToolMessage = messages[j]
        tc_id = getattr(tool_msg, "tool_call_id", None)
        if tc_id:
            sc = _extract_structured_content(tool_msg)
            results[tc_id] = {
                "content": tool_msg.content,
                "status": getattr(tool_msg, "status", "success"),
                "structured_content": sc,
            }
        j += 1
    return results


def _normalize_tool_result(content: Any) -> Any:
    """Normalize MCP tool content to a clean JSON value or string.

    MCP tools via ``langchain-mcp-adapters`` return content as a list of
    content parts like ``[{"type": "text", "text": "{...}"}]``.  This
    extracts the actual value.
    """
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            try:
                return json.loads(first["text"])
            except (json.JSONDecodeError, TypeError):
                return first.get("text", first)
        return first
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content
    return content


def _build_tool_invocation_part(
    tc_id: str,
    tc_name: str,
    tc_args: Any,
    result_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a ``tool-invocation`` UI part dict."""
    part: dict[str, Any] = {
        "type": "tool-invocation",
        "toolInvocationId": tc_id,
        "toolName": tc_name,
        "args": tc_args,
    }

    if result_info is None:
        part["state"] = "call"
        return part

    content = result_info["content"]
    status = result_info.get("status", "success")
    sc = result_info.get("structured_content")
    normalized = _normalize_tool_result(content)

    if status == "error":
        part["state"] = "error"
        part["error"] = str(normalized) if normalized else "Tool execution failed"
    else:
        part["state"] = "result"
        if sc is not None:
            part["result"] = {"_text": normalized, "structuredContent": sc}
        else:
            part["result"] = normalized

    return part


# ── Helpers ──────────────────────────────────────────────────────────────


def _infer_media_type_from_data_url(url: str, default: str = "") -> str:
    """Infer media type from a ``data:`` URL, or return *default*."""
    if url.startswith("data:"):
        # data:image/png;base64,...
        prefix = url.split(";", 1)[0]
        return prefix.replace("data:", "", 1)
    return default
