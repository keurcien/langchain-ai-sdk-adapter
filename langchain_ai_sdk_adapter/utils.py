"""
Utility functions ported from ``@ai-sdk/langchain`` utils.ts.

Contains all the message conversion helpers and stream chunk processors.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from ._types import LangGraphEventState

# ── chunk type: UIMessageChunk is just a plain dict in Python ────────────


def _chunk(d: dict[str, Any]) -> dict[str, Any]:
    return d


# ═══════════════════════════════════════════════════════════════════════════
# Message conversion helpers  (toBaseMessages / convertModelMessages path)
# ═══════════════════════════════════════════════════════════════════════════


def is_tool_result_part(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "tool-result"


def convert_tool_result_part(block: dict[str, Any]) -> ToolMessage:
    output = block.get("output", {})
    otype = output.get("type", "")
    if otype in ("text", "error-text"):
        content = output.get("value", "")
    elif otype in ("json", "error-json"):
        content = json.dumps(output.get("value", ""), separators=(",", ":"))
    elif otype == "content":
        content = "".join(b.get("text", "") for b in output.get(
            "value", []) if b.get("type") == "text")
    else:
        content = ""
    return ToolMessage(tool_call_id=block.get("toolCallId", ""), content=content)


def convert_assistant_content(content: Any) -> AIMessage:
    if isinstance(content, str):
        return AIMessage(content=content)
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in content or []:
        if part.get("type") == "text":
            text_parts.append(part.get("text", ""))
        elif part.get("type") == "tool-call":
            tool_calls.append(
                {
                    "id": part.get("toolCallId", ""),
                    "name": part.get("toolName", ""),
                    "args": part.get("input", {}),
                }
            )
    kwargs: dict[str, Any] = {"content": "".join(text_parts)}
    if tool_calls:
        kwargs["tool_calls"] = tool_calls
    return AIMessage(**kwargs)


def _get_default_filename(media_type: str, prefix: str = "file") -> str:
    ext = media_type.split("/")[-1] if "/" in media_type else "bin"
    return f"{prefix}.{ext}"


def convert_user_content(content: Any) -> HumanMessage:
    if isinstance(content, str):
        return HumanMessage(content=content)

    blocks: list[dict[str, Any]] = []
    for part in content or []:
        ptype = part.get("type", "")
        if ptype == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif ptype == "image":
            image = part.get("image", "")
            if isinstance(image, str):
                if image.startswith(("http://", "https://", "data:")):
                    blocks.append(
                        {"type": "image_url", "image_url": {"url": image}})
                else:
                    mt = part.get("mediaType", "image/png")
                    blocks.append({"type": "image_url", "image_url": {
                                  "url": f"data:{mt};base64,{image}"}})
            else:
                blocks.append(
                    {"type": "image_url", "image_url": {"url": str(image)}})
        elif ptype == "file":
            data = part.get("data", "")
            media_type = part.get("mediaType", "")
            is_image = media_type.startswith("image/")
            if is_image:
                if isinstance(data, str):
                    if data.startswith(("http://", "https://", "data:")):
                        blocks.append(
                            {"type": "image_url", "image_url": {"url": data}})
                    else:
                        blocks.append({"type": "image_url", "image_url": {
                                      "url": f"data:{media_type};base64,{data}"}})
                else:
                    blocks.append(
                        {"type": "image_url", "image_url": {"url": str(data)}})
            else:
                filename = part.get(
                    "filename") or _get_default_filename(media_type)
                if isinstance(data, str):
                    if data.startswith(("http://", "https://")):
                        blocks.append(
                            {"type": "file", "url": data, "mimeType": media_type, "filename": filename})
                    elif data.startswith("data:"):
                        import re

                        m = re.match(r"^data:([^;]+);base64,(.+)$", data)
                        if m:
                            blocks.append(
                                {"type": "file", "data": m.group(
                                    2), "mimeType": m.group(1), "filename": filename}
                            )
                        else:
                            blocks.append(
                                {"type": "file", "url": data, "mimeType": media_type, "filename": filename})
                    else:
                        blocks.append(
                            {"type": "file", "data": data, "mimeType": media_type, "filename": filename})
                else:
                    blocks.append({"type": "file", "data": str(
                        data), "mimeType": media_type, "filename": filename})

    if blocks and all(b["type"] == "text" for b in blocks):
        return HumanMessage(content="".join(b["text"] for b in blocks))
    return HumanMessage(content=blocks)


# ═══════════════════════════════════════════════════════════════════════════
# Message introspection helpers (isAIMessageChunk, isToolMessageType etc.)
# ═══════════════════════════════════════════════════════════════════════════


def is_plain_message_object(msg: Any) -> bool:
    """True if *msg* is a plain dict, not a LangChain class instance."""
    if isinstance(msg, BaseMessage):
        return False
    return isinstance(msg, dict)


def get_message_id(msg: Any) -> str | None:
    if msg is None:
        return None
    # Class instance
    if hasattr(msg, "id") and isinstance(msg.id, str):
        return msg.id
    # dict
    if isinstance(msg, dict):
        if isinstance(msg.get("id"), str):
            return msg["id"]
        # Serialized LangChain message
        if msg.get("type") == "constructor" and isinstance(msg.get("kwargs"), dict):
            kid = msg["kwargs"].get("id")
            if isinstance(kid, str):
                return kid
    return None


def is_ai_message_chunk(msg: Any) -> bool:
    if isinstance(msg, (AIMessageChunk, AIMessage)):
        return True
    if isinstance(msg, dict):
        if msg.get("type") == "ai":
            return True
        if (
            msg.get("type") == "constructor"
            and isinstance(msg.get("id"), list)
            and ("AIMessageChunk" in msg["id"] or "AIMessage" in msg["id"])
        ):
            return True
    return False


def is_tool_message_type(msg: Any) -> bool:
    if isinstance(msg, ToolMessage):
        return True
    if isinstance(msg, dict):
        if msg.get("type") == "tool":
            return True
        if msg.get("type") == "constructor" and isinstance(msg.get("id"), list) and "ToolMessage" in msg["id"]:
            return True
    return False


def _data_source(msg: Any) -> dict[str, Any]:
    """Return the actual data dict, handling serialized constructor format."""
    if isinstance(msg, dict):
        if msg.get("type") == "constructor" and isinstance(msg.get("kwargs"), dict):
            return msg["kwargs"]
        return msg
    # For class instances, convert to a dict-like view
    result: dict[str, Any] = {}
    for attr in (
        "content",
        "id",
        "tool_calls",
        "tool_call_chunks",
        "additional_kwargs",
        "tool_call_id",
        "status",
        "response_metadata",
        "artifact",
    ):
        if hasattr(msg, attr):
            result[attr] = getattr(msg, attr)
    return result


def _extract_structured_content(msg_or_output: Any) -> dict | None:
    """Extract structuredContent from a ToolMessage artifact, if present.

    MCP servers return structuredContent alongside tool results. The
    ``langchain-mcp-adapters`` library wraps this in the ToolMessage
    artifact as ``{"structured_content": {...}}``.

    Returns the structured_content dict, or None if not present.
    """
    artifact = (
        getattr(msg_or_output, "artifact", None)
        if not isinstance(msg_or_output, dict)
        else msg_or_output.get("artifact")
    )
    if isinstance(artifact, dict):
        sc = artifact.get("structured_content")
        if isinstance(sc, dict):
            return sc
    return None


def get_message_text(msg: Any) -> str:
    if isinstance(msg, AIMessageChunk):
        return getattr(msg, "text", "") or ""
    ds = _data_source(msg)
    content = ds.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Reasoning extraction
# ═══════════════════════════════════════════════════════════════════════════


def _is_reasoning_block(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "reasoning" and isinstance(obj.get("reasoning"), str)


def _is_thinking_block(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "thinking" and isinstance(obj.get("thinking"), str)


def _is_gpt5_reasoning_output(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "reasoning" and isinstance(obj.get("summary"), list)


def _get_content_blocks(msg: Any) -> list[dict[str, Any]] | None:
    """Get content blocks from a message (class or dict)."""
    # TODO: contentBlocks doesn't exist in Python langchain — remove once confirmed unused
    if hasattr(msg, "contentBlocks"):
        return msg.contentBlocks
    ds = _data_source(msg)
    if ds.get("contentBlocks") is not None:
        return ds["contentBlocks"]
    content = getattr(msg, "content", None) if not isinstance(
        msg, dict) else ds.get("content")
    if isinstance(content, list):
        return content
    return None


def _get_additional_kwargs(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "additional_kwargs"):
        return msg.additional_kwargs or {}
    ds = _data_source(msg)
    return ds.get("additional_kwargs", {})


def _get_response_metadata(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "response_metadata"):
        return msg.response_metadata or {}
    ds = _data_source(msg)
    return ds.get("response_metadata", {})


def extract_reasoning_id(msg: Any) -> str | None:
    """Extract reasoning block ID (GPT-5 format)."""
    ak = _get_additional_kwargs(msg)
    reasoning = ak.get("reasoning", {})
    if isinstance(reasoning, dict) and reasoning.get("id"):
        return reasoning["id"]
    rm = _get_response_metadata(msg)
    for item in rm.get("output", []) if isinstance(rm.get("output"), list) else []:
        if _is_gpt5_reasoning_output(item):
            return item.get("id")
    return None


def extract_reasoning_from_content_blocks(msg: Any) -> str | None:
    """Extract reasoning from ``contentBlocks`` or ``additional_kwargs.reasoning.summary``.

    For STREAMING chunks only — does NOT look at ``response_metadata`` to avoid duplication.
    """
    cb = _get_content_blocks(msg)
    if isinstance(cb, list):
        parts = []
        for block in cb:
            if _is_reasoning_block(block):
                parts.append(block["reasoning"])
            elif _is_thinking_block(block):
                parts.append(block["thinking"])
        if parts:
            return "".join(parts)

    ak = _get_additional_kwargs(msg)
    reasoning = ak.get("reasoning", {})
    if isinstance(reasoning, dict) and isinstance(reasoning.get("summary"), list):
        parts = []
        for item in reasoning["summary"]:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    return None


def extract_reasoning_from_values_message(msg: Any) -> str | None:
    """Extract accumulated reasoning from ``response_metadata.output`` (GPT-5 final)."""
    rm = _get_response_metadata(msg)
    if isinstance(rm.get("output"), list):
        parts = []
        for item in rm["output"]:
            if _is_gpt5_reasoning_output(item):
                for si in item.get("summary", []):
                    if isinstance(si, dict) and isinstance(si.get("text"), str) and si["text"]:
                        parts.append(si["text"])
        if parts:
            return "".join(parts)

    # Fallback: additional_kwargs.reasoning.summary
    ak = _get_additional_kwargs(msg)
    reasoning = ak.get("reasoning", {})
    if isinstance(reasoning, dict) and isinstance(reasoning.get("summary"), list):
        parts = []
        for item in reasoning["summary"]:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Image generation outputs
# ═══════════════════════════════════════════════════════════════════════════


def _is_image_generation_output(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "image_generation_call"


def extract_image_outputs(additional_kwargs: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not additional_kwargs:
        return []
    tool_outputs = additional_kwargs.get("tool_outputs")
    if not isinstance(tool_outputs, list):
        return []
    return [o for o in tool_outputs if _is_image_generation_output(o)]


# ═══════════════════════════════════════════════════════════════════════════
# processModelChunk  (direct model stream)
# ═══════════════════════════════════════════════════════════════════════════


def process_model_chunk(
    chunk: Any,
    state: dict[str, Any],
    emit: list[dict[str, Any]],
) -> None:
    """Process a single ``AIMessageChunk`` from a direct model stream.

    *emit* is a list to which UIMessageChunk dicts are appended.
    """
    if state.get("emitted_images") is None:
        state["emitted_images"] = set()

    chunk_id = getattr(chunk, "id", None)
    if chunk_id:
        state["message_id"] = chunk_id

    # Image generation outputs
    ak = _get_additional_kwargs(chunk)
    for img in extract_image_outputs(ak):
        if img.get("result") and img["id"] not in state["emitted_images"]:
            state["emitted_images"].add(img["id"])
            mt = f"image/{img.get('output_format', 'png')}"
            emit.append({"type": "file", "mediaType": mt,
                        "url": f"data:{mt};base64,{img['result']}"})
            state["started"] = True

    # Reasoning
    reasoning = extract_reasoning_from_content_blocks(
        chunk) or extract_reasoning_from_values_message(chunk)
    if reasoning:
        if not state.get("reasoning_started"):
            state["reasoning_message_id"] = state["message_id"]
            emit.append({"type": "reasoning-start", "id": state["message_id"]})
            state["reasoning_started"] = True
            state["started"] = True
        emit.append(
            {
                "type": "reasoning-delta",
                "delta": reasoning,
                "id": state.get("reasoning_message_id") or state["message_id"],
            }
        )

    # Text content
    content = getattr(chunk, "content", None) if not isinstance(
        chunk, dict) else chunk.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(b.get("text", "") for b in content if isinstance(
            b, dict) and b.get("type") == "text")
    else:
        text = ""

    if text:
        if state.get("reasoning_started") and not state.get("text_started"):
            emit.append(
                {
                    "type": "reasoning-end",
                    "id": state.get("reasoning_message_id") or state["message_id"],
                }
            )
            state["reasoning_started"] = False
        if not state.get("text_started"):
            state["text_message_id"] = state["message_id"]
            emit.append({"type": "text-start", "id": state["message_id"]})
            state["text_started"] = True
            state["started"] = True
        emit.append(
            {
                "type": "text-delta",
                "delta": text,
                "id": state.get("text_message_id") or state["message_id"],
            }
        )


# ═══════════════════════════════════════════════════════════════════════════
# processLangGraphEvent
# ═══════════════════════════════════════════════════════════════════════════


def process_langgraph_event(
    etype: str,
    data: Any,
    state: LangGraphEventState,
    emit: list[dict[str, Any]],
    interrupts: tuple[Any, ...] | None = None,
) -> None:
    """Process one LangGraph v2 StreamPart and append chunks to *emit*.

    *interrupts* — native interrupts from ``ValuesStreamPart.interrupts``.
    """

    if etype == "custom":
        custom_type = "custom"
        part_id = None
        if isinstance(data, dict):
            if isinstance(data.get("type"), str) and data["type"]:
                custom_type = data["type"]
            if isinstance(data.get("id"), str) and data["id"]:
                part_id = data["id"]
        emit.append(
            {
                "type": f"data-{custom_type}",
                "id": part_id,
                "transient": part_id is None,
                "data": data,
            }
        )
        return

    if etype == "messages":
        raw_msg, metadata = data if isinstance(
            data, (list, tuple)) and len(data) >= 2 else (data, {})
        if isinstance(data, (list, tuple)) and len(data) >= 1:
            raw_msg = data[0]
            metadata = data[1] if len(data) >= 2 else {}
        else:
            return

        msg = raw_msg
        msg_id = get_message_id(msg)
        if not msg_id:
            return

        # Step tracking
        lg_step = metadata.get("langgraph_step") if isinstance(
            metadata, dict) else None
        if isinstance(lg_step, (int, float)):
            lg_step = int(lg_step)
            if lg_step != state.current_step:
                if state.current_step is not None:
                    emit.append({"type": "finish-step"})
                emit.append({"type": "start-step"})
                state.current_step = lg_step

        # Accumulate chunks
        if isinstance(msg, AIMessageChunk):
            if msg_id in state.message_concat:
                # type: ignore[operator]
                state.message_concat[msg_id] = state.message_concat[msg_id] + msg
            else:
                state.message_concat[msg_id] = msg

        if is_ai_message_chunk(msg):
            concat_chunk = state.message_concat.get(msg_id)
            ds = _data_source(msg)

            # Image outputs
            ak = ds.get("additional_kwargs") or {}
            for img in extract_image_outputs(ak):
                if img.get("result") and img["id"] not in state.emitted_images:
                    state.emitted_images.add(img["id"])
                    mt = f"image/{img.get('output_format', 'png')}"
                    emit.append({"type": "file", "mediaType": mt,
                                "url": f"data:{mt};base64,{img['result']}"})

            # Tool call chunks
            tc_chunks = ds.get("tool_call_chunks") or (
                getattr(msg, "tool_call_chunks", None) if not isinstance(
                    msg, dict) else None
            )
            if tc_chunks:
                for tc in tc_chunks:
                    if not isinstance(tc, dict):
                        tc = {
                            "id": getattr(tc, "id", None),
                            "name": getattr(tc, "name", None),
                            "args": getattr(tc, "args", None),
                            "index": getattr(tc, "index", None),
                        }
                    idx = tc.get("index", 0) or 0

                    if tc.get("id"):
                        state.tool_call_info_by_index.setdefault(msg_id, {})[idx] = {
                            "id": tc["id"],
                            "name": tc.get("name") or _concat_tc_name(concat_chunk, idx) or "unknown",
                        }

                    tool_call_id = (
                        tc.get("id")
                        or state.tool_call_info_by_index.get(msg_id, {}).get(idx, {}).get("id")
                        or _concat_tc_id(concat_chunk, idx)
                    )
                    if not tool_call_id:
                        continue

                    tool_name = (
                        tc.get("name")
                        or state.tool_call_info_by_index.get(msg_id, {}).get(idx, {}).get("name")
                        or _concat_tc_name(concat_chunk, idx)
                        or "unknown"
                    )

                    seen_tool = state.message_seen.get(
                        msg_id, {}).get("tool", {})
                    if not seen_tool.get(tool_call_id):
                        emit.append(
                            {
                                "type": "tool-input-start",
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "dynamic": True,
                            }
                        )
                        state.message_seen.setdefault(msg_id, {}).setdefault("tool", {})[
                            tool_call_id] = True
                        state.emitted_tool_calls.add(tool_call_id)

                    if tc.get("args"):
                        emit.append(
                            {
                                "type": "tool-input-delta",
                                "toolCallId": tool_call_id,
                                "inputTextDelta": tc["args"],
                            }
                        )
                return  # ← early return, same as JS

            # Reasoning
            chunk_reasoning_id = extract_reasoning_id(msg)
            if chunk_reasoning_id:
                if msg_id not in state.message_reasoning_ids:
                    state.message_reasoning_ids[msg_id] = chunk_reasoning_id
                state.emitted_reasoning_ids.add(chunk_reasoning_id)

            reasoning = extract_reasoning_from_content_blocks(msg)
            if reasoning:
                reasoning_id = state.message_reasoning_ids.get(
                    msg_id) or chunk_reasoning_id or msg_id
                if not state.message_seen.get(msg_id, {}).get("reasoning"):
                    emit.append({"type": "reasoning-start", "id": msg_id})
                    state.message_seen.setdefault(
                        msg_id, {})["reasoning"] = True
                emit.append({"type": "reasoning-delta",
                            "delta": reasoning, "id": msg_id})
                state.emitted_reasoning_ids.add(reasoning_id)

            # Text
            text = get_message_text(msg)
            if text:
                if not state.message_seen.get(msg_id, {}).get("text"):
                    emit.append({"type": "text-start", "id": msg_id})
                    state.message_seen.setdefault(msg_id, {})["text"] = True
                emit.append(
                    {"type": "text-delta", "delta": text, "id": msg_id})

        elif is_tool_message_type(msg):
            ds = _data_source(msg)
            tool_call_id = ds.get("tool_call_id")
            status = ds.get("status")
            if tool_call_id:
                if status == "error":
                    emit.append(
                        {
                            "type": "tool-output-error",
                            "toolCallId": tool_call_id,
                            "errorText": (
                                ds.get("content", "Tool execution failed")
                                if isinstance(ds.get("content"), str)
                                else "Tool execution failed"
                            ),
                        }
                    )
                else:
                    output = ds.get("content")
                    sc = _extract_structured_content(msg)
                    if sc is not None:
                        output = {"_text": output, "structuredContent": sc}
                    emit.append(
                        {
                            "type": "tool-output-available",
                            "toolCallId": tool_call_id,
                            "output": output,
                        }
                    )
        return

    if etype == "values":
        # Finalize pending message chunks from messages mode
        for mid, seen in list(state.message_seen.items()):
            if seen.get("text"):
                emit.append({"type": "text-end", "id": mid})
            if seen.get("tool"):
                for tc_id, tc_seen in seen["tool"].items():
                    concat_msg = state.message_concat.get(mid)
                    tc_match = None
                    if concat_msg and hasattr(concat_msg, "tool_calls"):
                        for tc in concat_msg.tool_calls:
                            if tc.get("id") == tc_id:
                                tc_match = tc
                                break
                    if tc_seen and tc_match:
                        state.emitted_tool_calls.add(tc_id)
                        key = f"{tc_match['name']}:{json.dumps(tc_match.get('args', {}), separators=(',', ':'))}"
                        state.emitted_tool_calls_by_key[key] = tc_id
                        emit.append(
                            {
                                "type": "tool-input-available",
                                "toolCallId": tc_id,
                                "toolName": tc_match["name"],
                                "input": tc_match.get("args"),
                                "dynamic": True,
                            }
                        )
            if seen.get("reasoning"):
                emit.append({"type": "reasoning-end", "id": mid})
            del state.message_seen[mid]
            state.message_concat.pop(mid, None)
            state.message_reasoning_ids.pop(mid, None)

        # Scan for un-streamed tool calls in the full state
        if isinstance(data, dict) and "messages" in data:
            messages = data.get("messages", [])
            if isinstance(messages, list):
                # First pass: collect completed tool call IDs
                completed: set[str] = set()
                for m in messages:
                    if is_tool_message_type(m):
                        tcid = _data_source(m).get("tool_call_id")
                        if tcid:
                            completed.add(tcid)

                # Second pass: emit new tool calls
                for m in messages:
                    if not m or not isinstance(m, (dict, object)):
                        continue
                    mid2 = get_message_id(m)
                    if not mid2:
                        continue

                    tool_calls = _extract_tool_calls_from_msg(m)
                    if tool_calls:
                        for tc in tool_calls:
                            tc_id = tc.get("id", "")
                            if tc_id and tc_id not in state.emitted_tool_calls and tc_id not in completed:
                                state.emitted_tool_calls.add(tc_id)
                                key = f"{tc['name']}:{json.dumps(tc.get('args', {}), separators=(',', ':'))}"
                                state.emitted_tool_calls_by_key[key] = tc_id
                                emit.append(
                                    {
                                        "type": "tool-input-start",
                                        "toolCallId": tc_id,
                                        "toolName": tc["name"],
                                        "dynamic": True,
                                    }
                                )
                                emit.append(
                                    {
                                        "type": "tool-input-available",
                                        "toolCallId": tc_id,
                                        "toolName": tc["name"],
                                        "input": tc.get("args"),
                                        "dynamic": True,
                                    }
                                )

                    # Reasoning in values
                    rid = extract_reasoning_id(m)
                    was_streamed = mid2 in state.message_seen  # already cleaned, so check emitted
                    has_tc = bool(tool_calls)
                    should_emit = rid and rid not in state.emitted_reasoning_ids and (
                        was_streamed or not has_tc)
                    if should_emit:
                        reasoning_text = extract_reasoning_from_values_message(
                            m)
                        if reasoning_text:
                            emit.append(
                                {"type": "reasoning-start", "id": mid2})
                            emit.append({"type": "reasoning-delta",
                                        "delta": reasoning_text, "id": mid2})
                            emit.append({"type": "reasoning-end", "id": mid2})
                            state.emitted_reasoning_ids.add(rid)

        # HITL interrupts (v2 native)
        if isinstance(interrupts, (list, tuple)):
            for item in interrupts:
                # LangGraph Python uses Interrupt objects with .value attribute,
                # not plain dicts
                if isinstance(item, dict):
                    iv = item.get("value")
                else:
                    iv = getattr(item, "value", None)
                if not isinstance(iv, dict):
                    continue
                action_requests = iv.get(
                    "actionRequests") or iv.get("action_requests")
                if not isinstance(action_requests, (list, tuple)):
                    continue
                for ar in action_requests:
                    tool_name = ar.get("name", "")
                    tool_input = ar.get("args") if ar.get(
                        "args") is not None else ar.get("arguments")
                    key = f"{tool_name}:{json.dumps(tool_input, separators=(',', ':'))}" if tool_input is not None else ""
                    tc_id = state.emitted_tool_calls_by_key.get(
                        key) or ar.get("id") or f"hitl-{tool_name}-{int(time.time() * 1000)}"
                    if tc_id not in state.emitted_tool_calls:
                        state.emitted_tool_calls.add(tc_id)
                        if key:
                            state.emitted_tool_calls_by_key[key] = tc_id
                        emit.append(
                            {
                                "type": "tool-input-start",
                                "toolCallId": tc_id,
                                "toolName": tool_name,
                                "dynamic": True,
                            }
                        )
                        emit.append(
                            {
                                "type": "tool-input-available",
                                "toolCallId": tc_id,
                                "toolName": tool_name,
                                "input": tool_input,
                                "dynamic": True,
                            }
                        )
                    emit.append(
                        {
                            "type": "tool-approval-request",
                            "approvalId": tc_id,
                            "toolCallId": tc_id,
                        }
                    )
        return


# ── internal helpers ──────────────────────────────────────────────────────


def _concat_tc_id(concat: AIMessageChunk | None, idx: int) -> str | None:
    if concat is None:
        return None
    chunks = getattr(concat, "tool_call_chunks", None)
    if chunks and isinstance(chunks, list) and idx < len(chunks):
        return getattr(chunks[idx], "id", None) or (chunks[idx].get("id") if isinstance(chunks[idx], dict) else None)
    return None


def _concat_tc_name(concat: AIMessageChunk | None, idx: int) -> str | None:
    if concat is None:
        return None
    chunks = getattr(concat, "tool_call_chunks", None)
    if chunks and isinstance(chunks, list) and idx < len(chunks):
        return getattr(chunks[idx], "name", None) or (
            chunks[idx].get("name") if isinstance(chunks[idx], dict) else None
        )
    return None


def _extract_tool_calls_from_msg(msg: Any) -> list[dict[str, Any]]:
    """Extract normalized tool_calls from an AI message (class or dict)."""
    if isinstance(msg, (AIMessageChunk, AIMessage)):
        return getattr(msg, "tool_calls", []) or []

    ds = _data_source(msg)
    if isinstance(ds.get("tool_calls"), list):
        return ds["tool_calls"]

    # OpenAI format in additional_kwargs
    ak = ds.get("additional_kwargs", {})
    if isinstance(ak, dict) and isinstance(ak.get("tool_calls"), list):
        result = []
        for i, tc in enumerate(ak["tool_calls"]):
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}")
                                  ) if fn.get("arguments") else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            result.append(
                {
                    "id": tc.get("id") or f"call_{i}",
                    "name": fn.get("name", "unknown"),
                    "args": args,
                }
            )
        return result
    return []
