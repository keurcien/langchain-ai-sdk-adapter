"""
Core adapter — faithful port of ``@ai-sdk/langchain`` adapter.ts.

``to_ui_message_stream`` returns an async generator of UIMessageChunk dicts
(same structure the JS version pushes into its ReadableStream).
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from ._types import LangGraphEventState
from .callbacks import StreamCallbacks
from .utils import (
    _extract_structured_content,
    convert_assistant_content,
    convert_tool_result_part,
    convert_user_content,
    extract_reasoning_from_content_blocks,
    is_tool_result_part,
    process_langgraph_event,
    process_model_chunk,
)

# ═══════════════════════════════════════════════════════════════════════════
# toBaseMessages / convertModelMessages
# ═══════════════════════════════════════════════════════════════════════════

# Note: The JS version calls the ``ai`` package's ``convertToModelMessages``
# to turn UIMessage[] → ModelMessage[].  In Python we don't have that package,
# so ``to_lc_messages`` is a thin pass-through to ``convert_model_messages``
# when the caller already has the model-message format.  In practice, a
# FastAPI endpoint should call ``convert_model_messages`` directly on the
# parsed JSON.


def convert_model_messages(model_messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert AI SDK ``ModelMessage`` dicts into LangChain messages.

    Mirrors the JS ``convertModelMessages()`` from adapter.ts.
    """
    result: list[BaseMessage] = []
    for message in model_messages:
        role = message.get("role", "")
        content = message.get("content")

        if role == "tool":
            if isinstance(content, list):
                for item in content:
                    if is_tool_result_part(item):
                        result.append(convert_tool_result_part(item))
        elif role == "assistant":
            result.append(convert_assistant_content(content))
        elif role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(convert_user_content(content))
    return result


async def to_lc_messages(ui_messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert AI SDK ``UIMessage`` dicts into LangChain messages.

    The JS version calls ``convertToModelMessages`` from the ``ai`` package
    first, then feeds the result to ``convertModelMessages``.  Since we
    don't have that package in Python, this function does a simplified
    conversion that handles the common ``parts``-based format directly.
    """
    model_messages = _ui_messages_to_model_messages(ui_messages)
    return convert_model_messages(model_messages)


def _ui_messages_to_model_messages(ui_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simplified UIMessage → ModelMessage conversion.

    Handles: text, file parts, tool-invocation results on assistant messages.
    """
    result: list[dict[str, Any]] = []
    for msg in ui_messages:
        role = msg.get("role", "user")
        parts = msg.get("parts", [])

        if role == "system":
            text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
            result.append({"role": "system", "content": text or msg.get("content", "")})

        elif role == "user":
            content: list[dict[str, Any]] = []
            for p in parts:
                if p.get("type") == "text":
                    content.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "file":
                    media_type = p.get("mediaType", "")
                    if media_type.startswith("image/"):
                        content.append({"type": "image", "image": p.get("url", ""), "mediaType": media_type})
                    else:
                        content.append({"type": "file", "data": p.get("url", ""), "mediaType": media_type})
            if not content:
                content_val = msg.get("content", "")
                result.append({"role": "user", "content": content_val})
            elif len(content) == 1 and content[0].get("type") == "text":
                result.append({"role": "user", "content": content[0]["text"]})
            else:
                result.append({"role": "user", "content": content})

        elif role == "assistant":
            assistant_content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            for p in parts:
                if p.get("type") == "text":
                    assistant_content.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "tool-invocation":
                    assistant_content.append(
                        {
                            "type": "tool-call",
                            "toolCallId": p.get("toolInvocationId", ""),
                            "toolName": p.get("toolName", ""),
                            "input": p.get("input", {}),
                        }
                    )
                    if p.get("state") == "result":
                        tool_results.append(
                            {
                                "type": "tool-result",
                                "toolCallId": p.get("toolInvocationId", ""),
                                "toolName": p.get("toolName", ""),
                                "output": {"type": "json", "value": p.get("output")},
                            }
                        )
            if len(assistant_content) == 1 and assistant_content[0].get("type") == "text":
                result.append({"role": "assistant", "content": assistant_content[0]["text"]})
            else:
                result.append({"role": "assistant", "content": assistant_content or msg.get("content", "")})
            if tool_results:
                result.append({"role": "tool", "content": tool_results})

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Stream type detection
# ═══════════════════════════════════════════════════════════════════════════


def _is_stream_events_event(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "event" not in value or not isinstance(value["event"], str):
        return False
    if "data" not in value:
        return False
    return value["data"] is None or isinstance(value["data"], dict)


def _is_abort_error(error: BaseException) -> bool:
    name = type(error).__name__
    return "abort" in name.lower() or "cancel" in name.lower()


async def _maybe_await(fn: Any, *args: Any) -> None:
    if fn is None:
        return
    result = fn(*args)
    if inspect.isawaitable(result):
        await result


# ═══════════════════════════════════════════════════════════════════════════
# processStreamEventsEvent
# ═══════════════════════════════════════════════════════════════════════════


def _process_stream_events_event(
    event: dict[str, Any],
    state: dict[str, Any],
    emit: list[dict[str, Any]],
) -> None:
    """Process a single streamEvents v2 event."""
    run_id = event.get("run_id")
    if run_id and not state["started"]:
        state["message_id"] = run_id

    data = event.get("data")
    if data is None:
        return

    ev = event.get("event", "")

    if ev == "on_chat_model_start":
        rid = run_id or (data.get("run_id") if isinstance(data, dict) else None)
        if rid:
            state["message_id"] = rid

    elif ev == "on_chat_model_stream":
        chunk = data.get("chunk") if isinstance(data, dict) else None
        if chunk and isinstance(chunk, (dict, object)):
            chunk_id = chunk.get("id") if isinstance(chunk, dict) else getattr(chunk, "id", None)
            if chunk_id:
                state["message_id"] = chunk_id

            reasoning = extract_reasoning_from_content_blocks(chunk)
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

            content = chunk.get("content") if isinstance(chunk, dict) else getattr(chunk, "content", None)
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
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

    elif ev == "on_tool_start":
        rid = run_id or (data.get("run_id") if isinstance(data, dict) else None)
        name = event.get("name") or (data.get("name") if isinstance(data, dict) else None)
        if rid and name:
            emit.append(
                {
                    "type": "tool-input-start",
                    "toolCallId": rid,
                    "toolName": name,
                    "dynamic": True,
                }
            )

    elif ev == "on_tool_end":
        rid = run_id or (data.get("run_id") if isinstance(data, dict) else None)
        raw_output = data.get("output") if isinstance(data, dict) else None
        if rid:
            output = raw_output.content if hasattr(raw_output, "content") else raw_output

            sc = _extract_structured_content(raw_output)
            if sc is not None:
                output = {"_text": output, "structuredContent": sc}

            emit.append(
                {
                    "type": "tool-output-available",
                    "toolCallId": rid,
                    "output": output,
                }
            )


# ═══════════════════════════════════════════════════════════════════════════
# toUIMessageStream
# ═══════════════════════════════════════════════════════════════════════════


async def to_ui_message_stream(
    stream: AsyncIterable[Any],
    callbacks: StreamCallbacks | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Convert any LangChain / LangGraph async stream to UIMessageChunk dicts.

    This is the Python equivalent of the JS ``toUIMessageStream()``.
    It yields the *same* dict shapes that the JS version enqueues into
    its ``ReadableStream<UIMessageChunk>``.
    """
    cbs = callbacks or StreamCallbacks()
    text_chunks: list[str] = []
    last_values_data: Any = None

    model_state: dict[str, Any] = {
        "started": False,
        "message_id": "langchain-msg-1",
        "reasoning_started": False,
        "text_started": False,
        "text_message_id": None,
        "reasoning_message_id": None,
    }

    lg_state = LangGraphEventState()
    stream_type = None

    await _maybe_await(cbs.on_start)

    yield {"type": "start"}

    try:
        async for value in stream:
            # Detect stream type on first value
            if stream_type is None:
                if _is_stream_events_event(value):
                    stream_type = "streamEvents"
                elif isinstance(value, dict) and "ns" in value:
                    stream_type = "langgraph"
                else:
                    stream_type = "model"

            batch: list[dict[str, Any]] = []

            if stream_type == "model":
                process_model_chunk(value, model_state, batch)
            elif stream_type == "streamEvents":
                _process_stream_events_event(value, model_state, batch)
            else:
                # LangGraph v2 StreamPart dict
                ns = tuple(value.get("ns", ()))
                if ns != lg_state.current_ns:
                    lg_state.current_ns = ns
                    batch.append({"type": "data-namespace", "data": {"ns": list(ns)}})

                etype = value["type"]
                edata = value.get("data")
                if etype == "values":
                    last_values_data = edata
                process_langgraph_event(
                    etype, edata, lg_state, batch,
                    interrupts=value.get("interrupts"),
                )

            for chunk in batch:
                # Intercept text-delta for callbacks
                if chunk.get("type") == "text-delta" and chunk.get("delta"):
                    text_chunks.append(chunk["delta"])
                    await _maybe_await(cbs.on_token, chunk["delta"])
                    await _maybe_await(cbs.on_text, chunk["delta"])
                yield chunk

        # Finalize
        if stream_type in ("model", "streamEvents"):
            if model_state.get("reasoning_started"):
                yield {
                    "type": "reasoning-end",
                    "id": model_state.get("reasoning_message_id") or model_state["message_id"],
                }
            if model_state.get("text_started"):
                yield {
                    "type": "text-end",
                    "id": model_state.get("text_message_id") or model_state["message_id"],
                }
            yield {"type": "finish"}
        elif stream_type == "langgraph":
            if lg_state.current_step is not None:
                yield {"type": "finish-step"}
            yield {"type": "finish"}

        await _maybe_await(cbs.on_final, "".join(text_chunks))
        await _maybe_await(cbs.on_finish, last_values_data)

    except BaseException as error:
        await _maybe_await(cbs.on_final, "".join(text_chunks))

        if _is_abort_error(error):
            await _maybe_await(cbs.on_abort)
        else:
            err = error if isinstance(error, Exception) else Exception(str(error))
            await _maybe_await(cbs.on_error, err)

        yield {"type": "error", "errorText": str(error)}
