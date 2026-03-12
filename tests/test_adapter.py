"""
Port of adapter.test.ts from @ai-sdk/langchain.

Each test produces the SAME UIMessageChunk sequence as the JS version.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_ai_sdk_adapter.adapter import (
    convert_model_messages,
    to_base_messages,
    to_ui_message_stream,
)
from langchain_ai_sdk_adapter.callbacks import StreamCallbacks

# ── Helpers ───────────────────────────────────────────────────────────────


async def _to_list(stream: AsyncIterator[Any]) -> list[dict[str, Any]]:
    return [chunk async for chunk in stream]


async def _async_iter(*items: Any) -> AsyncIterator[Any]:
    for item in items:
        yield item


# ═══════════════════════════════════════════════════════════════════════════
# toUIMessageStream  —  LangGraph mode
# ═══════════════════════════════════════════════════════════════════════════


class TestToUIMessageStreamLangGraph:
    @pytest.mark.asyncio
    async def test_start_event(self):
        result = await _to_list(to_ui_message_stream(_async_iter(["values", {}])))
        assert result[0] == {"type": "start"}

    @pytest.mark.asyncio
    async def test_text_streaming_from_messages(self):
        """AIMessage chunks in messages mode — text content is skipped for non-AIMessageChunk."""
        chunk1 = AIMessage(content="Hello", id="msg-1")
        chunk2 = AIMessage(content=" World", id="msg-1")
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [chunk1]],
                    ["messages", [chunk2]],
                    ["values", {}],
                )
            )
        )
        # AIMessage (not AIMessageChunk) — JS adapter checks AIMessageChunk.isInstance
        # which returns false for AIMessage. The text shows only via text-start/delta/end
        # when using actual AIMessageChunk instances.
        assert result[0] == {"type": "start"}
        assert result[-1] == {"type": "finish"}

    @pytest.mark.asyncio
    async def test_tool_message_output(self):
        tool_msg = ToolMessage(tool_call_id="call-1", content="Sunny, 72°F", id="msg-1")
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [tool_msg]],
                    ["values", {}],
                )
            )
        )
        assert {"type": "tool-output-available", "toolCallId": "call-1", "output": "Sunny, 72°F"} in result

    @pytest.mark.asyncio
    async def test_custom_events(self):
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["custom", {"custom": "data"}],
                    ["values", {}],
                )
            )
        )
        assert {
            "type": "data-custom",
            "id": None,
            "transient": True,
            "data": {"custom": "data"},
        } in result

    @pytest.mark.asyncio
    async def test_three_element_arrays(self):
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["namespace", "custom", {"data": "value"}],
                    ["values", {}],
                )
            )
        )
        assert {
            "type": "data-custom",
            "id": None,
            "transient": True,
            "data": {"data": "value"},
        } in result

    @pytest.mark.asyncio
    async def test_plain_objects_remote_graph_ai(self):
        plain_msg = {
            "content": "Hello from RemoteGraph",
            "id": "chatcmpl-123",
            "type": "ai",
            "tool_call_chunks": [],
        }
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [plain_msg]],
                    ["values", {}],
                )
            )
        )
        assert {"type": "text-start", "id": "chatcmpl-123"} in result
        assert {"type": "text-delta", "delta": "Hello from RemoteGraph", "id": "chatcmpl-123"} in result
        assert {"type": "text-end", "id": "chatcmpl-123"} in result

    @pytest.mark.asyncio
    async def test_plain_objects_remote_graph_tool(self):
        plain_tool = {
            "content": "Tool result here",
            "id": "tool-msg-123",
            "type": "tool",
            "tool_call_id": "call-abc",
        }
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [plain_tool]],
                    ["values", {}],
                )
            )
        )
        assert {"type": "tool-output-available", "toolCallId": "call-abc", "output": "Tool result here"} in result

    @pytest.mark.asyncio
    async def test_tool_calls_in_values_only(self):
        values_data = {
            "messages": [
                {
                    "content": "",
                    "id": "ai-msg-1",
                    "type": "ai",
                    "tool_calls": [{"id": "call-123", "name": "get_weather", "args": {"city": "SF"}}],
                }
            ],
        }
        result = await _to_list(to_ui_message_stream(_async_iter(["values", values_data])))
        starts = [e for e in result if e.get("type") == "tool-input-start"]
        avails = [e for e in result if e.get("type") == "tool-input-available"]
        assert len(starts) == 1
        assert starts[0]["toolCallId"] == "call-123"
        assert starts[0]["toolName"] == "get_weather"
        assert len(avails) == 1
        assert avails[0]["input"] == {"city": "SF"}

    @pytest.mark.asyncio
    async def test_tool_calls_additional_kwargs_format(self):
        values_data = {
            "messages": [
                {
                    "content": "",
                    "id": "ai-msg-1",
                    "type": "ai",
                    "additional_kwargs": {
                        "tool_calls": [
                            {
                                "id": "call-456",
                                "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                            }
                        ],
                    },
                }
            ],
        }
        result = await _to_list(to_ui_message_stream(_async_iter(["values", values_data])))
        starts = [e for e in result if e.get("type") == "tool-input-start"]
        assert len(starts) == 1
        assert starts[0]["toolCallId"] == "call-456"
        assert starts[0]["toolName"] == "get_weather"

    @pytest.mark.asyncio
    async def test_no_duplicate_tool_calls(self):
        streamed = {
            "content": "",
            "id": "ai-msg-1",
            "type": "ai",
            "tool_call_chunks": [{"id": "call-789", "name": "get_weather", "args": '{"city":"LA"}', "index": 0}],
        }
        values = {
            "messages": [
                {
                    "content": "",
                    "id": "ai-msg-1",
                    "type": "ai",
                    "tool_calls": [{"id": "call-789", "name": "get_weather", "args": {"city": "LA"}}],
                }
            ],
        }
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [streamed]],
                    ["values", values],
                )
            )
        )
        tc_events = [e for e in result if e.get("type") in ("tool-input-start", "tool-input-available")]
        # tool-input-start from streaming, then no duplicate from values
        assert len(tc_events) == 1
        assert tc_events[0]["type"] == "tool-input-start"

    @pytest.mark.asyncio
    async def test_skip_tool_chunks_without_id(self):
        streamed = {
            "content": "",
            "id": "ai-msg-1",
            "type": "ai",
            "tool_call_chunks": [{"name": "get_weather", "args": '{"city":"LA"}', "index": 0}],  # no id!
        }
        values = {
            "messages": [
                {
                    "content": "",
                    "id": "ai-msg-1",
                    "type": "ai",
                    "tool_calls": [{"id": "call-real-id", "name": "get_weather", "args": {"city": "LA"}}],
                }
            ],
        }
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [streamed]],
                    ["values", values],
                )
            )
        )
        tc_events = [e for e in result if e.get("type") in ("tool-input-start", "tool-input-available")]
        assert len(tc_events) == 2
        assert tc_events[0] == {
            "type": "tool-input-start",
            "toolCallId": "call-real-id",
            "toolName": "get_weather",
            "dynamic": True,
        }
        assert tc_events[1] == {
            "type": "tool-input-available",
            "toolCallId": "call-real-id",
            "toolName": "get_weather",
            "input": {"city": "LA"},
            "dynamic": True,
        }

    @pytest.mark.asyncio
    async def test_reasoning_from_content_blocks(self):
        chunk = AIMessageChunk(id="msg-reason", content="")
        # Simulate contentBlocks (as done in JS tests with Object.defineProperty)
        chunk.contentBlocks = [{"type": "reasoning", "reasoning": "Let me think about this..."}]  # type: ignore[attr-defined]

        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [chunk]],
                    ["values", {}],
                )
            )
        )
        assert {"type": "reasoning-start", "id": "msg-reason"} in result
        assert {"type": "reasoning-delta", "delta": "Let me think about this...", "id": "msg-reason"} in result
        assert {"type": "reasoning-end", "id": "msg-reason"} in result

    @pytest.mark.asyncio
    async def test_thinking_content_blocks(self):
        chunk = AIMessageChunk(id="msg-think", content="")
        chunk.contentBlocks = [{"type": "thinking", "thinking": "First, I need to analyze...", "signature": "abc123"}]  # type: ignore[attr-defined]

        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [chunk]],
                    ["values", {}],
                )
            )
        )
        assert {"type": "reasoning-start", "id": "msg-think"} in result
        assert {"type": "reasoning-delta", "delta": "First, I need to analyze...", "id": "msg-think"} in result

    @pytest.mark.asyncio
    async def test_multiple_reasoning_chunks(self):
        c1 = AIMessageChunk(id="msg-reason", content="")
        c1.contentBlocks = [{"type": "reasoning", "reasoning": "First..."}]  # type: ignore[attr-defined]
        c2 = AIMessageChunk(id="msg-reason", content="")
        c2.contentBlocks = [{"type": "reasoning", "reasoning": "Second..."}]  # type: ignore[attr-defined]

        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [c1]],
                    ["messages", [c2]],
                    ["values", {}],
                )
            )
        )
        deltas = [e for e in result if e.get("type") == "reasoning-delta"]
        assert len(deltas) == 2
        assert deltas[0]["delta"] == "First..."
        assert deltas[1]["delta"] == "Second..."
        # Only one reasoning-start
        assert len([e for e in result if e.get("type") == "reasoning-start"]) == 1

    @pytest.mark.asyncio
    async def test_reasoning_followed_by_text(self):
        rc = AIMessageChunk(id="msg-1", content="")
        rc.contentBlocks = [{"type": "reasoning", "reasoning": "Thinking about this..."}]  # type: ignore[attr-defined]
        tc = AIMessageChunk(id="msg-1", content="Here is my answer.")

        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [rc]],
                    ["messages", [tc]],
                    ["values", {}],
                )
            )
        )
        assert result == [
            {"type": "start"},
            {"type": "reasoning-start", "id": "msg-1"},
            {"type": "reasoning-delta", "delta": "Thinking about this...", "id": "msg-1"},
            {"type": "text-start", "id": "msg-1"},
            {"type": "text-delta", "delta": "Here is my answer.", "id": "msg-1"},
            {"type": "text-end", "id": "msg-1"},
            {"type": "reasoning-end", "id": "msg-1"},
            {"type": "finish"},
        ]

    @pytest.mark.asyncio
    async def test_reasoning_with_tool_calls(self):
        """Reasoning before a tool call should emit reasoning then tool events."""
        reasoning_chunk = AIMessageChunk(id="msg-1", content="")
        reasoning_chunk.contentBlocks = [{"type": "reasoning", "reasoning": "I need to search for this..."}]  # type: ignore[attr-defined]

        tool_call_chunk = {
            "content": "",
            "id": "msg-1",
            "type": "ai",
            "tool_call_chunks": [{"id": "call-123", "name": "search", "args": '{"query":"test"}', "index": 0}],
        }

        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [reasoning_chunk]],
                    ["messages", [tool_call_chunk]],
                    ["values", {}],
                )
            )
        )
        assert result == [
            {"type": "start"},
            {"type": "reasoning-start", "id": "msg-1"},
            {"type": "reasoning-delta", "delta": "I need to search for this...", "id": "msg-1"},
            {"type": "tool-input-start", "toolCallId": "call-123", "toolName": "search", "dynamic": True},
            {"type": "tool-input-delta", "toolCallId": "call-123", "inputTextDelta": '{"query":"test"}'},
            {"type": "reasoning-end", "id": "msg-1"},
            {"type": "finish"},
        ]

    @pytest.mark.asyncio
    async def test_skip_messages_without_id(self):
        msg = AIMessage(content="No ID message")
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [msg]],
                    ["values", {}],
                )
            )
        )
        assert result == [{"type": "start"}, {"type": "finish"}]

    @pytest.mark.asyncio
    async def test_not_emit_historical_tool_calls(self):
        """Historical tool calls with a ToolMessage response should NOT be re-emitted."""
        values_data = {
            "messages": [
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "HumanMessage"],
                    "kwargs": {"id": "human-1", "content": "do maths"},
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "AIMessage"],
                    "kwargs": {
                        "id": "ai-1",
                        "content": "",
                        "tool_calls": [{"id": "call_H", "name": "maths", "args": {"input": 123}}],
                    },
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "ToolMessage"],
                    "kwargs": {"id": "tool-1", "tool_call_id": "call_H", "content": '{"result": "15.5"}'},
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "AIMessage"],
                    "kwargs": {"id": "ai-2", "content": "The result is 15.5"},
                },
            ],
        }
        result = await _to_list(to_ui_message_stream(_async_iter(["values", values_data])))
        assert len([e for e in result if e.get("type") == "tool-input-start"]) == 0

    @pytest.mark.asyncio
    async def test_emit_new_tool_calls_without_response(self):
        values_data = {
            "messages": [
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "HumanMessage"],
                    "kwargs": {"id": "human-1", "content": "do maths"},
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "AIMessage"],
                    "kwargs": {
                        "id": "ai-1",
                        "content": "",
                        "tool_calls": [{"id": "call_N", "name": "maths", "args": {"input": 456}}],
                    },
                },
            ],
        }
        result = await _to_list(to_ui_message_stream(_async_iter(["values", values_data])))
        starts = [e for e in result if e.get("type") == "tool-input-start"]
        avails = [e for e in result if e.get("type") == "tool-input-available"]
        assert len(starts) == 1
        assert starts[0]["toolCallId"] == "call_N"
        assert starts[0]["toolName"] == "maths"
        assert len(avails) == 1
        assert avails[0]["input"] == {"input": 456}

    @pytest.mark.asyncio
    async def test_mixed_historical_and_new_tool_calls(self):
        """Historical tool calls with ToolMessage + new pending tool call in same values event."""
        values_data = {
            "messages": [
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "HumanMessage"],
                    "kwargs": {"id": "human-1", "content": "do maths with 123"},
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "AIMessage"],
                    "kwargs": {
                        "id": "ai-1",
                        "content": "",
                        "tool_calls": [{"id": "call_HISTORICAL_789", "name": "maths", "args": {"input": 123}}],
                    },
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "ToolMessage"],
                    "kwargs": {"id": "tool-1", "tool_call_id": "call_HISTORICAL_789", "content": '{"result": "15.5"}'},
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "AIMessage"],
                    "kwargs": {"id": "ai-2", "content": "The result is 15.5"},
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "HumanMessage"],
                    "kwargs": {"id": "human-2", "content": "do it again with 999"},
                },
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "AIMessage"],
                    "kwargs": {
                        "id": "ai-3",
                        "content": "",
                        "tool_calls": [{"id": "call_CURRENT_999", "name": "maths", "args": {"input": 999}}],
                    },
                },
            ],
        }
        result = await _to_list(to_ui_message_stream(_async_iter(["values", values_data])))
        starts = [e for e in result if e.get("type") == "tool-input-start"]
        avails = [e for e in result if e.get("type") == "tool-input-available"]
        assert len(starts) == 1
        assert starts[0]["toolCallId"] == "call_CURRENT_999"
        assert starts[0]["toolName"] == "maths"
        assert len(avails) == 1
        assert avails[0]["toolCallId"] == "call_CURRENT_999"
        assert avails[0]["input"] == {"input": 999}
        # Verify historical tool call was NOT emitted
        historical = [e for e in result if e.get("toolCallId") == "call_HISTORICAL_789"]
        assert len(historical) == 0

    @pytest.mark.asyncio
    async def test_plain_object_historical_tool_calls(self):
        """Plain object format (RemoteGraph API) for historical tool calls."""
        values_data = {
            "messages": [
                {
                    "id": "ai-1",
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"id": "call_PLAIN_HISTORICAL", "name": "search", "args": {"q": "test"}}],
                },
                {
                    "id": "tool-1",
                    "type": "tool",
                    "tool_call_id": "call_PLAIN_HISTORICAL",
                    "content": "Search results...",
                },
                {
                    "id": "ai-2",
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"id": "call_PLAIN_CURRENT", "name": "search", "args": {"q": "new"}}],
                },
            ],
        }
        result = await _to_list(to_ui_message_stream(_async_iter(["values", values_data])))
        starts = [e for e in result if e.get("type") == "tool-input-start"]
        assert len(starts) == 1
        assert starts[0]["toolCallId"] == "call_PLAIN_CURRENT"

    @pytest.mark.asyncio
    async def test_multiple_historical_tool_calls_same_message(self):
        """AI message with multiple tool calls, all completed, plus a new one."""
        values_data = {
            "messages": [
                {
                    "id": "ai-1",
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_MULTI_1", "name": "tool_a", "args": {"a": 1}},
                        {"id": "call_MULTI_2", "name": "tool_b", "args": {"b": 2}},
                    ],
                },
                {"id": "tool-1", "type": "tool", "tool_call_id": "call_MULTI_1", "content": "Result A"},
                {"id": "tool-2", "type": "tool", "tool_call_id": "call_MULTI_2", "content": "Result B"},
                {
                    "id": "ai-2",
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"id": "call_MULTI_CURRENT", "name": "tool_c", "args": {"c": 3}}],
                },
            ],
        }
        result = await _to_list(to_ui_message_stream(_async_iter(["values", values_data])))
        starts = [e for e in result if e.get("type") == "tool-input-start"]
        assert len(starts) == 1
        assert starts[0]["toolCallId"] == "call_MULTI_CURRENT"


# ═══════════════════════════════════════════════════════════════════════════
# toUIMessageStream  —  model stream mode
# ═══════════════════════════════════════════════════════════════════════════


class TestToUIMessageStreamModel:
    @pytest.mark.asyncio
    async def test_non_array_as_model_stream(self):
        chunk = AIMessageChunk(content="Hello from model", id="test-1")
        result = await _to_list(to_ui_message_stream(_async_iter(chunk)))
        assert result == [
            {"type": "start"},
            {"type": "text-start", "id": "test-1"},
            {"type": "text-delta", "delta": "Hello from model", "id": "test-1"},
            {"type": "text-end", "id": "test-1"},
            {"type": "finish"},
        ]

    @pytest.mark.asyncio
    async def test_model_stream_reasoning(self):
        rc = AIMessageChunk(content="", id="test-1")
        rc.contentBlocks = [{"type": "reasoning", "reasoning": "Thinking..."}]  # type: ignore[attr-defined]
        tc = AIMessageChunk(content="Hello!", id="test-1")
        result = await _to_list(to_ui_message_stream(_async_iter(rc, tc)))
        assert result == [
            {"type": "start"},
            {"type": "reasoning-start", "id": "test-1"},
            {"type": "reasoning-delta", "delta": "Thinking...", "id": "test-1"},
            {"type": "reasoning-end", "id": "test-1"},
            {"type": "text-start", "id": "test-1"},
            {"type": "text-delta", "delta": "Hello!", "id": "test-1"},
            {"type": "text-end", "id": "test-1"},
            {"type": "finish"},
        ]


# ═══════════════════════════════════════════════════════════════════════════
# toUIMessageStream  —  streamEvents mode
# ═══════════════════════════════════════════════════════════════════════════


class TestToUIMessageStreamEvents:
    @pytest.mark.asyncio
    async def test_detect_and_handle_stream_events(self):
        """Full streamEvents format with start, stream, and end events."""
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    {"event": "on_chat_model_start", "data": {"input": "Hello"}},
                    {"event": "on_chat_model_stream", "data": {"chunk": {"id": "stream-msg-1", "content": "Hello"}}},
                    {"event": "on_chat_model_stream", "data": {"chunk": {"id": "stream-msg-1", "content": " World"}}},
                    {
                        "event": "on_chat_model_end",
                        "data": {"output": {"id": "stream-msg-1", "content": "Hello World"}},
                    },
                )
            )
        )
        assert result == [
            {"type": "start"},
            {"type": "text-start", "id": "stream-msg-1"},
            {"type": "text-delta", "delta": "Hello", "id": "stream-msg-1"},
            {"type": "text-delta", "delta": " World", "id": "stream-msg-1"},
            {"type": "text-end", "id": "stream-msg-1"},
            {"type": "finish"},
        ]

    @pytest.mark.asyncio
    async def test_basic_text(self):
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    {"event": "on_chat_model_stream", "data": {"chunk": {"id": "msg-1", "content": "Hello"}}},
                    {"event": "on_chat_model_stream", "data": {"chunk": {"id": "msg-1", "content": " World"}}},
                )
            )
        )
        assert {"type": "text-delta", "delta": "Hello", "id": "msg-1"} in result
        assert {"type": "text-delta", "delta": " World", "id": "msg-1"} in result

    @pytest.mark.asyncio
    async def test_tool_start_and_end(self):
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    {"event": "on_chat_model_start", "data": {"input": "What is the weather?"}},
                    {
                        "event": "on_tool_start",
                        "data": {"run_id": "tool-call-123", "name": "get_weather", "inputs": {"city": "SF"}},
                    },
                    {"event": "on_tool_end", "data": {"run_id": "tool-call-123", "output": "Sunny, 72°F"}},
                )
            )
        )
        assert result == [
            {"type": "start"},
            {"type": "tool-input-start", "toolCallId": "tool-call-123", "toolName": "get_weather", "dynamic": True},
            {"type": "tool-output-available", "toolCallId": "tool-call-123", "output": "Sunny, 72°F"},
            {"type": "finish"},
        ]

    @pytest.mark.asyncio
    async def test_reasoning(self):
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    {
                        "event": "on_chat_model_stream",
                        "data": {
                            "chunk": {
                                "id": "reasoning-msg-1",
                                "content": "",
                                "contentBlocks": [{"type": "reasoning", "reasoning": "Let me think..."}],
                            }
                        },
                    },
                    {
                        "event": "on_chat_model_stream",
                        "data": {
                            "chunk": {
                                "id": "reasoning-msg-1",
                                "content": "Here is my answer.",
                            }
                        },
                    },
                )
            )
        )
        assert {"type": "reasoning-start", "id": "reasoning-msg-1"} in result
        assert {"type": "reasoning-delta", "delta": "Let me think...", "id": "reasoning-msg-1"} in result
        assert {"type": "reasoning-end", "id": "reasoning-msg-1"} in result
        assert {"type": "text-start", "id": "reasoning-msg-1"} in result
        assert {"type": "text-delta", "delta": "Here is my answer.", "id": "reasoning-msg-1"} in result

    @pytest.mark.asyncio
    async def test_array_content(self):
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    {
                        "event": "on_chat_model_stream",
                        "data": {
                            "chunk": {
                                "id": "arr-1",
                                "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " from array"}],
                            }
                        },
                    },
                )
            )
        )
        assert {"type": "text-delta", "delta": "Hello from array", "id": "arr-1"} in result


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph finish events
# ═══════════════════════════════════════════════════════════════════════════


class TestLangGraphFinishEvents:
    @pytest.mark.asyncio
    async def test_finish_event(self):
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["values", {"messages": [{"id": "ai-1", "type": "ai", "content": "Hello!"}]}],
                )
            )
        )
        assert result[-1] == {"type": "finish"}

    @pytest.mark.asyncio
    async def test_step_events(self):
        chunk = AIMessageChunk(content="Hello", id="msg-1")
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [chunk, {"langgraph_step": 0}]],
                    ["values", {}],
                )
            )
        )
        assert len([e for e in result if e["type"] == "start-step"]) == 1
        assert len([e for e in result if e["type"] == "finish-step"]) == 1

    @pytest.mark.asyncio
    async def test_multiple_steps(self):
        c1 = AIMessageChunk(content="Step 0", id="msg-1")
        c2 = AIMessageChunk(content="Step 1", id="msg-2")
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["messages", [c1, {"langgraph_step": 0}]],
                    ["messages", [c2, {"langgraph_step": 1}]],
                    ["values", {}],
                )
            )
        )
        assert len([e for e in result if e["type"] == "start-step"]) == 2
        assert len([e for e in result if e["type"] == "finish-step"]) == 2

    @pytest.mark.asyncio
    async def test_no_finish_step_without_steps(self):
        result = await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["values", {"messages": [{"id": "ai-1", "type": "ai", "content": "Hello!"}]}],
                )
            )
        )
        assert len([e for e in result if e["type"] == "finish-step"]) == 0
        assert len([e for e in result if e["type"] == "finish"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════


class TestCallbacks:
    @pytest.mark.asyncio
    async def test_on_finish_with_langgraph_state(self):
        received = []
        cbs = StreamCallbacks(on_finish=lambda state: received.append(state))
        values = {"messages": [{"id": "ai-1", "type": "ai", "content": "Hello!"}]}
        await _to_list(to_ui_message_stream(_async_iter(["values", values]), callbacks=cbs))
        assert received == [values]

    @pytest.mark.asyncio
    async def test_on_finish_with_last_values(self):
        received = []
        cbs = StreamCallbacks(on_finish=lambda s: received.append(s))
        await _to_list(
            to_ui_message_stream(
                _async_iter(
                    ["values", {"messages": [], "step": 1}],
                    ["values", {"messages": [{"id": "ai-1"}], "step": 2}],
                ),
                callbacks=cbs,
            )
        )
        assert received == [{"messages": [{"id": "ai-1"}], "step": 2}]

    @pytest.mark.asyncio
    async def test_on_finish_undefined_for_model(self):
        received = []
        cbs = StreamCallbacks(on_finish=lambda s: received.append(s))
        chunk = AIMessageChunk(content="Hello", id="test-1")
        await _to_list(to_ui_message_stream(_async_iter(chunk), callbacks=cbs))
        assert received == [None]

    @pytest.mark.asyncio
    async def test_on_error(self):
        errors = []
        finishes = []
        cbs = StreamCallbacks(
            on_error=lambda e: errors.append(str(e)),
            on_finish=lambda s: finishes.append(s),
        )

        async def error_stream():
            yield ["values", {"messages": []}]
            raise Exception("Stream failed")

        await _to_list(to_ui_message_stream(error_stream(), callbacks=cbs))
        assert len(errors) == 1
        assert "Stream failed" in errors[0]
        assert finishes == []

    @pytest.mark.asyncio
    async def test_on_final_before_on_finish(self):
        order: list[str] = []
        cbs = StreamCallbacks(
            on_final=lambda _: order.append("final"),
            on_finish=lambda _: order.append("finish"),
        )
        await _to_list(to_ui_message_stream(_async_iter(["values", {"messages": []}]), callbacks=cbs))
        assert order == ["final", "finish"]

    @pytest.mark.asyncio
    async def test_on_final_before_on_error(self):
        order: list[str] = []
        cbs = StreamCallbacks(
            on_final=lambda _: order.append("final"),
            on_error=lambda _: order.append("error"),
        )

        async def err():
            yield ["values", {"messages": []}]
            raise Exception("fail")

        await _to_list(to_ui_message_stream(err(), callbacks=cbs))
        assert order == ["final", "error"]

    @pytest.mark.asyncio
    async def test_on_final_gets_accumulated_text_on_error(self):
        received = []
        cbs = StreamCallbacks(on_final=lambda t: received.append(t))

        async def partial():
            yield AIMessageChunk(content="Hello", id="msg-1")
            yield AIMessageChunk(content=" World", id="msg-1")
            raise Exception("Mid-stream error")

        await _to_list(to_ui_message_stream(partial(), callbacks=cbs))
        assert received == ["Hello World"]

    @pytest.mark.asyncio
    async def test_async_callbacks_order(self):
        results: list[str] = []

        async def on_start():
            results.append("start")

        async def on_final(t: str):
            results.append("final")

        async def on_finish(s: Any):
            results.append("finish")

        cbs = StreamCallbacks(on_start=on_start, on_final=on_final, on_finish=on_finish)
        await _to_list(to_ui_message_stream(_async_iter(["values", {"messages": []}]), callbacks=cbs))
        assert results == ["start", "final", "finish"]

    @pytest.mark.asyncio
    async def test_namespace_values_on_finish(self):
        received = []
        cbs = StreamCallbacks(on_finish=lambda s: received.append(s))
        values = {"messages": [{"id": "ai-1", "content": "Hello"}]}
        await _to_list(
            to_ui_message_stream(
                _async_iter(["some-namespace", "values", values]),
                callbacks=cbs,
            )
        )
        assert received == [values]

    @pytest.mark.asyncio
    async def test_on_finish_undefined_for_stream_events(self):
        received = []
        cbs = StreamCallbacks(on_finish=lambda s: received.append(s))
        await _to_list(
            to_ui_message_stream(
                _async_iter(
                    {"event": "on_chat_model_stream", "data": {"chunk": {"id": "msg-1", "content": "Hello"}}},
                ),
                callbacks=cbs,
            )
        )
        assert received == [None]

    @pytest.mark.asyncio
    async def test_on_abort(self):
        aborts = []
        errors = []
        finishes = []
        cbs = StreamCallbacks(
            on_abort=lambda: aborts.append(True),
            on_error=lambda e: errors.append(str(e)),
            on_finish=lambda s: finishes.append(s),
        )

        async def abort_stream():
            yield ["values", {"messages": []}]
            raise asyncio.CancelledError("Aborted")

        await _to_list(to_ui_message_stream(abort_stream(), callbacks=cbs))
        assert len(aborts) == 1
        assert errors == []
        assert finishes == []

    @pytest.mark.asyncio
    async def test_on_final_before_on_abort(self):
        order: list[str] = []
        cbs = StreamCallbacks(
            on_final=lambda _: order.append("final"),
            on_abort=lambda: order.append("abort"),
        )

        async def abort_stream():
            yield ["values", {"messages": []}]
            raise asyncio.CancelledError("Aborted")

        await _to_list(to_ui_message_stream(abort_stream(), callbacks=cbs))
        assert order == ["final", "abort"]


# ═══════════════════════════════════════════════════════════════════════════
# convertModelMessages
# ═══════════════════════════════════════════════════════════════════════════


class TestConvertModelMessages:
    def test_system(self):
        result = convert_model_messages([{"role": "system", "content": "You are helpful."}])
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are helpful."

    def test_user_text(self):
        result = convert_model_messages([{"role": "user", "content": "Hello!"}])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello!"

    def test_user_array_text(self):
        result = convert_model_messages([{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_user_image_url(self):
        result = convert_model_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "image": "https://example.com/image.jpg"},
                    ],
                }
            ]
        )
        assert result[0].content == [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        ]

    def test_user_image_base64(self):
        result = convert_model_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What?"},
                        {"type": "image", "image": "iVBORw0K", "mediaType": "image/png"},
                    ],
                }
            ]
        )
        assert result[0].content[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0K"},
        }

    def test_user_file_url(self):
        result = convert_model_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize."},
                        {"type": "file", "data": "https://example.com/doc.pdf", "mediaType": "application/pdf"},
                    ],
                }
            ]
        )
        assert result[0].content[1] == {
            "type": "file",
            "url": "https://example.com/doc.pdf",
            "mimeType": "application/pdf",
            "filename": "file.pdf",
        }

    def test_user_image_file(self):
        """image/jpeg file parts should use image_url format."""
        result = convert_model_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe."},
                        {"type": "file", "data": "https://example.com/photo.jpg", "mediaType": "image/jpeg"},
                    ],
                }
            ]
        )
        assert result[0].content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"},
        }

    def test_user_image_data_url(self):
        """Data URLs should be passed through directly."""
        result = convert_model_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image", "image": "data:image/png;base64,abc123"},
                    ],
                }
            ]
        )
        assert result[0].content == [
            {"type": "text", "text": "Describe this."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]

    def test_user_file_base64(self):
        result = convert_model_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this file?"},
                        {"type": "file", "data": "JVBERi0xLjQK", "mediaType": "application/pdf"},
                    ],
                }
            ]
        )
        assert result[0].content == [
            {"type": "text", "text": "What is in this file?"},
            {"type": "file", "data": "JVBERi0xLjQK", "mimeType": "application/pdf", "filename": "file.pdf"},
        ]

    def test_user_mixed_multimodal(self):
        result = convert_model_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these:"},
                        {"type": "image", "image": "https://example.com/image1.jpg"},
                        {"type": "text", "text": "And this document:"},
                        {"type": "file", "data": "https://example.com/doc.pdf", "mediaType": "application/pdf"},
                    ],
                }
            ]
        )
        assert result[0].content == [
            {"type": "text", "text": "Compare these:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "text", "text": "And this document:"},
            {
                "type": "file",
                "url": "https://example.com/doc.pdf",
                "mimeType": "application/pdf",
                "filename": "file.pdf",
            },
        ]

    def test_assistant_text(self):
        result = convert_model_messages([{"role": "assistant", "content": "Hello!"}])
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hello!"

    def test_assistant_tool_call(self):
        result = convert_model_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool-call", "toolCallId": "c1", "toolName": "get_weather", "input": {"city": "NYC"}}
                    ],
                }
            ]
        )
        tc = result[0].tool_calls[0]
        assert tc["id"] == "c1"
        assert tc["name"] == "get_weather"
        assert tc["args"] == {"city": "NYC"}

    def test_tool_text_output(self):
        result = convert_model_messages(
            [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "c1",
                            "toolName": "w",
                            "output": {"type": "text", "value": "Sunny"},
                        }
                    ],
                }
            ]
        )
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == "Sunny"

    def test_tool_json_output(self):
        result = convert_model_messages(
            [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "c1",
                            "toolName": "d",
                            "output": {"type": "json", "value": {"temp": 72}},
                        }
                    ],
                }
            ]
        )
        assert result[0].content == '{"temp":72}'

    def test_sequence(self):
        result = convert_model_messages(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)


# ═══════════════════════════════════════════════════════════════════════════
# toBaseMessages
# ═══════════════════════════════════════════════════════════════════════════


class TestToBaseMessages:
    @pytest.mark.asyncio
    async def test_basic(self):
        result = await to_base_messages(
            [
                {"id": "1", "role": "user", "parts": [{"type": "text", "text": "Hello!"}]},
                {"id": "2", "role": "assistant", "parts": [{"type": "text", "text": "Hi!"}]},
            ]
        )
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)

    @pytest.mark.asyncio
    async def test_system(self):
        result = await to_base_messages(
            [
                {"id": "1", "role": "system", "parts": [{"type": "text", "text": "Be helpful."}]},
            ]
        )
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "Be helpful."

    @pytest.mark.asyncio
    async def test_user_with_image_file(self):
        result = await to_base_messages(
            [
                {
                    "id": "1",
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "file", "url": "data:image/png;base64,abc123", "mediaType": "image/png"},
                    ],
                }
            ]
        )
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
