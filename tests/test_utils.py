"""
Port of utils.test.ts from @ai-sdk/langchain.

Tests for internal utility functions in langchain_ai_sdk_adapter.utils.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from langchain_ai_sdk_adapter._types import LangGraphEventState
from langchain_ai_sdk_adapter.utils import (
    _extract_structured_content,
    convert_assistant_content,
    convert_tool_result_part,
    convert_user_content,
    extract_reasoning_from_content_blocks,
    get_message_id,
    get_message_text,
    is_ai_message_chunk,
    is_plain_message_object,
    is_tool_message_type,
    is_tool_result_part,
    process_langgraph_event,
    process_model_chunk,
)

# ═══════════════════════════════════════════════════════════════════════════
# convertToolResultPart
# ═══════════════════════════════════════════════════════════════════════════


class TestConvertToolResultPart:
    def test_text_output(self):
        result = convert_tool_result_part(
            {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "search",
                "output": {"type": "text", "value": "Sunny, 72°F"},
            }
        )
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "c1"
        assert result.content == "Sunny, 72°F"

    def test_json_output(self):
        result = convert_tool_result_part(
            {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "get_data",
                "output": {"type": "json", "value": {"temperature": 72, "unit": "F"}},
            }
        )
        assert result.content == '{"temperature":72,"unit":"F"}'

    def test_error_text_output(self):
        result = convert_tool_result_part(
            {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "failing",
                "output": {"type": "error-text", "value": "Something went wrong"},
            }
        )
        assert result.content == "Something went wrong"

    def test_error_json_output(self):
        result = convert_tool_result_part(
            {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "failing",
                "output": {"type": "error-json", "value": {"code": 500}},
            }
        )
        assert result.content == '{"code":500}'

    def test_content_output(self):
        result = convert_tool_result_part(
            {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "multi",
                "output": {
                    "type": "content",
                    "value": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "World"},
                    ],
                },
            }
        )
        assert result.content == "Hello World"

    def test_empty_output(self):
        result = convert_tool_result_part(
            {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "noop",
                "output": {},
            }
        )
        assert result.content == ""


# ═══════════════════════════════════════════════════════════════════════════
# convertAssistantContent
# ═══════════════════════════════════════════════════════════════════════════


class TestConvertAssistantContent:
    def test_string_content(self):
        result = convert_assistant_content("Hello!")
        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"

    def test_array_with_text(self):
        result = convert_assistant_content(
            [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "World"},
            ]
        )
        assert result.content == "Hello World"

    def test_array_with_tool_calls(self):
        result = convert_assistant_content(
            [
                {"type": "tool-call", "toolCallId": "c1", "toolName": "get_weather", "input": {"city": "NYC"}},
            ]
        )
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["id"] == "c1"
        assert tc["name"] == "get_weather"
        assert tc["args"] == {"city": "NYC"}

    def test_mixed_text_and_tool_calls(self):
        result = convert_assistant_content(
            [
                {"type": "text", "text": "Let me check. "},
                {"type": "tool-call", "toolCallId": "c1", "toolName": "search", "input": {"q": "test"}},
            ]
        )
        assert result.content == "Let me check. "
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"


# ═══════════════════════════════════════════════════════════════════════════
# convertUserContent
# ═══════════════════════════════════════════════════════════════════════════


class TestConvertUserContent:
    def test_string_content(self):
        result = convert_user_content("Hello!")
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello!"

    def test_text_only_array(self):
        result = convert_user_content([{"type": "text", "text": "Hello"}])
        assert result.content == "Hello"

    def test_image_url(self):
        result = convert_user_content(
            [
                {"type": "text", "text": "What?"},
                {"type": "image", "image": "https://example.com/img.jpg"},
            ]
        )
        assert result.content == [
            {"type": "text", "text": "What?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]

    def test_image_base64(self):
        result = convert_user_content(
            [
                {"type": "image", "image": "iVBORw0K", "mediaType": "image/png"},
            ]
        )
        assert result.content == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}},
        ]

    def test_image_data_url(self):
        result = convert_user_content(
            [
                {"type": "image", "image": "data:image/png;base64,abc123"},
            ]
        )
        assert result.content == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]

    def test_file_url(self):
        result = convert_user_content(
            [
                {"type": "file", "data": "https://example.com/doc.pdf", "mediaType": "application/pdf"},
            ]
        )
        assert result.content == [
            {
                "type": "file",
                "url": "https://example.com/doc.pdf",
                "mimeType": "application/pdf",
                "filename": "file.pdf",
            },
        ]

    def test_file_image_uses_image_url(self):
        result = convert_user_content(
            [
                {"type": "file", "data": "https://example.com/photo.jpg", "mediaType": "image/jpeg"},
            ]
        )
        assert result.content == [
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ]

    def test_file_base64(self):
        result = convert_user_content(
            [
                {"type": "file", "data": "JVBERi0xLjQK", "mediaType": "application/pdf"},
            ]
        )
        assert result.content == [
            {"type": "file", "data": "JVBERi0xLjQK", "mimeType": "application/pdf", "filename": "file.pdf"},
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Type guards
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeGuards:
    def test_is_tool_result_part(self):
        assert is_tool_result_part({"type": "tool-result", "toolCallId": "c1"})
        assert not is_tool_result_part({"type": "text"})
        assert not is_tool_result_part("not a dict")

    def test_is_ai_message_chunk_with_class(self):
        assert is_ai_message_chunk(AIMessageChunk(content="Hello", id="1"))
        assert is_ai_message_chunk(AIMessage(content="Hello", id="1"))

    def test_is_ai_message_chunk_with_dict(self):
        assert is_ai_message_chunk({"type": "ai", "content": "Hello"})
        assert not is_ai_message_chunk({"type": "human", "content": "Hello"})

    def test_is_ai_message_chunk_with_constructor_format(self):
        assert is_ai_message_chunk(
            {
                "type": "constructor",
                "id": ["langchain_core", "messages", "AIMessage"],
                "kwargs": {"id": "ai-1", "content": "Hello"},
            }
        )
        assert is_ai_message_chunk(
            {
                "type": "constructor",
                "id": ["langchain_core", "messages", "AIMessageChunk"],
                "kwargs": {"id": "ai-1", "content": "Hello"},
            }
        )
        assert not is_ai_message_chunk(
            {
                "type": "constructor",
                "id": ["langchain_core", "messages", "HumanMessage"],
                "kwargs": {"id": "human-1", "content": "Hello"},
            }
        )

    def test_is_tool_message_type(self):
        assert is_tool_message_type(ToolMessage(tool_call_id="c1", content="result"))
        assert is_tool_message_type({"type": "tool", "content": "result"})
        assert is_tool_message_type(
            {
                "type": "constructor",
                "id": ["langchain_core", "messages", "ToolMessage"],
                "kwargs": {"tool_call_id": "c1", "content": "result"},
            }
        )
        assert not is_tool_message_type({"type": "ai", "content": "Hello"})

    def test_is_plain_message_object(self):
        assert is_plain_message_object({"type": "ai", "content": "Hello"})
        assert not is_plain_message_object(AIMessage(content="Hello"))
        assert not is_plain_message_object(AIMessageChunk(content="Hello", id="1"))
        assert not is_plain_message_object(ToolMessage(tool_call_id="c1", content="result"))


# ═══════════════════════════════════════════════════════════════════════════
# getMessageId / getMessageText
# ═══════════════════════════════════════════════════════════════════════════


class TestMessageHelpers:
    def test_get_message_id_from_class(self):
        msg = AIMessage(content="Hello", id="msg-1")
        assert get_message_id(msg) == "msg-1"

    def test_get_message_id_from_dict(self):
        assert get_message_id({"id": "msg-1", "type": "ai"}) == "msg-1"

    def test_get_message_id_from_constructor(self):
        assert (
            get_message_id(
                {
                    "type": "constructor",
                    "id": ["langchain_core", "messages", "AIMessage"],
                    "kwargs": {"id": "msg-1", "content": "Hello"},
                }
            )
            == "msg-1"
        )

    def test_get_message_id_none(self):
        assert get_message_id(None) is None
        assert get_message_id(AIMessage(content="Hello")) is None

    def test_get_message_text_from_string(self):
        assert get_message_text({"content": "Hello"}) == "Hello"

    def test_get_message_text_from_list(self):
        assert (
            get_message_text(
                {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "World"},
                    ]
                }
            )
            == "Hello World"
        )

    def test_get_message_text_from_chunk(self):
        chunk = AIMessageChunk(content="Hello", id="1")
        assert get_message_text(chunk) == "Hello"

    def test_get_message_text_empty(self):
        assert get_message_text({"content": 42}) == ""


# ═══════════════════════════════════════════════════════════════════════════
# extractReasoningFromContentBlocks
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractReasoning:
    def test_reasoning_block(self):
        chunk = AIMessageChunk(id="1", content="")
        # type: ignore[attr-defined]
        chunk.contentBlocks = [{"type": "reasoning", "reasoning": "Let me think..."}]
        assert extract_reasoning_from_content_blocks(chunk) == "Let me think..."

    def test_thinking_block(self):
        chunk = AIMessageChunk(id="1", content="")
        # type: ignore[attr-defined]
        chunk.contentBlocks = [{"type": "thinking", "thinking": "First, I need to analyze..."}]
        assert extract_reasoning_from_content_blocks(chunk) == "First, I need to analyze..."

    def test_thinking_block_in_content(self):
        """Anthropic/Gemini put thinking blocks in the content field."""
        chunk = AIMessageChunk(
            id="1",
            content=[
                {"type": "thinking", "thinking": "Let me reason..."},
                {"type": "text", "text": "The answer is 42"},
            ],
        )
        assert extract_reasoning_from_content_blocks(chunk) == "Let me reason..."

    def test_no_reasoning(self):
        chunk = AIMessageChunk(id="1", content="Hello")
        assert extract_reasoning_from_content_blocks(chunk) is None

    def test_no_reasoning_text_only_list(self):
        chunk = AIMessageChunk(
            id="1",
            content=[{"type": "text", "text": "Hello"}],
        )
        assert extract_reasoning_from_content_blocks(chunk) is None

    def test_dict_with_content_blocks(self):
        msg = {"contentBlocks": [{"type": "reasoning", "reasoning": "Hmm..."}]}
        assert extract_reasoning_from_content_blocks(msg) == "Hmm..."


# ═══════════════════════════════════════════════════════════════════════════
# processModelChunk
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessModelChunk:
    def _make_state(self) -> dict[str, Any]:
        return {
            "started": False,
            "message_id": "default-id",
            "reasoning_started": False,
            "text_started": False,
            "text_message_id": None,
            "reasoning_message_id": None,
        }

    def test_text_chunk(self):
        state = self._make_state()
        emit: list[dict[str, Any]] = []
        chunk = AIMessageChunk(content="Hello", id="msg-1")
        process_model_chunk(chunk, state, emit)
        assert {"type": "text-start", "id": "msg-1"} in emit
        assert {"type": "text-delta", "delta": "Hello", "id": "msg-1"} in emit

    def test_reasoning_then_text(self):
        state = self._make_state()
        emit: list[dict[str, Any]] = []

        rc = AIMessageChunk(content="", id="msg-1")
        # type: ignore[attr-defined]
        rc.contentBlocks = [{"type": "reasoning", "reasoning": "Thinking..."}]
        process_model_chunk(rc, state, emit)
        assert {"type": "reasoning-start", "id": "msg-1"} in emit
        assert {"type": "reasoning-delta", "delta": "Thinking...", "id": "msg-1"} in emit

        tc = AIMessageChunk(content="Answer.", id="msg-1")
        process_model_chunk(tc, state, emit)
        assert {"type": "reasoning-end", "id": "msg-1"} in emit
        assert {"type": "text-start", "id": "msg-1"} in emit
        assert {"type": "text-delta", "delta": "Answer.", "id": "msg-1"} in emit

    def test_consistent_ids(self):
        """Message ID should track the first chunk that sets it."""
        state = self._make_state()
        emit: list[dict[str, Any]] = []

        c1 = AIMessageChunk(content="First", id="id-A")
        process_model_chunk(c1, state, emit)

        c2 = AIMessageChunk(content="Second", id="id-B")
        process_model_chunk(c2, state, emit)

        text_starts = [e for e in emit if e["type"] == "text-start"]
        assert len(text_starts) == 1
        assert text_starts[0]["id"] == "id-A"


# ═══════════════════════════════════════════════════════════════════════════
# processLangGraphEvent
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessLangGraphEvent:
    def test_custom_event(self):
        state = LangGraphEventState()
        emit: list[dict[str, Any]] = []
        process_langgraph_event("custom", {"custom": "data"}, state, emit)
        assert emit == [
            {
                "type": "data-custom",
                "id": None,
                "transient": True,
                "data": {"custom": "data"},
            }
        ]

    def test_ai_message_text(self):
        state = LangGraphEventState()
        emit: list[dict[str, Any]] = []
        chunk = AIMessageChunk(content="Hello", id="msg-1")
        process_langgraph_event("messages", (chunk, {}), state, emit)
        assert {"type": "text-start", "id": "msg-1"} in emit
        assert {"type": "text-delta", "delta": "Hello", "id": "msg-1"} in emit

    def test_tool_message(self):
        state = LangGraphEventState()
        emit: list[dict[str, Any]] = []
        tool_msg = ToolMessage(tool_call_id="call-1", content="Sunny", id="msg-1")
        process_langgraph_event("messages", (tool_msg, {}), state, emit)
        assert emit == [{"type": "tool-output-available", "toolCallId": "call-1", "output": "Sunny"}]

    def test_values_event_finalizes(self):
        state = LangGraphEventState()
        emit: list[dict[str, Any]] = []

        chunk = AIMessageChunk(content="Hello", id="msg-1")
        process_langgraph_event("messages", (chunk, {}), state, emit)
        emit.clear()

        process_langgraph_event("values", {}, state, emit)
        assert {"type": "text-end", "id": "msg-1"} in emit

    def test_step_tracking(self):
        state = LangGraphEventState()
        emit: list[dict[str, Any]] = []

        chunk = AIMessageChunk(content="Hello", id="msg-1")
        process_langgraph_event("messages", (chunk, {"langgraph_step": 0}), state, emit)
        assert {"type": "start-step"} in emit
        assert state.current_step == 0

    def test_skip_messages_without_id(self):
        state = LangGraphEventState()
        emit: list[dict[str, Any]] = []
        msg = AIMessage(content="No ID")
        process_langgraph_event("messages", (msg, {}), state, emit)
        assert emit == []

    def test_v2_native_interrupts(self):
        class FakeInterrupt:
            def __init__(self, value):
                self.value = value

        state = LangGraphEventState()
        emit: list[dict[str, Any]] = []
        interrupts = (FakeInterrupt(value={"action_requests": [{"name": "tool_x", "args": {"a": 1}, "id": "call-x1"}]}),)
        process_langgraph_event("values", {"messages": []}, state, emit, interrupts=interrupts)
        approvals = [e for e in emit if e.get("type") == "tool-approval-request"]
        assert len(approvals) == 1
        assert approvals[0]["toolCallId"] == "call-x1"


# ═══════════════════════════════════════════════════════════════════════════
# _extract_structured_content
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractStructuredContent:
    def test_from_tool_message_with_artifact(self):
        msg = ToolMessage(
            tool_call_id="c1",
            content="ok",
            artifact={"structured_content": {"data": [1, 2]}},
        )
        assert _extract_structured_content(msg) == {"data": [1, 2]}

    def test_from_tool_message_without_artifact(self):
        msg = ToolMessage(tool_call_id="c1", content="ok")
        assert _extract_structured_content(msg) is None

    def test_from_tool_message_artifact_no_sc(self):
        msg = ToolMessage(tool_call_id="c1", content="ok", artifact={"other": 1})
        assert _extract_structured_content(msg) is None

    def test_from_dict_with_artifact(self):
        d = {"artifact": {"structured_content": {"items": []}}}
        assert _extract_structured_content(d) == {"items": []}

    def test_from_none(self):
        assert _extract_structured_content(None) is None

    def test_non_dict_structured_content_ignored(self):
        msg = ToolMessage(
            tool_call_id="c1",
            content="ok",
            artifact={"structured_content": "not a dict"},
        )
        assert _extract_structured_content(msg) is None
