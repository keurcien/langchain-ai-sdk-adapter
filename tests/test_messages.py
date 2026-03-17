"""Tests for to_ui_messages — batch conversion of persisted LangChain messages."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_ai_sdk_adapter.messages import to_ui_messages

# ═══════════════════════════════════════════════════════════════════════════
# Basic message types
# ═══════════════════════════════════════════════════════════════════════════


class TestHumanMessages:
    def test_simple_text(self):
        result = to_ui_messages([HumanMessage(content="Hello")])
        assert result == [{"role": "user", "parts": [{"type": "text", "text": "Hello"}]}]

    def test_empty_content_skipped(self):
        result = to_ui_messages([HumanMessage(content="")])
        assert result == []

    def test_list_content_text_only(self):
        result = to_ui_messages(
            [HumanMessage(content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": " World"}])]
        )
        assert result == [
            {"role": "user", "parts": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " World"}]}
        ]

    def test_image_url_content(self):
        result = to_ui_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                    ]
                )
            ]
        )
        assert len(result) == 1
        parts = result[0]["parts"]
        # Files come before text
        assert parts[0] == {"type": "file", "url": "data:image/png;base64,abc123", "mediaType": "image/png"}
        assert parts[1] == {"type": "text", "text": "What is this?"}

    def test_image_with_source(self):
        result = to_ui_messages(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": "abcdef"},
                            "filename": "photo.jpg",
                        },
                    ]
                )
            ]
        )
        parts = result[0]["parts"]
        assert parts[0] == {
            "type": "file",
            "url": "data:image/jpeg;base64,abcdef",
            "mediaType": "image/jpeg",
            "filename": "photo.jpg",
        }

    def test_file_with_data_and_mime(self):
        """LangChain file format: {type: file, data: ..., mimeType: ...}."""
        result = to_ui_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "file", "data": "JVBERi0xLjQK", "mimeType": "application/pdf", "filename": "doc.pdf"},
                    ]
                )
            ]
        )
        parts = result[0]["parts"]
        assert parts[0] == {
            "type": "file",
            "url": "data:application/pdf;base64,JVBERi0xLjQK",
            "mediaType": "application/pdf",
            "filename": "doc.pdf",
        }

    def test_file_with_url(self):
        result = to_ui_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "file", "url": "https://example.com/file.pdf", "mimeType": "application/pdf"},
                    ]
                )
            ]
        )
        parts = result[0]["parts"]
        assert parts[0] == {
            "type": "file",
            "url": "https://example.com/file.pdf",
            "mediaType": "application/pdf",
        }

    def test_string_items_in_list(self):
        result = to_ui_messages([HumanMessage(content=["Hello", "World"])])
        parts = result[0]["parts"]
        assert parts == [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]


class TestSystemMessages:
    def test_simple(self):
        result = to_ui_messages([SystemMessage(content="Be helpful.")])
        assert result == [{"role": "system", "parts": [{"type": "text", "text": "Be helpful."}]}]

    def test_empty_skipped(self):
        result = to_ui_messages([SystemMessage(content="")])
        assert result == []


class TestAssistantMessages:
    def test_simple_text(self):
        result = to_ui_messages([AIMessage(content="Hello!", id="ai-1")])
        assert result == [{"role": "assistant", "parts": [{"type": "text", "text": "Hello!"}]}]

    def test_empty_content_with_no_tools_skipped(self):
        result = to_ui_messages([AIMessage(content="", id="ai-1")])
        assert result == []

    def test_list_content_text(self):
        result = to_ui_messages([AIMessage(content=[{"type": "text", "text": "The answer is 42."}], id="ai-1")])
        assert result == [{"role": "assistant", "parts": [{"type": "text", "text": "The answer is 42."}]}]


# ═══════════════════════════════════════════════════════════════════════════
# Reasoning / thinking content
# ═══════════════════════════════════════════════════════════════════════════


class TestReasoning:
    def test_thinking_block(self):
        result = to_ui_messages(
            [
                AIMessage(
                    content=[
                        {"type": "thinking", "thinking": "Let me reason about this..."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                    id="ai-1",
                )
            ]
        )
        parts = result[0]["parts"]
        assert parts[0] == {"type": "reasoning", "text": "Let me reason about this..."}
        assert parts[1] == {"type": "text", "text": "The answer is 42."}


# ═══════════════════════════════════════════════════════════════════════════
# Tool calls and results
# ═══════════════════════════════════════════════════════════════════════════


class TestToolCalls:
    def test_tool_call_with_result(self):
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "get_weather", "args": {"city": "Paris"}}],
                ),
                ToolMessage(
                    content='{"temperature": 22, "unit": "celsius"}',
                    tool_call_id="call-1",
                    id="tool-1",
                ),
            ]
        )

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        tool_part = result[0]["parts"][0]
        assert tool_part["type"] == "tool-invocation"
        assert tool_part["toolInvocationId"] == "call-1"
        assert tool_part["toolName"] == "get_weather"
        assert tool_part["args"] == {"city": "Paris"}
        assert tool_part["state"] == "result"
        assert tool_part["result"] == {"temperature": 22, "unit": "celsius"}

    def test_tool_call_without_result(self):
        """Tool call with no corresponding ToolMessage gets state=call."""
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "get_weather", "args": {"city": "Paris"}}],
                ),
            ]
        )

        tool_part = result[0]["parts"][0]
        assert tool_part["state"] == "call"
        assert "result" not in tool_part

    def test_tool_call_with_error(self):
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "failing_tool", "args": {}}],
                ),
                ToolMessage(
                    content="Connection timeout",
                    tool_call_id="call-1",
                    id="tool-1",
                    status="error",
                ),
            ]
        )

        tool_part = result[0]["parts"][0]
        assert tool_part["state"] == "error"
        assert tool_part["error"] == "Connection timeout"
        assert "result" not in tool_part

    def test_multiple_tool_calls(self):
        result = to_ui_messages(
            [
                AIMessage(
                    content="Let me check both.",
                    id="ai-1",
                    tool_calls=[
                        {"id": "call-1", "name": "get_weather", "args": {"city": "Paris"}},
                        {"id": "call-2", "name": "get_weather", "args": {"city": "London"}},
                    ],
                ),
                ToolMessage(content='{"temp": 22}', tool_call_id="call-1", id="tool-1"),
                ToolMessage(content='{"temp": 15}', tool_call_id="call-2", id="tool-2"),
            ]
        )

        parts = result[0]["parts"]
        assert parts[0] == {"type": "text", "text": "Let me check both."}
        assert parts[1]["toolInvocationId"] == "call-1"
        assert parts[1]["result"] == {"temp": 22}
        assert parts[2]["toolInvocationId"] == "call-2"
        assert parts[2]["result"] == {"temp": 15}

    def test_tool_call_then_text_response(self):
        """Full flow: AI calls tool → tool result → AI responds with text."""
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "get_weather", "args": {"city": "Paris"}}],
                ),
                ToolMessage(content='{"temp": 22}', tool_call_id="call-1", id="tool-1"),
                AIMessage(content="The weather in Paris is 22°C.", id="ai-2"),
            ]
        )

        # All grouped into one assistant message
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        parts = result[0]["parts"]
        assert parts[0]["type"] == "tool-invocation"
        assert parts[0]["state"] == "result"
        assert parts[1] == {"type": "text", "text": "The weather in Paris is 22°C."}

    def test_mcp_content_parts_normalized(self):
        """MCP tools return [{"type": "text", "text": "{...}"}] — should be normalized."""
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "search", "args": {"q": "test"}}],
                ),
                ToolMessage(
                    content=[{"type": "text", "text": '{"results": [1, 2, 3]}'}],
                    tool_call_id="call-1",
                    id="tool-1",
                ),
            ]
        )

        tool_part = result[0]["parts"][0]
        assert tool_part["result"] == {"results": [1, 2, 3]}

    def test_mcp_content_plain_text(self):
        """MCP text content that isn't JSON stays as string."""
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "echo", "args": {}}],
                ),
                ToolMessage(
                    content=[{"type": "text", "text": "Hello world"}],
                    tool_call_id="call-1",
                    id="tool-1",
                ),
            ]
        )

        tool_part = result[0]["parts"][0]
        assert tool_part["result"] == "Hello world"


# ═══════════════════════════════════════════════════════════════════════════
# Structured content
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredContent:
    def test_structured_content_in_result(self):
        """structured_content from ToolMessage.artifact is wrapped with camelCase key."""
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "render_chart", "args": {}}],
                ),
                ToolMessage(
                    content="[Chart rendered]",
                    tool_call_id="call-1",
                    id="tool-1",
                    artifact={"structured_content": {"type": "chart", "data": [1, 2, 3]}},
                ),
            ]
        )

        tool_part = result[0]["parts"][0]
        assert tool_part["state"] == "result"
        assert tool_part["result"] == {
            "_text": "[Chart rendered]",
            "structuredContent": {"type": "chart", "data": [1, 2, 3]},
        }

    def test_no_structured_content(self):
        """Without structured_content in artifact, result is plain."""
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "search", "args": {}}],
                ),
                ToolMessage(
                    content="plain result",
                    tool_call_id="call-1",
                    id="tool-1",
                    artifact={"other_key": "value"},
                ),
            ]
        )

        tool_part = result[0]["parts"][0]
        assert tool_part["result"] == "plain result"

    def test_structured_content_with_json_text(self):
        """structured_content + MCP text content: _text is normalized."""
        result = to_ui_messages(
            [
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "widget", "args": {}}],
                ),
                ToolMessage(
                    content=[{"type": "text", "text": '{"status": "ok"}'}],
                    tool_call_id="call-1",
                    id="tool-1",
                    artifact={"structured_content": {"type": "widget"}},
                ),
            ]
        )

        tool_part = result[0]["parts"][0]
        assert tool_part["result"] == {
            "_text": {"status": "ok"},
            "structuredContent": {"type": "widget"},
        }


# ═══════════════════════════════════════════════════════════════════════════
# Full conversations
# ═══════════════════════════════════════════════════════════════════════════


class TestFullConversations:
    def test_multi_turn(self):
        """Multi-turn conversation: system → user → assistant → user → assistant."""
        result = to_ui_messages(
            [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="Hi"),
                AIMessage(content="Hello! How can I help?", id="ai-1"),
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4", id="ai-2"),
            ]
        )

        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"
        assert result[4]["role"] == "assistant"

    def test_tool_use_conversation(self):
        """User asks question → AI calls tool → tool result → AI responds."""
        result = to_ui_messages(
            [
                HumanMessage(content="Weather in Paris?"),
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[{"id": "call-1", "name": "get_weather", "args": {"city": "Paris"}}],
                ),
                ToolMessage(content='{"temp": 22}', tool_call_id="call-1", id="tool-1"),
                AIMessage(content="It's 22°C in Paris.", id="ai-2"),
            ]
        )

        assert len(result) == 2  # user + assistant (grouped)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

        assistant_parts = result[1]["parts"]
        assert assistant_parts[0]["type"] == "tool-invocation"
        assert assistant_parts[0]["state"] == "result"
        assert assistant_parts[1] == {"type": "text", "text": "It's 22°C in Paris."}

    def test_empty_list(self):
        assert to_ui_messages([]) == []

    def test_unknown_message_type_skipped(self):
        """Unknown message types are silently skipped."""

        class CustomMessage(HumanMessage):
            pass

        # CustomMessage is a subclass of HumanMessage, so it will be processed.
        # Let's test with a truly unknown BaseMessage subclass.
        from langchain_core.messages import ChatMessage

        result = to_ui_messages([ChatMessage(content="hi", role="custom")])
        assert result == []
