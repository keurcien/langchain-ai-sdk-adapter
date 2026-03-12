"""
Port of stream-callbacks.test.ts from @ai-sdk/langchain.

The TS version tests ``createCallbacksTransformer`` which processes string
chunks through onStart/onToken/onText/onFinal callbacks.  In Python there
is no ``createCallbacksTransformer`` — those callbacks are invoked inline
by ``to_ui_message_stream``.  These tests exercise the same callback
semantics through the Python adapter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from langchain_ai_sdk_adapter.adapter import to_ui_message_stream
from langchain_ai_sdk_adapter.callbacks import StreamCallbacks

# ── Helpers ───────────────────────────────────────────────────────────────


async def _to_list(stream: AsyncIterator[Any]) -> list[dict[str, Any]]:
    return [chunk async for chunk in stream]


async def _model_stream(*texts: str) -> AsyncIterator[Any]:
    """Yield AIMessageChunks that each carry a text delta."""
    for _i, text in enumerate(texts):
        yield AIMessageChunk(content=text, id="cb-test-1")


# ═══════════════════════════════════════════════════════════════════════════
# StreamCallbacks via to_ui_message_stream  (mirrors stream-callbacks.test.ts)
# ═══════════════════════════════════════════════════════════════════════════


class TestStreamCallbacks:
    @pytest.mark.asyncio
    async def test_pass_through_without_callbacks(self):
        result = await _to_list(to_ui_message_stream(_model_stream("Hello", " ", "World")))
        deltas = [e["delta"] for e in result if e.get("type") == "text-delta"]
        assert deltas == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_pass_through_with_empty_callbacks(self):
        result = await _to_list(to_ui_message_stream(_model_stream("Hello", " ", "World"), callbacks=StreamCallbacks()))
        deltas = [e["delta"] for e in result if e.get("type") == "text-delta"]
        assert deltas == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_on_start_called_once(self):
        calls = []
        cbs = StreamCallbacks(on_start=lambda: calls.append("start"))
        await _to_list(to_ui_message_stream(_model_stream("Hello", " ", "World"), callbacks=cbs))
        assert calls == ["start"]

    @pytest.mark.asyncio
    async def test_async_on_start(self):
        calls = []

        async def on_start():
            calls.append("start")

        cbs = StreamCallbacks(on_start=on_start)
        await _to_list(to_ui_message_stream(_model_stream("Hello"), callbacks=cbs))
        assert calls == ["start"]

    @pytest.mark.asyncio
    async def test_on_token_for_each_delta(self):
        tokens = []
        cbs = StreamCallbacks(on_token=lambda t: tokens.append(t))
        await _to_list(to_ui_message_stream(_model_stream("Hello", " ", "World"), callbacks=cbs))
        assert tokens == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_async_on_token(self):
        tokens = []

        async def on_token(t: str):
            tokens.append(t)

        cbs = StreamCallbacks(on_token=on_token)
        await _to_list(to_ui_message_stream(_model_stream("Hello", "World"), callbacks=cbs))
        assert tokens == ["Hello", "World"]

    @pytest.mark.asyncio
    async def test_on_text_for_each_delta(self):
        texts = []
        cbs = StreamCallbacks(on_text=lambda t: texts.append(t))
        await _to_list(to_ui_message_stream(_model_stream("Hello", " ", "World"), callbacks=cbs))
        assert texts == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_async_on_text(self):
        texts = []

        async def on_text(t: str):
            texts.append(t)

        cbs = StreamCallbacks(on_text=on_text)
        await _to_list(to_ui_message_stream(_model_stream("Hello"), callbacks=cbs))
        assert texts == ["Hello"]

    @pytest.mark.asyncio
    async def test_on_final_with_aggregated_text(self):
        received = []
        cbs = StreamCallbacks(on_final=lambda t: received.append(t))
        await _to_list(to_ui_message_stream(_model_stream("Hello", " ", "World"), callbacks=cbs))
        assert received == ["Hello World"]

    @pytest.mark.asyncio
    async def test_async_on_final(self):
        received = []

        async def on_final(t: str):
            received.append(t)

        cbs = StreamCallbacks(on_final=on_final)
        await _to_list(to_ui_message_stream(_model_stream("Hello", "World"), callbacks=cbs))
        assert received == ["HelloWorld"]

    @pytest.mark.asyncio
    async def test_on_final_empty_when_no_text(self):
        received = []
        cbs = StreamCallbacks(on_final=lambda t: received.append(t))

        async def empty_stream() -> AsyncIterator[Any]:
            yield ["values", {}]

        await _to_list(to_ui_message_stream(empty_stream(), callbacks=cbs))
        assert received == [""]

    @pytest.mark.asyncio
    async def test_all_callbacks_in_correct_order(self):
        call_order: list[str] = []
        cbs = StreamCallbacks(
            on_start=lambda: call_order.append("start"),
            on_token=lambda t: call_order.append(f"token:{t}"),
            on_text=lambda t: call_order.append(f"text:{t}"),
            on_final=lambda t: call_order.append(f"final:{t}"),
        )
        await _to_list(to_ui_message_stream(_model_stream("A", "B"), callbacks=cbs))
        assert call_order == [
            "start",
            "token:A",
            "text:A",
            "token:B",
            "text:B",
            "final:AB",
        ]

    @pytest.mark.asyncio
    async def test_single_character_messages(self):
        tokens = []
        received_final = []
        cbs = StreamCallbacks(
            on_token=lambda t: tokens.append(t),
            on_final=lambda t: received_final.append(t),
        )
        await _to_list(to_ui_message_stream(_model_stream("a", "b", "c"), callbacks=cbs))
        assert len(tokens) == 3
        assert received_final == ["abc"]

    @pytest.mark.asyncio
    async def test_special_characters(self):
        received = []
        cbs = StreamCallbacks(on_final=lambda t: received.append(t))
        await _to_list(to_ui_message_stream(_model_stream("Hello\n", "World\t", "!"), callbacks=cbs))
        assert received == ["Hello\nWorld\t!"]

    @pytest.mark.asyncio
    async def test_unicode_characters(self):
        received = []
        cbs = StreamCallbacks(on_final=lambda t: received.append(t))
        await _to_list(
            to_ui_message_stream(
                _model_stream("こんにちは", " ", "🌍"),
                callbacks=cbs,
            )
        )
        assert received == ["こんにちは 🌍"]
