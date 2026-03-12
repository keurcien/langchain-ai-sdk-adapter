"""HTTP response helpers for FastAPI / Starlette."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from starlette.responses import StreamingResponse

from .adapter import to_ui_message_stream
from .callbacks import StreamCallbacks


def _encode_sse(chunk: dict[str, Any]) -> str:
    """Encode a UIMessageChunk dict as an SSE ``data:`` frame."""
    return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"


async def _sse_stream(
    stream: AsyncIterator[Any],
    callbacks: StreamCallbacks | None = None,
) -> AsyncIterator[str]:
    """Yield SSE frames from a LangChain/LangGraph stream."""
    async for chunk in to_ui_message_stream(stream, callbacks=callbacks):
        yield _encode_sse(chunk)
    yield "data: [DONE]\n\n"


def create_ui_message_stream_response(
    stream: AsyncIterator[Any],
    *,
    callbacks: StreamCallbacks | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """Wrap a LangChain / LangGraph stream in a ``StreamingResponse``
    with AI SDK UI Message Stream headers."""
    h = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "x-vercel-ai-ui-message-stream": "v1",
    }
    if headers:
        h.update(headers)
    return StreamingResponse(
        _sse_stream(stream, callbacks=callbacks),
        status_code=status_code,
        headers=h,
        media_type="text/event-stream",
    )
