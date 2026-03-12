"""Callback definitions matching ``@ai-sdk/langchain`` stream-callbacks.ts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

MaybeAwaitable = Awaitable[None] | None
Callback = Callable[..., MaybeAwaitable]


@dataclass
class StreamCallbacks:
    """Lifecycle callbacks for ``to_ui_message_stream``."""

    on_start: Callable[[], MaybeAwaitable] | None = None
    on_token: Callable[[str], MaybeAwaitable] | None = None
    on_text: Callable[[str], MaybeAwaitable] | None = None
    on_final: Callable[[str], MaybeAwaitable] | None = None
    on_finish: Callable[[Any], MaybeAwaitable] | None = None
    on_error: Callable[[Exception], MaybeAwaitable] | None = None
    on_abort: Callable[[], MaybeAwaitable] | None = None
