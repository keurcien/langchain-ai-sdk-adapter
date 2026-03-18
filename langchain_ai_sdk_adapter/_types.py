"""Type definitions matching ``@ai-sdk/langchain`` types.ts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessageChunk


@dataclass
class LangGraphEventState:
    """Mutable state bag carried across ``processLangGraphEvent`` calls."""

    message_seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    message_concat: dict[str, AIMessageChunk] = field(default_factory=dict)
    emitted_tool_calls: set[str] = field(default_factory=set)
    emitted_images: set[str] = field(default_factory=set)
    emitted_reasoning_ids: set[str] = field(default_factory=set)
    message_reasoning_ids: dict[str, str] = field(default_factory=dict)
    # Maps msgId → {index: {id, name}} for tool-call chunks that arrive without id
    tool_call_info_by_index: dict[str, dict[int, dict[str, str]]] = field(default_factory=dict)
    current_step: int | None = None
    emitted_tool_calls_by_key: dict[str, str] = field(default_factory=dict)
    current_ns: tuple[str, ...] = field(default_factory=tuple)
