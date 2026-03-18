# langchain-ai-sdk-adapter

Python port of @ai-sdk/langchain.

Converts LangChain / LangGraph streams into Vercel AI SDK `UIMessageStream` chunks, so you can use `useChat` on the frontend with a LangGraph agent on the backend.

## Installation

```bash
pip install langchain-ai-sdk-adapter
```

## Quick start

```python
from langchain_ai_sdk_adapter import to_ui_message_stream, create_ui_message_stream_response

# In your FastAPI / Starlette endpoint:
async def chat(request):
    body = await request.json()
    messages = body["messages"]

    # Your LangGraph agent — must use stream v2
    stream = agent.astream(
        {"messages": messages},
        stream_mode=["messages", "values"],
        version="v2",
    )

    return create_ui_message_stream_response(stream)
```

## LangGraph stream v2

**v0.3.0+ requires LangGraph stream v2.** Pass `version="v2"` to `.stream()` / `.astream()`.

v2 events are typed dicts with a consistent shape:

```python
{
    "type": "messages" | "values" | "custom" | ...,
    "ns": ("tools:abc-123",),  # subgraph namespace tuple
    "data": ...,               # payload (varies by type)
    "interrupts": (...),       # only on "values" events
}
```

### Namespace tracking (`ns`)

When streaming with `subgraphs=True`, events carry an `ns` tuple identifying which subgraph produced them (e.g. `("tools:ef4e3406-acaf-6fe2-657e-63b7bd1872a1",)`).

The adapter emits `data-namespace` parts on namespace transitions:

```python
{"type": "data-namespace", "data": {"ns": ["tools:abc-123"]}}
{"type": "text-start", "id": "msg-1"}
{"type": "text-delta", "delta": "Exploring tables...", "id": "msg-1"}
# ns changes back to root
{"type": "data-namespace", "data": {"ns": []}}
{"type": "text-start", "id": "msg-2"}
{"type": "text-delta", "delta": "Based on the analysis...", "id": "msg-2"}
```

These are emitted as AI SDK [data parts](https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data), so they appear in `message.parts[]` on the frontend. You can use them to build collapsible subagent UIs:

```tsx
let currentNs: string[] = [];
message.parts.map(part => {
  if (part.type === 'data-namespace') {
    currentNs = part.data.ns;
    return null;
  }
  if (part.type === 'text' && currentNs.length > 0) {
    return <CollapsibleSubagent ns={currentNs}>{part.text}</CollapsibleSubagent>;
  }
  return <Markdown>{part.text}</Markdown>;
});
```

### Interrupts (human-in-the-loop)

v2 carries interrupts as a dedicated field on `values` events (instead of `__interrupt__` in the state dict). The adapter reads them natively and emits `tool-approval-request` chunks for each action request.

### Supported event types

| v2 `type` | Adapter behaviour |
|---|---|
| `messages` | Text streaming, tool call chunks, reasoning, tool outputs |
| `values` | Finalization, un-streamed tool calls, interrupts |
| `custom` | Passed through as `data-{type}` parts |
| `updates`, `checkpoints`, `tasks`, `debug` | Ignored (no-op) |

## Message conversion

```python
from langchain_ai_sdk_adapter import convert_model_messages, to_lc_messages, to_ui_messages

# AI SDK ModelMessage[] → LangChain BaseMessage[]
lc_messages = convert_model_messages(model_messages)

# AI SDK UIMessage[] → LangChain BaseMessage[]
lc_messages = await to_lc_messages(ui_messages)

# LangChain BaseMessage[] → AI SDK UIMessage[]
ui_messages = to_ui_messages(lc_messages)
```
