"""
langchain-ai-sdk-adapter — Python port of ``@ai-sdk/langchain``.
"""

from importlib.metadata import version

__version__ = version("langchain-ai-sdk-adapter")

from .adapter import convert_model_messages, to_lc_messages, to_ui_message_stream
from .callbacks import StreamCallbacks
from .messages import to_ui_messages
from .response import create_ui_message_stream_response

__all__ = [
    "convert_model_messages",
    "to_lc_messages",
    "to_ui_message_stream",
    "to_ui_messages",
    "create_ui_message_stream_response",
    "StreamCallbacks",
]
