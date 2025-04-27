"""Utility & helper functions."""

import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models.base import ChatOpenAI


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    if provider == "ollama":
        ollama_url = os.environ.get("OLLAMA_URL")
        ollama_key = os.environ.get("OLLAMA_KEY")
        if ollama_url == "":
            ollama_url = "http://localhost:11434/v1"
            print(f"Warning: OLLAMA_URL not set, use default URL: {ollama_url}")
        return ChatOpenAI(
            model=model,
            api_key=ollama_key,
            base_url=ollama_url,
        )
    return init_chat_model(model, model_provider=provider)
