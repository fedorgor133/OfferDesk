"""Deprecated: conversation splitting from uploads has been removed."""


class ConversationSplitter:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ConversationSplitter has been removed. Use config/agent_prompt.json as the documentation source."
        )
