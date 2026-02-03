"""Deprecated: document loading from uploads has been removed."""


class DocumentLoader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DocumentLoader has been removed. Use config/agent_prompt.json as the documentation source."
        )
