from __future__ import annotations

import os
from uuid import UUID


def _normalize_langsmith_workspace_id() -> None:
    workspace_id = (os.getenv("LANGSMITH_WORKSPACE_ID") or "").strip()
    if not workspace_id:
        return

    try:
        UUID(workspace_id)
    except ValueError:
        # Fall back to the API key's default workspace when a human-readable
        # slug was provided instead of the UUID LangSmith expects.
        os.environ.pop("LANGSMITH_WORKSPACE_ID", None)
        print(
            "[observability] Ignoring invalid LANGSMITH_WORKSPACE_ID. "
            "Expected a UUID; falling back to the API key default workspace."
        )


_normalize_langsmith_workspace_id()


try:
    from langsmith import traceable as _traceable
    from langsmith.wrappers import wrap_openai as _wrap_openai
except ImportError:
    def traceable(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    def wrap_openai(client):
        return client
else:
    def traceable(*args, **kwargs):
        return _traceable(*args, **kwargs)

    def wrap_openai(client):
        return _wrap_openai(client)
