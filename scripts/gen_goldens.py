#!/usr/bin/env python3
"""Generate golden fixtures from LiteLLM v1.100.1 adapters.

Usage:
  pip install litellm==1.100.1
  python scripts/gen_goldens.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "goldens"


def write(name: str, obj: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def main() -> int:
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )
    except ImportError as e:
        print("install litellm==1.100.1 first:", e, file=sys.stderr)
        return 1

    anthropic_req = {
        "model": "gpt-4o",
        "max_tokens": 32,
        "system": "be brief",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "name": "lookup",
                "description": "d",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "stop_sequences": ["END"],
        "metadata": {"user_id": "u1"},
    }

    adapter = LiteLLMAnthropicMessagesAdapter()
    chat_req, _ = adapter.translate_anthropic_to_openai(anthropic_req)
    write("messages_to_chat.request.json", chat_req)

    resp_adapter = LiteLLMAnthropicToResponsesAPIAdapter()
    responses_req = resp_adapter.translate_request(anthropic_req)
    write("messages_to_responses.request.json", responses_req)

    responses_as_chat = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="gpt-4o",
        input=responses_req["input"],
        responses_api_request={
            k: v
            for k, v in responses_req.items()
            if k not in {"model", "input"}
        },
    )
    write("responses_to_chat.request.json", responses_as_chat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
