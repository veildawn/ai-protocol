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

    write_gemini_goldens(chat_req, responses_as_chat)
    return 0


def write_gemini_goldens(messages_as_chat: dict, responses_as_chat: dict) -> None:
    """Generate the gemini-dialect goldens through the reference's own paths.

    The reference has no direct messages/gemini or responses/gemini fold: both
    compose through a chat body first (the anthropic passthrough adapter, the
    responses bridge), and the chat body then goes through the vertex config's
    parameter mapping and request builder. The goldens pin that composition.
    """
    import types

    import httpx
    import litellm
    from litellm.llms.vertex_ai.gemini.transformation import sync_transform_request_body
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
    from litellm.types.utils import ModelResponse

    def to_gemini_request(chat_body: dict) -> dict:
        model = "gemini-2.5-pro"
        nd = {k: v for k, v in chat_body.items() if k not in ("messages", "model")}
        opt = litellm.get_optional_params(
            model=model, custom_llm_provider="vertex_ai", drop_params=False, **nd
        )
        return sync_transform_request_body(
            gemini_api_key="dummy",
            messages=[dict(m) for m in chat_body["messages"]],
            api_base=None,
            model=model,
            client=None,
            timeout=None,
            extra_headers=None,
            optional_params=dict(opt),
            logging_obj=types.SimpleNamespace(optional_params=dict(opt)),
            custom_llm_provider="vertex_ai",
            litellm_params={},
            vertex_project=None,
            vertex_location=None,
            vertex_auth_header=None,
        )

    chat_direct = {
        "model": "gemini-2.5-pro",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Boston"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        "temperature": 0.5,
        "max_tokens": 32,
        "stop": "END",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "d",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": "required",
        "reasoning_effort": "high",
    }
    write("chat_to_gemini.request.json", to_gemini_request(chat_direct))
    write("messages_to_gemini.request.json", to_gemini_request(messages_as_chat))
    write("responses_to_gemini.request.json", to_gemini_request(responses_as_chat))

    gemini_resp = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "let me think", "thought": True},
                        {"functionCall": {"name": "lookup", "args": {"q": "x"}}},
                    ],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8,
        },
        "responseId": "resp-1",
    }
    raw = httpx.Response(
        200, json=gemini_resp, request=httpx.Request("POST", "https://example.test")
    )
    logging_obj = types.SimpleNamespace(
        post_call=lambda **kw: None, optional_params={}, stream=False, verbose=False
    )
    chat_out = VertexGeminiConfig()._transform_google_generate_content_to_openai_model_response(
        completion_response=gemini_resp,
        model_response=ModelResponse(),
        model="gemini-2.5-pro",
        logging_obj=logging_obj,
        raw_response=raw,
    ).model_dump(exclude_none=True)
    write("gemini_to_chat.response.json", scrub_volatile(chat_out))


def scrub_volatile(obj):
    """Replace the clock and uuid values the reference mints per run, so the
    golden is stable across regenerations. Everything else is the reference's
    output verbatim."""
    import re

    call_id = re.compile(r"^call_[0-9a-f]{8,}$")

    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str) and call_id.match(v):
            return "call_<uuid>"
        if isinstance(v, int) and v > 1600000000:
            return 0
        return v

    return walk(obj)


if __name__ == "__main__":
    raise SystemExit(main())
