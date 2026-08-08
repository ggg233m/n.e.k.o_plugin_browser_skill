from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plugin.plugins.browser_skill.runtime.agent_loop as agent_loop_module
import pytest
from plugin.plugins.browser_skill.runtime.agent_loop import LLMPlanner
from plugin.plugins.browser_skill.runtime.models import NavigateAction, RuntimeSettings


class Config:
    def get_model_api_config(self, purpose: str) -> dict[str, str]:
        assert purpose == "agent"
        return {
            "model": "fake-agent",
            "base_url": "https://endpoint.invalid/v1",
            "api_key": "test",
            "provider_type": "openai_compatible",
        }


class Response:
    content = '{"action":"navigate","url":"https://example.com"}'
    usage_metadata = {
        "input_tokens": 321,
        "output_tokens": 17,
        "total_tokens": 338,
    }


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages: list[Any] = []

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Response:
        self.calls.append(kwargs)
        self.messages.append(messages)
        if "response_format" in kwargs:
            raise ValueError("unsupported parameter: response_format")
        return Response()

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_structured_output_capability_fallback_is_cached_by_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeLLM] = []

    async def factory(*args: Any, **kwargs: Any) -> FakeLLM:
        llm = FakeLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(agent_loop_module, "create_chat_llm_async", factory)
    LLMPlanner._structured_capabilities.clear()
    prompts = Path(__file__).parent.parent / "prompts"

    first = LLMPlanner(
        config_manager=Config(),
        settings=RuntimeSettings(),
        prompts_dir=prompts,
        language="en",
    )
    first_action = await first.decide(
        instruction="Open Example and read only the main heading",
        raw_request="Please open Example",
        observation="Current URL: about:blank",
        history=[],
        verification_required=False,
    )
    await first.close()
    assert first.total_usage == {
        "input_tokens": 321,
        "output_tokens": 17,
        "total_tokens": 338,
        "calls": 1,
        "estimated_calls": 0,
    }

    second = LLMPlanner(
        config_manager=Config(),
        settings=RuntimeSettings(),
        prompts_dir=prompts,
        language="en",
    )
    second_action = await second.decide(
        instruction="Open Example and read only the main heading",
        raw_request="Please open Example",
        observation="Current URL: about:blank",
        history=[],
        verification_required=False,
    )
    await second.close()
    assert second.last_usage["source"] == "provider"
    assert second.last_usage["total_tokens"] == 338

    assert isinstance(first_action, NavigateAction)
    assert isinstance(second_action, NavigateAction)
    assert created[0].calls == [{"response_format": {"type": "json_object"}}, {}]
    assert created[1].calls == [{}]
    payload = json.loads(created[0].messages[0][1]["content"])
    assert payload["execution_goal"] == "Open Example and read only the main heading"
    assert payload["latest_user_request"] == "Please open Example"
    assert "without paraphrasing" in payload["completion_evidence_contract"]
    assert "visible_evidence_ref" in payload["completion_evidence_contract"]
    assert "original_user_request" not in payload
