from __future__ import annotations

import json

import pytest
from plugin.plugins.browser_skill.runtime.models import (
    ClickAction,
    FillAction,
    NavigateAction,
    RuntimeSettings,
    ScrollAction,
    action_schema_json,
    parse_agent_action,
)
from plugin.plugins.browser_skill.runtime.policy import (
    PolicyViolation,
    additional_agent_tab_requested,
    is_search_fill,
    is_sensitive_fill,
    requested_agent_tab_count,
    requires_critical_confirmation,
    requires_persistent_session,
    user_tab_requested,
    validate_action,
    validate_http_url,
)
from pydantic import ValidationError


@pytest.mark.parametrize(
    "url",
    ["chrome://settings", "edge://flags", "file:///tmp/a", "javascript:alert(1)"],
)
def test_rejects_non_http_navigation(url: str) -> None:
    with pytest.raises(PolicyViolation, match="http/https"):
        validate_http_url(url)


def test_rejects_credentials_embedded_in_url() -> None:
    with pytest.raises(PolicyViolation, match="认证信息"):
        validate_action(NavigateAction(action="navigate", url="https://alice:secret@example.com"))


def test_parser_accepts_fenced_json_but_rejects_unknown_action() -> None:
    action = parse_agent_action(
        '```json\n{"action":"navigate","url":"https://example.com"}\n```'
    )
    assert isinstance(action, NavigateAction)
    with pytest.raises(ValidationError):
        parse_agent_action('{"action":"evaluate","script":"document.cookie"}')


def test_scroll_action_is_bounded_to_one_page() -> None:
    action = parse_agent_action(
        '{"action":"scroll","direction":"down","pages":1,"until":"下一条结果"}'
    )
    assert isinstance(action, ScrollAction)
    assert action.pages == 1
    with pytest.raises(ValidationError):
        parse_agent_action('{"action":"scroll","direction":"down","pages":2}')
    with pytest.raises(ValidationError):
        parse_agent_action('{"action":"scroll","direction":"bottom","pages":1}')


def test_agent_prompt_schema_is_compact_without_losing_constraints() -> None:
    schema_text = action_schema_json()
    schema = json.loads(schema_text)

    assert len(schema_text) < 10_000
    request_help = schema["$defs"]["RequestHelpAction"]["properties"]
    assert "title" in request_help
    assert request_help["prompt"]["maxLength"] == 2000
    assert "default" not in request_help["help_kind"]


def test_password_fill_is_always_human_work() -> None:
    action = FillAction(action="fill", target="@e7", value="must-not-be-sent")
    assert is_sensitive_fill(action, 'textbox "Password" @e7')


def test_fill_submit_is_limited_to_observed_search_fields() -> None:
    action = FillAction(action="fill", target="@e7", value="cats", submit=True)
    assert is_search_fill(action, 'textbox "搜索" @e7')
    assert not is_search_fill(action, 'textbox "昵称" @e7')


def test_critical_click_requires_confirmation() -> None:
    action = ClickAction(action="click", target="@e9", reason="提交订单")
    assert requires_critical_confirmation(action, "购买商品", 'button "提交订单" @e9')


@pytest.mark.parametrize(
    "instruction",
    ["操作我已经打开的页面", "请在 current tab 中继续", "使用现有标签完成任务"],
)
def test_explicit_user_tab_intent(instruction: str) -> None:
    assert user_tab_requested(instruction)


def test_media_playback_requires_persistent_session() -> None:
    assert requires_persistent_session("请播放这个视频")
    assert requires_persistent_session("watch the live stream")
    assert not requires_persistent_session("停止播放视频")


def test_additional_agent_tab_requires_explicit_user_intent() -> None:
    assert additional_agent_tab_requested("请在新标签页打开结果")
    assert additional_agent_tab_requested("在另一个标签页里搜索猫娘计划")
    assert additional_agent_tab_requested("分别在两个标签页打开这两个页面")
    assert additional_agent_tab_requested("open the result in a new tab")
    assert not additional_agent_tab_requested("打开结果页面")
    assert not additional_agent_tab_requested("不要打开新标签页")
    assert requested_agent_tab_count("在另一个标签页打开 GitHub") == 2
    assert requested_agent_tab_count("分别在三个标签页打开页面") == 3
    assert requested_agent_tab_count("open both pages in 2 tabs") == 2
    assert additional_agent_tab_requested("总共 2 个标签页")
    assert requested_agent_tab_count("总共 2 个标签页") == 2


def test_runtime_defaults_to_one_plugin_owned_agent_window() -> None:
    settings = RuntimeSettings()
    assert settings.session_scope == "plugin"
    assert settings.release_control_when_idle is True
    assert settings.reuse_existing_window is True
    assert settings.allow_additional_agent_tabs is False
    assert settings.snapshot_max_tokens == 8000
    assert settings.scroll_max_pages == 1
    assert settings.scroll_snapshot_max_tokens == 2000
    assert settings.live_page_max_chars == 1200


def test_navigation_defaults_to_dom_content_loaded() -> None:
    action = NavigateAction(action="navigate", url="https://www.bing.com/search?q=neko")
    assert action.wait_until == "domcontentloaded"
