from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from typing import Any

import pytest
from plugin.plugins.browser_skill.runtime.agent_loop import AgentLoop, LoopFailure
from plugin.plugins.browser_skill.runtime.bsk_client import BskCommandError, BskCommandResult
from plugin.plugins.browser_skill.runtime.control import BrowserTaskControl
from plugin.plugins.browser_skill.runtime.models import (
    BorrowTabAction,
    ClickAction,
    DoneAction,
    FailAction,
    FillAction,
    NavigateAction,
    ObserveAction,
    PressAction,
    RuntimeSettings,
    ScrollAction,
    SnapshotAction,
    TabCreateAction,
    TabListAction,
)
from plugin.plugins.browser_skill.runtime.session_manager import (
    SessionManager,
)


class FakeConfig:
    def get_model_api_config(self, purpose: str) -> dict[str, str]:
        return {"model": f"fake-{purpose}", "base_url": "https://llm.invalid"}


class FakePlanner:
    def __init__(self, *actions: Any, autofill_verified_done: bool = True) -> None:
        self.actions = deque(actions)
        self.observations: list[str] = []
        self.histories: list[list[str]] = []
        self.autofill_verified_done = autofill_verified_done

    async def decide(self, **kwargs: Any) -> Any:
        observation = kwargs["observation"]
        self.observations.append(observation)
        self.histories.append(list(kwargs.get("history") or []))
        action = self.actions.popleft()
        if (
            self.autofill_verified_done
            and kwargs.get("verification_required") is True
            and isinstance(action, DoneAction)
        ):
            evidence = "semantic page" if "semantic page" in observation else "page"
            return action.model_copy(
                update={
                    "primary_content_visible": True,
                    "visible_evidence": evidence,
                }
            )
        return action

    async def close(self) -> None:
        return None


class FakeBsk:
    def __init__(self) -> None:
        self.started = 0
        self.stopped: list[str] = []
        self.returned: list[int] = []
        self.borrowed: list[int] = []
        self.help_count = 0
        self.clicks: list[str] = []
        self.fills: list[tuple[str, str]] = []
        self.url = "https://example.com/"
        self.session_ids: list[str] = []
        self.snapshot_counter = 0
        self.agent_tab_list_count = 0
        self.created_tabs: list[str | None] = []
        self.selected_tabs: list[int] = []
        self.cancel_count = 0
        self.observe_count = 0
        self.observe_token_limits: list[int] = []
        self.html_count = 0
        self.pressed: list[tuple[str, str | None]] = []
        self.snapshot_token_limits: list[int] = []

    async def cancel_active(self) -> None:
        self.cancel_count += 1

    async def start_session(self, browser_id: str) -> dict[str, Any]:
        self.started += 1
        session_id = f"s-{self.started}"
        self.session_ids.append(session_id)
        return {"session_id": session_id, "browser_instance_id": browser_id}

    async def stop_session(self, session_id: str) -> dict[str, Any]:
        self.stopped.append(session_id)
        return {"stopped": [session_id]}

    async def status(self) -> dict[str, Any]:
        return {"sessions": [{"session_id": value} for value in self.session_ids if value not in self.stopped]}

    async def tab_list(self, session_id: str, *, scope: str) -> dict[str, Any]:
        if scope == "user":
            return {
                "tabs": [
                    {
                        "tab_id": 42,
                        "active": True,
                        "url": "https://signed-in.example/account",
                        "title": "Account",
                        "scope": "user",
                    }
                ]
            }
        self.agent_tab_list_count += 1
        return {"tabs": [{"tab_id": 11, "active": True, "url": self.url}]}

    async def snapshot(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        self.snapshot_counter += 1
        self.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {
            "text": f'page version {self.snapshot_counter}; textbox "Password" @e7; button "提交订单" @e9',
            "tab_id": 11,
        }

    async def observe(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        self.observe_count += 1
        self.observe_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {"text": 'semantic page; textbox "搜索" @e12; button "百度一下" @e13'}

    async def get_html(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        self.html_count += 1
        return {"html": '<input name="wd"><button>百度一下</button>'}

    async def press(
        self,
        session_id: str,
        key: str,
        *,
        target: str | None = None,
    ) -> dict[str, Any]:
        self.pressed.append((key, target))
        return {}

    async def navigate(self, session_id: str, url: str, **kwargs: Any) -> dict[str, Any]:
        self.url = url
        return {"final_url": url}

    async def click(self, session_id: str, target: str, **kwargs: Any) -> dict[str, Any]:
        self.clicks.append(target)
        return {}

    async def fill(self, session_id: str, target: str, value: str) -> dict[str, Any]:
        self.fills.append((target, value))
        return {}

    async def request_help(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        self.help_count += 1
        return {"outcome": "completed"}

    async def tab_borrow(self, session_id: str, tab_id: int) -> dict[str, Any]:
        self.borrowed.append(tab_id)
        return {"tab_id": tab_id}

    async def tab_return(self, session_id: str, tab_id: int) -> dict[str, Any]:
        self.returned.append(tab_id)
        return {"tab_id": tab_id}

    async def tab_create(self, session_id: str, *, url: str | None = None) -> dict[str, Any]:
        self.created_tabs.append(url)
        if url:
            self.url = url
        return {"tab_id": 12, "url": url or "about:blank"}

    async def tab_select(self, session_id: str, tab_id: int) -> dict[str, Any]:
        self.selected_tabs.append(tab_id)
        return {"tab_id": tab_id}


def make_loop(
    client: FakeBsk,
    sessions: SessionManager,
    settings: RuntimeSettings | None = None,
) -> AgentLoop:
    return AgentLoop(
        client=client,  # type: ignore[arg-type]
        sessions=sessions,
        config_manager=FakeConfig(),
        settings=settings or RuntimeSettings(max_steps=10, session_keepalive_seconds=0),
        prompts_dir=Path(__file__).parent.parent / "prompts",
    )


@pytest.mark.asyncio
async def test_conversation_session_reuses_live_session_and_close_cleans_it() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    first = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    second = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    assert first is second
    assert client.started == 1
    await sessions.close_conversation("chat-1")
    assert client.stopped == ["s-1"]


@pytest.mark.asyncio
async def test_different_context_keys_reuse_the_one_existing_agent_window() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    native = await sessions.get_or_create(
        conversation_id="browser-skill:main-dialog",
        browser_id="browser-1",
    )
    fallback = await sessions.get_or_create(
        conversation_id="lanlan:another-context",
        browser_id="browser-1",
        reuse_existing=True,
    )

    assert native is fallback
    assert fallback.conversation_id == "lanlan:another-context"
    assert client.started == 1
    assert list(sessions.sessions) == ["lanlan:another-context"]


@pytest.mark.asyncio
async def test_terminated_task_can_preserve_and_reuse_the_same_session() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    first = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    first.current_url = "https://example.com/playing"
    first.last_observation = "video playing"

    kept = await sessions.preserve_session(first, interval_seconds=0)
    second = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")

    assert kept is True
    assert second is first
    assert second.current_url == "https://example.com/playing"
    assert client.started == 1
    assert client.stopped == []


@pytest.mark.asyncio
async def test_idle_session_releases_to_user_and_next_task_reacquires() -> None:
    class IdleHelpClient:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def request_help(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
            assert session_id == "s-1"
            assert kwargs["timeout_seconds"] == 86400
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return {}

    class HandoffBsk(FakeBsk):
        def __init__(self) -> None:
            super().__init__()
            self.peer = IdleHelpClient()

        def spawn_peer(self) -> IdleHelpClient:
            return self.peer

    client = HandoffBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    first = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")

    kept = await sessions.preserve_session(
        first,
        interval_seconds=120,
        release_control=True,
    )
    await asyncio.wait_for(client.peer.started.wait(), timeout=1)
    assert kept is True
    assert first.control_owner == "user"

    reused = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    assert reused is first
    assert reused.control_owner == "agent"
    assert client.peer.cancelled.is_set()
    assert client.started == 1
    assert client.stopped == []


@pytest.mark.asyncio
async def test_transient_status_failure_does_not_create_second_agent_window() -> None:
    class TransientStatusBsk(FakeBsk):
        def __init__(self) -> None:
            super().__init__()
            self.status_calls = 0

        async def status(self) -> dict[str, Any]:
            self.status_calls += 1
            if self.status_calls == 1:
                raise RuntimeError("temporary status failure")
            return await super().status()

    client = TransientStatusBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    first = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    second = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")

    assert first is second
    assert client.started == 1
    assert client.status_calls == 2


@pytest.mark.asyncio
async def test_loop_navigates_verifies_and_keeps_reusable_session() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        NavigateAction(action="navigate", url="https://example.com/result"),
        DoneAction(action="done", summary="完成", session_disposition="keep_session"),
        DoneAction(action="done", summary="已复核", session_disposition="keep_session"),
    )
    result = await make_loop(client, sessions).run(
        instruction="打开结果页",
        raw_request="打开结果页",
        session=session,
        planner=planner,
    )
    assert result.success and result.session_state == "kept"
    assert result.session_decision_required is True
    assert result.current_url == "https://example.com/result"
    assert client.snapshot_counter >= 2
    assert client.stopped == []


@pytest.mark.asyncio
async def test_element_not_found_fail_is_recovered_before_task_ends() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        FailAction(
            action="fail",
            summary="没有找到百度搜索框元素",
            error_code="ELEMENT_NOT_FOUND",
            retryable=False,
        ),
        FillAction(action="fill", target="@e12", value="可爱小猫视频"),
        DoneAction(action="done", summary="已经完成搜索"),
        DoneAction(action="done", summary="已复核搜索结果"),
    )

    result = await make_loop(client, sessions).run(
        instruction="在百度搜索可爱小猫视频",
        raw_request="在百度搜索可爱小猫视频",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert client.observe_count == 2
    assert client.fills == [("@e12", "可爱小猫视频")]
    assert result.session_state == "kept"


@pytest.mark.asyncio
async def test_link_click_fail_is_replanned_inside_agent_loop() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        FailAction(
            action="fail",
            summary="没法直接触发页面上的链接点击操作",
            error_code="CLICK_FAILED",
            retryable=False,
        ),
        ClickAction(action="click", target="@e13", reason="打开开源仓库"),
        DoneAction(action="done", summary="已进入开源仓库"),
        DoneAction(action="done", summary="已复核开源仓库页面"),
    )

    result = await make_loop(client, sessions).run(
        instruction="找到猫娘计划开源仓库并点进去",
        raw_request="找到猫娘计划开源仓库并点进去",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert client.observe_count == 2
    assert client.clicks == ["@e13"]


@pytest.mark.asyncio
async def test_bsk_link_interaction_error_is_replanned_inside_agent_loop() -> None:
    client = FakeBsk()
    click_attempts = 0

    async def transient_click(
        _session_id: str,
        target: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal click_attempts
        click_attempts += 1
        if click_attempts <= 2:
            result = BskCommandResult(
                ("click", target),
                3,
                '{"code":"ELEMENT_NOT_INTERACTABLE","message":"link is not clickable"}',
                "",
                {
                    "code": "ELEMENT_NOT_INTERACTABLE",
                    "message": "link is not clickable",
                },
            )
            raise BskCommandError(result)
        client.clicks.append(target)
        return {}

    client.click = transient_click  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        ClickAction(action="click", target="@e9", reason="打开搜索结果"),
        ClickAction(action="click", target="@e13", reason="改用重新定位后的仓库链接"),
        DoneAction(action="done", summary="已进入开源仓库"),
        DoneAction(action="done", summary="已复核开源仓库页面"),
    )

    result = await make_loop(client, sessions).run(
        instruction="找到猫娘计划开源仓库并点进去",
        raw_request="找到猫娘计划开源仓库并点进去",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert click_attempts == 3
    assert client.observe_count == 2
    assert client.clicks == ["@e13"]


@pytest.mark.asyncio
async def test_dom_failure_exhaustion_gets_one_automatic_alternate_route() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    dom_failure = FailAction(
        action="fail",
        summary="浏览器提示 DOM 错误没法完成",
        error_code="DOM_ERROR",
        retryable=False,
    )
    planner = FakePlanner(
        dom_failure,
        dom_failure,
        dom_failure,
        NavigateAction(action="navigate", url="https://github.com/example/neko-project"),
        DoneAction(action="done", summary="已通过替代路径进入开源仓库"),
        DoneAction(action="done", summary="已复核开源仓库页面"),
    )
    settings = RuntimeSettings(
        max_steps=10,
        session_keepalive_seconds=0,
        enable_vision_fallback=False,
    )

    result = await make_loop(client, sessions, settings).run(
        instruction="找到猫娘计划开源仓库并点进去",
        raw_request="找到猫娘计划开源仓库并点进去",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert client.observe_count == 2
    assert client.html_count == 1
    assert client.url == "https://github.com/example/neko-project"


@pytest.mark.asyncio
async def test_snapshot_link_ref_uses_href_navigation_instead_of_dom_click() -> None:
    client = FakeBsk()
    client.url = "https://www.bing.com/search?q=neko"
    html_reads: list[str | None] = []

    async def link_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {
            "text": 'link "Project N.E.K.O 开源仓库" @e5',
            "tab_id": 11,
        }

    async def link_html(
        _session_id: str,
        *,
        ref: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        html_reads.append(ref)
        return {
            "html": '<a href="https://github.com/project-neko/neko">Project N.E.K.O</a>'
        }

    client.snapshot = link_snapshot  # type: ignore[method-assign]
    client.get_html = link_html  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        ClickAction(action="click", target="@e5", reason="进入开源仓库"),
        DoneAction(action="done", summary="已进入开源仓库"),
        DoneAction(action="done", summary="已复核开源仓库页面"),
    )

    result = await make_loop(client, sessions).run(
        instruction="找到猫娘计划开源仓库并点进去",
        raw_request="找到猫娘计划开源仓库并点进去",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert html_reads == ["@e5"]
    assert client.clicks == []
    assert client.url == "https://github.com/project-neko/neko"


@pytest.mark.asyncio
async def test_scroll_returns_to_llm_after_each_page() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        ScrollAction(action="scroll", direction="down", pages=1, until="目标内容"),
        DoneAction(action="done", summary="完成长页面查找"),
        DoneAction(action="done", summary="已复核长页面"),
    )
    settings = RuntimeSettings(
        max_steps=10,
        session_keepalive_seconds=0,
        scroll_snapshot_max_tokens=1200,
        scroll_settle_ms=50,
    )

    result = await make_loop(client, sessions, settings).run(
        instruction="向下滚动查找目标内容",
        raw_request="向下滚动查找目标内容",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert client.pressed == [("PageDown", None)]
    assert client.snapshot_token_limits[1:2] == [1200]
    assert len(planner.observations) == 3
    assert "Scroll batch: pages=1" in planner.observations[1]


@pytest.mark.asyncio
async def test_compact_snapshot_escalates_to_full_observe_only_when_agent_requests_it() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        ObserveAction(action="observe", reason="紧凑视图里没有足够正文"),
        DoneAction(
            action="done",
            summary="已在深度观察中看到目标",
            visible_evidence="semantic page",
        ),
        autofill_verified_done=False,
    )
    settings = RuntimeSettings(snapshot_max_tokens=7000, session_keepalive_seconds=0)

    result = await make_loop(client, sessions, settings).run(
        instruction="读取页面目标内容",
        raw_request="读取页面目标内容",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert client.snapshot_token_limits[0] == 4000
    assert client.observe_token_limits == [7000, 7000]
    assert "Runtime observation mode: compact snapshot" in planner.observations[0]
    assert "Runtime observation mode: compact snapshot" not in planner.observations[1]


@pytest.mark.asyncio
async def test_content_entry_done_requires_current_viewport_evidence() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        DoneAction(action="done", summary="目标页面已经打开"),
        DoneAction(action="done", summary="只看到了页面标题"),
        ScrollAction(action="scroll", direction="down", pages=1),
        DoneAction(action="done", summary="滚动后准备复核正文"),
        DoneAction(
            action="done",
            summary="已复核正文",
            primary_content_visible=True,
            visible_evidence="semantic page",
        ),
        autofill_verified_done=False,
    )

    result = await make_loop(client, sessions).run(
        instruction="搜索目标文章并点进去",
        raw_request="搜索目标文章并点进去",
        session=session,
        planner=planner,
    )

    assert result.success is True
    # The runtime rejects unsupported completion but never scrolls on the
    # model's behalf. Only the one explicit ScrollAction is executed.
    assert client.pressed == [("PageDown", None)]
    assert "Completion semantic observation" in planner.observations[1]
    assert "viewport_offset=observed+0" in planner.observations[1]
    assert "runtime did not scroll" in " ".join(planner.histories[2]).casefold()
    assert "viewport_offset=observed+1" in planner.observations[3]
    assert len(planner.observations) == 5


@pytest.mark.asyncio
async def test_repeated_done_with_invalid_evidence_is_fused_after_navigation() -> None:
    client = FakeBsk()

    async def content_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {
            "text": (
                "heading 111；维基百科，自由的百科全书；"
                "111（一百一十一）是 110 与 112 之间的自然数。"
                "数学：第81个合数，正因数有1、3、37和111。"
            ),
            "tab_id": 11,
        }

    async def content_observe(_session_id: str, **_kwargs: Any) -> dict[str, Any]:
        client.observe_count += 1
        return {
            "text": (
                "维基百科条目 111。一百一十一位于110和112之间；"
                "当前页面还显示数学、命名和识别等正文栏目。"
            )
        }

    client.snapshot = content_snapshot  # type: ignore[method-assign]
    client.observe = content_observe  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        NavigateAction(action="navigate", url="https://zh.wikipedia.org/wiki/111"),
        DoneAction(
            action="done",
            summary="已进入维基百科 111 条目",
            primary_content_visible=True,
            visible_evidence="页面已经展示111条目的数学和命名正文",
        ),
        DoneAction(
            action="done",
            summary="已进入维基百科 111 条目并看到正文",
            primary_content_visible=True,
            visible_evidence="当前是111的百科正文页面",
        ),
        DoneAction(
            action="done",
            summary="再次声明已经看到正文",
            primary_content_visible=True,
            visible_evidence="这段证据同样不在当前页面",
        ),
        autofill_verified_done=False,
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="搜索 111 然后打开维基百科条目",
            raw_request="搜索 111 然后打开维基百科条目",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user"
    assert len(planner.observations) == 4
    assert client.observe_count == 1


@pytest.mark.asyncio
async def test_third_consecutive_unverifiable_done_is_fused() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        DoneAction(action="done", summary="第一次完成声明"),
        DoneAction(action="done", summary="第二次完成声明"),
        DoneAction(action="done", summary="第三次完成声明"),
        autofill_verified_done=False,
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="搜索目标文章并点进去",
            raw_request="搜索目标文章并点进去",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user"
    assert len(planner.observations) == 3


def test_visible_evidence_matching_ignores_punctuation_and_width() -> None:
    observation = "111（一百一十一）是 110 与 112 之间的自然数。"
    evidence = "111 (一百一十一) 是110与112之间的自然数"

    assert AgentLoop._evidence_is_visible(evidence, observation, minimum_chars=8)


def test_visible_evidence_does_not_match_runtime_url_header() -> None:
    observation = (
        "Current URL: https://zh.wikipedia.org/wiki/111\n"
        "页面仍在加载，当前还没有正文。"
    )

    assert not AgentLoop._evidence_is_visible(
        "wikipediawiki111",
        observation,
        minimum_chars=8,
    )


@pytest.mark.asyncio
async def test_observation_actions_cannot_reset_rejected_done_fuse() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        DoneAction(action="done", summary="第一次无证据完成"),
        SnapshotAction(action="snapshot"),
        DoneAction(action="done", summary="第二次无证据完成"),
        ObserveAction(action="observe"),
        DoneAction(action="done", summary="第三次无证据完成"),
        autofill_verified_done=False,
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="搜索目标文章并点进去",
            raw_request="搜索目标文章并点进去",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user"
    assert len(planner.observations) == 5


@pytest.mark.asyncio
async def test_strong_first_done_finishes_without_second_llm_call() -> None:
    client = FakeBsk()

    async def content_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {"text": "目标文章正文已经显示在当前视口", "tab_id": 11}

    async def content_observe(_session_id: str, **_kwargs: Any) -> dict[str, Any]:
        client.observe_count += 1
        return {"text": "目标文章正文已经显示在当前视口"}

    client.snapshot = content_snapshot  # type: ignore[method-assign]
    client.observe = content_observe  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        DoneAction(
            action="done",
            summary="已打开并看到目标文章正文",
            primary_content_visible=True,
            visible_evidence="目标文章正文已经显示",
        ),
        autofill_verified_done=False,
    )

    result = await make_loop(client, sessions).run(
        instruction="搜索目标文章并点进去",
        raw_request="搜索目标文章并点进去",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert result.steps == 1
    assert client.observe_count == 1
    assert len(planner.observations) == 1


@pytest.mark.asyncio
async def test_stable_element_ref_can_ground_first_done_without_second_llm_call() -> None:
    client = FakeBsk()

    async def content_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {"text": 'status "消息已发送" @e77', "tab_id": 11}

    async def content_observe(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.observe_count += 1
        client.observe_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {"text": 'status "消息已发送" @e77'}

    client.snapshot = content_snapshot  # type: ignore[method-assign]
    client.observe = content_observe  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        DoneAction(
            action="done",
            summary="消息已发送",
            visible_evidence_ref="@e77",
        ),
        autofill_verified_done=False,
    )

    result = await make_loop(client, sessions).run(
        instruction="确认消息已经发送",
        raw_request="确认消息已经发送",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert result.steps == 1
    assert client.observe_count == 1
    assert len(planner.observations) == 1


@pytest.mark.asyncio
async def test_search_fill_can_submit_with_enter_in_one_agent_action() -> None:
    client = FakeBsk()

    async def search_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {"text": 'textbox "搜索" @e12', "tab_id": 11}

    client.snapshot = search_snapshot  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        FillAction(
            action="fill",
            target="@e12",
            value="可爱小猫视频",
            submit=True,
        ),
        DoneAction(action="done", summary="搜索已提交"),
        DoneAction(action="done", summary="已复核搜索结果"),
    )

    result = await make_loop(client, sessions).run(
        instruction="搜索可爱小猫视频",
        raw_request="搜索可爱小猫视频",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert client.fills == [("@e12", "可爱小猫视频")]
    assert client.pressed == [("Enter", None)]


@pytest.mark.asyncio
async def test_non_search_fill_submit_is_replanned_as_separate_actions() -> None:
    client = FakeBsk()
    client.url = "https://chat.deepseek.com/"
    state = {"sent": False}

    def page_text() -> str:
        sent = '; StaticText "测试消息已经成功发送"' if state["sent"] else ""
        return f'textbox "给 DeepSeek 发送消息" [empty] @e111{sent}'

    async def chat_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {"text": page_text(), "tab_id": 11}

    async def chat_observe(_session_id: str, **_kwargs: Any) -> dict[str, Any]:
        client.observe_count += 1
        return {"text": page_text()}

    async def send_message(
        _session_id: str,
        key: str,
        *,
        target: str | None = None,
    ) -> dict[str, Any]:
        client.pressed.append((key, target))
        state["sent"] = True
        return {}

    client.snapshot = chat_snapshot  # type: ignore[method-assign]
    client.observe = chat_observe  # type: ignore[method-assign]
    client.press = send_message  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        FillAction(
            action="fill",
            target="@e111",
            value="测试消息",
            submit=True,
        ),
        FillAction(
            action="fill",
            target="@e111",
            value="测试消息",
            submit=False,
        ),
        PressAction(action="press", key="Enter", target="@e111", reason="发送消息"),
        DoneAction(
            action="done",
            summary="消息已发送",
            visible_evidence="测试消息已经成功发送",
        ),
        autofill_verified_done=False,
    )

    result = await make_loop(client, sessions).run(
        instruction="在 DeepSeek 中发送测试消息",
        raw_request="在 DeepSeek 中发送测试消息",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert result.steps == 4
    assert client.fills == [("@e111", "测试消息")]
    assert client.pressed == [("Enter", "@e111")]
    assert client.help_count == 1
    assert any(
        "fill.submit is only a search-box shortcut" in item
        for item in planner.histories[1]
    )


@pytest.mark.asyncio
async def test_repeated_non_search_fill_submit_is_fused() -> None:
    client = FakeBsk()

    async def chat_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {
            "text": (
                'textbox "给 DeepSeek 发送消息" @e111; '
                'textbox "另一个聊天输入框" @e112; '
                'textbox "第三个聊天输入框" @e113'
            ),
            "tab_id": 11,
        }

    client.snapshot = chat_snapshot  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        FillAction(action="fill", target="@e111", value="一", submit=True),
        FillAction(action="fill", target="@e112", value="二", submit=True),
        FillAction(action="fill", target="@e113", value="三", submit=True),
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="在聊天页面发送消息",
            raw_request="在聊天页面发送消息",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user"
    assert client.fills == []
    assert client.pressed == []


@pytest.mark.asyncio
async def test_search_result_completion_is_decided_by_agent() -> None:
    client = FakeBsk()
    client.url = "https://www.bing.com/"

    async def search_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        text = (
            'textbox "搜索" @e12'
            if "search?" not in client.url
            else 'textbox "搜索" value="111" @e4; link "111 - 搜索结果" @e9'
        )
        return {"text": text, "tab_id": 11}

    async def submit_search(
        _session_id: str,
        key: str,
        *,
        target: str | None = None,
    ) -> dict[str, Any]:
        client.pressed.append((key, target))
        client.url = "https://www.bing.com/search?q=111&form=QBRE"
        return {}

    client.snapshot = search_snapshot  # type: ignore[method-assign]
    client.press = submit_search  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        FillAction(action="fill", target="@e12", value="111", submit=True),
        DoneAction(action="done", summary="搜索结果页已经显示"),
        DoneAction(action="done", summary="复核后确认搜索结果页已经显示"),
    )

    result = await make_loop(client, sessions).run(
        instruction="按照用户要求使用 Bing 搜索 111",
        raw_request="用 Bing 搜索 111",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert result.steps == 3
    assert result.current_url == "https://www.bing.com/search?q=111&form=QBRE"
    assert len(planner.observations) == 3
    assert client.observe_count == 1
    assert client.fills == [("@e12", "111")]
    assert client.pressed == [("Enter", None)]


@pytest.mark.asyncio
async def test_compound_search_goal_does_not_finish_after_search_submission() -> None:
    client = FakeBsk()
    client.url = "https://www.bing.com/"

    async def search_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        text = (
            'textbox "搜索" @e12'
            if "search?" not in client.url
            else 'textbox "搜索" value="猫娘计划" @e4; link "GitHub" @e9'
        )
        return {"text": text, "tab_id": 11}

    async def submit_search(
        _session_id: str,
        key: str,
        *,
        target: str | None = None,
    ) -> dict[str, Any]:
        client.pressed.append((key, target))
        client.url = "https://www.bing.com/search?q=%E7%8C%AB%E5%A8%98%E8%AE%A1%E5%88%92"
        return {}

    client.snapshot = search_snapshot  # type: ignore[method-assign]
    client.press = submit_search  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        FillAction(action="fill", target="@e12", value="猫娘计划", submit=True),
    )
    settings = RuntimeSettings(max_steps=1, session_keepalive_seconds=0)

    result = await make_loop(client, sessions, settings).run(
        instruction="搜索猫娘计划然后打开 GitHub 仓库",
        raw_request="搜索猫娘计划然后打开 GitHub 仓库",
        session=session,
        planner=planner,
    )

    assert result.success is False
    assert result.error is not None and result.error.code == "STEP_LIMIT"
    assert result.steps == 1


def test_observation_hash_ignores_volatile_query_args_and_element_refs() -> None:
    first = (
        "Current URL: https://www.bing.com/search?q=111&form=QBRE&cvid=one\n"
        'textbox "搜索" value="111" @e4; link "结果" @e9'
    )
    second = (
        "Current URL: https://www.bing.com/search?q=111&form=ANSPH1&cvid=two\n"
        'textbox "搜索" value="111" @e17; link "结果" @e28'
    )
    changed = (
        "Current URL: https://www.bing.com/search?q=222\n"
        'textbox "搜索" value="222" @e4; link "其他结果" @e9'
    )

    assert AgentLoop._observation_hash(first) == AgentLoop._observation_hash(second)
    assert AgentLoop._observation_hash(first) != AgentLoop._observation_hash(changed)


def test_page_route_key_ignores_queries_but_preserves_spa_routes() -> None:
    assert AgentLoop._page_route_key(
        "https://www.bing.com/search?q=111&form=QBRE&cvid=one"
    ) == AgentLoop._page_route_key(
        "https://www.bing.com/search?cvid=two&q=222&form=ANSPH1"
    )
    assert AgentLoop._page_route_key(
        "https://music.163.com/#/search/m/?s=%E6%99%B4%E5%A4%A9&type=1"
    ) != AgentLoop._page_route_key(
        "https://music.163.com/#/song?id=1"
    )
    assert AgentLoop._page_route_key(
        "https://github.com/example/project#readme"
    ) == AgentLoop._page_route_key(
        "https://github.com/example/project#installation"
    )


def test_observation_hash_ignores_url_query_change_without_body_change() -> None:
    first = "Current URL: https://www.bing.com/search?q=111\n相同页面正文"
    second = "Current URL: https://www.bing.com/search?q=222\n相同页面正文"

    assert AgentLoop._observation_hash(first) == AgentLoop._observation_hash(second)


def test_observation_hash_ignores_compact_and_deep_runtime_wrappers() -> None:
    compact = (
        "Current URL: https://chat.deepseek.com/a/chat/s/1\n"
        "[Runtime observation mode: compact snapshot, max_tokens=4000; "
        "choose observe only when deeper semantic page detail is needed.]\n"
        'StaticText "加载失败，你可以"\nbutton "重试加载" @e114'
    )
    deep = (
        "Current URL: https://chat.deepseek.com/a/chat/s/1\n"
        "[Completion semantic observation: DOM text may contain off-screen nodes.]\n"
        'StaticText "加载失败，你可以"\nbutton "重试加载" @e114'
    )

    assert AgentLoop._observation_hash(compact) == AgentLoop._observation_hash(deep)


def test_action_url_effect_uses_action_context_instead_of_parameter_names() -> None:
    navigation = NavigateAction(
        action="navigate",
        url="https://example.com/?form=checkout",
    )
    search_submit = FillAction(action="fill", target="@e1", value="111", submit=True)

    assert AgentLoop._action_url_effect_confirmed(
        navigation,
        previous_url="https://example.com/?form=login",
        current_url="https://example.com/?form=checkout",
        direct_result=True,
    )
    assert not AgentLoop._action_url_effect_confirmed(
        search_submit,
        previous_url="https://www.bing.com/search?q=111&form=old&cvid=one",
        current_url="https://www.bing.com/search?q=111&form=new&cvid=two",
        direct_result=False,
    )


def test_fill_action_signature_ignores_renumbered_snapshot_ref() -> None:
    first = FillAction(action="fill", target="@e4", value="111", submit=True)
    second = FillAction(action="fill", target="@e17", value="111", submit=True)
    different = FillAction(action="fill", target="@e17", value="222", submit=True)

    assert AgentLoop._action_signature(first) == AgentLoop._action_signature(second)
    assert AgentLoop._action_signature(first) != AgentLoop._action_signature(different)


def test_click_action_signature_ignores_reworded_reason() -> None:
    first = ClickAction(action="click", target="@e114", reason="重试加载")
    second = ClickAction(action="click", target="@e114", reason="再次尝试恢复页面")

    assert AgentLoop._action_signature(first) == AgentLoop._action_signature(second)


@pytest.mark.asyncio
async def test_click_observe_click_cycle_is_fused_on_unchanged_page() -> None:
    client = FakeBsk()

    async def failed_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {
            "text": 'StaticText "加载失败，你可以"\nbutton "重试加载" @e114',
            "tab_id": 11,
        }

    async def failed_observe(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.observe_count += 1
        client.observe_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {"text": 'StaticText "加载失败，你可以"\nbutton "重试加载" @e114'}

    client.snapshot = failed_snapshot  # type: ignore[method-assign]
    client.observe = failed_observe  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        ObserveAction(action="observe", reason="确认加载失败状态"),
        ClickAction(action="click", target="@e114", reason="重试加载"),
        ClickAction(action="click", target="@e114", reason="再次尝试恢复页面"),
        ClickAction(action="click", target="@e114", reason="最后再试一次"),
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="确认当前页面回复内容是否真实",
            raw_request="确认当前页面回复内容是否真实",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user"
    assert client.clicks == ["@e114", "@e114"]
    assert client.observe_count == 2


@pytest.mark.asyncio
async def test_semantically_unchanged_mutations_stop_after_one_observation_recovery() -> None:
    client = FakeBsk()

    async def unchanged_search_snapshot(
        _session_id: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        client.snapshot_counter += 1
        return {
            "text": f'textbox "搜索" @e{client.snapshot_counter}',
            "tab_id": 11,
        }

    client.snapshot = unchanged_search_snapshot  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        FillAction(action="fill", target="@e1", value="第一次", submit=True),
        FillAction(action="fill", target="@e2", value="第二次", submit=True),
        FillAction(action="fill", target="@e12", value="第三次", submit=True),
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="搜索候选项，然后打开其中一个结果",
            raw_request="搜索候选项，然后打开其中一个结果",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user", str(caught.value)
    assert len(client.fills) == 3
    assert client.observe_count == 1


@pytest.mark.asyncio
async def test_real_navigation_resets_stagnation_recovery_fuse() -> None:
    client = FakeBsk()
    client.url = "https://example.com/a"

    async def page_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        page = "A" if client.url.endswith("/a") else "B"
        return {"text": f"页面 {page} 已经显示稳定内容", "tab_id": 11}

    async def page_observe(_session_id: str, **_kwargs: Any) -> dict[str, Any]:
        client.observe_count += 1
        page = "A" if client.url.endswith("/a") else "B"
        return {"text": f"页面 {page} 已经显示稳定内容"}

    client.snapshot = page_snapshot  # type: ignore[method-assign]
    client.observe = page_observe  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        NavigateAction(action="navigate", url="https://example.com/a"),
        NavigateAction(action="navigate", url="https://example.com/a"),
        NavigateAction(action="navigate", url="https://example.com/b"),
        NavigateAction(action="navigate", url="https://example.com/b"),
        DoneAction(
            action="done",
            summary="已切换到目标地址",
            visible_evidence="页面 B 已经显示稳定内容",
        ),
        autofill_verified_done=False,
    )

    result = await make_loop(client, sessions).run(
        instruction="将地址切换为 https://example.com/b",
        raw_request="将地址切换为 https://example.com/b",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert result.steps == 5
    assert client.url == "https://example.com/b"
    assert client.observe_count == 2


@pytest.mark.asyncio
async def test_scroll_explicitly_tells_agent_when_page_bottom_is_reached() -> None:
    client = FakeBsk()

    async def unchanged_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        return {"text": 'button "末尾内容" @e1', "tab_id": 11}

    client.snapshot = unchanged_snapshot  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        ScrollAction(action="scroll", direction="down", pages=1),
        DoneAction(action="done", summary="已经检查到页面底部"),
        DoneAction(action="done", summary="已复核页面底部"),
    )

    result = await make_loop(client, sessions).run(
        instruction="检查页面剩余内容",
        raw_request="检查页面剩余内容",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert "stopped_by=page_bottom_reached" in planner.observations[1]
    assert "PAGE BOTTOM REACHED" in planner.observations[1]


@pytest.mark.asyncio
async def test_tab_create_reuses_current_agent_tab_for_ordinary_navigation() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        TabCreateAction(action="tab_create", url="https://example.com/reused"),
        DoneAction(action="done", summary="完成"),
        DoneAction(action="done", summary="已复核"),
    )

    result = await make_loop(client, sessions).run(
        instruction="打开结果页面",
        raw_request="打开结果页面",
        session=session,
        planner=planner,
    )

    assert result.success
    assert client.created_tabs == []
    assert client.selected_tabs == [11]
    assert client.url == "https://example.com/reused"


@pytest.mark.asyncio
async def test_auth_modal_is_handed_to_user_before_the_next_agent_plan() -> None:
    client = FakeBsk()
    help_titles: list[str] = []

    async def auth_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        text = (
            "dialog 登录；扫码登录；使用音乐 APP 或微信扫码登录"
            if client.help_count == 0
            else "歌曲搜索结果已经显示"
        )
        return {"text": text, "tab_id": 11}

    async def auth_observe(_session_id: str, **_kwargs: Any) -> dict[str, Any]:
        client.observe_count += 1
        return {"text": "歌曲搜索结果已经显示"}

    async def complete_auth(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.help_count += 1
        help_titles.append(str(kwargs.get("title") or ""))
        return {"outcome": "completed"}

    client.snapshot = auth_snapshot  # type: ignore[method-assign]
    client.observe = auth_observe  # type: ignore[method-assign]
    client.request_help = complete_auth  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        DoneAction(
            action="done",
            summary="已显示歌曲搜索结果",
            visible_evidence="歌曲搜索结果已经显示",
        ),
        autofill_verified_done=False,
    )

    result = await make_loop(client, sessions).run(
        instruction="搜索并播放指定歌曲",
        raw_request="搜索并播放指定歌曲",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert result.steps == 2
    assert client.help_count == 1
    assert help_titles == ["页面需要扫码登录"]
    assert client.created_tabs == []
    assert len(planner.observations) == 1


def test_generic_login_marketing_copy_without_dialog_is_not_blocking_auth() -> None:
    assert AgentLoop._human_auth_challenge(
        "登录获取更好的服务；二维码；选择其他登录模式"
    ) is None


def test_generic_login_dialog_is_detected_without_site_specific_rules() -> None:
    assert AgentLoop._human_auth_challenge(
        'dialog "登录获取更好的服务；二维码；选择其他登录模式"; '
        'button "关闭" @e5'
    ) in {"qr_login", "login"}


@pytest.mark.asyncio
async def test_unresolved_auth_dialog_stops_without_returning_to_agent_planning() -> None:
    client = FakeBsk()

    async def auth_snapshot(_session_id: str, **kwargs: Any) -> dict[str, Any]:
        client.snapshot_token_limits.append(int(kwargs.get("max_tokens") or 0))
        if not client.clicks:
            return {
                "text": 'dialog "扫码登录"; button "关闭" @e5',
                "tab_id": 11,
            }
        return {"text": 'heading "晴天 搜索结果"', "tab_id": 11}

    client.snapshot = auth_snapshot  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(TabCreateAction(action="tab_create"))
    settings = RuntimeSettings(
        max_steps=5,
        session_keepalive_seconds=0,
        allow_additional_agent_tabs=True,
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions, settings).run(
            instruction="搜索晴天，然后在另一个标签页打开 GitHub",
            raw_request="搜索晴天，然后在另一个标签页打开 GitHub",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "AUTHENTICATION_REQUIRED"
    assert caught.value.status == "needs_user"
    assert client.help_count == 1
    assert client.created_tabs == []
    assert client.clicks == []
    assert planner.observations == []


def test_plain_login_navigation_text_is_not_treated_as_blocking_auth() -> None:
    assert AgentLoop._human_auth_challenge("导航：首页 我的音乐 登录") is None


@pytest.mark.asyncio
async def test_identical_action_on_unchanged_page_is_fused_on_third_plan() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        TabCreateAction(action="tab_create"),
        TabCreateAction(action="tab_create"),
        TabCreateAction(action="tab_create"),
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="读取当前页面",
            raw_request="读取当前页面",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user"
    assert client.created_tabs == []


@pytest.mark.asyncio
async def test_repeated_tab_create_after_requested_count_is_satisfied_stops_on_second_try() -> None:
    class ExistingTwoTabsBsk(FakeBsk):
        async def tab_list(self, session_id: str, *, scope: str) -> dict[str, Any]:
            if scope == "user":
                return await super().tab_list(session_id, scope=scope)
            self.agent_tab_list_count += 1
            return {
                "tabs": [
                    {"tab_id": 11, "active": True, "url": "https://music.163.com/search"},
                    {"tab_id": 12, "active": False, "url": "https://github.com/example/repo"},
                ]
            }

    client = ExistingTwoTabsBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        TabCreateAction(action="tab_create"),
        TabCreateAction(action="tab_create"),
    )
    settings = RuntimeSettings(
        max_steps=5,
        session_keepalive_seconds=0,
        allow_additional_agent_tabs=True,
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions, settings).run(
            instruction="总共 2 个标签页",
            raw_request="总共 2 个标签页",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user"
    assert client.created_tabs == []
    assert len(planner.observations) == 2


@pytest.mark.asyncio
async def test_explicit_new_tab_cannot_report_success_while_ui_setting_disables_it() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        TabCreateAction(action="tab_create", url="https://example.com/reused"),
        DoneAction(action="done", summary="完成"),
        DoneAction(action="done", summary="已复核"),
    )

    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="请在新标签页打开结果",
            raw_request="请在新标签页打开结果",
            session=session,
            planner=planner,
        )

    assert caught.value.code == "ACTION_REJECTED"
    assert caught.value.status == "needs_user"
    assert client.created_tabs == []
    assert client.url == "https://example.com/reused"


@pytest.mark.asyncio
async def test_another_tab_wording_creates_exactly_one_extra_tab_and_verifies_two_pages() -> None:
    class TwoTabBsk(FakeBsk):
        def __init__(self) -> None:
            super().__init__()
            self.tabs = {
                11: "https://music.163.com/search/m/?s=勾指启誓&type=1",
            }
            self.active_tab_id = 11

        async def tab_list(self, session_id: str, *, scope: str) -> dict[str, Any]:
            if scope == "user":
                return await super().tab_list(session_id, scope=scope)
            self.agent_tab_list_count += 1
            return {
                "tabs": [
                    {
                        "tab_id": tab_id,
                        "active": tab_id == self.active_tab_id,
                        "url": url,
                    }
                    for tab_id, url in self.tabs.items()
                ]
            }

        async def tab_create(self, session_id: str, *, url: str | None = None) -> dict[str, Any]:
            self.created_tabs.append(url)
            self.active_tab_id = 12
            self.tabs[12] = url or "about:blank"
            return {"tab_id": 12, "url": self.tabs[12]}

        async def navigate(self, session_id: str, url: str, **kwargs: Any) -> dict[str, Any]:
            self.url = url
            self.tabs[self.active_tab_id] = url
            return {"final_url": url}

    client = TwoTabBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        TabCreateAction(action="tab_create"),
        TabCreateAction(action="tab_create"),
        NavigateAction(action="navigate", url="https://github.com/Project-N-E-K-O/N.E.K.O"),
        DoneAction(action="done", summary="两个页面都已打开"),
        DoneAction(action="done", summary="已复核两个页面"),
    )
    settings = RuntimeSettings(
        max_steps=10,
        session_keepalive_seconds=0,
        allow_additional_agent_tabs=True,
    )

    result = await make_loop(client, sessions, settings).run(
        instruction="用浏览器打开网易云搜索勾指启誓，然后在另一个标签页里搜索猫娘计划然后打开github",
        raw_request="用浏览器打开网易云搜索勾指启誓，然后在另一个标签页里搜索猫娘计划然后打开github",
        session=session,
        planner=planner,
    )

    assert result.success is True
    assert client.created_tabs == ["about:blank"]
    assert len(client.tabs) == 2
    assert client.tabs[11].startswith("https://music.163.com/")
    assert client.tabs[12] == "https://github.com/Project-N-E-K-O/N.E.K.O"


@pytest.mark.asyncio
async def test_default_defer_ignores_browser_agent_close_decision() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        DoneAction(action="done", summary="视频正在播放", session_disposition="close_session"),
        DoneAction(action="done", summary="已复核播放", session_disposition="close_session"),
    )
    result = await make_loop(client, sessions).run(
        instruction="播放这个视频",
        raw_request="播放这个视频",
        session=session,
        planner=planner,
    )
    assert result.success and result.session_state == "kept"
    assert result.session_decision_required is True
    assert client.stopped == []
    await sessions.close_all()


@pytest.mark.asyncio
async def test_high_level_explicit_close_closes_session_after_success() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        DoneAction(action="done", summary="完成", session_disposition="keep_session"),
        DoneAction(action="done", summary="已复核", session_disposition="keep_session"),
    )

    result = await make_loop(client, sessions).run(
        instruction="读取页面",
        raw_request="读取页面，完成后关闭",
        session=session,
        planner=planner,
        final_session_action="close",
    )

    assert result.success and result.session_state == "closed"
    assert result.session_decision_required is False
    assert client.stopped == ["s-1"]


@pytest.mark.asyncio
async def test_kept_session_uses_read_only_keepalive() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    sessions.start_keepalive(session, interval_seconds=0.01)
    try:
        await asyncio.sleep(0.05)
        assert client.agent_tab_list_count >= 1
    finally:
        await sessions.close_session(session)
    assert client.stopped == ["s-1"]


@pytest.mark.asyncio
async def test_sensitive_fill_is_delegated_without_sending_value() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id=None, browser_id="browser-1")
    planner = FakePlanner(
        FillAction(action="fill", target="@e7", value="super-secret"),
        DoneAction(action="done", summary="完成", session_disposition="keep_session"),
        DoneAction(action="done", summary="已复核", session_disposition="keep_session"),
    )
    result = await make_loop(client, sessions).run(
        instruction="登录",
        raw_request="登录",
        session=session,
        planner=planner,
    )
    assert result.session_state == "closed"
    assert client.help_count == 1
    assert client.fills == []
    assert client.stopped == ["s-1"]


@pytest.mark.asyncio
async def test_critical_click_is_confirmed_once_before_execution() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        ClickAction(action="click", target="@e9", reason="提交订单"),
        DoneAction(action="done", summary="完成", session_disposition="close_session"),
        DoneAction(action="done", summary="已复核", session_disposition="close_session"),
    )
    await make_loop(client, sessions).run(
        instruction="购买商品",
        raw_request="购买商品",
        session=session,
        planner=planner,
    )
    assert client.help_count == 1
    assert client.clicks == ["@e9"]
    assert client.snapshot_counter >= 3


@pytest.mark.asyncio
async def test_borrowed_user_tab_is_confirmed_and_returned_at_task_end() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        TabListAction(action="tab_list", scope="user"),
        BorrowTabAction(action="borrow_tab", tab_id=42, purpose="读取当前标签"),
        DoneAction(action="done", summary="完成", session_disposition="keep_session"),
        DoneAction(action="done", summary="已复核", session_disposition="keep_session"),
    )
    result = await make_loop(client, sessions).run(
        instruction="读取我已经打开的页面",
        raw_request="读取我已经打开的页面",
        session=session,
        planner=planner,
    )
    assert result.session_state == "kept"
    assert client.help_count == 1
    assert client.borrowed == [42]
    assert client.returned == [42]
    assert session.borrowed_tab_ids == set()


@pytest.mark.asyncio
async def test_second_consecutive_stale_ref_returns_stable_error() -> None:
    client = FakeBsk()

    async def stale_click(session_id: str, target: str, **kwargs: Any) -> dict[str, Any]:
        result = BskCommandResult(
            ("click", target),
            1,
            '{"code":"REF_NOT_FOUND","message":"snapshot ref not found"}',
            "",
            {"code": "REF_NOT_FOUND", "message": "snapshot ref not found"},
        )
        raise BskCommandError(result)

    client.click = stale_click  # type: ignore[method-assign]
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        ClickAction(action="click", target="@e2", reason="展开详情"),
        ClickAction(action="click", target="@e2", reason="展开详情"),
    )
    with pytest.raises(LoopFailure) as caught:
        await make_loop(client, sessions).run(
            instruction="展开详情",
            raw_request="展开详情",
            session=session,
            planner=planner,
        )
    assert caught.value.code == "STALE_REF"


@pytest.mark.asyncio
async def test_replace_steering_discards_action_planned_for_old_goal() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    control = BrowserTaskControl("chat-1", "打开旧页面", "打开旧页面")

    class SteeringPlanner(FakePlanner):
        async def decide(self, **kwargs: Any) -> Any:
            self.observations.append(kwargs["instruction"])
            if len(self.observations) == 1:
                control.submit(
                    "replace",
                    "打开新页面并读取标题",
                    user_request="不要旧页面了，改看新页面",
                )
                return NavigateAction(action="navigate", url="https://old.example/")
            action = self.actions.popleft()
            if kwargs.get("verification_required") is True and isinstance(action, DoneAction):
                return action.model_copy(
                    update={
                        "primary_content_visible": True,
                        "visible_evidence": "semantic page",
                    }
                )
            return action

    planner = SteeringPlanner(
        NavigateAction(action="navigate", url="https://new.example/"),
        DoneAction(action="done", summary="完成", session_disposition="keep_session"),
        DoneAction(action="done", summary="已复核", session_disposition="keep_session"),
    )
    result = await make_loop(client, sessions).run(
        instruction="打开旧页面",
        raw_request="打开旧页面",
        session=session,
        planner=planner,
        control=control,
    )
    assert result.success
    assert client.url == "https://new.example/"
    assert "https://old.example/" != client.url
    assert planner.observations[1] == "打开新页面并读取标题"
    assert control.applied_revision == 1
    assert control.original_request == "不要旧页面了，改看新页面"


@pytest.mark.asyncio
async def test_steering_invalidates_critical_confirmation_before_click() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    control = BrowserTaskControl("chat-1", "购买商品", "购买商品")

    async def steer_during_confirmation(session_id: str, **kwargs: Any) -> dict[str, Any]:
        control.submit("replace", "只阅读商品信息，不要购买")
        return {"outcome": "completed"}

    client.request_help = steer_during_confirmation  # type: ignore[method-assign]
    planner = FakePlanner(
        ClickAction(action="click", target="@e9", reason="提交订单"),
        DoneAction(action="done", summary="未购买", session_disposition="close_session"),
        DoneAction(action="done", summary="已复核未购买", session_disposition="close_session"),
    )
    result = await make_loop(client, sessions).run(
        instruction="购买商品",
        raw_request="购买商品",
        session=session,
        planner=planner,
        control=control,
    )
    assert result.success
    assert client.clicks == []
    assert control.goal == "只阅读商品信息，不要购买"


@pytest.mark.asyncio
async def test_open_ended_progress_reports_action_limit_as_metric_not_total() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    loop = make_loop(client, sessions)
    updates: list[dict[str, Any]] = []

    async def capture(**payload: Any) -> None:
        updates.append(payload)

    await loop._progress(capture, "planning", "Agent 正在规划下一步", 3)

    assert updates == [
        {
            "stage": "planning",
            "message": "Agent 正在规划下一步",
            "step": 3,
            "metrics": {"actions_used": 3, "action_limit": 10},
        }
    ]


@pytest.mark.asyncio
async def test_step_limit_keeps_reusable_session_for_high_level_continuation() -> None:
    client = FakeBsk()
    sessions = SessionManager(client)  # type: ignore[arg-type]
    session = await sessions.get_or_create(conversation_id="chat-1", browser_id="browser-1")
    planner = FakePlanner(
        *(SnapshotAction(action="snapshot") for _ in range(10))
    )

    result = await make_loop(client, sessions).run(
        instruction="继续检查页面直到完成",
        raw_request="检查这个页面",
        session=session,
        planner=planner,
    )

    assert result.status == "needs_user"
    assert result.error is not None and result.error.code == "STEP_LIMIT"
    assert result.continuation_available is True
    assert result.session_state == "kept"
    assert client.stopped == []
    assert sessions.sessions["chat-1"] is session
