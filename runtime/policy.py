"""Safety policy independent from model-generated claims."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, unquote, urlsplit

from .models import AgentAction, ClickAction, FillAction, PressAction, TabCreateAction

_CRITICAL_PATTERNS = (
    r"提交订单|立即购买|确认购买|付款|支付|转账|充值|发送消息|发送邮件|发表评论",
    r"发布|删除|注销|修改密码|更改密码|授权|授予权限|提交表单|下载文件",
    r"\b(pay|purchase|buy now|place order|checkout|transfer|send|publish|post|delete|remove)\b",
    r"\b(change password|grant access|authorize|submit|download|install)\b",
)
_SENSITIVE_PATTERNS = (
    r"密码|口令|验证码|短信码|动态码|支付密码|安全码|银行卡|信用卡|身份证",
    r"\b(password|passwd|passcode|otp|captcha|verification code|security code)\b",
    r"\b(card number|credit card|cvv|cvc|social security|ssn)\b",
)
_EXPLICIT_USER_TAB_PATTERNS = (
    r"当前(标签|页面)|已经打开|我打开的|现有标签|用户标签",
    r"\b(current tab|open tab|existing tab|tab I opened|my tab)\b",
)
_ADDITIONAL_AGENT_TAB_PATTERNS = (
    r"新标签页|新标签|另开(?:一个)?标签|另(?:一|一个)标签页|第[二三四五六七八九十\d]+\s*个?标签页|[两二三四五六七八九十\d]+\s*个?标签页|多个标签|分别.*标签页|保留当前页面.*(?:另开|新开)",
    r"\b(new|another|additional|multiple|separate) tabs?\b|"
    r"\bopen .* in (?:a )?new tab\b|\b\d+ tabs?\b",
)
_NEGATED_ADDITIONAL_TAB_PATTERNS = (
    r"(?:不要|无需|不必|禁止|(?<!分)别)(?:再|使用|打开|创建|新建)?[^。；，,.]{0,8}(?:新|另|额外|多个|标签页)",
    r"\b(?:do not|don't|without|no need to)\b[^.]{0,24}\b(?:new|another|additional|multiple) tabs?\b",
)
_ZH_TAB_NUMBERS = {
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_PERSISTENT_MEDIA_PATTERNS = (
    r"播放|放一?(?:首|个|段)?(?:歌|音乐|视频|电影)|看(?:视频|电影|直播)|听(?:歌|音乐)|直播|连续播放",
    r"\b(play|watch|listen(?: to)?|stream)\b.*\b(video|movie|music|song|audio|live|stream)?\b",
    r"\bkeep (?:the )?(?:video|music|audio|stream|browser) (?:playing|open)\b",
)
_STOP_MEDIA_PATTERNS = (
    r"停止播放|暂停播放|关掉(?:视频|音乐|播放器)|关闭播放",
    r"\b(stop|pause|close)\b.*\b(video|music|audio|playback|stream)\b",
)
_SEARCH_FIELD_PATTERNS = (
    r"搜索|搜一搜|查询|检索|查找",
    r"\b(search|query|find)\b",
)
_TEXT_ENTRY_CONTROL_PATTERN = re.compile(
    r"(?i)(?:\b(?:textbox|searchbox|combobox)\b|文本框|搜索框|组合框)"
)
_CONSEQUENTIAL_TEXT_ENTRY_PATTERNS = (
    r"消息|邮件|评论|回复|聊天|帖子|订单|表单|申请|转账|付款|支付",
    r"\b(?:message|email|comment|reply|chat|post|order|form|application|payment|transfer)\b",
)
_FINALIZATION_CONTROL_PATTERNS = (
    r"确认|提交|发送|支付|付款|购买|下单|删除|发布|下载|授权|转账|充值",
    r"\b(?:confirm|submit|send|pay|purchase|buy|order|delete|publish|post|download|authorize|transfer)\b",
)
_ACTIVATION_KEYS = {"enter", "return", "numpadenter"}
_CRITICAL_FINALIZATION_INTENT_PATTERNS = (
    r"(?:发送|提交|支付|付款|购买|下单|删除|发布|下载|授权|转账|充值)"
    r".{0,12}(?:消息|邮件|评论|回复|订单|表单|文件|权限|款项|商品|内容|帖子|申请)",
    r"^(?:请|帮我|替我|现在)?\s*(?:支付|付款|购买|下单|删除|发布|下载|授权|转账|充值)(?:吧|。)?$",
    r"\b(?:send|submit|pay(?:\s+for)?|buy|delete|publish|post|download|authorize|transfer)\b"
    r"(?:\s+(?:this|the|that|my|a|an))?\s+"
    r"(?:message|email|comment|reply|order|form|file|payment|item|product|post|application)\b",
    r"\b(?:place|confirm)\s+(?:this|the|that|my|a|an)?\s*(?:order|payment|purchase)\b",
    r"\bpurchase\s+(?:this|the|that|my|a|an)\s+(?:item|product|order)\b",
    r"^(?:please\s+)?(?:send|submit|pay|buy|delete|publish|post|download|authorize|transfer)\s+"
    r"(?:it|this|that)(?:\s+now)?[.!]?$",
    r"^(?:请|帮我|替我|现在)?\s*(?:发送|提交|支付|付款|购买|下单|删除|发布|下载|授权|转账|充值)"
    r"(?:它|这个|这条|该项|出去)(?:吧|。)?$",
)
_CRITICAL_ENTER_INTENT_PATTERNS = _CRITICAL_FINALIZATION_INTENT_PATTERNS + (
    r"(?:把|将).{0,16}(?:消息|邮件|评论|回复|订单|表单|文件|权限|款项|商品|内容|帖子|申请)"
    r".{0,12}(?:发(?:送|出去)?|提交|支付|付款|购买|下单|删除|发布|下载|授权|转账|充值)",
    r"(?:发|发送)(?:个|一条|这条|该条|一封)?(?:消息|邮件|评论|回复)",
    r"\b(?:send|submit|pay|buy|delete|publish|post|download|authorize|transfer)\b"
    r"(?:\s+\S+){1,12}",
)
_INFORMATIONAL_LINK_PATTERNS = (
    r"帮助|文档|指南|说明|政策|教程|常见问题|了解更多|查看详情|阅读",
    r"\b(?:help|docs?|documentation|guide|manual|policy|tutorial|faq|learn more|read more)\b",
)
_INFORMATIONAL_PATH_SEGMENTS = {
    "article",
    "articles",
    "blog",
    "doc",
    "docs",
    "documentation",
    "faq",
    "guide",
    "guides",
    "help",
    "learn",
    "manual",
    "policy",
    "support",
    "tutorial",
}
_SIDE_EFFECT_PATH_SEGMENTS = {
    "authorize",
    "buy",
    "cancel",
    "checkout",
    "deactivate",
    "delete",
    "destroy",
    "download",
    "logout",
    "pay",
    "purchase",
    "remove",
    "revoke",
    "signout",
    "submit",
    "transfer",
    "unsubscribe",
}


class PolicyViolation(ValueError):
    def __init__(self, code: str, message: str, *, replan_hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.replan_hint = str(replan_hint or "").strip()


def validate_http_url(url: str) -> str:
    value = str(url or "").strip()
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise PolicyViolation("ACTION_REJECTED", "网址格式无效") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise PolicyViolation("ACTION_REJECTED", "仅允许访问 http/https 页面")
    if parsed.username is not None or parsed.password is not None:
        raise PolicyViolation("ACTION_REJECTED", "网址中不得包含认证信息")
    return value


def _matching_line(target: str | None, observation: str) -> str:
    return _observed_matching_line(target, observation) or str(target or "").strip()


def _observation_controls(observation: str) -> list[str]:
    controls: list[str] = []
    for line in str(observation or "").splitlines():
        controls.extend(part.strip() for part in line.split(";") if part.strip())
    return controls


def observed_target_control(target: str | None, observation: str) -> str:
    if not target:
        return ""
    target = target.strip()
    ref_match = re.fullmatch(r"@?e(\d+)", target, flags=re.IGNORECASE)
    if ref_match:
        normalized_ref = f"@e{ref_match.group(1)}"
        ref_pattern = re.compile(
            rf"(?<![\w@]){re.escape(normalized_ref)}(?!\w)",
            flags=re.IGNORECASE,
        )
        for control in _observation_controls(observation):
            if ref_pattern.search(control):
                return control[:1000]
        return ""
    return ""


def observed_controls_match(
    target: str | None,
    before_observation: str,
    after_observation: str,
) -> bool:
    """Return whether a ref still names the same observed semantic control."""
    before = observed_target_control(target, before_observation)
    after = observed_target_control(target, after_observation)
    if not before or not after:
        return False

    def signature(value: str) -> str:
        value = re.sub(r"(?i)(?<![\w@])@e\d+(?!\w)", " ", value)
        # Focus/value/state markers may legitimately change while the user is
        # confirming. The stable role and accessible name define identity.
        value = re.sub(r"\[[^\]\r\n]*\]", " ", value)
        return re.sub(r"\s+", " ", value).strip().casefold()

    return signature(before) == signature(after)


def _observed_matching_line(target: str | None, observation: str) -> str:
    return observed_target_control(target, observation)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def is_sensitive_fill(action: FillAction, observation: str) -> bool:
    context = f"{_matching_line(action.target, observation)} {action.reason}"
    return _matches(_SENSITIVE_PATTERNS, context)


def is_search_fill(action: FillAction, observation: str) -> bool:
    # submit=True performs an Enter press, so only the observed control may
    # authorize the shortcut. A model-authored reason such as "search" cannot
    # turn an ordinary form field into a harmless search box.
    target_line = _observed_matching_line(action.target, observation)
    return bool(target_line and _matches(_SEARCH_FIELD_PATTERNS, target_line))


def requires_critical_confirmation(
    action: AgentAction,
    instruction: str,
    observation: str,
    *,
    recent_fill_target: str | None = None,
    recent_fill_was_search: bool = False,
) -> bool:
    if not isinstance(action, (ClickAction, PressAction)):
        return False
    target = getattr(action, "target", None)
    target_line = _observed_matching_line(target, observation)
    # Model-authored reasons and unmatched selectors are not authorization
    # evidence. Only the observed target control can make a click critical.
    if isinstance(action, ClickAction):
        search_controls = [
            control
            for control in _observation_controls(observation)
            if _TEXT_ENTRY_CONTROL_PATTERN.search(control)
            and _matches(_SEARCH_FIELD_PATTERNS, control)
        ]
        search_button = _matches(
            (r"搜索|查询|检索|查找", r"\b(?:search|query|find)\b"),
            target_line,
        )
        if (
            target_line
            and search_controls
            and _matches(_SEARCH_FIELD_PATTERNS, instruction)
            and not _matches(_CRITICAL_FINALIZATION_INTENT_PATTERNS, instruction)
            and search_button
        ):
            return False
        return bool(
            target_line
            and (
                _matches(_CRITICAL_PATTERNS, target_line)
                or (
                    _matches(_CRITICAL_FINALIZATION_INTENT_PATTERNS, instruction)
                    and _matches(_FINALIZATION_CONTROL_PATTERNS, target_line)
                )
            )
        )
    if not isinstance(action, PressAction) or action.key.strip().casefold() not in _ACTIVATION_KEYS:
        return False
    if not _matches(_CRITICAL_ENTER_INTENT_PATTERNS, instruction):
        return False

    selected_from_recent_fill = bool(
        recent_fill_target and (not target or target == recent_fill_target)
    )
    if not target_line and selected_from_recent_fill:
        target_line = _observed_matching_line(recent_fill_target, observation)
    if target_line and _matches(_SEARCH_FIELD_PATTERNS, target_line):
        return False
    if selected_from_recent_fill and recent_fill_was_search:
        return False
    if selected_from_recent_fill and not target_line:
        # Refs may be regenerated by the snapshot taken after fill. The
        # immediately preceding non-search fill is still sufficient focus
        # evidence for a clearly consequential Enter intent.
        return True

    # A target-less Enter acts on browser focus. Without a recent fill, use a
    # control only when the snapshot has one unambiguous text-entry candidate.
    # Buttons and links elsewhere are ignored so incidental purchase/help copy
    # does not create confirmation prompts.
    action_has_observed_target = bool(target and target_line)
    if not target_line:
        text_entry_lines = [
            control
            for control in _observation_controls(observation)
            if _TEXT_ENTRY_CONTROL_PATTERN.search(control)
        ]
        if len(text_entry_lines) != 1:
            return False
        target_line = text_entry_lines[0]
        if _matches(_SEARCH_FIELD_PATTERNS, target_line):
            return False

    if not _TEXT_ENTRY_CONTROL_PATTERN.search(target_line):
        return False
    if _matches(_CRITICAL_PATTERNS + _CONSEQUENTIAL_TEXT_ENTRY_PATTERNS, target_line):
        return True
    # A generic field is confirmed only when Enter is tied to an observed
    # target or to the field just filled by the agent. This protects unlabeled
    # editors while leaving ambiguous target-less pages unblocked.
    return action_has_observed_target or selected_from_recent_fill


def http_link_requires_confirmation(
    action: ClickAction,
    instruction: str,
    observation: str,
    resolved_url: str,
) -> bool:
    """Classify an HTTP(S) href without treating every action word as a side effect."""
    target_line = observed_target_control(action.target, observation)
    try:
        parsed = urlsplit(resolved_url)
    except ValueError:
        return True
    segments = {
        unquote(part).strip().casefold()
        for part in parsed.path.split("/")
        if part.strip()
    }
    # Exact action endpoints are never exempt merely because a parent path is
    # called docs/help. A URL such as /docs/delete-account remains ordinary
    # informational navigation because "delete-account" is not an endpoint verb.
    if segments & _SIDE_EFFECT_PATH_SEGMENTS:
        return True
    action_query_keys = {"action", "command", "do", "method", "operation", "task"}
    if any(
        key.casefold() in _SIDE_EFFECT_PATH_SEGMENTS
        or (
            key.casefold() in action_query_keys
            and unquote(value).strip().casefold() in _SIDE_EFFECT_PATH_SEGMENTS
        )
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    ):
        return True
    informational = bool(
        _matches(_INFORMATIONAL_LINK_PATTERNS, target_line)
        or segments & _INFORMATIONAL_PATH_SEGMENTS
    )
    if informational:
        return False
    return requires_critical_confirmation(action, instruction, observation)


def user_tab_requested(instruction: str) -> bool:
    return _matches(_EXPLICIT_USER_TAB_PATTERNS, instruction)


def additional_agent_tab_requested(instruction: str) -> bool:
    text = str(instruction or "")
    return not _matches(_NEGATED_ADDITIONAL_TAB_PATTERNS, text) and _matches(
        _ADDITIONAL_AGENT_TAB_PATTERNS,
        text,
    )


def requested_agent_tab_count(instruction: str) -> int | None:
    """Return an explicit total Agent-tab count, if the user supplied one."""
    text = str(instruction or "")
    if not additional_agent_tab_requested(text):
        return None
    match = re.search(r"([2-9]|10)\s*个?标签页", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([2-9]|10)\s+tabs?\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"([二两三四五六七八九十])\s*个?标签页", text)
    if match:
        return _ZH_TAB_NUMBERS[match.group(1)]
    if _matches(
        (
            r"新标签页|新标签|另开(?:一个)?标签|另(?:一|一个)标签页|第二个标签页",
            r"\b(?:new|another|additional|separate) tab\b|\bopen .* in (?:a )?new tab\b",
        ),
        text,
    ):
        return 2
    return None


def requires_persistent_session(instruction: str) -> bool:
    text = str(instruction or "")
    return not _matches(_STOP_MEDIA_PATTERNS, text) and _matches(
        _PERSISTENT_MEDIA_PATTERNS,
        text,
    )


def validate_action(action: AgentAction) -> None:
    if hasattr(action, "url"):
        url = getattr(action, "url")
        if url:
            validate_http_url(url)
    if isinstance(action, TabCreateAction) and action.url:
        validate_http_url(action.url)
