"""Safety policy independent from model-generated claims."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

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
    if not target:
        return ""
    target = target.strip()
    for line in observation.splitlines():
        if target in line:
            return line[:1000]
    return target


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def is_sensitive_fill(action: FillAction, observation: str) -> bool:
    context = f"{_matching_line(action.target, observation)} {action.reason}"
    return _matches(_SENSITIVE_PATTERNS, context)


def is_search_fill(action: FillAction, observation: str) -> bool:
    context = f"{_matching_line(action.target, observation)} {action.reason}"
    return _matches(_SEARCH_FIELD_PATTERNS, context)


def requires_critical_confirmation(
    action: AgentAction,
    instruction: str,
    observation: str,
) -> bool:
    if not isinstance(action, (ClickAction, PressAction)):
        return False
    target = getattr(action, "target", None)
    context = f"{_matching_line(target, observation)} {action.reason}"
    # The page control itself must look consequential. The broad user goal is
    # supplementary only, otherwise a task such as "read purchase policy"
    # would confirm every harmless click.
    return _matches(_CRITICAL_PATTERNS, context) or (
        _matches(_CRITICAL_PATTERNS, instruction) and _matches((r"确认|提交|发送|支付|删除|publish|submit|send|pay|delete",), context)
    )


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
