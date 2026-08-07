from __future__ import annotations

import tomllib
from pathlib import Path

from plugin.plugins.browser_skill import BrowserSkillPlugin
from plugin.plugins.browser_skill.runtime import BrowserSkillRuntime


def test_plugin_manifest_and_public_runtime_load() -> None:
    root = Path(__file__).parent.parent
    manifest = tomllib.loads((root / "plugin.toml").read_text(encoding="utf-8"))
    assert manifest["plugin"]["id"] == "browser_skill"
    assert manifest["plugin"]["entry"].endswith(":BrowserSkillPlugin")
    assert manifest["plugin"]["ui"]["panel"][0]["entry"] == "static/panel.html"
    assert manifest["browser_skill"]["bsk_executable"] == "bin/bsk.exe"
    bundled = root / "bin" / "bsk.exe"
    assert bundled.is_file()
    assert bundled.stat().st_size == 10_568_192
    assert BrowserSkillPlugin.__name__ == "BrowserSkillPlugin"
    assert BrowserSkillRuntime.__name__ == "BrowserSkillRuntime"
    context_meta = getattr(
        BrowserSkillPlugin.get_dashboard_ui_context,
        "__neko_ui_context__",
    )
    assert context_meta["id"] == "browser_skill"

    assert (root / "static" / "panel.html").is_file()
    offline_entry = (root / "static" / "index.html").read_text(encoding="utf-8")
    assert "panel.html" in offline_entry

    script = (root / "static" / "app.js").read_text(encoding="utf-8")
    assert "/hosted-ui/context" in script
    assert "setInterval(refreshSilently,1000)" in script

    panel = (root / "static" / "panel.html").read_text(encoding="utf-8")
    style = (root / "static" / "style.css").read_text(encoding="utf-8")
    assert 'id="scrollMaxPages" type="number" min="1" max="1" readonly' in panel
    assert 'for="scrollTokens">滚动观察 token</label><button type="button" class="help-tip open-right"' in panel
    assert 'for="livePageChars">实时页面摘要字符</label><button type="button" class="help-tip open-right"' in panel
    assert 'id="tokenCount"' in panel
    assert "estimated_calls" in script
    assert "markPluginOffline" in script
    assert ".help-tip.open-right::after" in style
