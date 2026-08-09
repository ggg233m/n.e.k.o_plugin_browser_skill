# BrowserSkill 插件 PoC

本 N.E.K.O 用户插件通过本地 `bsk` CLI 及其 Chrome/Edge 扩展，将浏览器任务委托给
腾讯 BrowserSkill 执行。运行时不会联网下载或自动更新 BrowserSkill。

插件直接内置官方 **bsk CLI 0.1.10** 的 macOS Intel/Apple Silicon、Linux x64/ARM64
和 Windows x64 可执行文件。默认 `bsk_executable = "bundled"` 会识别当前平台、校验
对应文件的 SHA-256 并直接执行，不会解压文件或运行错误平台的二进制。`bundled` 只作为
内部配置标记，控制台显示解析后的插件内真实绝对路径。仍需安装协议匹配的 BrowserSkill
浏览器扩展。启用插件前请确认扩展显示浏览器已连接，并配置好 N.E.K.O 的 Agent 模型。
若连接了多个浏览器，请设置 `browser_label`。也可把 `bsk_executable` 改为自定义绝对
路径；留空时从系统 `PATH` 查找 `bsk`。

源码仓库不保存这些大体积二进制。开发或本地打包前运行
`python scripts/fetch_bsk.py`，脚本会从锁定的官方 Release 下载归档，并同时校验归档与
解出的可执行文件 SHA-256。发布 CI 也会执行同一步骤；最终插件包仍内置五个平台文件，
插件运行时不会下载任何内容。

从插件 UI 打开 **BrowserSkill 控制台**，可配置可执行文件、浏览器实例、路由模式、
守护进程启动、会话范围、动作检查点、超时、标签页借用以及视觉/媒体行为。推荐使用
`routing_mode = "auto"`：已知具备工具调用能力的对话模型使用原生主 LLM 工具；
不确定或免费路由也会暴露宿主回退通道。两套入口共享同一运行时并对同一用户回合去重，
因此不会产生两个浏览器队列。`native`、`fallback` 与 `hybrid` 仍保留用于诊断。

修改设置会先关闭本插件的活跃任务与已注册会话，再应用新的运行时。控制面板还可刷新
诊断信息、启动已安装的本地守护进程，以及显式关闭插件会话；但不会下载或安装
BrowserSkill。停用或关闭插件时只清理本插件登记的会话，不会停止可能由其他客户端共享
的 BrowserSkill daemon。

从控制面板保存的设置存储于 BrowserSkill 插件数据目录中，并覆盖 `[browser_skill]`
默认值。保存操作不需要当前存在活动的 N.E.K.O 配置档案。

默认 `session_scope = "plugin"` 与 `reuse_existing_window = true` 使原生 LLM 工具
调用与回退调用复用同一插件专属的 Agent Window。`allow_additional_agent_tabs = false`
还会将模型生成的 `tab_create` 转换为对现有 Agent 标签页的导航。当工作流确实需要时，
用户可在面板中显式启用额外标签页。

本插件默认使用专属的 Agent Window。借用普通用户标签页以及具有外部后果的操作需要
任务级确认。密码、OTP 验证码、CAPTCHA、支付认证以及浏览器权限提示，始终由用户通过
BrowserSkill 的 `request-help` 流程完成。

---

# BrowserSkill plugin PoC

This N.E.K.O user plugin delegates browser tasks to Tencent BrowserSkill through
the local `bsk` CLI and its Chrome/Edge extension. It never downloads or updates
BrowserSkill at runtime.

The plugin directly bundles the official **bsk CLI 0.1.10** executables for macOS
(Intel and Apple Silicon), Linux (x64 and ARM64), and Windows x64. The default
`bsk_executable = "bundled"` detects the current platform, verifies that file's
SHA-256, and executes it directly without extracting anything. The selector
remains internal; the control panel displays the resolved absolute path inside
the plugin. It never attempts to run a binary for another platform. A compatible
BrowserSkill extension is still required. Make sure the extension reports a
connected browser and configure N.E.K.O's Agent model. If more than one browser
is connected, set `browser_label`. You may instead configure an absolute CLI path,
or leave the field empty to find `bsk` on the system `PATH`.

The source repository does not store these large binaries. Run
`python scripts/fetch_bsk.py` before local testing or packaging. The script
downloads pinned official release archives and verifies both the archive and
extracted executable SHA-256. Release CI performs the same step, so the final
plugin package still embeds all five files and never downloads at runtime.

Open **BrowserSkill 控制台** from the plugin UI to configure the executable,
browser instance, routing mode, daemon startup, session scope, action checkpoint,
timeouts, tab borrowing, and vision/media behavior. `routing_mode = "auto"` is
recommended: known tool-capable conversation models use the native main-LLM tool;
uncertain/free routes also expose the host fallback. Both surfaces share one
runtime and deduplicate the same user turn, so they cannot create two browser
queues. `native`, `fallback`, and `hybrid` remain available for diagnostics.

Changing settings closes this plugin's active tasks and registered sessions
before applying the new runtime. The control panel can also refresh diagnostics,
start the already-installed local daemon, and explicitly close plugin sessions;
it never downloads or installs BrowserSkill. Disabling or shutting down the
plugin only cleans up sessions registered by this plugin; it does not stop the
BrowserSkill daemon, which may be shared by other clients.

Settings saved from the control panel are stored in the BrowserSkill plugin data
directory and override the `[browser_skill]` defaults. Saving does not require an
active N.E.K.O configuration profile.

The default `session_scope = "plugin"` and `reuse_existing_window = true` keep
native LLM-tool calls and fallback calls on the same plugin-owned Agent Window.
`allow_additional_agent_tabs = false` also converts a model-generated
`tab_create` into navigation of the existing Agent tab. Users may explicitly
enable additional tabs in the panel when a workflow genuinely needs them.

The plugin uses a dedicated Agent Window by default. Borrowing a normal user tab
and consequential external actions require task-scoped confirmation. Passwords,
OTP codes, CAPTCHAs, payment authentication, and browser permission prompts are
always completed by the user through BrowserSkill's `request-help` flow.
