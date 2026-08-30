# UrbanFly desktop runtime

## Entry point

Run `desktop/publish/win-x64/UrbanFly.Desktop.exe` or
`scripts/launch_desktop.ps1`. The native Windows window has no browser address
bar. Loopback HTTP/WebSocket remains the private local transport; changing a URL
does not itself improve performance. The existing engine is reused when healthy.

Build with `scripts/build_desktop.ps1` after closing the native window. It builds
the production frontend, restores .NET dependencies, and publishes the shell.
Build fails instead of overwriting an executable that is running.

Current prerequisites: .NET 9 desktop runtime, WebView2 runtime and the project's
Python environment. This is a framework-dependent Windows build, not a standalone
installer. Dependencies/assets are local at runtime, so ordinary collection does
not require internet; the local sensor view must remain alive. A fresh dependency
restore may need network access.

Optional environment configuration:

- `URBANFLY_ROOT`: project directory (otherwise discovered from executable path).
- `URBANFLY_PYTHON`: Python executable.
- `URBANFLY_START_MINIMIZED=1`: diagnostic/background start; do not use for the
  ordinary visible launcher. It also hides the test window from the taskbar.
- `URBANFLY_DESKTOP_DEVTOOLS=1`: developer tools, disabled by default.

## Lifecycle and safety

The shell checks the runtime health schema, reuses healthy engines and starts
an engine only when the endpoint is not already occupied. HTTP timeout is not
proof that a process died: a live engine is not automatically killed/restarted.
Unknown collector counts are not treated as zero.

Closing the sensor window while a policy client is active (or its health is
unknown) is blocked, because the view still owns real RGB-D. A normal idle close
detaches the shell and leaves the local engine warm; it does not kill the Python
engine. Diagnostic shell processes were terminated only after verifying no
policy client. Backend stdout/stderr files are owned by Python, not a pipe that
breaks when the shell exits.

Logs: `outputs/runtime_logs/desktop_supervisor.log` and
`outputs/runtime_logs/desktop_backend_YYYYMMDD_HHMMSS.log`.

ProcessFailed handling distinguishes a dead browser process (recreate WebView2)
from a dead renderer (reload). Automatic recovery must not imply that a dataset
episode affected by the crash is still valid. Crash injection is not yet tested.

## Performance boundaries

The presentation plane has independent budgets and native minimize signaling.
Idle unchanged cities render on demand. Smooth mode uses display DPR <= 1 and
no bloom; detail mode uses DPR <= 1.5 and bloom. These settings do not change the
160x90 sensor targets, sensor lighting or source geometry. Bridge capture is
driven by real state events, independently of display rAF.

Smooth mode also defaults to presentation-only tile LOD: original L21 tiles
within 200 m of the display camera's distance to tile bounds, L18 overview
outside, with a 240 m exit threshold to avoid boundary flicker. Uncheck
`近精远简（仅展示）` to compare the original full-detail display at the same DPR;
`精细展示` also forces full display detail. Sensor cameras always see the
original full L21 city, regardless of these controls. Overview assets stay
resident, so this trades extra memory for fewer draw calls, not lower memory.

`渲染测速 · 10 秒` forces continuous presentation for a controlled measurement
after one second of warmup; it does not start simulation. Capture or a hidden
window cancels it. Compare with the same camera, canvas and quality, toggling
only LOD. Results include actual FPS, frame-interval P95, draw calls and
triangles; CPU submission time is not GPU completion time.

The shared WebGL context and synchronous sensor readback are still on the main
render thread. Full-city interaction is not yet a proven stable 60 FPS path.
See section 29 of `PROJECT_STATE.md` for actual benchmark/probe results and the
remaining P1 performance gap. Do not resume formal data collection from a
successful hover diagnostic alone.

## Upstream references used

- [Three.js BufferAttribute](https://threejs.org/docs/pages/BufferAttribute.html):
  fixed buffers and dynamic usage instead of repeatedly creating geometry.
- [Microsoft WebView2 process events](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/process-related-events):
  process-specific recovery and view recreation.
